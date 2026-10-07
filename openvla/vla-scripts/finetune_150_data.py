"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...

    
"""

from datetime import datetime
import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np

import draccus
import torch
import torch.distributed as dist
import torch.nn.functional as F
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset, EpisodicRLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import compute_stratified_split_indices, save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

import ot

# === Local GPU ===
# Note: If you want to run on local CPU, comment out the following lines.in terminal, and run following command on terminal 
'''
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=12355
'''
local_debug = False  # Set to True if you want to run on local GPU for debugging

if local_debug:
    import torch.distributed as dist

    dist.init_process_group(
        backend='nccl',  # or 'gloo' for CPU
        init_method='env://'
    )
    # Sane Defaults
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
# === Local GPU ===


# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"    # Path to OpenVLA model (on HuggingFace Hub)
    adapter_path: str = "None"
    if local_debug:
        data_root_dir_string = "/home/zhangy50/RL/Spot_VLA/dataset/modified_libero_rlds"
    else:
        data_root_dir_string = "/scratch/work/zhangy50/RL/Spot_VLA/dataset/modified_libero_rlds"
    data_root_dir: Path = Path(data_root_dir_string)        # Path to Open-X dataset directory
    # data_root_dir: Path = Path("/scratch/work/zhangy50/RL/Spot_VLA/dataset/tensorflow_datasets/")        # Path to Open-X dataset directory
    dataset_name: str = "libero_goal_no_noops"                    # already updated in openvla/prismatic/vla/datasets/rlds/oxe/configs.py and transform.py. dont include /!
    run_root_dir: Path = Path("runs")                              # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    if local_debug:
        batch_size: int = 2  # 16 is good, 24 to big for H100, for H200, 32 is a good target                                     # Fine-tuning batch size
    else:
        batch_size: int = 56 # 40 only Power 85%, MEM 74%, 56 is good for H200, 98% MEM, 86% Power. 64& too big. 
    max_steps: int = 50000 # 10_000                                        # Max number of fine-tuning steps
    save_steps: int = 1000                                          # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1   # or 4?                             # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 1000 # 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = False                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)
    resume_optimizer_state: bool = True                              # Save/restore AdamW moments (m, v) alongside each adapter
                                                                    #   checkpoint so continual-finetuning phases resume without
                                                                    #   the optimizer-reset loss spike (see optimizer setup below).
                                                                    #   NOTE: writes a ~0.6 GB optimizer_state.pt per saved
                                                                    #   checkpoint (2 fp32 moments x ~80M LoRA params). Set False
                                                                    #   if disk/inode quota is tight (cf. the disk-quota-exceeded
                                                                    #   crash on 2026-07-09).
    # Validation
    validation_flag: bool = False            # Whether to run validation during training
    validation_interval = 500       # Validate every 500 optimizer steps
    patience = 3                    # Early stopping after 5 validation checks

    # Data Split (stratified by task, fixed across runs so val/test stay comparable across checkpoints)
    train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    data_split_seed: int = 0
    # Optionally cap the *train* split at this many trajectories (evenly across tasks). The full
    # 80/10/10 split gives 342 train for libero_goal; set to 150 to train on a 15/task subset while
    # leaving the 44 val / 42 test trajectories unchanged (so eval stays comparable). None => no cap.
    max_train_trajectories: Optional[int] = 150

    if local_debug:
        chunk_size: int = 8              # Trajectory segment, Process 8 steps at a time to avoid OOM
    else:
        chunk_size: int = 8             # Trajectory segment, Process 16 steps at a time to avoid OOM

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32 # 64 is good                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance
    ot_epsilon: float = 0.05                                        # Entropic regularization for OT evaluation

    # Tracking Parameters
    wandb_project: str = "openvla_spot"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "yibo-zhang"                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # fmt: on



def compute_optimal_transport_distance(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Compute optimal transport distance between two trajectory sequences.
    
    Args:
        gt: Ground truth trajectory actions (shape: [num_steps, 7])
        pred: Predicted trajectory actions (shape: [num_steps, 7])
    
    Returns:
        Optimal transport distance
    """
    # Flatten the trajectories to 1D for optimal transport computation
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()
    
    # Normalize to obtain probability distributions (histograms)
    a = np.ones(len(gt_flat)) / len(gt_flat)
    b = np.ones(len(pred_flat)) / len(pred_flat)
    
    # Cost matrix: squared Euclidean distance
    M = ot.dist(gt_flat.reshape(-1, 1), pred_flat.reshape(-1, 1), metric='euclidean') ** 2
    
    return ot.emd2(a, b, M)
    

@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`. Dataset path: {cfg.data_root_dir}")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"time{datetime.now().strftime('%Y-%m-%d_%H-%M-%S') }"  # Add timestamp for uniqueness
        f"+{cfg.vla_path.split('/')[-1]}+{cfg.data_root_dir_string.split('/')[-2]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
        f"+shf{cfg.shuffle_buffer_size}"
        f"+adapter-{cfg.adapter_path.split('/')[-1] if cfg.adapter_path else 'None'}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> target specific linear modules to avoid Identity layers
    if cfg.use_lora:
        if cfg.adapter_path and os.path.exists(cfg.adapter_path):
            # Load existing LoRA adapter weights
            print(f"Loading LoRA adapter weights from {cfg.adapter_path}")
            vla = PeftModel.from_pretrained(vla, cfg.adapter_path, is_trainable=True)
            vla.print_trainable_parameters()
        else:
            print("Initializing new LoRA adapter")
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=min(cfg.lora_rank, 16),
                lora_dropout=cfg.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                init_lora_weights="gaussian",
            )
            vla = get_peft_model(vla, lora_config)
            vla.print_trainable_parameters()

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    print(f'distributed_state: {distributed_state}')
    # if distributed_state:
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    #
    # ------------------------------------------------------------------------------------------------
    # [ISSUE] Optimizer-reset shock across continual-finetuning phases
    #   When a run resumes from a saved LoRA adapter (`cfg.adapter_path` set), the *weights* are
    #   restored correctly, but historically a brand-new AdamW was created here with empty moment
    #   estimates (m, v), and training ran at the full constant LR (5e-4, no warmup). Restarting Adam
    #   cold on an already-converged minimum kicks the weights straight off it within the first ~10
    #   steps. Diagnosed on the 34000-step chain: the loaded checkpoint scored train-loss 0.155 on its
    #   very first batch (before any update), then spiked back to ~0.51 by step 500 and only re-
    #   converged to ~0.14 over the next 10k steps. Because this shock dominates the start of every
    #   phase, each phase replays the same "blow up, then recover" curve regardless of how good the
    #   input checkpoint was -- wiping out most of the accumulated continual-training benefit and
    #   making successive phases look like repeated finetunes from the same checkpoint.
    #
    # [FIX] Persist Adam's moments alongside each adapter checkpoint (see save_model) and restore them
    #   here on resume, so momentum/variance carry over across phases and the loaded minimum is
    #   preserved instead of being destroyed. Backward compatible: if no optimizer_state.pt is found
    #   (first phase, or an older checkpoint) we simply fall back to the previous cold-start behavior.
    # ------------------------------------------------------------------------------------------------
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    if cfg.resume_optimizer_state and cfg.use_lora and cfg.adapter_path and os.path.exists(cfg.adapter_path):
        optimizer_state_path = os.path.join(cfg.adapter_path, "optimizer_state.pt")
        if os.path.exists(optimizer_state_path):
            print(f"Restoring optimizer state (Adam moments) from {optimizer_state_path}")
            optimizer.load_state_dict(torch.load(optimizer_state_path, map_location=f"cuda:{device_id}"))
        else:
            print(f"[WARN] resume_optimizer_state=True but no optimizer_state.pt found at "
                  f"{cfg.adapter_path}; starting Adam cold (expect a transient loss spike at resume).")

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)


    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    

    # Stratified train/val/test split by task, fixed across runs (cached alongside the dataset) so every
    # checkpoint and every finetuning run is compared against the exact same held-out trajectories.
    split_indices = compute_stratified_split_indices(
        name=cfg.dataset_name,
        data_dir=str(cfg.data_root_dir),
        language_key="language_instruction",
        split_fractions=cfg.train_val_test_split,
        seed=cfg.data_split_seed,
        max_train_trajectories=cfg.max_train_trajectories,
    )

    # Create train, validation, and test datasets
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        train=True,
        split_indices=split_indices,
        split_name="train",
    )


    episodic_dataset = EpisodicRLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        # shuffle_buffer_size=10000,  # Increased shuffle buffer for better randomization
        train=False,  # Use validation split
        image_aug=False,
        split_indices=split_indices,
        split_name="val",
    )

    # Held out until final reporting -- not touched anywhere in the training/early-stopping loop below.
    episodic_test_dataset = EpisodicRLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        train=False,
        image_aug=False,
        split_indices=split_indices,
        split_name="test",
    )

    # Save dataset statistics (only need to do this once, using train stats)
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    # Per-dimension action z-score normalization (train-set stats only), applied before any distance
    # metric is computed: \bar{a}_d = (a_d - mu_d) / (sigma_d + eps). Translation, rotation, and gripper
    # commands live on very different numerical scales, so metrics computed on raw decoded actions would
    # be dominated by whichever dimension happens to have the largest range.
    _train_action_stats = next(iter(train_dataset.dataset_statistics.values()))["action"]
    ACTION_MEAN = np.array(_train_action_stats["mean"], dtype=np.float32)
    ACTION_STD = np.array(_train_action_stats["std"], dtype=np.float32)
    ACTION_NORM_EPS = 1e-6

    def normalize_actions(actions: np.ndarray) -> np.ndarray:
        return (actions - ACTION_MEAN) / (ACTION_STD + ACTION_NORM_EPS)


    # Create collator (shared between train and val)
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, 
        processor.tokenizer.pad_token_id, 
        padding_side="right"
    )

    # Train DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
        # shuffle=True  # Shuffle at DataLoader level if needed
    )



    # Note: We don't need a DataLoader for episodic_dataset because it returns 
    # trajectories (lists of steps) rather than individual steps, so we iterate 
    # over it directly in the validation loop


    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")
        # Persist this run's wandb run id next to its checkpoints (mirrors lerobot's
        # finetune_pi05_metric_action_norm.py) so checkpoint_selection.py's wandb fallback can
        # auto-discover it later without a manual `wandb.Api().runs(...)` search -- each resumed
        # fine-tuning stage here gets its own run_dir (exp_id embeds a fresh timestamp) and its
        # own wandb run whose logged step count restarts from 0, matching that run_dir's own
        # step_<k> checkpoint numbering exactly.
        (run_dir / "wandb_run_id.txt").write_text(wandb.run.id)

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l2_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_cos_distances = deque(maxlen=cfg.grad_accumulation_steps)

    def action_eval(action_7_pred, action_7_gt):

        tensor_pred = torch.tensor(action_7_pred, dtype=torch.float32).to(device_id)
        tensor_gt = torch.tensor(action_7_gt, dtype=torch.float32).to(device_id)

        # Cosine distance: 1 - mean cosine similarity across action vectors.
        cos_similarities = torch.nn.functional.cosine_similarity(tensor_pred, tensor_gt, dim=1, eps=1e-8)
        mean_cos_distance = 1.0 - torch.mean(cos_similarities)

        # Compute L1 and L2 losses on predicted continuous actions.
        action_l1_loss = torch.nn.functional.l1_loss(tensor_pred, tensor_gt)
        action_l2_loss = F.mse_loss(tensor_pred, tensor_gt, reduction='mean')

        return action_l1_loss.item(), action_l2_loss.item(), mean_cos_distance.item()

    def compute_cosine_distance(action_7_pred, action_7_gt):
        tensor_pred = torch.tensor(action_7_pred, dtype=torch.float32).to(device_id)
        tensor_gt = torch.tensor(action_7_gt, dtype=torch.float32).to(device_id)
        cos_similarities = torch.nn.functional.cosine_similarity(tensor_pred, tensor_gt, dim=1, eps=1e-8)
        return float(1.0 - torch.mean(cos_similarities).item())

    def compute_exact_dtw(action_7_pred, action_7_gt):
        pred = np.asarray(action_7_pred, dtype=np.float64)
        gt = np.asarray(action_7_gt, dtype=np.float64)
        if pred.size == 0 or gt.size == 0:
            return 0.0, 0.0, 0

        n, m = len(pred), len(gt)
        dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
        back = np.full((n + 1, m + 1), -1, dtype=np.int8)
        dp[0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.linalg.norm(pred[i - 1] - gt[j - 1]) ** 2  # c(x,y) = ||x-y||_2^2, per tex
                prev_options = (dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
                step = int(np.argmin(prev_options))
                dp[i, j] = cost + prev_options[step]
                back[i, j] = step

        path_len = 0
        i, j = n, m
        while i > 0 and j > 0:
            path_len += 1
            step = back[i, j]
            if step == 0:
                i -= 1
            elif step == 1:
                j -= 1
            else:
                i -= 1
                j -= 1

        normalized_dtw = float(dp[n, m] / max(path_len, 1))
        return normalized_dtw, float(dp[n, m]), path_len

    def compute_entropic_ot(action_7_pred, action_7_gt):
        pred = np.asarray(action_7_pred, dtype=np.float64)
        gt = np.asarray(action_7_gt, dtype=np.float64)
        if pred.size == 0 or gt.size == 0:
            return 0.0

        a = np.ones(len(pred), dtype=np.float64) / len(pred)
        b = np.ones(len(gt), dtype=np.float64) / len(gt)
        cost_matrix = ot.dist(pred, gt, metric='sqeuclidean')  # c(x,y) = ||x-y||_2^2, per tex
        return float(ot.sinkhorn2(a, b, cost_matrix, reg=cfg.ot_epsilon))

    def compute_action_log_likelihood(action_logits, action_gt, mask):
        flat_mask = mask.reshape(-1)
        num_actions = int(flat_mask.sum().item())
        if num_actions == 0:
            return 0.0, 0

        log_probs = F.log_softmax(action_logits, dim=-1)
        flat_log_probs = log_probs.reshape(-1, log_probs.shape[-1])[flat_mask]
        flat_targets = action_gt.reshape(-1)[flat_mask].long()
        token_log_likelihood = flat_log_probs.gather(1, flat_targets.unsqueeze(1)).squeeze(1)
        return float(token_log_likelihood.sum().item()), num_actions

    def save_model(subfolder: str = "default", val: float = 0.0) -> None:
        import datetime
        if distributed_state.is_main_process:
            print(f"Saving Model Checkpoint for Step {gradient_step_idx}")
            
            # Write to log file
            log_file = run_dir / "model_saves.log"
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, "a") as f:
                f.write(f"[{timestamp}] Saving {subfolder} at step: {gradient_step_idx}, value: {val:.4f} \n")

        # Wait for processor and adapter weights to be saved by main process
        dist.barrier()

        # Merge LoRA weights into model backbone for faster inference
        #   =>> Note that merging is slow and can be done post-hoc to speed up training
        if cfg.use_lora:
            # base_vla = AutoModelForVision2Seq.from_pretrained(
            #     cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
            # )
            # merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
            # merged_vla = merged_vla.merge_and_unload()
            if distributed_state.is_main_process:
                if subfolder == "default" and cfg.save_latest_checkpoint_only == False:
                    # Prepare to save checkpoint in new directory
                    checkpoint_dir = run_dir / f"--{gradient_step_idx}_chkpt-{val:.4f}"
                else:
                    # Save in subfolder
                    checkpoint_dir = run_dir / subfolder
                    
                os.makedirs(checkpoint_dir, exist_ok=True)
                save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
                # Save processor and model weights
                print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")
                processor.save_pretrained(checkpoint_dir)
                vla.module.save_pretrained(checkpoint_dir)
                # merged_vla.save_pretrained(checkpoint_dir)

                # Persist AdamW moments (m, v) next to the adapter so a later continual-finetuning
                # phase that resumes from this `step_*` dir can restore the optimizer and avoid the
                # reset shock documented at the optimizer setup above. Lives beside
                # adapter_model.safetensors, so a resume only needs `cfg.adapter_path` = this dir.
                # WARNING: ~0.6 GB per checkpoint -- gated by cfg.resume_optimizer_state because the
                # per-validation checkpointing here already exhausted the disk/inode quota once.
                if cfg.resume_optimizer_state:
                    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer_state.pt")

                # Log successful save to file
                log_file = run_dir / "model_saves.log"
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      
                print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")
        else:
            if distributed_state.is_main_process:
                processor.save_pretrained(run_dir)
                
                # Log successful save to file
                log_file = run_dir / "model_saves.log"
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
             
        # Block on Main Process Checkpointing
        dist.barrier()

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_dataloader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )
                loss = output.loss

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # get orginal action predictions and ground truth, then normalize before any distance metric
            action_7_pred = normalize_actions(action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy()).reshape(-1, 7))
            action_7_gt = normalize_actions(action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy()).reshape(-1, 7))


            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            
            # Push Metrics to W&B (every 10 gradient steps)
            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                
                # Compute Accuracy and L1 Loss for Logging
                action_l1_loss, action_l2_loss, mean_cos_distance = action_eval(action_7_pred, action_7_gt)
                
                # Store recent train metrics
                recent_losses.append(loss.item())
                recent_action_accuracies.append(action_accuracy.item())
                recent_l1_losses.append(action_l1_loss)
                recent_l2_losses.append(action_l2_loss)
                recent_cos_distances.append(mean_cos_distance)
                # recent_dtws.append(mean_dtw)

                # Compute gradient step index

                # Compute smoothened train metrics
                #   =>> Equal to current step metrics when not using gradient accumulation
                #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
                smoothened_loss = sum(recent_losses) / len(recent_losses)
                smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
                smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)  
                smoothened_l2_loss = sum(recent_l2_losses) / len(recent_l2_losses)
                smoothened_cos_distance = sum(recent_cos_distances) / len(recent_cos_distances)
                # smoothened_dtw = sum(recent_dtws) / len(recent_dtws)
                # smoothen, recent losses is defined as deque, no need to manually deque.  

                print(smoothened_loss, type(smoothened_loss))
                wandb.log(
                    {
                        "train_cross_entropy": smoothened_loss,
                        "action_accuracy": smoothened_action_accuracy,
                        "l1_loss": smoothened_l1_loss,
                        "l2_loss": smoothened_l2_loss,
                        "vector_cosine": smoothened_cos_distance,
                    },
                    step=gradient_step_idx,
                )


            # Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress.update()

                # ----- Validation & early stopping ----- 

                if (gradient_step_idx + 1) % cfg.validation_interval == 0:
                    # Persist a full checkpoint at every validation step (no per-metric best/tolerance
                    # copies) so theta_0,...,theta_K are all independently reloadable later. Checkpoint
                    # selection happens afterward by plotting the logged offline metrics vs. step and
                    # picking the corresponding `step_{k}` checkpoint directory.
                    save_model(f"step_{gradient_step_idx}", val=smoothened_loss)

                    if cfg.validation_flag == True:
                            
                        vla.eval()
                        print(f"Validating at step {gradient_step_idx}...")
                        val_loss_list = []
                        val_accuracy_list = []
                        val_l1_list = []
                        val_l2_list = []
                        val_cos_distance_list = []
                        val_dtw_list = []
                        val_ot_list = []
                        val_log_likelihood_list = []
                        val_nll_list = []
                        traj_action_preds = []
                        traj_action_gt = []

                        # Use the full validation set every time so m̄(k) is comparable across checkpoints
                        for traj_i, trajectory in enumerate(episodic_dataset):

                            # Process the trajectory in smaller chunks to avoid OOM
                            # Split trajectory into manageable chunks and compute DTW/OT for each chunk

                            traj_loss = 0.0
                            traj_accuracy = 0.0
                            traj_log_lik = 0.0
                            traj_num_actions = 0
                            traj_action_preds = []
                            traj_action_gt = []
                            
                            for chunk_start in range(0, len(trajectory), cfg.chunk_size):
                                chunk_end = min(chunk_start + cfg.chunk_size, len(trajectory))
                                trajectory_chunk = trajectory[chunk_start:chunk_end]
                                chunk_length = len(trajectory_chunk)

                                if chunk_length == 0:
                                    continue
                                
                                # Process this chunk as a batch
                                batch_input_ids = torch.stack([step["input_ids"] for step in trajectory_chunk]).to(device_id)
                                
                                # Check if attention_mask exists, if not create it (all ones)
                                if "attention_mask" in trajectory_chunk[0]:
                                    batch_attention_mask = torch.stack([step["attention_mask"] for step in trajectory_chunk]).to(device_id)
                                else:
                                    batch_attention_mask = torch.ones_like(batch_input_ids).to(device_id)
                                
                                batch_pixel_values = torch.stack([step["pixel_values"] for step in trajectory_chunk]).to(torch.bfloat16).to(device_id)
                                batch_labels = torch.stack([step["labels"] for step in trajectory_chunk])
                                
                                # Process chunk with gradient checkpointing to save memory
                                with torch.no_grad():
                                    with torch.autocast("cuda", dtype=torch.bfloat16):
                                        output: CausalLMOutputWithPast = vla(
                                            input_ids=batch_input_ids,
                                            attention_mask=batch_attention_mask,
                                            pixel_values=batch_pixel_values,
                                            labels=batch_labels,
                                        )
                                        chunk_loss = output.loss

                                        # Extract action predictions for this chunk
                                        action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
                                        action_preds = action_logits.argmax(dim=2)
                                        action_gt = batch_labels[:, 1:].to(action_preds.device)
                                        mask = action_gt > action_tokenizer.action_token_begin_idx

                                        correct_preds = (action_preds == action_gt) & mask
                                        chunk_accuracy = correct_preds.sum().float() / mask.sum().float()

                                        chunk_log_lik, chunk_num_actions = compute_action_log_likelihood(
                                            action_logits, action_gt, mask
                                        )

                                        # get orginal action predictions and ground truth, then normalize before any distance metric
                                        action_7_pred = normalize_actions(action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy()).reshape(-1, 7))
                                        action_7_gt = normalize_actions(action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy()).reshape(-1, 7))

                                traj_loss += chunk_loss.item() * chunk_length
                                traj_accuracy += chunk_accuracy.item() * chunk_length
                                traj_log_lik += chunk_log_lik
                                traj_num_actions += chunk_num_actions
                                traj_action_gt.append(action_7_gt)
                                traj_action_preds.append(action_7_pred)

                            traj_action_gt = np.concatenate(traj_action_gt) if len(traj_action_gt) > 0 else np.empty((0, 7))
                            traj_action_preds = np.concatenate(traj_action_preds) if len(traj_action_preds) > 0 else np.empty((0, 7))

                            traj_l1 = float(np.mean(np.abs(traj_action_preds - traj_action_gt))) if traj_action_gt.size else 0.0
                            traj_l2 = float(np.mean((traj_action_preds - traj_action_gt) ** 2)) if traj_action_gt.size else 0.0
                            traj_cos_distance = compute_cosine_distance(traj_action_preds, traj_action_gt) if traj_action_gt.size else 0.0
                            traj_dtw, traj_dtw_raw, traj_dtw_path_len = compute_exact_dtw(traj_action_preds, traj_action_gt)
                            traj_ot = compute_entropic_ot(traj_action_preds, traj_action_gt)
                            traj_nll = -traj_log_lik / traj_num_actions if traj_num_actions > 0 else 0.0

                            traj_length = len(trajectory)
                            val_loss_list.append(traj_loss / traj_length)
                            val_accuracy_list.append(traj_accuracy / traj_length)
                            val_l1_list.append(traj_l1)
                            val_l2_list.append(traj_l2)
                            val_cos_distance_list.append(traj_cos_distance)
                            val_dtw_list.append(traj_dtw)
                            val_ot_list.append(traj_ot)
                            val_log_likelihood_list.append(traj_log_lik)
                            val_nll_list.append(traj_nll)
                            

                        print(f"step: {gradient_step_idx}, "
                            f"CE loss: {val_loss_list}, "
                            f"acc: {val_accuracy_list}, "
                            f"L1 loss: {val_l1_list}, "
                            f"L2 loss: {val_l2_list}, "
                            f"cosine distance: {val_cos_distance_list}, "
                            f"DTW: {val_dtw_list}, "
                            f"OT: {val_ot_list}, "
                            f"log_likelihood: {val_log_likelihood_list}, "
                            f"nll: {val_nll_list}")

                        val_ce = np.mean(val_loss_list)
                        val_l1 = np.mean(val_l1_list)
                        val_l2 = np.mean(val_l2_list)
                        val_accuracy = np.mean(val_accuracy_list)
                        val_cos_distance = np.mean(val_cos_distance_list)
                        val_dtw = np.mean(val_dtw_list)
                        val_ot = np.mean(val_ot_list)
                        val_log_likelihood = np.mean(val_log_likelihood_list)
                        val_nll = np.mean(val_nll_list)
                        # Log validation metrics
                        val_metrics = {
                            "val_ce_loss": val_ce,
                            "val_accuracy": val_accuracy,
                            "val_l1_loss": val_l1,
                            "val_l2_loss": val_l2,
                            "val_cosine_distance": val_cos_distance,
                            "val_dtw_distance": val_dtw,
                            "val_ot_distance": val_ot,
                            "val_log_likelihood": val_log_likelihood,
                            "val_nll": val_nll,
                        }
                        if distributed_state.is_main_process:
                            wandb.log(val_metrics, step=gradient_step_idx)

                            # Local, network-independent record of every checkpoint's offline metrics
                            # (tex section 9's Metric Curve Table), so checkpoint_selection.py's
                            # early-stopping rule (tex eq. 437-451) doesn't require wandb API access.
                            metrics_history_path = run_dir / "metrics_history.json"
                            history = (
                                json.loads(metrics_history_path.read_text())
                                if metrics_history_path.exists()
                                else []
                            )
                            history.append({"step": gradient_step_idx, **val_metrics})
                            metrics_history_path.write_text(json.dumps(history, indent=2))

                    
            # # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            # if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
            #     save_model(val=smoothened_loss)


if __name__ == "__main__":
    finetune()
