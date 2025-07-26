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

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

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
local_debug = True  # Set to True if you want to run on local GPU for debugging

if local_debug == True:
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
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir_string = "/home/rllab/spot_vla/Spot_VLA/dataset/modified_libero_rlds" # Path to Open-X dataset directory
    data_root_dir: Path = Path(data_root_dir_string)        # Path to Open-X dataset directory
    # data_root_dir: Path = Path("/scratch/work/zhangy50/RL/Spot_VLA/dataset/tensorflow_datasets/")        # Path to Open-X dataset directory
    dataset_name: str = "libero_goal_no_noops"                    # already updated in openvla/prismatic/vla/datasets/rlds/oxe/configs.py and transform.py. dont include /!
    run_root_dir: Path = Path("runs")                              # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    if local_debug == True:
        batch_size: int = 1  # 16 is good, 24 to big for H100                                        # Fine-tuning batch size
    else:
        batch_size: int = 16
    max_steps: int = 1000 # 10_000                                        # Max number of fine-tuning steps
    save_steps: int = 1000                                          # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1   # or 4?                             # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 1000 # 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)
    # Validation
    validation_interval = 10       # Validate every 10 optimizer steps
    patience = 5                    # Early stopping after 5 validation checks


    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32 # 64 is good                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    wandb_project: str = "openvla_spot"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "yibo-zhang"                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # fmt: on


def evaluate(model, dataloader, device_id, action_tokenizer):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_l1 = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = model(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )
                loss = output.loss

            # Compute Accuracy and L1 Loss for Logging
            action_logits = output.logits[:, model.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
            
            num_samples = len(batch["input_ids"])

            # Accumulate
            total_loss += loss * num_samples
            total_accuracy += action_accuracy * num_samples
            total_l1 += action_l1_loss * num_samples
            total_samples += num_samples

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_accuracy / total_samples,
        "l1_loss": total_l1 / total_samples,
    }

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
    
def evaluate_dtw_on_trajectories(model, device_id, action_tokenizer, processor, cfg, num_trajectories=5):
    """
    Evaluate DTW distance between predicted and ground truth action trajectories.
    
    Args:
        model: The VLA model
        device_id: Device ID
        action_tokenizer: Action tokenizer
        processor: Model processor
        cfg: Configuration
        num_trajectories: Number of trajectories to evaluate
        step_idx: Current training step (used for random seed)
    
    Returns:
        Average DTW distance across trajectories
    """
    from accelerate import PartialState
    import random
    import time
    distributed_state = PartialState()
    
    # Clear GPU cache before starting evaluation
    torch.cuda.empty_cache()
    
    random.seed(int(time.time()))  # Use current time as seed
    
    # Create episodic dataset for trajectory-level evaluation
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    
    episodic_dataset = EpisodicRLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(model.module.config.image_sizes),
        # shuffle_buffer_size=10000,  # Increased shuffle buffer for better randomization
        train=False,  # Use validation split
        image_aug=False,

    )
    
    model.eval()
    
    if distributed_state.is_main_process:
        print(f"Sampling {num_trajectories} trajectories randomly from validation set...")
    
    # Use reservoir sampling or random skip-based sampling to avoid memory waste
    # This approach samples truly randomly from the entire validation set without loading all trajectories
    
    with torch.no_grad(): 
        total_trajectories = sum(1 for _ in episodic_dataset)
        print(f"Total val traj: {total_trajectories}")
        if total_trajectories < num_trajectories:
            num_trajectories = total_trajectories
            if distributed_state.is_main_process:
                print(f"Adjusted number of trajectories to sample: {num_trajectories}")
        target_indices = sorted(random.sample(range(total_trajectories), num_trajectories))
        sampled_trajectories = []
        current_target_idx = 0
        for traj_idx, trajectory in enumerate(episodic_dataset):
            if current_target_idx >= num_trajectories:
                break  # Got all trajectories we need
            print(f"trajectory {traj_idx}, target_idx={target_indices[current_target_idx]}, {current_target_idx}/{num_trajectories}, length={len(trajectory)}")
            if traj_idx == target_indices[current_target_idx]:
                sampled_trajectories.append(trajectory)
                current_target_idx += 1
                
                if distributed_state.is_main_process:
                    print(f"Collected trajectory {current_target_idx}/{num_trajectories} (index {traj_idx})")
        
        all_ce = []
        all_accuracy = []
        all_cos_distances = []
        all_l1_losses = []
        all_l2_losses = []
        all_dtw = []
        all_ot = []

        chunk_size = 8  # Process 8 steps at a time to avoid OOM
        # Now process the selected trajectories
        # Extract ground truth actions from trajectory
        for trajectory in sampled_trajectories:        
            # Process the trajectory in smaller chunks to avoid OOM
            # Split trajectory into manageable chunks and compute DTW/OT for each chunk
          
            for chunk_start in range(0, len(trajectory), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(trajectory))
                trajectory_chunk = trajectory[chunk_start:chunk_end]
                
                if len(trajectory_chunk) == 0:
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
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output: CausalLMOutputWithPast = model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        pixel_values=batch_pixel_values,
                        labels=batch_labels,
                    )
                    loss = output.loss
                
                # Extract action predictions for this chunk
                action_logits = output.logits[:, model.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
                action_preds = action_logits.argmax(dim=2)
                action_gt = batch_labels[:, 1:].to(action_preds.device)
                mask = action_gt > action_tokenizer.action_token_begin_idx

                if mask.sum() == 0:
                    torch.cuda.empty_cache()
                    continue  # Skip this chunk if no valid action tokens

                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()

                # get orginal action predictions and ground truth
                action_7_pred = action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy()).reshape(-1, 7)
                action_7_gt = action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy()).reshape(-1, 7)
                

                tensor_pred = torch.tensor(action_7_pred, dtype=torch.float32).to(device_id)
                tensor_gt = torch.tensor(action_7_gt, dtype=torch.float32).to(device_id)

                # cosine distance between predicted and ground truth actions, then calculate mean
                cos_distances = torch.nn.functional.cosine_similarity(tensor_pred, tensor_gt, dim=1)
                mean_cos_distance = torch.mean(cos_distances)
                
                # Compute L1 Loss on Predicted (Continuous) Actions
                action_l1_loss = torch.nn.functional.l1_loss(tensor_pred, tensor_gt, reduction='mean')
                action_l2_loss = F.mse_loss(tensor_pred, tensor_gt, reduction='mean')

                all_ce = loss.item()
                all_accuracy.append(action_accuracy.item())
                all_cos_distances.append(mean_cos_distance.item())
                all_l1_losses.append(action_l1_loss.item())
                all_l2_losses.append(action_l2_loss.item())
                
                
                min_len = min(len(action_7_pred), len(action_7_gt))
                # Compute DTW and OT for this chunk directly
                if min_len > 1:
                    chunk_gt_trimmed = np.array(action_7_gt[:min_len])
                    chunk_pred_trimmed = np.array(action_7_pred[:min_len])
                    
                    # Calculate DTW and OT for this chunk
                    chunk_dtw_distance, _ = fastdtw(chunk_gt_trimmed, chunk_pred_trimmed, dist=euclidean)
                    chunk_ot_distance = compute_optimal_transport_distance(chunk_gt_trimmed, chunk_pred_trimmed)
                    
                    # Normalize by chunk length
                    normalized_chunk_dtw = chunk_dtw_distance / min_len
                    normalized_chunk_ot = chunk_ot_distance / min_len

                    all_dtw.append(normalized_chunk_dtw)
                    all_ot.append(normalized_chunk_ot)

                # Clear cache after each chunk to free memory
                torch.cuda.empty_cache()
    return {np.mean(all_ce) if all_ce else float('inf'),
            np.mean(all_accuracy) if all_accuracy else 0.0,
            np.mean(all_cos_distances) if all_cos_distances else -1.0,
            np.mean(all_l1_losses) if all_l1_losses else float('inf'),
            np.mean(all_l2_losses) if all_l2_losses else float('inf'), 
            np.mean(all_dtw) if all_dtw else float('inf'), 
            np.mean(all_ot) if all_ot else float('inf')}


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
        f"{cfg.vla_path.split('/')[-1]}+{cfg.data_root_dir_string.split('/')[-2]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
        f"+shf{cfg.shuffle_buffer_size}"
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

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    print(f'distributed_state: {distributed_state}')
    # if distributed_state:
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )

    '''

    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )
    '''

    # Create train and validation datasets
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        train=True
    )

    val_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=100,  # No shuffling for validation
        image_aug=False,  # Typically no augmentation for validation
        train=False
    )

    # Save dataset statistics (only need to do this once, using train stats)
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)
        

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

    # Validation DataLoader
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,  # Can use different batch size if needed
        sampler=None,
        collate_fn=collator,
        num_workers=0,
        shuffle=False  # Never shuffle validation
    )


    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l2_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_cos_distances = deque(maxlen=cfg.grad_accumulation_steps)

    def action_eval(output):
        # Compute Accuracy and L1 Loss for Logging
        action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
        action_preds = action_logits.argmax(dim=2)
        action_gt = batch["labels"][:, 1:].to(action_preds.device)
        mask = action_gt > action_tokenizer.action_token_begin_idx

        # Compute Accuracy
        correct_preds = (action_preds == action_gt) & mask
        action_accuracy = correct_preds.sum().float() / mask.sum().float()

        # get orginal action predictions and ground truth
        action_7_pred = action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy()).reshape(-1, 7)
        action_7_gt = action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy()).reshape(-1, 7)
        
        tensor_pred = torch.tensor(action_7_pred, dtype=torch.float32).to(device_id)
        tensor_gt = torch.tensor(action_7_gt, dtype=torch.float32).to(device_id)
        
        # cosine distance between predicted and ground truth actions, then calculate mean
        cos_distances = torch.nn.functional.cosine_similarity(tensor_pred, tensor_gt, dim=1)
        mean_cos_distance = torch.mean(cos_distances)
        
        # Compute L1 Loss on Predicted (Continuous) Actions
        action_l1_loss = torch.nn.functional.l1_loss(tensor_pred, tensor_gt)
        action_l2_loss = F.mse_loss(tensor_pred, tensor_gt, reduction='mean')

        return action_accuracy, action_l1_loss, action_l2_loss, mean_cos_distance

    def save_model(subfolder: str = "default", val: float = 0.0) -> None:
        import datetime
        if distributed_state.is_main_process:
            print(f"Saving Model Checkpoint for Step {gradient_step_idx}")
            
            # Write to log file
            log_file = run_dir / "model_saves.log"
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, "a") as f:
                f.write(f"[{timestamp}] Saving {subfolder} at step: {gradient_step_idx}, value: {val:.4f} \n")

            # # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
            # save_dir = adapter_dir if cfg.use_lora else run_dir

            # # Save Processor & Weights
            # vla.module.save_pretrained(save_dir)

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
                    checkpoint_dir = run_dir / f"--{gradient_step_idx}_chkpt--loss-{val_loss:.3f}"
                else:
                    # Save in subfolder
                    checkpoint_dir = run_dir / subfolder
                save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
                os.makedirs(checkpoint_dir, exist_ok=True)
                # Save processor and model weights
                print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")
                processor.save_pretrained(checkpoint_dir)
                vla.module.save_pretrained(checkpoint_dir)
                # merged_vla.save_pretrained(checkpoint_dir)

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
    
    val_accuracy = 0.0
    val_l1 = 0.0
    val_loss = 0.0

    best_val_loss = float('inf')
    best_val_l1 = float('inf')
    best_val_l2 = float('inf')
    best_val_dtw = float('inf')  # Add DTW tracking
    best_val_cos_distance = -1.0  # ideal cosine distance is 1.0, so we start from -1.0
    best_val_accuracy = 0.0

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

            # Compute Accuracy and L1 Loss for Logging
            action_accuracy, action_l1_loss, action_l2_loss, mean_cos_distance = action_eval(output)

            # Store recent train metrics
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())
            recent_l2_losses.append(action_l2_loss.item())
            recent_cos_distances.append(mean_cos_distance.item())
            # recent_dtws.append(mean_dtw)

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

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

            # Push Metrics to W&B (every 10 gradient steps)
            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
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

                    with torch.no_grad():
                        # DTW evaluation on full trajectories (with memory management)
                        torch.cuda.empty_cache()  # Clear cache before DTW evaluation
                        val_loss, val_accuracy, val_l1, val_l2, val_cos_distance, val_dtw, val_ot = evaluate_dtw_on_trajectories(vla, device_id, action_tokenizer, processor, cfg, num_trajectories=3)
                        torch.cuda.empty_cache()  # Clear cache after DTW evaluation

                    # Log validation metrics
                    if distributed_state.is_main_process:
                        wandb.log({
                            "val_cross_entropy": val_loss,
                            "val_action_accuracy": val_accuracy,
                            "val_l1_loss": val_l1,
                            "val_l2_loss": val_l2,
                            "val_vector_cosine": val_cos_distance,
                            "val_normalized_dtw_distance": val_dtw,
                            "val_normalized_ot_distance": val_ot,
                        }, step=gradient_step_idx)
                

                    # Early Stopping Check
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        # patience_counter = 0
                        
                        # Save best model checkpoint
                        if distributed_state.is_main_process:
                            print(f"New best validation loss: {best_val_loss:.4f}")
                            save_model("val_loss", best_val_loss)

                    if val_l1 < best_val_l1:
                        best_val_l1 = val_l1
                        if distributed_state.is_main_process:
                            print(f"New best validation l1 loss: {best_val_l1:.4f}")
                            save_model("val_l1", best_val_l1)

                    if val_l2 < best_val_l2:
                        best_val_l2 = val_l2
                        if distributed_state.is_main_process:
                            print(f"New best validation l2 loss: {best_val_l2:.4f}")
                            save_model("val_l2", best_val_l2)

                    if val_cos_distance > best_val_cos_distance:
                        best_val_cos_distance = val_cos_distance

                        # Save best model checkpoint
                        if distributed_state.is_main_process:
                            print(f"New best validation cosine distance: {best_val_cos_distance:.4f}")
                            save_model("val_cosine_distance", best_val_cos_distance)

                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        
                        # Save best model checkpoint
                        if distributed_state.is_main_process:
                            print(f"New best validation accuracy: {best_val_accuracy:.4f}")
                            save_model("val_accuracy", best_val_accuracy)

                    if val_dtw < best_val_dtw:
                        best_val_dtw = val_dtw
                        
                        # Save best model checkpoint
                        if distributed_state.is_main_process:
                            print(f"New best validation normalized DTW: {best_val_dtw:.4f}")
                            save_model("val_dtw", best_val_dtw)


            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                save_model()


if __name__ == "__main__":
    finetune()
