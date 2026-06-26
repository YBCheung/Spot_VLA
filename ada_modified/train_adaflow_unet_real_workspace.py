if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import hydra
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import random
import tqdm
import torch
import wandb
import json
import pickle
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from adaflow.workspace.base_workspace import BaseWorkspace
from adaflow.dataset.base_dataset import BaseImageDataset
from adaflow.dataset.mimicplay_dataset import *
from adaflow.env_runner.base_image_runner import BaseImageRunner
from adaflow.common.checkpoint_util import TopKCheckpointManager
from adaflow.common.json_logger import JsonLogger
from adaflow.common.pytorch_util import dict_apply, optimizer_to
from adaflow.model.diffusion.ema_model import EMAModel
from adaflow.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

BASE_PATH = '/media/spot/T71'

def quaternion_to_euler(q):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw)
    Input shape: (..., 4) where q = [w, x, y, z]
    Output shape: (..., 3) where angles = [roll, pitch, yaw]
    """
    # Extract quaternion components
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1,
                     np.sign(sinp) * np.pi / 2,
                     np.arcsin(sinp))
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.stack([roll, pitch, yaw], axis=-1)

def euler_to_quaternion(euler):
    """
    Convert Euler angles to quaternion
    Input shape: (..., 3) where euler = [roll, pitch, yaw]
    Output shape: (..., 4) where q = [w, x, y, z]
    """
    roll, pitch, yaw = euler[..., 0], euler[..., 1], euler[..., 2]
    
    # Compute trigonometric functions once
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    # Compute quaternion components
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.stack([w, x, y, z], axis=-1)


class ActionNormalizer:
    def __init__(self):
        self.min_vals = None
        self.max_vals = None
        
    def fit(self, actions):
        """
        Fit the normalizer to the action data
        actions: list of N×7 arrays or a single combined array
        """
        if isinstance(actions, list):
            # Combine all demonstrations
            combined_actions = np.concatenate(actions, axis=0)
        else:
            combined_actions = actions
            
        # Compute min and max values along each dimension
        self.min_vals = np.min(combined_actions, axis=0)
        self.max_vals = np.max(combined_actions, axis=0)
        
        # Handle constant dimensions
        eps = 1e-5
        constant_dims = np.abs(self.max_vals - self.min_vals) < eps
        self.min_vals[constant_dims] = self.max_vals[constant_dims] - 1
        self.max_vals[constant_dims] = self.max_vals[constant_dims] + 1
        
    def normalize(self, actions):
        """
        Normalize actions to [-1, 1] range
        """
        if self.min_vals is None or self.max_vals is None:
            raise ValueError("Normalizer needs to be fitted first")
            
        return 2 * (actions - self.min_vals) / (self.max_vals - self.min_vals) - 1
        
    def denormalize(self, normalized_actions):
        """
        Convert normalized actions back to original range
        """
        if self.min_vals is None or self.max_vals is None:
            raise ValueError("Normalizer needs to be fitted first")
            
        return (normalized_actions + 1) / 2 * (self.max_vals - self.min_vals) + self.min_vals
    
    def save(self, file_path):
        """
        Save the normalizer parameters
        """
        params = {
            'min_vals': self.min_vals,
            'max_vals': self.max_vals
        }
        with open(file_path, 'wb') as f:
            pickle.dump(params, f)
            
    @classmethod
    def load(cls, file_path):
        """
        Load a saved normalizer
        """
        with open(file_path, 'rb') as f:
            params = pickle.load(f)
        
        # Create a new instance
        normalizer = cls()
        # Set the parameters
        normalizer.min_vals = params['min_vals']
        normalizer.max_vals = params['max_vals']
        print("loading data", normalizer.min_vals, normalizer.max_vals)
        return normalizer



class TrainAdaflowUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)    
        random.seed(seed)

        # configure model
        self.model: AdaflowUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: AdaflowUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        if not os.path.exists(cfg.policy.pretrained_rf_ckpt):
            print(f"Pretrained RF checkpoint not found at {cfg.policy.pretrained_rf_ckpt}")
        else: 
            # load pretrained RF checkpoint
            pretrained_rf_ckpt = torch.load(cfg.policy.pretrained_rf_ckpt)
            ema_state_dict = pretrained_rf_ckpt['state_dicts']["ema_model"]
        
            self.model.load_state_dict(ema_state_dict, strict=False)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0
    
    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        #'''
        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        #'''

        # Set parameters
        '''
        dataset_path = "../../../../../2-PDDL/retrieval_PDDL/retrieval_PDDL/dataset/obj_segmentation_demonstration/mimic/image_demo_local.hdf5"
        camera_names = ["agentview_image", "robot0_eye_in_hand_image"]
        batch_size_train = 128
        batch_size_val = 64
        
        # Test batch loading
        normalizer, train_dataloader, val_dataloader, norm_stats = load_mimic_data(
            dataset_path=dataset_path,
            camera_names=camera_names,
            batch_size_train=batch_size_train,
            batch_size_val=batch_size_val,
            min_goal_horizon=50,
            max_goal_horizon=1200,
            max_action_chunk=16,  # Changed to match action_horizon
            train_ratio = 0.99
        )
        '''


        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)
        '''
         # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)
        '''

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                #with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                #        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                #    for batch_idx, batch in enumerate(tepoch):
                
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                               leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    total_batches = len(train_dataloader)
                    one_third_batches = total_batches   # Integer division to get 1/3 of total batches
                    
                    for batch_idx, batch in enumerate(tepoch):
                        if batch_idx >= one_third_batches:
                            break  # Skip to next epoch
                        # device transfer
                        all_batch = {'obs': {key: [] for key in batch[0]['obs'].keys()}, 
                                     'action': []}

                        for b in batch:                  
                            all_batch['action'].append(b['action'])

                            for key in batch[0]['obs'].keys():
                                all_batch['obs'][key].append(b['obs'][key])
                        
                        for key in all_batch['obs'].keys():
                            all_batch['obs'][key] = torch.cat(all_batch['obs'][key], dim=0)

                        # Concatenate action tensors
                        all_batch['action'] = torch.cat(all_batch['action'], dim=0)  # Add this line
    
                        batch = dict_apply(all_batch, lambda x: x.to(device, non_blocking=True))
                            
                        '''
                        for k in batch['obs']:
                            print(k)
                            print(batch[batch['obs'][k].shape])

                        for k in batch['action']:
                            print(k)
                            print(batch[batch['action'][k].shape])
                        a()
                        '''
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss, mse = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0],
                            'mse': mse
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                #if (self.epoch % cfg.training.rollout_every) == 0:# and self.epoch!=0:
                    #print('eval', self.epoch)
                    #runner_log = env_runner.run(policy)
                    #step_log.update(runner_log)


                # run validation
                if (self.epoch % cfg.training.val_every) == 0 and self.epoch!=0:
                    with torch.no_grad():
                        val_losses = list()
                        val_mse_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            # Save one sample from the first batch
                            sample_saved = False
                            

                            for batch_idx, batch in enumerate(tepoch):
                                # device transfer
                                all_batch = {'obs': {key: [] for key in batch[0]['obs'].keys()}, 
                                             'action': []}

                                for b in batch:                  
                                    all_batch['action'].append(b['action'])

                                    for key in batch[0]['obs'].keys():
                                        all_batch['obs'][key].append(b['obs'][key])
                                
                                for key in all_batch['obs'].keys():
                                    all_batch['obs'][key] = torch.cat(all_batch['obs'][key], dim=0)

                                # Concatenate action tensors
                                all_batch['action'] = torch.cat(all_batch['action'], dim=0)  # Add this line
            
                                batch = dict_apply(all_batch, lambda x: x.to(device, non_blocking=True))
                                    
                                loss, mse = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                val_mse_losses.append(mse)
                                
                                # Save first batch sample prediction and ground truth
                                if not sample_saved:
                                    # Get prediction for first item in batch
                                    nobs = self.model.normalizer.normalize(batch['obs'])
                                    nactions = self.model.normalizer['action'].normalize(batch['action'])
                                    batch_size = nactions.shape[0]
                                    horizon = nactions.shape[1]
                                    
                                    # handle different ways of passing observation
                                    local_cond = None
                                    global_cond = None
                                    trajectory = nactions
                                    cond_data = trajectory
                                    if self.model.obs_as_global_cond:
                                        # reshape B, T, ... to B*T
                                        this_nobs = dict_apply(nobs, 
                                            lambda x: x[:,:self.model.n_obs_steps,...].reshape(-1,*x.shape[2:]))
                                        nobs_features = self.model.obs_encoder(this_nobs)
                                        # reshape back to B, Do
                                        global_cond = nobs_features.reshape(batch_size, -1)
                                    else:
                                        # reshape B, T, ... to B*T
                                        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
                                        nobs_features = self.model.obs_encoder(this_nobs)
                                        # reshape back to B, T, Do
                                        nobs_features = nobs_features.reshape(batch_size, horizon, -1)
                                        cond_data = torch.cat([nactions, nobs_features], dim=-1)
                                        trajectory = cond_data.detach()

                                    # Sample noise that we'll add to the images
                                    noise = torch.randn(trajectory.shape, device=trajectory.device)

                                    z_t, t, target = self.model.get_train_tuple(z0=noise, z1=nactions)
                                    
                                    # compute loss mask
                                    velocity_pred, log_sqrt_var_pred = self.model.model(z_t, t.squeeze()*self.model.kwargs["pos_emb_scale"], 
                                        local_cond=local_cond, global_cond=global_cond, freeze_rf=self.model.freeze_rf )
                                    #print(velocity_pred.shape, target.shape)                    
                                    # Create log entry
                                    log_entry = {
                                        'epoch': self.epoch,
                                        'sample_idx': batch_idx,
                                        'prediction': velocity_pred[0].cpu().numpy().tolist(),  # First sample
                                        'ground_truth': target[0].cpu().numpy().tolist(),  # First sample
                                        'diff': (target - velocity_pred).cpu().numpy()[0].tolist()
                                    }
                                    
                                    # Save to log file
                                    log_file = os.path.join(self.output_dir, 'validation_samples.jsonl')
                                    with open(log_file, 'a') as f:
                                        json.dump(log_entry, f)
                                        f.write('\n')
                                    
                                    sample_saved = True
                                
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                        break
                                        
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                        if len(val_mse_losses) > 0:
                            val_mse_losses = torch.mean(torch.tensor(val_mse_losses)).item()
                            # log epoch average validation loss
                            step_log['val_mse'] = val_mse_losses

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0 and self.epoch!=0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0 and self.epoch!=0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        if self.epoch%15==0:
                            tag = 'latest_'+str(self.epoch)
                            self.save_checkpoint(tag=tag)
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_mimic_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None and (self.epoch % 2000)==0:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
    
    def get_evaluate_only_dir(self): 
        print(os.path.join(self.output_dir, f"eval_{self.cfg.evaluate_mode}_inference_step={self.cfg.policy.num_inference_steps}"))
        #return os.path.join(self.output_dir, f"eval_{self.cfg.evaluate_mode}_inference_step={self.cfg.policy.num_inference_steps}_eta={self.cfg.policy.eta}")
        return os.path.join(self.output_dir, "eval_rand_start_inference_step=5_eta=0.5")
   
    def init_model(self, output_dir=None, act_normalizer_path=None): 
        cfg = copy.deepcopy(self.cfg)
      
        ckpt_paths = os.listdir(os.path.join(output_dir, "checkpoints"))

        self.orig_act_normalizer = ActionNormalizer.load(act_normalizer_path)
        print(self.orig_act_normalizer.min_vals)

        for ckpt_path in ckpt_paths: 
            print(ckpt_path)
            if ckpt_path.endswith(".ckpt") and "latest" not in ckpt_path: 
                pass
            else: 
                continue
            
            print("Evaluating checkpoint: ", ckpt_path)
            self.load_checkpoint(path=os.path.join(self.output_dir, "checkpoints", ckpt_path))
            self._output_dir = output_dir
            
            # device transfer
            device = torch.device(cfg.training.device)
            self.model.to(device)
            if self.ema_model is not None:
                self.ema_model.to(device)

            self.policy = self.model
            if cfg.training.use_ema: 
                self.policy = self.ema_model
            self.policy.eval()
            self.policy.reset()


    '''
    def load_test_data(self, fpath):
        agentview_image torch.Size([10, 2, 3, 84, 84])
        robot0_eye_in_hand_image torch.Size([10, 2, 3, 84, 84])
        goal_image torch.Size([10, 2, 3, 84, 84])
        robot0_eef_pos torch.Size([10, 2, 3])
        robot0_eef_quat torch.Size([10, 2, 4])
        robot0_gripper_qpos torch.Size([10, 2, 2])
    '''

    def load_test_prompt(self, fpath):
        from PIL import Image
        import numpy as np

        demo_id = 'demo_3'
        start_id = '00000'
        goal_id = '00160'

        start_im = np.array(Image.open(BASE_PATH+'/2-PDDL/retrieval_PDDL/retrieval_PDDL/dataset/real/real_converted_dataset/'+demo_id+'/agentview_image/'+start_id+'.jpg'))  # Returns RGB format
        hand_im = np.array(Image.open(BASE_PATH+'/2-PDDL/retrieval_PDDL/retrieval_PDDL/dataset/real/real_converted_dataset/'+demo_id+'/robot0_eye_in_hand_image/'+start_id+'.jpg'))  # Returns RGB format
        goal_im = np.array(Image.open(BASE_PATH+'/2-PDDL/retrieval_PDDL/retrieval_PDDL/dataset/real/real_converted_dataset/'+demo_id+'/agentview_image/'+goal_id+'.jpg'))  # Returns RGB format

        ee_pos = h5py.File(fpath, 'r')['data'][demo_id]['obs']['robot0_eef_pos'][int(start_id)]
        ee_quat = h5py.File(fpath, 'r')['data'][demo_id]['obs']['robot0_eef_quat'][int(start_id)]
        gripper = h5py.File(fpath, 'r')['data'][demo_id]['obs']['robot0_gripper_qpos'][int(start_id)]
        action = h5py.File(fpath, 'r')['data'][demo_id]['actions'][int(start_id)]
        
        return start_im, hand_im, goal_im, ee_pos, ee_quat, gripper, action

    def model_predict(self, start_im, hand_im, goal_im, ee_pos, ee_quat, gripper):
        # Convert to torch tensors and reshape
        try:
            obs_dict = {
                'agentview_image': torch.from_numpy(np.transpose(start_im, (2,0,1))).float().unsqueeze(0).unsqueeze(0),  # [1,1,3,300,300]
                'robot0_eye_in_hand_image': torch.from_numpy(np.transpose(hand_im, (2,0,1))).float().unsqueeze(0).unsqueeze(0),  # [1,1,3,300,300]
                'goal_image': torch.from_numpy(np.transpose(goal_im, (2,0,1))).float().unsqueeze(0).unsqueeze(0),  # [1,1,3,300,300]
                'robot0_eef_pos': torch.from_numpy(ee_pos).float().unsqueeze(0).unsqueeze(0),  # [1,1,3]
                'robot0_eef_quat': torch.from_numpy(ee_quat).float().unsqueeze(0).unsqueeze(0),  # [1,1,4]
                'robot0_gripper_qpos': torch.from_numpy(gripper).float().unsqueeze(0).unsqueeze(0),  # [1,1,2]
            }
        except:
            for i in range(2):
                print('obs_dict err')

        #try:
        step_list_chunk_i = []
        # run policy
        with torch.no_grad():
            self.policy.sampling_method = "adaptive"
            action_dict = self.policy.predict_action(obs_dict)

        if "nfe" in action_dict:
            nfe = action_dict["nfe"]
            step_list_chunk_i.append(nfe)
    
        

        # device_transfer
        np_action_dict = dict_apply(action_dict,
            lambda x: x.detach().to('cpu').numpy())
        pred_action = np_action_dict['action'][0]

        return pred_action

    def test_model(self):
        
        start_im, hand_im, goal_im, ee_pos, ee_quat, gripper, test_act  = self.load_test_prompt(BASE_PATH+'/2-PDDL/retrieval_PDDL/retrieval_PDDL/dataset/real/real_converted_dataset.hdf5')

        convert_act = self.model_predict(start_im, hand_im, goal_im, ee_pos, ee_quat, gripper)

       
        # Convert Euler angles to quaternions (assuming pred_action[:, 3:6] contains Euler angles)
        ee_quat = euler_to_quaternion(convert_act[:, 3:6])
        print(ee_quat.shape, convert_act[..., :3].shape, convert_act[..., -1:].shape)
        # Concatenate position, quaternion and gripper components
        final_act = np.concatenate([
            convert_act[..., :3],  # Position
            ee_quat,              # Quaternion
            convert_act[..., -1:] # Gripper
        ], axis=-1)

        print(pred_action, test_act, final_act)


## The short horizon experiment
"""
eval_exp_dir="exps/outputs/2025.01.27/13.48.49_train_adaflow_unet_real_mimic_image_real"
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
python real_eval.py --num_inference_steps=5 --eval_exp_dir=$eval_exp_dir 
"""

"""
def undo_transform_action(self, action):
    raw_shape = action.shape
    if raw_shape[-1] == 20:
        # dual arm
        action = action.reshape(-1,2,10)

    d_rot = action.shape[-1] - 4
    pos = action[...,:3]
    rot = action[...,3:3+d_rot]
    gripper = action[...,[-1]]
    rot = self.rotation_transformer.inverse(rot)
    uaction = np.concatenate([
        pos, rot, gripper
    ], axis=-1)

    if raw_shape[-1] == 20:
        # dual arm
        uaction = uaction.reshape(*raw_shape[:-1], 14)

    return uaction
"""
    