import os
import time
import torch
import tqdm
import numpy as np
import logging
from typing import Any, Tuple, Dict, Optional

from mmd_model import MMDModel
from mmd_utils import MMDu

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Simplified training pipeline with basic MMD training."""
    
    def __init__(self, args, config: Any, timestamp: str, device: str = "cuda:0"):
        self.args = args
        self.config = config
        self.device = device
        
        self.timestamp = timestamp
        self.net = MMDModel(
            config=self.config,
            state_dict=None,
            device=self.device
        ).to(self.device)
        self._setup_optimizer()
        
        self.sigma, self.sigma0_u, self.ep = None, None, None
        self.best_auroc = 0.0
        self.best_power = 0.0
        self.best_epoch = 0
        
    def _initialize_mmd_parameters(self, args):
        """Initialize MMD parameters."""
        sigma = torch.tensor(args.sigma**2, device=self.device, dtype=torch.float)
        sigma0_u = torch.tensor(args.sigma0**2, device=self.device, dtype=torch.float)
        ep = torch.tensor(args.epsilon**2, device=self.device, dtype=torch.float)
        
        return sigma, sigma0_u, ep
    
    def _setup_optimizer(self):
        """Setup optimizer with trainable MMD parameters."""
        sigma, sigma0_u, ep = self._initialize_mmd_parameters(self.args)
        
        self.epsilonOPT = torch.from_numpy(
            np.ones(1) * np.sqrt(ep.detach().cpu().numpy())
        ).to(self.device, torch.float)
        self.epsilonOPT.requires_grad = True
        
        self.sigmaOPT = torch.from_numpy(
            np.ones(1) * np.sqrt(sigma.detach().cpu().numpy())
        ).to(self.device, torch.float)
        self.sigmaOPT.requires_grad = True
        
        self.sigma0OPT = torch.from_numpy(
            np.ones(1) * np.sqrt(sigma0_u.detach().cpu().numpy())
        ).to(self.device, torch.float)
        self.sigma0OPT.requires_grad = True
        
        self.optimizer = torch.optim.Adam(
            list(self.net.parameters()) + [self.epsilonOPT, self.sigmaOPT, self.sigma0OPT], 
            lr=self.args.lr,
            weight_decay=1e-4
        )
    
    def _train_epoch(self, epoch: int, fea_train_real: torch.Tensor, fea_train_generated: torch.Tensor):
        """Train for one epoch."""
        logger.info(f"Training epoch {epoch}")
        self.net.train()

        fea_train_real_shuffled = fea_train_real[torch.randperm(fea_train_real.shape[0])]
        fea_train_generated_shuffled = fea_train_generated[torch.randperm(fea_train_generated.shape[0])]

        min_len = min(len(fea_train_real_shuffled), len(fea_train_generated_shuffled))
        fea_real = fea_train_real_shuffled[:min_len]
        fea_generated = fea_train_generated_shuffled[:min_len]

        epoch_loss = 0.0
        batch_count = 0
        batch_size = min(self.args.batch_size, len(fea_generated))

        data_size = len(fea_generated)
        num_batches = max(1, data_size // batch_size)

        accumulation_steps = max(1, 32 // batch_size)

        for batch in tqdm.tqdm(range(num_batches), desc="Training batches"):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, data_size)

            inputs = fea_real[start_idx:end_idx].detach()
            x_adv = fea_generated[start_idx:end_idx].detach()

            if inputs.shape[0] != x_adv.shape[0]:
                break

            inputs = inputs.to(self.device, non_blocking=True)
            x_adv = x_adv.to(self.device, non_blocking=True)
            X = torch.cat([inputs, x_adv], dim=0)

            if batch % accumulation_steps == 0:
                self.optimizer.zero_grad()

            outputs = self.net(X)

            ep_t = self.epsilonOPT**2
            sigma_t = self.sigmaOPT**2
            sigma0_u_t = self.sigma0OPT**2

            if self.args.metric == "auroc" and hasattr(self.args, "mmdo_flag") and self.args.mmdo_flag:
                ep_t = torch.ones(1, device=ep_t.device, dtype=ep_t.dtype, requires_grad=True)

            TEMP = MMDu(
                outputs,
                inputs.shape[0],
                X.view(X.shape[0], -1),
                sigma_t,  # type: ignore
                sigma0_u_t,  # type: ignore
                ep_t,  # type: ignore
                coeff_xy=self.args.coeff_xy,
                is_yy_zero=self.args.is_yy_zero,
                is_xx_zero=self.args.is_xx_zero,
            )
            mmd_value_temp = -1 * (TEMP[0])

            if TEMP[1] is not None:
                mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
            else:
                mmd_std_temp = torch.sqrt(torch.tensor(10 ** (-8), device=mmd_value_temp.device))

            STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
            loss = STAT_u / accumulation_steps
            loss.backward()

            if (batch + 1) % accumulation_steps == 0:
                self.optimizer.step()

            epoch_loss += float(loss.detach().cpu()) * accumulation_steps
            batch_count += 1

            last_sigma = float(sigma_t.detach().cpu())
            last_sigma0_u = float(sigma0_u_t.detach().cpu())
            last_ep = float(ep_t.detach().cpu())

            del inputs, x_adv, X, outputs, STAT_u

        if data_size % (batch_size * accumulation_steps) != 0:
            self.optimizer.step()

        self.sigma = last_sigma
        self.sigma0_u = last_sigma0_u
        self.ep = last_ep

        torch.cuda.empty_cache()
        if hasattr(self.net, 'module'):
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
    
    
    def train(
        self, 
        epochs: int,
        data_splits: DataSplits,
        checkpoint_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        fea_train_real = torch.cat(data_splits.train_real, dim=0)
        fea_train_generated = torch.cat(data_splits.train_generated, dim=0)
        
        logger.info(f"Starting training for {epochs} epochs...")
        logger.info(f"Training data: Real={fea_train_real.shape}, Generated={fea_train_generated.shape}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            self._train_epoch(epoch, fea_train_real, fea_train_generated)
            
            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch}: {epoch_time:.2f}s")
            
            if (epoch + 1) % 5 == 0:
                is_best = self._validate(epoch, data_splits, checkpoint_dir)
                if is_best:
                    logger.info(f"New best model at epoch {epoch}")
                    
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        
        return {
            'epochs_completed': epochs,
            'total_time': total_time,
            'best_epoch': self.best_epoch,
            'best_auroc': self.best_auroc,
            'best_power': self.best_power,
            'final_sigma': float(self.sigma),
            'final_sigma0_u': float(self.sigma0_u),
            'final_ep': float(self.ep)
        }
    
    def _validate(self, epoch: int, data_splits: DataSplits, checkpoint_dir: Optional[str]) -> bool:
        """Validation using current model."""      
        tester = MMDTester(self.device)
            
        try:
            fea_real = torch.cat(data_splits.val_real, dim=0)
            fea_generated = torch.cat(data_splits.val_generated, dim=0)
            auroc_score = tester.single_instance_test(
                net=self.net,
                fea_reference=data_splits.reference,
                fea_real=fea_real,
                fea_generated=fea_generated,
                sigma=float(self.sigma) if self.sigma is not None else 0.0,
                sigma0_u=float(self.sigma0_u) if self.sigma0_u is not None else 0.0,
                ep=float(self.ep) if self.ep is not None else 0.0
            )
        except Exception as e:
            logger.error(f"Error in single_instance_test: {e}")
            auroc_score = 0.0
        
        try:
            power_score = tester.two_sample_test(
                net=self.net,
                fea_real_ls=data_splits.val_real,
                fea_generated_ls=data_splits.val_generated,
                sigma=float(self.sigma) if self.sigma is not None else 0.0,
                sigma0_u=float(self.sigma0_u) if self.sigma0_u is not None else 0.0,
                ep=float(self.ep) if self.ep is not None else 0.0,
                N=20  # Reduced for speed
            )
        except Exception as e:
            logger.error(f"Error in two_sample_test: {e}")
            power_score = 0.0

        # try:
        #     eer = tester.three_sample_test(
        #         net=self.net,
        #         fea_real_ls=data_splits.val_real,
        #         fea_generated_ls=data_splits.val_generated,
        #         fea_test_real_ls=[],
        #         fea_test_generated_ls=data_splits.test_generated,
        #         round=5
        #     )
        # except:
        #     eer = float('inf')
        
        # Validation metrics logging removed (TensorBoard)

        # Check if best
        is_best = False
        
        if self.args.metric == 'auroc' and auroc_score > self.best_auroc:
            self.best_auroc = auroc_score
            self.best_epoch = epoch
            is_best = True
        elif self.args.metric == 'power' and power_score > self.best_power:
            self.best_power = power_score
            self.best_epoch = epoch
            is_best = True
        # elif self.args.metric == 'eer' and eer < self.best_auroc:  # Lower EER is better
        #     self.best_auroc = eer
        #     self.best_epoch = epoch
        #     is_best = True
            
        # Always track best of both
        if auroc_score > self.best_auroc:
            self.best_auroc = auroc_score
        if power_score > self.best_power:
            self.best_power = power_score
        # if eer < self.best_auroc:  # Lower EER is better
        #     self.best_auroc = eer
        
        logger.info(f"Validation - AUROC: {auroc_score:.4f}, Power: {power_score:.4f}")
        
        # Save best model
        if is_best and checkpoint_dir:
            self._save_checkpoint(checkpoint_dir, "best_ckpt.pth")
            
        return is_best
    
    def _save_checkpoint(self, checkpoint_dir: str, filename: str):
        """Save model checkpoint."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        state = {
            "net": self.net.state_dict(),
            "sigma": float(self.sigma) if self.sigma is not None else 0.0,
            "sigma0_u": float(self.sigma0_u) if self.sigma0_u is not None else 0.0,
            "ep": float(self.ep) if self.ep is not None else 0.0,
        }
        
        filepath = f"{checkpoint_dir}/{filename}"
        torch.save(state, filepath)

    def enable_multi_gpu(self, num_gpus: int):
        """Enable multi-GPU training using DataParallel."""
        if num_gpus > 1 and torch.cuda.device_count() >= num_gpus:
            logger.info(f"Enabling DataParallel training across {num_gpus} GPUs")
            
            # Ensure model is on cuda:0 for DataParallel
            if not self.device.startswith('cuda:0'):
                logger.info(f"Moving model from {self.device} to cuda:0 for DataParallel")
                self.device = "cuda:0"
                self.net = self.net.to(self.device)
                
                # Also move optimizer parameters to cuda:0
                self.epsilonOPT = self.epsilonOPT.to(self.device)
                self.sigmaOPT = self.sigmaOPT.to(self.device) 
                self.sigma0OPT = self.sigma0OPT.to(self.device)
                
                # Recreate optimizer with moved parameters
                self.optimizer = torch.optim.Adam(
                    list(self.net.parameters()) + [self.epsilonOPT, self.sigmaOPT, self.sigma0OPT], 
                    lr=self.args.lr
                )
            
            self.net = torch.nn.DataParallel(self.net)
            
            return True
        return False
