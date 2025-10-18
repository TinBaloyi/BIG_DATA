"""
Training Script for BraTS Glioma Segmentation
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
import argparse
from datetime import datetime
import json

# Import custom modules (assuming they're in the same project)
# from models.unet import get_model
# from training.dataset import get_dataloaders
# from training.losses import CombinedLoss, DiceLoss
# from training.metrics import BraTSMetrics


class BraTSTrainer:
    """
    Trainer class for BraTS Glioma Segmentation
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup directories
        self.setup_directories()
        
        # Initialize model
        self.model = self.build_model()
        
        # Initialize loss function
        self.criterion = self.build_loss()
        
        # Initialize optimizer
        self.optimizer = self.build_optimizer()
        
        # Initialize scheduler
        self.scheduler = self.build_scheduler()
        
        # Initialize data loaders
        self.dataloaders = self.build_dataloaders()
        
        # Initialize metrics
        self.metrics_calculator = BraTSMetrics()
        
        # Tensorboard writer
        self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_dice = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
    
    def setup_directories(self):
        """Create necessary directories"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = self.config.get('experiment_name', 'brats_experiment')
        
        self.exp_dir = Path(self.config['output_dir']) / f'{exp_name}_{timestamp}'
        self.checkpoint_dir = self.exp_dir / 'checkpoints'
        self.log_dir = self.exp_dir / 'logs'
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.exp_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)
    
    def build_model(self):
        """Build and initialize model"""
        model_config = self.config['model']
        model = get_model(
            model_name=model_config['name'],
            in_channels=model_config['in_channels'],
            num_classes=model_config['num_classes'],
            base_channels=model_config['base_channels'],
            depth=model_config.get('depth', 4)
        )
        
        # Move to device
        model = model.to(self.device)
        
        # Multi-GPU training
        if torch.cuda.device_count() > 1 and self.config.get('use_multi_gpu', False):
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
        
        # Load pretrained weights if specified
        if 'pretrained_weights' in self.config:
            self.load_checkpoint(self.config['pretrained_weights'], load_optimizer=False)
        
        return model
    
    def build_loss(self):
        """Build loss function"""
        loss_type = self.config['training'].get('loss', 'combined')
        
        if loss_type == 'dice':
            return DiceLoss()
        elif loss_type == 'combined':
            return CombinedLoss(
                dice_weight=self.config['training'].get('dice_weight', 0.5),
                ce_weight=self.config['training'].get('ce_weight', 0.5)
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def build_optimizer(self):
        """Build optimizer"""
        opt_config = self.config['training']['optimizer']
        
        if opt_config['name'] == 'adamw':
            return AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['name']}")
    
    def build_scheduler(self):
        """Build learning rate scheduler"""
        sched_config = self.config['training'].get('scheduler', {})
        
        if sched_config.get('name') == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs'],
                eta_min=sched_config.get('min_lr', 1e-6)
            )
        elif sched_config.get('name') == 'reduce_on_plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            return None
    
    def build_dataloaders(self):
        """Build data loaders"""
        return get_dataloaders(
            processed_data_dir=self.config['data']['processed_dir'],
            splits_dir=self.config['data']['splits_dir'],
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['training'].get('num_workers', 4),
            patch_size=tuple(self.config['data']['patch_size'])
        )
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        progress_bar = tqdm(self.dataloaders['train'], 
                           desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch['image'].to(self.device)
            targets = batch['segmentation'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('gradient_clip', None):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log to tensorboard
            global_step = self.current_epoch * len(self.dataloaders['train']) + batch_idx
            self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
        
        # Calculate average loss
        avg_loss = epoch_loss / len(self.dataloaders['train'])
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
        epoch_loss = 0.0
        all_metrics = []
        
        progress_bar = tqdm(self.dataloaders['val'], desc='Validation')
        
        for batch in progress_bar:
            # Move data to device
            images = batch['image'].to(self.device)
            targets = batch['segmentation'].to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            epoch_loss += loss.item()
            
            # Get predictions
            predictions = torch.argmax(outputs, dim=1)
            
            # Calculate metrics for each sample in batch
            for i in range(predictions.shape[0]):
                pred_np = predictions[i].cpu().numpy()
                target_np = targets[i].cpu().numpy()
                
                metrics = self.metrics_calculator.calculate_metrics(pred_np, target_np)
                all_metrics.append(metrics)
        
        # Calculate average loss
        avg_loss = epoch_loss / len(self.dataloaders['val'])
        self.val_losses.append(avg_loss)
        
        # Aggregate metrics
        aggregated_metrics = self.metrics_calculator.aggregate_metrics(all_metrics)
        self.val_metrics.append(aggregated_metrics)
        
        return avg_loss, aggregated_metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_dice': self.best_val_dice,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint with Dice: {self.best_val_dice:.4f}")
        
        # Save periodic checkpoint
        if (self.current_epoch + 1) % self.config['training'].get('save_freq', 10) == 0:
            periodic_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch + 1}.pth'
            torch.save(checkpoint, periodic_path)
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_dice = checkpoint.get('best_val_dice', 0.0)
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self):
        """Main training loop"""
        num_epochs = self.config['training']['num_epochs']
        
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate()
            
            # Get mean Dice for main regions
            mean_dice = np.mean([
                val_metrics.get('ET_Dice_mean', 0),
                val_metrics.get('TC_Dice_mean', 0),
                val_metrics.get('WT_Dice_mean', 0)
            ])
            
            # Log to tensorboard
            self.writer.add_scalar('Train/Loss', train_loss, epoch)
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Val/MeanDice', mean_dice, epoch)
            
            for metric_name, metric_value in val_metrics.items():
                if 'mean' in metric_name:
                    self.writer.add_scalar(f'Val/{metric_name}', metric_value, epoch)
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/LearningRate', current_lr, epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Mean Dice: {mean_dice:.4f}")
            print(f"  ET Dice: {val_metrics.get('ET_Dice_mean', 0):.4f}")
            print(f"  TC Dice: {val_metrics.get('TC_Dice_mean', 0):.4f}")
            print(f"  WT Dice: {val_metrics.get('WT_Dice_mean', 0):.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Update learning rate scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(mean_dice)
                else:
                    self.scheduler.step()
            
            # Save checkpoint
            is_best = mean_dice > self.best_val_dice
            if is_best:
                self.best_val_dice = mean_dice
            
            self.save_checkpoint(is_best=is_best)
            
            # Save training history
            history = {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_metrics': self.val_metrics,
                'best_val_dice': self.best_val_dice
            }
            
            with open(self.exp_dir / 'training_history.json', 'w') as f:
                json.dump(history, f, indent=2)
        
        print("\nTraining completed!")
        print(f"Best validation Dice: {self.best_val_dice:.4f}")
        
        self.writer.close()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train BraTS Glioma Segmentation Model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create trainer
    trainer = BraTSTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()