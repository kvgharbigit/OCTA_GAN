import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time
from tqdm import tqdm
from datetime import datetime
import os
import json
import argparse

from base import (
    Generator, Discriminator,
    PerceptualLoss, SSIMLoss,
    init_weights, get_scheduler, save_checkpoint
)
from hsi_octa_dataset_cropped import HSI_OCTA_Dataset_Cropped
from config_utils import load_config, setup_directories, validate_directories
from visualization_utils import save_sample_visualizations


# Custom JSON encoder to handle Path objects
class PathEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


class Trainer:
    def __init__(self, config_path: str, exp_id: str = None, use_circle_crop: bool = True):
        # Load and validate configuration
        self.config = load_config(config_path)

        # Set option for circle cropping
        self.use_circle_crop = use_circle_crop

        # Update config with circle crop option
        if 'preprocessing' not in self.config:
            self.config['preprocessing'] = {}
        self.config['preprocessing']['circle_crop'] = use_circle_crop
        self.config['preprocessing']['crop_padding'] = 10  # Default padding

        # Set experiment ID
        if exp_id:
            self.exp_id = exp_id
        else:
            # Generate a timestamp-based experiment ID if none provided
            self.exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"Running experiment: {self.exp_id}")
        print(f"Circle cropping: {'enabled' if use_circle_crop else 'disabled'}")

        # Create a parent experiment directory
        self.exp_dir = Path(self.config['output']['base_dir']) / f"experiment_{self.exp_id}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created experiment directory: {self.exp_dir}")

        # Modify output paths to be within the experiment directory
        for key in ['checkpoint_dir', 'results_dir', 'tensorboard_dir', 'visualization_dir']:
            if key in self.config['output']:
                orig_name = Path(self.config['output'][key]).name
                self.config['output'][key] = self.exp_dir / orig_name

        setup_directories(self.config)
        validate_directories(self.config)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize early stopping variables
        self.early_stop_counter = 0
        self.best_val_loss = float('inf')

        # Initialize models
        self.generator = Generator(spectral_channels=self.config['model']['spectral_channels']).to(self.device)
        self.discriminator = Discriminator().to(self.device)

        # Initialize weights
        init_weights(self.generator)
        init_weights(self.discriminator)

        # Setup data normalization
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Initialize loss functions
        self.criterion_gan = nn.BCELoss()
        self.criterion_pixel = nn.L1Loss()
        self.criterion_perceptual = PerceptualLoss().to(self.device)
        self.criterion_ssim = SSIMLoss().to(self.device)

        # Setup optimizers
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config['learning_rate'],
            betas=(self.config['beta1'], self.config['beta2']),
            weight_decay=self.config['weight_decay']
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config['learning_rate'],
            betas=(self.config['beta1'], self.config['beta2']),
            weight_decay=self.config['weight_decay']
        )

        # Setup schedulers
        self.scheduler_G = get_scheduler(self.optimizer_G, self.config)
        self.scheduler_D = get_scheduler(self.optimizer_D, self.config)

        # Setup tensorboard
        self.writer = SummaryWriter(str(self.config['output']['tensorboard_dir']))

        # Add experiment info to tensorboard
        self.writer.add_text('Experiment/ID', self.exp_id, 0)
        self.writer.add_text('Experiment/Config', str(self.config), 0)
        self.writer.add_text('Experiment/CircleCrop', str(use_circle_crop), 0)

        # Save configuration to the experiment directory using the custom encoder
        with open(self.exp_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=4, cls=PathEncoder)

    def setup_data(self):
        """Setup datasets and dataloaders."""
        print("Setting up datasets...")

        # Get crop padding from config
        crop_padding = self.config.get('preprocessing', {}).get('crop_padding', 10)

        # Get approved participants CSV path from config
        approved_csv_path = self.config.get('data', {}).get('approved_csv_path')
        if approved_csv_path:
            print(f"Using approved participants from: {approved_csv_path}")

        # Create datasets from the same directory with different splits
        self.train_dataset = HSI_OCTA_Dataset_Cropped(
            data_dir=str(self.config['data']['data_dir']),
            approved_csv_path=approved_csv_path,  # Use path from config
            transform=self.transform,
            split='train',
            target_size=self.config['data']['target_size'],
            val_ratio=self.config['data']['val_ratio'],
            test_ratio=self.config['data']['test_ratio'],
            crop_padding=crop_padding,
            circle_crop=self.use_circle_crop
        )

        self.val_dataset = HSI_OCTA_Dataset_Cropped(
            data_dir=str(self.config['data']['data_dir']),
            approved_csv_path=approved_csv_path,  # Use path from config
            transform=self.transform,
            split='val',
            target_size=self.config['data']['target_size'],
            val_ratio=self.config['data']['val_ratio'],
            test_ratio=self.config['data']['test_ratio'],
            augment=False,  # No augmentation for validation
            crop_padding=crop_padding,
            circle_crop=self.use_circle_crop
        )

        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")

        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory']
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory']
        )

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()

        total_g_loss = 0
        total_d_loss = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for i, (hsi, octa, _) in enumerate(pbar):
            batch_size = hsi.size(0)

            # Move data to device
            hsi = hsi.to(self.device)
            octa = octa.to(self.device)

            # Train discriminator
            self.optimizer_D.zero_grad()

            real_label = torch.ones(batch_size, 1, 30, 30).to(self.device)
            fake_label = torch.zeros(batch_size, 1, 30, 30).to(self.device)

            # Generate fake image
            fake_octa = self.generator(hsi)

            # Real loss
            real_output = self.discriminator(octa)
            d_real_loss = self.criterion_gan(real_output, real_label)

            # Fake loss
            fake_output = self.discriminator(fake_octa.detach())
            d_fake_loss = self.criterion_gan(fake_output, fake_label)

            # Combined D loss
            d_loss = (d_real_loss + d_fake_loss) * 0.5
            d_loss.backward()
            self.optimizer_D.step()

            # Train generator
            self.optimizer_G.zero_grad()

            fake_output = self.discriminator(fake_octa)
            g_gan_loss = self.criterion_gan(fake_output, real_label)
            g_pixel_loss = self.criterion_pixel(fake_octa, octa) * self.config['lambda_pixel']
            g_perceptual_loss = self.criterion_perceptual(fake_octa, octa) * self.config['lambda_perceptual']
            g_ssim_loss = self.criterion_ssim(fake_octa, octa) * self.config['lambda_ssim']

            g_loss = g_gan_loss + g_pixel_loss + g_perceptual_loss + g_ssim_loss
            g_loss.backward()
            self.optimizer_G.step()

            # Update statistics
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

            pbar.set_postfix({
                'G_loss': g_loss.item(),
                'D_loss': d_loss.item()
            })

            if i % self.config['logging']['print_interval'] == 0:
                step = epoch * len(self.train_loader) + i
                self.writer.add_scalar('Train/G_loss', g_loss.item(), step)
                self.writer.add_scalar('Train/D_loss', d_loss.item(), step)
                self.writer.add_scalar('Train/Pixel_loss', g_pixel_loss.item(), step)
                self.writer.add_scalar('Train/Perceptual_loss', g_perceptual_loss.item(), step)
                self.writer.add_scalar('Train/SSIM_loss', g_ssim_loss.item(), step)

        return total_g_loss / len(self.train_loader), total_d_loss / len(self.train_loader)

    def validate(self, epoch: int):
        """Run validation."""
        self.generator.eval()
        self.discriminator.eval()

        total_val_loss = 0

        with torch.no_grad():
            for hsi, octa, _ in self.val_loader:
                hsi = hsi.to(self.device)
                octa = octa.to(self.device)

                fake_octa = self.generator(hsi)

                # ... continuing from where your code cut off
                val_pixel_loss = self.criterion_pixel(fake_octa, octa)
                val_ssim_loss = self.criterion_ssim(fake_octa, octa)
                val_loss = val_pixel_loss + val_ssim_loss

                total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(self.val_loader)
            self.writer.add_scalar('Validation/Loss', avg_val_loss, epoch)

            return avg_val_loss

    def train(self):
        """Main training loop."""
        self.setup_data()

        # Record start time
        self.start_time = time.time()
        self.start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.completed_epochs = 0
        self.early_stopped = False

        print(f"Starting training for {self.config['num_epochs']} epochs")
        print(f"Checkpoints will be saved to {self.config['output']['checkpoint_dir']}")

        # Create a directory for visual samples
        vis_dir = self.exp_dir / 'visual_samples'
        vis_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.config['num_epochs']):
            start_time = time.time()

            # Train for one epoch
            train_g_loss, train_d_loss = self.train_epoch(epoch)
            self.completed_epochs = epoch + 1

            # Visualize samples at configured interval
            save_images_interval = self.config.get('logging', {}).get('save_images_interval',
                                                                      5)  # Default to 5 if not specified
            save_images_start = self.config.get('logging', {}).get('save_images_interval_start',
                                                                   0)  # Default to 0 if not specified

            if epoch >= save_images_start and epoch % save_images_interval == 0:
                save_sample_visualizations(
                    generator=self.generator,
                    val_loader=self.val_loader,
                    device=self.device,
                    writer=self.writer,
                    epoch=epoch,
                    output_dir=vis_dir
                )

            # Validate
            if epoch % self.config['validate_interval'] == 0:
                val_loss = self.validate(epoch)

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss

                    # Convert Path objects to strings for serialization
                    serializable_config = json.loads(json.dumps(self.config, cls=PathEncoder))

                    save_checkpoint({
                        'epoch': epoch,
                        'exp_id': self.exp_id,
                        'generator_state_dict': self.generator.state_dict(),
                        'discriminator_state_dict': self.discriminator.state_dict(),
                        'optimizer_G_state_dict': self.optimizer_G.state_dict(),
                        'optimizer_D_state_dict': self.optimizer_D.state_dict(),
                        'val_loss': val_loss,
                        'config': serializable_config,
                        'circle_crop': self.use_circle_crop
                    }, str(self.config['output']['checkpoint_dir'] / f'best_model.pth'))

            # Save regular checkpoint
            if epoch % self.config['save_interval'] == 0:
                # Convert Path objects to strings for serialization
                serializable_config = json.loads(json.dumps(self.config, cls=PathEncoder))

                save_checkpoint({
                    'epoch': epoch,
                    'exp_id': self.exp_id,
                    'generator_state_dict': self.generator.state_dict(),
                    'discriminator_state_dict': self.discriminator.state_dict(),
                    'optimizer_G_state_dict': self.optimizer_G.state_dict(),
                    'optimizer_D_state_dict': self.optimizer_D.state_dict(),
                    'val_loss': val_loss if epoch % self.config['validate_interval'] == 0 else None,
                    'config': serializable_config,
                    'circle_crop': self.use_circle_crop
                }, str(self.config['output']['checkpoint_dir'] / f'checkpoint_epoch_{epoch}.pth'))

            # Check for early stopping
            if self.config['early_stopping']['enabled']:
                if val_loss > (self.best_val_loss - self.config['early_stopping']['min_delta']):
                    self.early_stop_counter += 1
                    if self.early_stop_counter >= self.config['early_stopping']['patience']:
                        print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                        self.early_stopped = True
                        break
                else:
                    self.early_stop_counter = 0

            # Update learning rate
            self.scheduler_G.step()
            self.scheduler_D.step()

            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f'\nEpoch {epoch + 1}/{self.config["num_epochs"]} Summary:')
            print(f'Time: {epoch_time:.2f}s')
            print(f'Generator Loss: {train_g_loss:.4f}')
            print(f'Discriminator Loss: {train_d_loss:.4f}')
            if epoch % self.config['validate_interval'] == 0:
                print(f'Validation Loss: {val_loss:.4f}')
                print(f'Best Validation Loss: {self.best_val_loss:.4f}')
            print(f'Learning Rate: {self.optimizer_G.param_groups[0]["lr"]:.6f}')
            print()

        # Calculate total training time
        total_training_time = time.time() - self.start_time

        # Generate and save training summary
        summary_path = self.generate_training_summary(total_training_time)

        # Final cleanup
        self.writer.close()
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Model checkpoints saved in: {self.config['output']['checkpoint_dir']}")
        print(f"Tensorboard logs saved in: {self.config['output']['tensorboard_dir']}")
        print(f"Training summary saved to: {summary_path}")
        print(f"All experiment files saved in: {self.exp_dir}")

    def generate_training_summary(self, training_time_seconds):
        """
        Generate a comprehensive training summary text file with all important information.

        Args:
            training_time_seconds: Total training time in seconds
        """
        # Convert training time to hours, minutes, seconds
        hours, remainder = divmod(training_time_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        summary_path = self.exp_dir / 'training_summary.txt'
        print(f"\nGenerating training summary at: {summary_path}")

        with open(summary_path, 'w') as f:
            # Write header
            f.write("=" * 80 + "\n")
            f.write(f"TRAINING SUMMARY: {self.exp_id}\n")
            f.write("=" * 80 + "\n\n")

            # Write experiment information
            f.write("EXPERIMENT INFORMATION\n")
            f.write("-" * 50 + "\n")
            f.write(f"Experiment ID: {self.exp_id}\n")
            f.write(f"Start time: {self.start_time_str}\n")
            f.write(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s\n")
            f.write(f"Circle crop enabled: {self.use_circle_crop}\n\n")

            # Write dataset information
            f.write("DATASET INFORMATION\n")
            f.write("-" * 50 + "\n")
            f.write(f"Data directory: {self.config['data']['data_dir']}\n")

            if hasattr(self.train_dataset, 'approved_ids') and self.train_dataset.approved_ids:
                f.write(f"Approved participants file: {self.train_dataset.approved_csv_path}\n")
                f.write(f"Total approved participants: {len(self.train_dataset.approved_ids)}\n")

                # Get unique patient IDs from the file pairs
                train_patients = set(pair['patient_id'] for pair in self.train_dataset.file_pairs)
                val_patients = set(pair['patient_id'] for pair in self.val_dataset.file_pairs)
                all_patients = train_patients.union(val_patients)

                f.write(f"Approved participants found: {len(all_patients)}\n")
                f.write(f"Approved participants missing: {len(self.train_dataset.approved_ids) - len(all_patients)}\n")

            f.write(f"Training samples: {len(self.train_dataset)}\n")
            f.write(f"Validation samples: {len(self.val_dataset)}\n")
            f.write(f"Image size: {self.config['data']['target_size']}x{self.config['data']['target_size']}\n\n")

            # Write model information
            f.write("MODEL INFORMATION\n")
            f.write("-" * 50 + "\n")
            f.write(f"Spectral channels: {self.config['model']['spectral_channels']}\n")

            # Count generator parameters
            gen_params = sum(p.numel() for p in self.generator.parameters())
            disc_params = sum(p.numel() for p in self.discriminator.parameters())
            f.write(f"Generator parameters: {gen_params:,}\n")
            f.write(f"Discriminator parameters: {disc_params:,}\n\n")

            # Write training parameters
            f.write("TRAINING PARAMETERS\n")
            f.write("-" * 50 + "\n")
            f.write(f"Batch size: {self.config['batch_size']}\n")
            f.write(f"Learning rate: {self.config['learning_rate']}\n")
            f.write(f"Number of epochs: {self.config['num_epochs']}\n")
            f.write(f"Early stopping: {self.config['early_stopping']['enabled']}\n")
            if self.config['early_stopping']['enabled']:
                f.write(f"  - Patience: {self.config['early_stopping']['patience']}\n")
                f.write(f"  - Min delta: {self.config['early_stopping']['min_delta']}\n")

            # Write loss weights
            f.write("\nLOSS WEIGHTS\n")
            f.write("-" * 50 + "\n")
            f.write(f"Pixel loss (L1): {self.config['lambda_pixel']}\n")
            f.write(f"Perceptual loss: {self.config['lambda_perceptual']}\n")
            f.write(f"SSIM loss: {self.config['lambda_ssim']}\n")
            f.write(f"Adversarial loss: {self.config['lambda_adv']}\n\n")

            # Write final training results
            f.write("TRAINING RESULTS\n")
            f.write("-" * 50 + "\n")
            f.write(f"Completed epochs: {self.completed_epochs}\n")
            if hasattr(self, 'early_stopped') and self.early_stopped:
                f.write("Training stopped early: Yes\n")
            f.write(f"Best validation loss: {self.best_val_loss:.6f}\n")
            f.write(f"Final learning rate: {self.optimizer_G.param_groups[0]['lr']:.6f}\n\n")

            # Write output directories
            f.write("OUTPUT DIRECTORIES\n")
            f.write("-" * 50 + "\n")
            f.write(f"Base directory: {self.exp_dir}\n")
            f.write(f"Checkpoints: {self.config['output']['checkpoint_dir']}\n")
            f.write(f"TensorBoard logs: {self.config['output']['tensorboard_dir']}\n")
            f.write(
                f"Visualizations: {self.config['output']['visualization_dir'] if 'visualization_dir' in self.config['output'] else self.exp_dir / 'visual_samples'}\n\n")

            # Write recommendations for evaluation
            f.write("NEXT STEPS\n")
            f.write("-" * 50 + "\n")
            f.write("To evaluate this model, run:\n")
            f.write(f"python evaluation-script.py --config eval_config.json --exp_id {self.exp_id}_eval\n\n")
            f.write("Make sure your eval_config.json points to the best model checkpoint at:\n")
            f.write(f"{self.config['output']['checkpoint_dir'] / 'best_model.pth'}\n\n")

            # Write footer
            f.write("=" * 80 + "\n")
            f.write("End of training summary\n")
            f.write("=" * 80 + "\n")

        print(f"Training summary saved to: {summary_path}")
        return summary_path

if __name__ == '__main__':
        # Create argument parser correctly
        parser = argparse.ArgumentParser(description='Train HSI to OCTA translation model')
        parser.add_argument('--config', type=str, required=True,
                            help='Path to config JSON file')
        parser.add_argument('--resume', type=str, default=None,
                            help='Path to checkpoint for resuming training')
        parser.add_argument('--exp_id', type=str, default=None,
                            help='Experiment ID (will default to timestamp if not provided)')
        parser.add_argument('--circle_crop', action='store_true', default=True,
                            help='Enable circle detection and cropping')
        parser.add_argument('--no_circle_crop', action='store_true',
                            help='Disable circle detection and cropping')

        args = parser.parse_args()

        # Determine circle crop option (default is True, but can be disabled with --no_circle_crop)
        use_circle_crop = True
        if args.no_circle_crop:
            use_circle_crop = False

        try:
            # Create trainer instance
            trainer = Trainer(config_path=args.config, exp_id=args.exp_id, use_circle_crop=use_circle_crop)

            # Resume from checkpoint if specified
            if args.resume:
                print(f"Resuming training from checkpoint: {args.resume}")
                checkpoint = torch.load(args.resume, map_location=trainer.device)
                trainer.generator.load_state_dict(checkpoint['generator_state_dict'])
                trainer.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                trainer.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
                trainer.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
                trainer.best_val_loss = checkpoint.get('val_loss', float('inf'))

                # If resuming, we can use the experiment ID from the checkpoint
                if 'exp_id' in checkpoint and not args.exp_id:
                    trainer.exp_id = checkpoint['exp_id']
                    print(f"Using experiment ID from checkpoint: {trainer.exp_id}")

                # Check if the checkpoint was trained with circle cropping
                if 'circle_crop' in checkpoint:
                    saved_crop = checkpoint['circle_crop']
                    if saved_crop != use_circle_crop:
                        print(f"WARNING: Checkpoint was trained with circle_crop={saved_crop}, "
                              f"but current setting is circle_crop={use_circle_crop}")

                start_epoch = checkpoint['epoch'] + 1
                print(f"Resuming from epoch {start_epoch}")

            # Start training
            trainer.train()

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"\nError occurred during training: {str(e)}")
            raise