import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time
from tqdm import tqdm

from base import (
    HSI_OCTA_Dataset, Generator, Discriminator,
    PerceptualLoss, SSIMLoss, TrainingConfig,
    init_weights, get_scheduler, save_checkpoint
)
from config_utils import load_config, setup_directories, validate_directories


class Trainer:
    def __init__(self, config_path: str):
        # Load and validate configuration
        self.config = load_config(config_path)
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

    def setup_data(self):
        """Setup datasets and dataloaders."""
        print("Setting up datasets...")

        # Create datasets from the same directory with different splits
        self.train_dataset = HSI_OCTA_Dataset(
            data_dir=str(self.config['data']['data_dir']),
            transform=self.transform,
            split='train',
            target_size=self.config['data']['target_size'],
            val_ratio=self.config['data']['val_ratio'],
            test_ratio=self.config['data']['test_ratio']
        )

        self.val_dataset = HSI_OCTA_Dataset(
            data_dir=str(self.config['data']['data_dir']),
            transform=self.transform,
            split='val',
            target_size=self.config['data']['target_size'],
            val_ratio=self.config['data']['val_ratio'],
            test_ratio=self.config['data']['test_ratio'],
            augment=False  # No augmentation for validation
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

        print(f"Starting training for {self.config['num_epochs']} epochs")
        print(f"Checkpoints will be saved to {self.config['output']['checkpoint_dir']}")

        for epoch in range(self.config['num_epochs']):
            start_time = time.time()

            # Train for one epoch
            train_g_loss, train_d_loss = self.train_epoch(epoch)

            # Validate
            if epoch % self.config['validate_interval'] == 0:
                val_loss = self.validate(epoch)

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    save_checkpoint({
                        'epoch': epoch,
                        'generator_state_dict': self.generator.state_dict(),
                        'discriminator_state_dict': self.discriminator.state_dict(),
                        'optimizer_G_state_dict': self.optimizer_G.state_dict(),
                        'optimizer_D_state_dict': self.optimizer_D.state_dict(),
                        'val_loss': val_loss,
                        'config': self.config,  # Save configuration with checkpoint
                    }, str(self.config['output']['checkpoint_dir'] / 'best_model.pth'))

            # Save regular checkpoint
            if epoch % self.config['save_interval'] == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'generator_state_dict': self.generator.state_dict(),
                    'discriminator_state_dict': self.discriminator.state_dict(),
                    'optimizer_G_state_dict': self.optimizer_G.state_dict(),
                    'optimizer_D_state_dict': self.optimizer_D.state_dict(),
                    'val_loss': val_loss if epoch % self.config['validate_interval'] == 0 else None,
                    'config': self.config,  # Save configuration with checkpoint
                }, str(self.config['output']['checkpoint_dir'] / f'checkpoint_epoch_{epoch}.pth'))

            # Check for early stopping
            if self.config['early_stopping']['enabled']:
                if val_loss > (self.best_val_loss - self.config['early_stopping']['min_delta']):
                    self.early_stop_counter += 1
                    if self.early_stop_counter >= self.config['early_stopping']['patience']:
                        print(f"\nEarly stopping triggered after {epoch + 1} epochs")
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

        # Final cleanup
        self.writer.close()
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Model checkpoints saved in: {self.config['output']['checkpoint_dir']}")
        print(f"Tensorboard logs saved in: {self.config['output']['tensorboard_dir']}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train HSI to OCTA translation model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config JSON file')
    parser.add_argument('--resume', type=str,
                        help='Path to checkpoint for resuming training')

    args = parser.parse_args()

    try:
        # Create trainer instance
        trainer = Trainer(config_path=args.config)

        # Resume from checkpoint if specified
        if args.resume:
            print(f"Resuming training from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=trainer.device)
            trainer.generator.load_state_dict(checkpoint['generator_state_dict'])
            trainer.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            trainer.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            trainer.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            trainer.best_val_loss = checkpoint.get('val_loss', float('inf'))
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")

        # Start training
        trainer.train()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError occurred during training: {str(e)}")
        raise