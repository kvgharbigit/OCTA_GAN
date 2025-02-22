import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time
import json
from tqdm import tqdm
import numpy as np

from base import (
    HSI_OCTA_Dataset, Generator, Discriminator,
    PerceptualLoss, SSIMLoss, TrainingConfig,
    init_weights, get_scheduler, save_checkpoint
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time
import json
from tqdm import tqdm
import numpy as np
import sys

from base import (
    HSI_OCTA_Dataset, Generator, Discriminator,
    PerceptualLoss, SSIMLoss, TrainingConfig,
    init_weights, get_scheduler, save_checkpoint
)


class Trainer:
    def __init__(self, config_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = TrainingConfig()

        # Load custom config if provided
        if config_path:
            with open(config_path, 'r') as f:
                self.custom_config = json.load(f)
                # Update training parameters from config
                for key, value in self.custom_config.items():
                    if not key.startswith('//') and not isinstance(value, dict):
                        setattr(self.config, key, value)

        # Initialize models
        self.generator = Generator().to(self.device)
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
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay
        )

        # Setup schedulers
        self.scheduler_G = get_scheduler(self.optimizer_G, self.config)
        self.scheduler_D = get_scheduler(self.optimizer_D, self.config)

        # Setup tensorboard
        self.writer = SummaryWriter()

        # Initialize best validation loss
        self.best_val_loss = float('inf')

    def setup_data(self, data_dir: str = None, batch_size: int = None):
        # Use data directory from config if not provided
        if data_dir is None and hasattr(self, 'custom_config'):
            data_dir = self.custom_config.get('data', {}).get('train_dir')
            if data_dir is None:
                raise ValueError("Data directory must be provided either in config or as argument")

        if batch_size is None:
            batch_size = self.config.batch_size

        print(f"Setting up datasets from {data_dir}")

        # Add extensive debugging for dataset setup
        print("Debugging dataset creation:")
        print(f"Data directory contents:")
        data_path = Path(data_dir)
        for item in data_path.iterdir():
            print(f"  - {item}")

        # Create datasets
        self.train_dataset = HSI_OCTA_Dataset(
            data_dir=data_dir,
            transform=self.transform,
            split='train',
            target_size=500,
            val_ratio=0.5,  # Explicitly set validation ratio
            test_ratio=0.15  # Explicitly set test ratio
        )

        self.val_dataset = HSI_OCTA_Dataset(
            data_dir=data_dir,
            transform=self.transform,
            split='val',
            target_size=500,
            val_ratio=0.5,  # Match the ratio from train dataset
            test_ratio=0.15,
            augment=False
        )

        # Debug dataset sizes
        print(f"\nTrain dataset size: {len(self.train_dataset)} samples")
        print(f"Validation dataset size: {len(self.val_dataset)} samples")

        # If validation dataset is empty, adjust strategy
        if len(self.val_dataset) == 0:
            print("\nWARNING: Validation dataset is empty!")
            print("Fallback strategy: Using train dataset for validation")
            self.val_dataset = self.train_dataset

            # If train dataset is also empty, raise an error
            if len(self.train_dataset) == 0:
                raise ValueError(f"No valid data found in {data_dir}. Please check your data directory.")

        # Get number of workers from config if available
        num_workers = 4
        if hasattr(self, 'custom_config'):
            num_workers = self.custom_config.get('hardware', {}).get('num_workers', 4)

        print(f"Creating dataloaders with {num_workers} workers")

        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        print(f"Train loader size: {len(self.train_loader)} batches")
        print(f"Val loader size: {len(self.val_loader)} batches")

    def train_epoch(self, epoch: int):
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

            # Prepare labels
            real_label = torch.ones(batch_size, 1, 30, 30).to(self.device)
            fake_label = torch.zeros(batch_size, 1, 30, 30).to(self.device)

            # -----------------
            #  Train Generator
            # -----------------
            self.optimizer_G.zero_grad()

            # Generate fake image
            fake_octa = self.generator(hsi)

            # Discriminator output on fake images
            fake_output = self.discriminator(fake_octa)

            # Calculate losses
            g_gan_loss = self.criterion_gan(fake_output, real_label)
            g_pixel_loss = self.criterion_pixel(fake_octa, octa) * self.config.lambda_pixel
            g_perceptual_loss = self.criterion_perceptual(fake_octa, octa) * self.config.lambda_perceptual
            g_ssim_loss = self.criterion_ssim(fake_octa, octa) * self.config.lambda_ssim

            # Combined loss
            g_loss = g_gan_loss + g_pixel_loss + g_perceptual_loss + g_ssim_loss

            # Backward pass
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.config.gradient_clip)
            self.optimizer_G.step()

            # -------------------
            #  Train Discriminator
            # -------------------
            self.optimizer_D.zero_grad()

            # Real loss
            real_output = self.discriminator(octa)
            d_real_loss = self.criterion_gan(real_output, real_label)

            # Fake loss
            fake_output = self.discriminator(fake_octa.detach())
            d_fake_loss = self.criterion_gan(fake_output, fake_label)

            # Combined loss
            d_loss = (d_real_loss + d_fake_loss) * 0.5

            # Backward pass
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config.gradient_clip)
            self.optimizer_D.step()

            # Update statistics
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

            # Update progress bar
            pbar.set_postfix({
                'G_loss': g_loss.item(),
                'D_loss': d_loss.item()
            })

            # Log to tensorboard
            if i % self.config.print_interval == 0:
                step = epoch * len(self.train_loader) + i
                self.writer.add_scalar('Train/G_loss', g_loss.item(), step)
                self.writer.add_scalar('Train/D_loss', d_loss.item(), step)
                self.writer.add_scalar('Train/Pixel_loss', g_pixel_loss.item(), step)
                self.writer.add_scalar('Train/Perceptual_loss', g_perceptual_loss.item(), step)
                self.writer.add_scalar('Train/SSIM_loss', g_ssim_loss.item(), step)

        return total_g_loss / len(self.train_loader), total_d_loss / len(self.train_loader)

    def validate(self, epoch: int):
        self.generator.eval()
        self.discriminator.eval()

        total_val_loss = 0

        with torch.no_grad():
            for hsi, octa, _ in self.val_loader:
                hsi = hsi.to(self.device)
                octa = octa.to(self.device)

                # Generate fake image
                fake_octa = self.generator(hsi)

                # Calculate validation loss (using only L1 and SSIM for simplicity)
                val_pixel_loss = self.criterion_pixel(fake_octa, octa)
                val_ssim_loss = self.criterion_ssim(fake_octa, octa)
                val_loss = val_pixel_loss + val_ssim_loss

                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(self.val_loader)
        self.writer.add_scalar('Validation/Loss', avg_val_loss, epoch)

        return avg_val_loss

    def train(self, data_dir: str, checkpoint_dir: str):
        """Main training loop."""
        # Setup data
        self.setup_data(data_dir)

        # Create checkpoint directory
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Checkpoints will be saved to {checkpoint_dir}")

        # Training loop
        for epoch in range(self.config.num_epochs):
            start_time = time.time()

            # Train for one epoch
            train_g_loss, train_d_loss = self.train_epoch(epoch)

            # Validate
            if epoch % self.config.validate_interval == 0:
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
                    }, str(checkpoint_dir / 'best_model.pth'))

            # Save regular checkpoint
            if epoch % self.config.save_interval == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'generator_state_dict': self.generator.state_dict(),
                    'discriminator_state_dict': self.discriminator.state_dict(),
                    'optimizer_G_state_dict': self.optimizer_G.state_dict(),
                    'optimizer_D_state_dict': self.optimizer_D.state_dict(),
                    'val_loss': val_loss if epoch % self.config.validate_interval == 0 else None,
                }, str(checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'))

            # Update learning rate
            self.scheduler_G.step()
            self.scheduler_D.step()

            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f'\nEpoch {epoch} Summary:')
            print(f'Time: {epoch_time:.2f}s')
            print(f'Generator Loss: {train_g_loss:.4f}')
            print(f'Discriminator Loss: {train_d_loss:.4f}')
            if epoch % self.config.validate_interval == 0:
                print(f'Validation Loss: {val_loss:.4f}')
            print()

        self.writer.close()
        print("Training completed!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train HSI to OCTA translation model')
    parser.add_argument('--data_dir', type=str, help='Path to data directory')
    parser.add_argument('--checkpoint_dir', type=str, help='Path to save checkpoints')
    parser.add_argument('--config', type=str, help='Path to config JSON file', required=True)

    args = parser.parse_args()

    trainer = Trainer(config_path=args.config)

    # Use paths from config if not provided in arguments
    data_dir = args.data_dir
    checkpoint_dir = args.checkpoint_dir

    if data_dir is None and hasattr(trainer, 'custom_config'):
        data_dir = trainer.custom_config.get('data', {}).get('train_dir')
    if checkpoint_dir is None and hasattr(trainer, 'custom_config'):
        checkpoint_dir = trainer.custom_config.get('checkpoint_dir')

    if data_dir is None:
        raise ValueError("Data directory must be provided either in config or as argument")
    if checkpoint_dir is None:
        raise ValueError("Checkpoint directory must be provided either in config or as argument")

    trainer.train(data_dir, checkpoint_dir)