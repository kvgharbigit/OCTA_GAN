import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from tqdm import tqdm
import wandb
from datetime import datetime

from base import (
    Generator,
    Discriminator,
    HSI_OCTA_Dataset,
    PerceptualLoss,
    SSIMLoss
)


class Trainer:
    def __init__(self, config):
        """
        Initialize the trainer with configuration

        Args:
            config: dictionary containing training parameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize models
        self.generator = Generator(spectral_channels=config['spectral_channels']).to(self.device)
        self.discriminator = Discriminator().to(self.device)

        # Initialize optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config['lr'],
            betas=(config['beta1'], 0.999)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config['lr'],
            betas=(config['beta1'], 0.999)
        )

        # Initialize loss functions
        self.adversarial_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss().to(self.device)
        self.ssim_loss = SSIMLoss().to(self.device)

        # Setup dataloaders
        self.train_dataset = HSI_OCTA_Dataset(
            config['train_dir'],
            augment=True
        )
        self.val_dataset = HSI_OCTA_Dataset(
            config['val_dir'],
            augment=False
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )

        # Initialize directories
        self.setup_directories()

        # Initialize logging
        if config['use_wandb']:
            wandb.init(project=config['project_name'])
            wandb.config.update(config)

    def setup_directories(self):
        """Create necessary directories for saving results"""
        self.run_dir = os.path.join(
            self.config['save_dir'],
            datetime.now().strftime('%Y%m%d_%H%M%S')
        )
        self.checkpoint_dir = os.path.join(self.run_dir, 'checkpoints')
        self.sample_dir = os.path.join(self.run_dir, 'samples')

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
        }

        # Save regular checkpoint
        path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, path)

        # Save best model if specified
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)

        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])

        return checkpoint['epoch']

    def train_step(self, hsi, octa):
        """Single training step"""
        batch_size = hsi.size(0)
        real_label = torch.ones(batch_size, 1, 16, 16).to(self.device)
        fake_label = torch.zeros(batch_size, 1, 16, 16).to(self.device)

        # -----------------
        # Train Generator
        # -----------------
        self.g_optimizer.zero_grad()

        # Generate fake OCTA
        fake_octa = self.generator(hsi)

        # Adversarial loss
        pred_fake = self.discriminator(fake_octa)
        g_loss_adv = self.adversarial_loss(pred_fake, real_label)

        # Pixel-wise L1 loss
        g_loss_pixel = self.l1_loss(fake_octa, octa)

        # Perceptual loss
        g_loss_perceptual = self.perceptual_loss(fake_octa, octa)

        # SSIM loss
        g_loss_ssim = self.ssim_loss(fake_octa, octa)

        # Combined generator loss
        g_loss = (
                self.config['lambda_adv'] * g_loss_adv +
                self.config['lambda_pixel'] * g_loss_pixel +
                self.config['lambda_perceptual'] * g_loss_perceptual +
                self.config['lambda_ssim'] * g_loss_ssim
        )

        g_loss.backward()
        self.g_optimizer.step()

        # -----------------
        # Train Discriminator
        # -----------------
        self.d_optimizer.zero_grad()

        # Real loss
        pred_real = self.discriminator(octa)
        d_loss_real = self.adversarial_loss(pred_real, real_label)

        # Fake loss
        pred_fake = self.discriminator(fake_octa.detach())
        d_loss_fake = self.adversarial_loss(pred_fake, fake_label)

        # Combined discriminator loss
        d_loss = (d_loss_real + d_loss_fake) / 2

        d_loss.backward()
        self.d_optimizer.step()

        return {
            'g_loss': g_loss.item(),
            'g_loss_adv': g_loss_adv.item(),
            'g_loss_pixel': g_loss_pixel.item(),
            'g_loss_perceptual': g_loss_perceptual.item(),
            'g_loss_ssim': g_loss_ssim.item(),
            'd_loss': d_loss.item()
        }

    @torch.no_grad()
    def validate(self):
        """Validation step"""
        self.generator.eval()
        val_losses = []

        for hsi, octa in tqdm(self.val_loader, desc='Validation'):
            hsi = hsi.to(self.device)
            octa = octa.to(self.device)

            # Generate fake OCTA
            fake_octa = self.generator(hsi)

            # Calculate validation losses
            val_loss = self.l1_loss(fake_octa, octa)
            val_losses.append(val_loss.item())

            # Save some validation samples
            if len(val_losses) == 1:  # Save first batch
                save_image(
                    fake_octa[:4],
                    os.path.join(self.sample_dir, f'val_samples_epoch_{self.current_epoch}.png'),
                    nrow=2, normalize=True
                )

        self.generator.train()
        return sum(val_losses) / len(val_losses)

    def train(self):
        """Main training loop"""
        best_val_loss = float('inf')

        for epoch in range(self.config['num_epochs']):
            self.current_epoch = epoch

            # Training phase
            self.generator.train()
            self.discriminator.train()

            train_losses = []
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

            for hsi, octa in pbar:
                hsi = hsi.to(self.device)
                octa = octa.to(self.device)

                losses = self.train_step(hsi, octa)
                train_losses.append(losses)

                # Update progress bar
                pbar.set_postfix({
                    'g_loss': f"{losses['g_loss']:.4f}",
                    'd_loss': f"{losses['d_loss']:.4f}"
                })

            # Validation phase
            val_loss = self.validate()

            # Log metrics
            if self.config['use_wandb']:
                metrics = {
                    'epoch': epoch,
                    'val_loss': val_loss,
                    **{k: sum(d[k] for d in train_losses) / len(train_losses)
                       for k in train_losses[0].keys()}
                }
                wandb.log(metrics)

            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            self.save_checkpoint(epoch, is_best)

            print(f'Epoch {epoch} - Val Loss: {val_loss:.4f}')


def main():
    """Main function to start training"""
    config = {
        'spectral_channels': 100,
        'batch_size': 8,
        'num_epochs': 100,
        'lr': 2e-4,
        'beta1': 0.5,
        'num_workers': 4,
        'lambda_adv': 1.0,
        'lambda_pixel': 100.0,
        'lambda_perceptual': 10.0,
        'lambda_ssim': 5.0,
        'train_dir': 'path/to/train/data',
        'val_dir': 'path/to/val/data',
        'save_dir': 'runs',
        'use_wandb': True,
        'project_name': 'hsi-to-octa'
    }

    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()