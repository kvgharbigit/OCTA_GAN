import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import sys

# Add the parent directory to the path so we can import base.py
sys.path.append(str(Path(__file__).parent))

from base import (
    HSI_OCTA_Dataset, Generator, Discriminator,
    PerceptualLoss, SSIMLoss, TrainingConfig,
    init_weights, get_scheduler
)

def test_dataset():
    """Test dataset loading and iteration with standardized 500x500 dimensions"""
    print("\nTesting Dataset...")

    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = HSI_OCTA_Dataset(
        data_dir="/Users/pc/Library/CloudStorage/GoogleDrive-kvgharbi99@gmail.com/My Drive/2025 Kayvan/MPhil Opthal/Projects/Python projects/OCTA_GAN/DummyData",
        transform=transform,
        split='train',
        target_size=500
    )

    val_dataset = HSI_OCTA_Dataset(
        data_dir="/Users/pc/Library/CloudStorage/GoogleDrive-kvgharbi99@gmail.com/My Drive/2025 Kayvan/MPhil Opthal/Projects/Python projects/OCTA_GAN/DummyData",
        transform=transform,
        split='val',
        target_size=500
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    hsi, octa, patient_id = next(iter(train_loader))
    print(f"HSI shape: {hsi.shape}")  # Should be [1, 31, 500, 500]
    print(f"OCTA shape: {octa.shape}")  # Should be [1, 1, 500, 500]
    print(f"Sample patient ID: {patient_id[0]}")

    return train_loader, val_dataset

def test_models():
    """Test model creation and forward pass with 500x500 spatial dimensions"""
    print("\nTesting Models...")

    generator = Generator(spectral_channels=31)
    discriminator = Discriminator()

    init_weights(generator)
    init_weights(discriminator)

    batch_size = 1
    test_input = torch.randn(batch_size, 31, 500, 500)

    gen_output = generator(test_input)
    print(f"Generator output shape: {gen_output.shape}")  # Should be [1, 1, 500, 500]

    disc_output = discriminator(gen_output)
    print(f"Discriminator output shape: {disc_output.shape}")  # Should be [1, 1, 30, 30]

    return generator, discriminator

def test_training_loop():
    """Test a small training loop with standardized dimensions"""
    print("\nTesting Training Loop...")

    # Get dataset and models
    train_loader, _ = test_dataset()
    generator, discriminator = test_models()

    # Set up loss functions
    criterion_gan = nn.BCELoss()
    criterion_pixel = nn.L1Loss()
    criterion_perceptual = PerceptualLoss()
    criterion_ssim = SSIMLoss()

    # Set up optimizers
    config = TrainingConfig()
    optimizer_G = torch.optim.Adam(
        generator.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2)
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2)
    )

    # Train for a few iterations
    print("Running mini training loop...")
    generator.train()
    discriminator.train()

    for i, (hsi, octa, _) in enumerate(train_loader):
        if i >= 2:  # Only test a couple batches
            break

        # Train discriminator
        optimizer_D.zero_grad()

        # Adjusted discriminator label sizes for 500x500 input
        real_label = torch.ones(hsi.size(0), 1, 30, 30)  # Updated size to match discriminator output
        fake_label = torch.zeros(hsi.size(0), 1, 30, 30)  # Updated size to match discriminator output

        # Generate fake image
        fake_octa = generator(hsi)

        # Real loss
        real_output = discriminator(octa)
        d_real_loss = criterion_gan(real_output, real_label)

        # Fake loss
        fake_output = discriminator(fake_octa.detach())
        d_fake_loss = criterion_gan(fake_output, fake_label)

        # Combined D loss
        d_loss = (d_real_loss + d_fake_loss) * 0.5
        d_loss.backward()
        optimizer_D.step()

        # Train generator
        optimizer_G.zero_grad()

        # GAN loss
        fake_output = discriminator(fake_octa)
        g_gan_loss = criterion_gan(fake_output, real_label)

        # Pixel loss
        g_pixel_loss = criterion_pixel(fake_octa, octa) * config.lambda_pixel

        # Perceptual loss
        g_perceptual_loss = criterion_perceptual(fake_octa, octa) * config.lambda_perceptual

        # SSIM loss
        g_ssim_loss = criterion_ssim(fake_octa, octa) * config.lambda_ssim

        # Combined G loss
        g_loss = g_gan_loss + g_pixel_loss + g_perceptual_loss + g_ssim_loss
        g_loss.backward()
        optimizer_G.step()

        print(f"Batch {i + 1}: D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

    print("Training loop completed successfully!")

if __name__ == "__main__":
    print("Starting tests...")

    try:
        # Run all tests
        test_dataset()
        test_models()
        test_training_loop()
        print("\nAll tests completed successfully!")

    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        raise