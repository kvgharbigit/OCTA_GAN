#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Mini Pipeline (No TensorBoard version)

A script to test the full OCTA-GAN training pipeline with a minimal dataset for sanity checking.
Performs one epoch of training, validation, and generates visualizations using a small subset
of data from the approved participants CSV, without requiring TensorBoard.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from pathlib import Path
import time
from tqdm import tqdm
from datetime import datetime
import argparse
import csv
import pandas as pd
import shutil
import numpy as np
import matplotlib.pyplot as plt

from base import (
    Generator, Discriminator,
    PerceptualLoss, SSIMLoss,
    init_weights, get_scheduler, save_checkpoint,
    print_model_structure, get_model_summary_string, save_model_structure
)
from hsi_octa_dataset_cropped import HSI_OCTA_Dataset_Cropped
from config_utils import load_config, setup_directories, validate_directories
from visualization_utils import (
    save_sample_visualizations, save_loss_plots, 
    log_metrics, save_image_grid
)


def create_mini_csv(input_csv_path, output_csv_path, num_samples=10):
    """
    Create a small subset CSV with the first N samples from the original CSV.
    
    Args:
        input_csv_path: Path to the original CSV file
        output_csv_path: Path to save the mini CSV file
        num_samples: Number of samples to include (default: 10)
    
    Returns:
        Path to the created mini CSV file
    """
    try:
        # Read the original CSV
        df = pd.read_csv(input_csv_path)
        
        # Take the first num_samples rows
        mini_df = df.head(num_samples)
        
        # Save to a new CSV file
        mini_df.to_csv(output_csv_path, index=False)
        
        print(f"Created mini dataset CSV with {len(mini_df)} samples at {output_csv_path}")
        return output_csv_path
    except Exception as e:
        print(f"Error creating mini CSV: {str(e)}")
        raise


# We're now using the same save_sample_visualizations function from visualization_utils.py
# that is used in the main training pipeline


# We're now using the save_loss_plots function from visualization_utils.py
# that is used in the main training pipeline


def run_mini_test(config_path, num_samples=10, use_circle_crop=True):
    """
    Run a mini test of the training pipeline with a small subset of data.
    
    Args:
        config_path: Path to the configuration JSON file
        num_samples: Number of samples to include in the mini test
        use_circle_crop: Whether to use circle cropping
    """
    # Initialize metrics history dictionary to track all losses
    metrics_history = {
        'epoch': [],
        'g_loss_total': [],
        'd_loss': [],
        'gan_loss_unweighted': [],
        'pixel_loss_unweighted': [],
        'perceptual_loss_unweighted': [],
        'ssim_loss_unweighted': [],
        'gan_loss_weighted': [],
        'pixel_loss_weighted': [],
        'perceptual_loss_weighted': [],
        'ssim_loss_weighted': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    # Load configuration first
    config = load_config(config_path)
    
    # Get model size from config
    model_size = config.get('model', {}).get('size', 'medium')
    
    # Print model size
    print(f"\n{'='*50}")
    print(f"RUNNING TEST WITH MODEL SIZE: {model_size.upper()}")
    print(f"{'='*50}\n")
    
    # Setup experiment name with timestamp and model size
    # Format: MMDD_HHMMSS_test_modelSize (month, day, hour, minute, second, test indicator, model size)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    exp_id = f"{timestamp}_test_{model_size}"
    
    # Create mini dataset CSV
    original_csv_path = config.get('data', {}).get('approved_csv_path', '')
    if not original_csv_path:
        raise ValueError("No approved_csv_path found in config!")
    
    # Convert to Path object if it's a string
    if isinstance(original_csv_path, str):
        # Replace variable placeholders in the CSV path if needed
        if "${base_dir}" in original_csv_path:
            base_dir = config.get('data', {}).get('base_dir', '')
            original_csv_path = original_csv_path.replace("${base_dir}", base_dir)
        original_csv_path = Path(original_csv_path)
    else:
        # It's already a Path object, convert to string to check for placeholders
        original_csv_path_str = str(original_csv_path)
        if "${base_dir}" in original_csv_path_str:
            base_dir = config.get('data', {}).get('base_dir', '')
            original_csv_path = Path(original_csv_path_str.replace("${base_dir}", base_dir))
    
    mini_csv_path = Path(original_csv_path).parent / f"test_data_{num_samples}_{timestamp}.csv"
    create_mini_csv(original_csv_path, mini_csv_path, num_samples)
    
    # Update config with mini dataset path
    config['data']['approved_csv_path'] = str(mini_csv_path)
    
    # Create experiment directory (just use the ID as the folder name)
    exp_dir = Path(config['output']['base_dir']) / f"{exp_id}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created experiment directory: {exp_dir}")
    
    # Modify output paths to be within the experiment directory
    for key in ['checkpoint_dir', 'results_dir', 'tensorboard_dir', 'visualization_dir']:
        if key in config['output']:
            orig_name = Path(config['output'][key]).name
            config['output'][key] = exp_dir / orig_name
    
    # Create a logs directory for our own logs (without tensorboard)
    log_dir = exp_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup directories
    setup_directories(config)
    validate_directories(config)
    
    # Setup hardware
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Add more visible device information
    print("\n" + "=" * 50)
    if device.type == 'cuda':
        print(f"🔥 USING GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️ USING CPU: GPU NOT AVAILABLE")
        print("Training will be significantly slower on CPU.")
        print("Consider using a machine with an NVIDIA GPU for faster training.")
    print("=" * 50 + "\n")
    
    print(f"Device: {device}")
    
    # Initialize models
    generator = Generator(spectral_channels=config['model']['spectral_channels']).to(device)
    discriminator = Discriminator().to(device)
    
    # Initialize weights
    init_weights(generator)
    init_weights(discriminator)
    
    # Print and save model structures
    print("\nGenerator Architecture:")
    print_model_structure(generator)
    
    print("\nDiscriminator Architecture:")
    print_model_structure(discriminator)
    
    # Save model structures to files
    generator_structure_path = exp_dir / 'generator_structure.txt'
    discriminator_structure_path = exp_dir / 'discriminator_structure.txt'
    
    # Save with input shape information
    save_model_structure(
        generator, 
        generator_structure_path, 
        input_shape=(1, config['model']['spectral_channels'], config['data']['target_size'], config['data']['target_size'])
    )
    
    save_model_structure(
        discriminator, 
        discriminator_structure_path, 
        input_shape=(1, 1, config['data']['target_size'], config['data']['target_size'])
    )
    
    # Setup data normalization
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Initialize loss functions
    criterion_gan = nn.BCELoss()
    criterion_pixel = nn.L1Loss()
    criterion_perceptual = PerceptualLoss().to(device)
    criterion_ssim = SSIMLoss().to(device)
    
    # Setup loss component toggles from config
    loss_components = {
        'pixel_enabled': config.get('loss_components', {}).get('pixel_enabled', True),
        'perceptual_enabled': config.get('loss_components', {}).get('perceptual_enabled', True),
        'ssim_enabled': config.get('loss_components', {}).get('ssim_enabled', True),
        'adversarial_enabled': config.get('loss_components', {}).get('adversarial_enabled', True)
    }
    
    # Print active loss components
    print("\nActive loss components:")
    for component, enabled in loss_components.items():
        component_name = component.replace('_enabled', '')
        print(f"  - {component_name}: {'Enabled' if enabled else 'Disabled'}")
        if enabled:
            weight_name = f"lambda_{component_name}"
            if weight_name in config:
                print(f"    Weight: {config[weight_name]}")
    
    # Setup optimizers
    optimizer_G = torch.optim.Adam(
        generator.parameters(),
        lr=config['learning_rate'],
        betas=(config['beta1'], config['beta2']),
        weight_decay=config['weight_decay']
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(),
        lr=config['learning_rate'],
        betas=(config['beta1'], config['beta2']),
        weight_decay=config['weight_decay']
    )
    
    # Setup schedulers
    scheduler_G = get_scheduler(optimizer_G, config)
    scheduler_D = get_scheduler(optimizer_D, config)
    
    # Create the CSV file with headers
    csv_path = exp_dir / 'training_metrics.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        writer_csv = csv.writer(csvfile)
        writer_csv.writerow([
            'epoch',
            'g_loss_total',
            'd_loss',
            'gan_loss_unweighted',
            'pixel_loss_unweighted',
            'perceptual_loss_unweighted',
            'ssim_loss_unweighted',
            'gan_loss_weighted',
            'pixel_loss_weighted',
            'perceptual_loss_weighted',
            'ssim_loss_weighted',
            'val_loss',
            'learning_rate'
        ])
    
    # Get crop padding from config
    crop_padding = config.get('preprocessing', {}).get('crop_padding', 10)
    
    # Create datasets
    print("\nSetting up datasets...")
    train_dataset = HSI_OCTA_Dataset_Cropped(
        data_dir=str(config['data']['data_dir']),
        approved_csv_path=config['data']['approved_csv_path'],  # Use our mini CSV
        transform=transform,
        split='train',
        target_size=config['data']['target_size'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        crop_padding=crop_padding,
        circle_crop=use_circle_crop
    )
    
    val_dataset = HSI_OCTA_Dataset_Cropped(
        data_dir=str(config['data']['data_dir']),
        approved_csv_path=config['data']['approved_csv_path'],  # Use our mini CSV
        transform=transform,
        split='val',
        target_size=config['data']['target_size'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        augment=False,  # No augmentation for validation
        crop_padding=crop_padding,
        circle_crop=use_circle_crop
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Train for one epoch
    print("\n" + "=" * 80)
    print(f"RUNNING ONE EPOCH OF TRAINING")
    print("=" * 80)
    
    # Initialize loss tracking for this epoch
    epoch_losses = {
        'g_loss': [],
        'd_loss': [],
        'gan_loss': [],
        'pixel_loss': [],
        'perceptual_loss': [],
        'ssim_loss': [],
        'gan_loss_weighted': [],
        'pixel_loss_weighted': [],
        'perceptual_loss_weighted': [],
        'ssim_loss_weighted': []
    }
    
    # Set to training mode
    generator.train()
    discriminator.train()
    
    total_g_loss = 0
    total_d_loss = 0
    
    epoch = 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for i, (hsi, octa, _) in enumerate(pbar):
        batch_size = hsi.size(0)
        
        # Move data to device
        hsi = hsi.to(device)
        octa = octa.to(device)
        
        # Train discriminator
        if loss_components['adversarial_enabled']:
            optimizer_D.zero_grad()
            
            real_label = torch.ones(batch_size, 1, 30, 30).to(device)
            fake_label = torch.zeros(batch_size, 1, 30, 30).to(device)
            
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
        else:
            # Skip discriminator training if adversarial loss is disabled
            d_loss = torch.tensor(0.0, device=device)
            # Still need to generate fake OCTA for the generator training
            fake_octa = generator(hsi)
        
        # Train generator
        optimizer_G.zero_grad()
        
        fake_output = discriminator(fake_octa)
        
        # Calculate individual loss components (unweighted)
        g_gan_loss = torch.tensor(0.0, device=device)
        g_pixel_loss_unweighted = torch.tensor(0.0, device=device)
        g_perceptual_loss_unweighted = torch.tensor(0.0, device=device)
        g_ssim_loss_unweighted = torch.tensor(0.0, device=device)
        
        # Only compute enabled loss components
        if loss_components['adversarial_enabled']:
            g_gan_loss = criterion_gan(fake_output, real_label)
        
        if loss_components['pixel_enabled']:
            g_pixel_loss_unweighted = criterion_pixel(fake_octa, octa)
        
        if loss_components['perceptual_enabled']:
            g_perceptual_loss_unweighted = criterion_perceptual(fake_octa, octa)
        
        if loss_components['ssim_enabled']:
            g_ssim_loss_unweighted = criterion_ssim(fake_octa, octa)
        
        # Apply weights to get the weighted loss components
        g_pixel_loss = g_pixel_loss_unweighted * config['lambda_pixel'] if loss_components['pixel_enabled'] else torch.tensor(0.0, device=device)
        g_perceptual_loss = g_perceptual_loss_unweighted * config['lambda_perceptual'] if loss_components['perceptual_enabled'] else torch.tensor(0.0, device=device)
        g_ssim_loss = g_ssim_loss_unweighted * config['lambda_ssim'] if loss_components['ssim_enabled'] else torch.tensor(0.0, device=device)
        g_gan_loss_weighted = g_gan_loss * config['lambda_adv'] if loss_components['adversarial_enabled'] else torch.tensor(0.0, device=device)
        
        # Combined G loss
        g_loss = g_gan_loss_weighted + g_pixel_loss + g_perceptual_loss + g_ssim_loss
        g_loss.backward()
        optimizer_G.step()
        
        # Update statistics
        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()
        
        # Store unweighted losses for recording
        epoch_losses['g_loss'].append(g_loss.item())
        epoch_losses['d_loss'].append(d_loss.item())
        epoch_losses['gan_loss'].append(g_gan_loss.item() if loss_components['adversarial_enabled'] else 0.0)
        epoch_losses['pixel_loss'].append(g_pixel_loss_unweighted.item() if loss_components['pixel_enabled'] else 0.0)
        epoch_losses['perceptual_loss'].append(g_perceptual_loss_unweighted.item() if loss_components['perceptual_enabled'] else 0.0)
        epoch_losses['ssim_loss'].append(g_ssim_loss_unweighted.item() if loss_components['ssim_enabled'] else 0.0)
        
        # Store weighted losses for recording
        epoch_losses['gan_loss_weighted'].append(g_gan_loss_weighted.item() if loss_components['adversarial_enabled'] else 0.0)
        epoch_losses['pixel_loss_weighted'].append(g_pixel_loss.item() if loss_components['pixel_enabled'] else 0.0)
        epoch_losses['perceptual_loss_weighted'].append(g_perceptual_loss.item() if loss_components['perceptual_enabled'] else 0.0)
        epoch_losses['ssim_loss_weighted'].append(g_ssim_loss.item() if loss_components['ssim_enabled'] else 0.0)
        
        # Update progress bar
        pbar.set_postfix({
            'G_loss': g_loss.item(),
            'D_loss': d_loss.item()
        })
    
    # Calculate average losses for the epoch
    avg_g_loss = total_g_loss / len(train_loader)
    avg_d_loss = total_d_loss / len(train_loader)
    
    # Calculate averages of all loss components
    avg_gan_loss = sum(epoch_losses['gan_loss']) / len(epoch_losses['gan_loss']) if epoch_losses['gan_loss'] else 0
    avg_pixel_loss = sum(epoch_losses['pixel_loss']) / len(epoch_losses['pixel_loss']) if epoch_losses['pixel_loss'] else 0
    avg_perceptual_loss = sum(epoch_losses['perceptual_loss']) / len(epoch_losses['perceptual_loss']) if epoch_losses['perceptual_loss'] else 0
    avg_ssim_loss = sum(epoch_losses['ssim_loss']) / len(epoch_losses['ssim_loss']) if epoch_losses['ssim_loss'] else 0
    
    # Calculate averages of weighted loss components
    avg_gan_loss_weighted = sum(epoch_losses['gan_loss_weighted']) / len(epoch_losses['gan_loss_weighted']) if epoch_losses['gan_loss_weighted'] else 0
    avg_pixel_loss_weighted = sum(epoch_losses['pixel_loss_weighted']) / len(epoch_losses['pixel_loss_weighted']) if epoch_losses['pixel_loss_weighted'] else 0
    avg_perceptual_loss_weighted = sum(epoch_losses['perceptual_loss_weighted']) / len(epoch_losses['perceptual_loss_weighted']) if epoch_losses['perceptual_loss_weighted'] else 0
    avg_ssim_loss_weighted = sum(epoch_losses['ssim_loss_weighted']) / len(epoch_losses['ssim_loss_weighted']) if epoch_losses['ssim_loss_weighted'] else 0
    
    # Update the metrics history for tracking
    # Add current epoch metrics
    metrics_history['epoch'].append(epoch)
    metrics_history['g_loss_total'].append(avg_g_loss)
    metrics_history['d_loss'].append(avg_d_loss)
    metrics_history['gan_loss_unweighted'].append(avg_gan_loss)
    metrics_history['pixel_loss_unweighted'].append(avg_pixel_loss)
    metrics_history['perceptual_loss_unweighted'].append(avg_perceptual_loss)
    metrics_history['ssim_loss_unweighted'].append(avg_ssim_loss)
    metrics_history['gan_loss_weighted'].append(avg_gan_loss_weighted)
    metrics_history['pixel_loss_weighted'].append(avg_pixel_loss_weighted)
    metrics_history['perceptual_loss_weighted'].append(avg_perceptual_loss_weighted)
    metrics_history['ssim_loss_weighted'].append(avg_ssim_loss_weighted)
    metrics_history['learning_rate'].append(optimizer_G.param_groups[0]['lr'])
    
    # Generate and update the loss plots
    update_loss_plots(metrics_history, exp_dir, log_dir)
    
    # Log losses to a file
    with open(log_dir / 'training_log.txt', 'w') as f:
        f.write(f"Training Epoch {epoch} Losses:\n")
        f.write(f"Generator Loss: {avg_g_loss:.6f}\n")
        f.write(f"Discriminator Loss: {avg_d_loss:.6f}\n")
        f.write(f"GAN Loss (unweighted): {avg_gan_loss:.6f}\n")
        f.write(f"Pixel Loss (unweighted): {avg_pixel_loss:.6f}\n")
        f.write(f"Perceptual Loss (unweighted): {avg_perceptual_loss:.6f}\n")
        f.write(f"SSIM Loss (unweighted): {avg_ssim_loss:.6f}\n")
        f.write(f"GAN Loss (weighted): {avg_gan_loss_weighted:.6f}\n")
        f.write(f"Pixel Loss (weighted): {avg_pixel_loss_weighted:.6f}\n")
        f.write(f"Perceptual Loss (weighted): {avg_perceptual_loss_weighted:.6f}\n")
        f.write(f"SSIM Loss (weighted): {avg_ssim_loss_weighted:.6f}\n")
    
    # Print epoch summary
    print(f'\nTraining Summary:')
    print(f'Generator Loss: {avg_g_loss:.4f}')
    print(f'Discriminator Loss: {avg_d_loss:.4f}')
    
    # Run validation
    print("\n" + "=" * 80)
    print(f"RUNNING VALIDATION")
    print("=" * 80)
    
    generator.eval()
    discriminator.eval()
    
    total_val_loss = 0
    
    with torch.no_grad():
        for hsi, octa, _ in tqdm(val_loader, desc="Validation"):
            hsi = hsi.to(device)
            octa = octa.to(device)
            
            fake_octa = generator(hsi)
            
            val_pixel_loss = criterion_pixel(fake_octa, octa)
            val_ssim_loss = criterion_ssim(fake_octa, octa)
            val_loss = val_pixel_loss + val_ssim_loss
            
            total_val_loss += val_loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    
    # Log validation loss to a file
    with open(log_dir / 'validation_log.txt', 'w') as f:
        f.write(f"Validation Epoch {epoch} Loss: {avg_val_loss:.6f}\n")
    
    # Update the CSV with metrics
    with open(csv_path, 'a', newline='') as csvfile:
        writer_csv = csv.writer(csvfile)
        writer_csv.writerow([
            epoch,
            avg_g_loss,
            avg_d_loss,
            avg_gan_loss,
            avg_pixel_loss,
            avg_perceptual_loss,
            avg_ssim_loss,
            avg_gan_loss_weighted,
            avg_pixel_loss_weighted,
            avg_perceptual_loss_weighted,
            avg_ssim_loss_weighted,
            avg_val_loss,
            optimizer_G.param_groups[0]['lr']
        ])
    
    print(f'Validation Loss: {avg_val_loss:.4f}')
    
    # Update validation loss in metrics history
    metrics_history['val_loss'].append(avg_val_loss)
    
    # Update the loss plots after validation using the same function as the main pipeline
    save_loss_plots(metrics_history, exp_dir / 'plots', log_dir)
    
    # Generate sample visualizations
    print("\n" + "=" * 80)
    print(f"GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    vis_dir = exp_dir / 'visual_samples'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate and save visualizations using the same function as the main pipeline
    save_sample_visualizations(
        generator=generator,
        val_loader=val_loader,
        device=device,
        epoch=epoch,
        output_dir=vis_dir,
        log_dir=log_dir,
        num_samples=3
    )
    
    # Save checkpoint
    checkpoint_path = config['output']['checkpoint_dir'] / 'mini_test_checkpoint.pth'
    save_checkpoint({
        'epoch': epoch,
        'exp_id': exp_id,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'val_loss': avg_val_loss,
        'config': config,
        'circle_crop': use_circle_crop
    }, str(checkpoint_path))
    
    # Generate summary report
    print("\n" + "=" * 80)
    print(f"MINI TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Mini test results:")
    print(f"- Experiment ID: {exp_id}")
    print(f"- Generator Loss: {avg_g_loss:.4f}")
    print(f"- Discriminator Loss: {avg_d_loss:.4f}")
    print(f"- Validation Loss: {avg_val_loss:.4f}")
    print(f"- Output directory: {exp_dir}")
    print(f"- Checkpoint saved: {checkpoint_path}")
    print(f"- Visualizations saved in: {vis_dir}")
    print(f"- Logs saved in: {log_dir}")
    
    # Save summary to a file
    with open(exp_dir / 'test_summary.txt', 'w') as f:
        f.write(f"MINI TEST SUMMARY\n")
        f.write(f"===============================\n")
        f.write(f"Experiment ID: {exp_id}\n")
        f.write(f"Dataset: {config['data']['approved_csv_path']}\n")
        f.write(f"Num samples used: {num_samples}\n")
        f.write(f"Circle cropping: {'enabled' if use_circle_crop else 'disabled'}\n")
        f.write(f"Model size: {config.get('model', {}).get('size', 'medium')}\n\n")
        f.write(f"Training Results:\n")
        f.write(f"- Generator Loss: {avg_g_loss:.6f}\n")
        f.write(f"- Discriminator Loss: {avg_d_loss:.6f}\n")
        f.write(f"- Validation Loss: {avg_val_loss:.6f}\n\n")
        f.write(f"Output Locations:\n")
        f.write(f"- Experiment directory: {exp_dir}\n")
        f.write(f"- Model checkpoint: {checkpoint_path}\n")
        f.write(f"- Visualizations: {vis_dir}\n")
        f.write(f"- Loss logs: {log_dir}\n")
    
    # Save a copy of the config file to the experiment directory
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    return exp_dir, metrics_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run mini test of OCTA-GAN training pipeline (no TensorBoard)')
    parser.add_argument('--config', type=str, default='config.json',
                      help='Path to config JSON file')
    parser.add_argument('--num_samples', type=int, default=10,
                      help='Number of samples to use for mini test')
    parser.add_argument('--circle_crop', action='store_true', default=True,
                      help='Enable circle detection and cropping')
    parser.add_argument('--no_circle_crop', action='store_true',
                      help='Disable circle detection and cropping')
    
    args = parser.parse_args()
    
    # Determine circle crop option
    use_circle_crop = True
    if args.no_circle_crop:
        use_circle_crop = False
    
    try:
        exp_dir, metrics_history = run_mini_test(args.config, args.num_samples, use_circle_crop)
        print(f"Mini test completed successfully! Results saved to {exp_dir}")
    except Exception as e:
        print(f"Error in mini test: {str(e)}")
        raise