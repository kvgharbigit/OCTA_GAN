import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import time
from tqdm import tqdm
from datetime import datetime
import os
import json
import argparse
import csv
import shutil
import matplotlib.pyplot as plt
import numpy as np

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


# Custom JSON encoder to handle Path objects
class PathEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


class Trainer:
    def __init__(self, config_path: str, exp_id: str = None):
        # Load and validate configuration
        self.config = load_config(config_path)

        # Get circle cropping setting from config
        self.use_circle_crop = self.config.get('preprocessing', {}).get('circle_crop', False)

        # Set up mixed precision training if available and enabled in config
        self.use_mixed_precision = self.config.get('memory_optimization', {}).get('use_amp', False)
        self.scaler = None
        if self.use_mixed_precision and torch.cuda.is_available():
            # Use torch.cuda.amp.GradScaler for handling mixed precision gradients
            self.scaler = torch.cuda.amp.GradScaler()
            print("[INFO] Mixed precision training enabled (reduces memory usage by ~50%)")
        else:
            if self.use_mixed_precision:
                print("[WARNING] Mixed precision requested but not available - using full precision")
            self.use_mixed_precision = False

        # Initialize dictionary to track losses for epoch averaging
        self.epoch_losses = {
            'g_loss': [],
            'd_loss': [],
            'pixel_loss': [],
            'perceptual_loss': [],
            'ssim_loss': [],
            'gan_loss': []
        }

        # Initialize lists to store metrics history
        self.metrics_history = {
            'epoch': [],
            'g_loss_total': [],
            'd_loss': [],
            'total_train_loss': [],
            'val_loss': [],
            # Training unweighted losses
            'pixel_loss_unweighted': [],
            'ssim_loss_unweighted': [],
            'perceptual_loss_unweighted': [],
            'gan_loss_unweighted': [],
            # Training weighted losses
            'pixel_loss_weighted': [],
            'ssim_loss_weighted': [],
            'perceptual_loss_weighted': [],
            'gan_loss_weighted': [],
            # Validation unweighted losses
            'val_pixel_loss_unweighted': [],
            'val_ssim_loss_unweighted': [],
            'val_perceptual_loss_unweighted': [],
            'val_gan_loss_unweighted': [],
            # Validation weighted losses
            'val_pixel_loss_weighted': [],
            'val_ssim_loss_weighted': [],
            'val_perceptual_loss_weighted': [],
            'val_gan_loss_weighted': [],
            'learning_rate': []
        }

        # Set up loss component toggles from config
        self.loss_components = {
            'pixel_enabled': self.config.get('loss_components', {}).get('pixel_enabled', True),
            'perceptual_enabled': self.config.get('loss_components', {}).get('perceptual_enabled', True),
            'ssim_enabled': self.config.get('loss_components', {}).get('ssim_enabled', True),
            'adversarial_enabled': self.config.get('loss_components', {}).get('adversarial_enabled', True)
        }

        # Print active loss components
        print("\nActive loss components:")
        for component, enabled in self.loss_components.items():
            component_name = component.replace('_enabled', '')
            print(f"  - {component_name}: {'Enabled' if enabled else 'Disabled'}")
            if enabled:
                weight_name = f"lambda_{component_name}"
                if weight_name in self.config:
                    print(f"    Weight: {self.config[weight_name]}")

        # Ensure preprocessing config exists and has default values if not set
        if 'preprocessing' not in self.config:
            self.config['preprocessing'] = {}
        if 'crop_padding' not in self.config['preprocessing']:
            self.config['preprocessing']['crop_padding'] = 10  # Default padding

        # Get model size for including in experiment ID
        model_size = self.config['model'].get('size', 'medium')
        
        # Set experiment ID
        if exp_id:
            self.exp_id = exp_id
        else:
            # Generate a timestamp-based experiment ID if none provided
            # Format: MMDD_HHMMSS_modelSize (month, day, hour, minute, second, size)
            timestamp = datetime.now().strftime("%m%d_%H%M%S")
            self.exp_id = f"{timestamp}_{model_size}"

        print(f"Running experiment: {self.exp_id}")
        print(f"Model size: {model_size}")
        print(f"Circle cropping: {'enabled' if self.use_circle_crop else 'disabled'}")

        # Create a parent experiment directory (just use the ID as the folder name)
        self.exp_dir = Path(self.config['output']['base_dir']) / f"{self.exp_id}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created experiment directory: {self.exp_dir}")

        # Modify output paths to be within the experiment directory
        for key in ['checkpoint_dir', 'results_dir', 'visualization_dir']:
            if key in self.config['output']:
                orig_name = Path(self.config['output'][key]).name
                self.config['output'][key] = self.exp_dir / orig_name

        # Create logs directory
        self.log_dir = self.exp_dir / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        setup_directories(self.config)
        validate_directories(self.config)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Add more visible device information
        print("\n" + "=" * 50)
        if self.device.type == 'cuda':
            print(f"🔥 USING GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠️ USING CPU: GPU NOT AVAILABLE")
            print("Training will be significantly slower on CPU.")
            print("Consider using a machine with an NVIDIA GPU for faster training.")
        print("=" * 50 + "\n")
        
        print(f"Device: {self.device}")

        # Initialize early stopping variables
        self.early_stop_counter = 0
        self.best_val_loss = float('inf')

        # Initialize models with the specified size
        model_size = self.config['model'].get('size', 'medium')
        print(f"\nUsing model size: {model_size}")
        self.generator = Generator(
            spectral_channels=self.config['model']['spectral_channels'],
            model_size=model_size
        ).to(self.device)
        
        # Create standard discriminator
        self.discriminator = Discriminator(model_size=model_size).to(self.device)
        
        # If using mixed precision, modify the discriminator to remove the final sigmoid
        # This is necessary because BCEWithLogitsLoss includes sigmoid internally
        if self.use_mixed_precision:
            # Remove the sigmoid layer from the discriminator
            # The discriminator model is a Sequential, and we need to remove the last layer (sigmoid)
            model_layers = list(self.discriminator.model)
            if isinstance(model_layers[-1], nn.Sigmoid):
                # Create a new model without the sigmoid
                self.discriminator.model = nn.Sequential(*model_layers[:-1])
                print("[INFO] Removed sigmoid from discriminator for mixed precision compatibility")

        # Initialize weights
        init_weights(self.generator)
        init_weights(self.discriminator)
        
        # Print and save model structures
        print("\nGenerator Architecture:")
        print_model_structure(self.generator)
        
        print("\nDiscriminator Architecture:")
        print_model_structure(self.discriminator)
        
        # Save model structures to files
        generator_structure_path = self.exp_dir / 'generator_structure.txt'
        discriminator_structure_path = self.exp_dir / 'discriminator_structure.txt'
        
        # Save with input shape information
        save_model_structure(
            self.generator, 
            generator_structure_path, 
            input_shape=(1, self.config['model']['spectral_channels'], self.config['data']['target_size'], self.config['data']['target_size'])
        )
        
        save_model_structure(
            self.discriminator, 
            discriminator_structure_path, 
            input_shape=(1, 1, self.config['data']['target_size'], self.config['data']['target_size'])
        )

        # Setup data normalization
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Initialize loss functions
        # For mixed precision training, use BCEWithLogitsLoss which is numerically stable
        if self.use_mixed_precision:
            # BCEWithLogitsLoss is safe for mixed precision (combines sigmoid + BCE)
            self.criterion_gan = nn.BCEWithLogitsLoss()
            print("[INFO] Using BCEWithLogitsLoss for mixed precision compatibility")
        else:
            # Standard BCE loss when not using mixed precision
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

        # Add experiment info to log file
        with open(self.log_dir / 'experiment_info.txt', 'w') as f:
            f.write(f"Experiment ID: {self.exp_id}\n")
            f.write(f"Config file: {config_path}\n")
            f.write(f"Circle crop: {self.use_circle_crop}\n")
            f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {self.device}\n\n")
            f.write("Configuration:\n")
            f.write(json.dumps(self.config, indent=2, cls=PathEncoder))

        # Save configuration to the experiment directory using the custom encoder
        with open(self.exp_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=4, cls=PathEncoder)

        # Record start time
        self.start_time = time.time()
        self.start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.completed_epochs = 0
        self.early_stopped = False

        # Create the CSV file with headers
        self.csv_path = self.exp_dir / 'training_metrics.csv'
        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'epoch',
                'g_loss_total',
                'd_loss',
                'total_train_loss',
                'val_loss',
                # Training unweighted losses
                'train_pixel_loss_unweighted',
                'train_ssim_loss_unweighted',
                'train_perceptual_loss_unweighted',
                'train_gan_loss_unweighted',
                # Training weighted losses
                'train_pixel_loss_weighted',
                'train_ssim_loss_weighted',
                'train_perceptual_loss_weighted',
                'train_gan_loss_weighted',
                # Validation unweighted losses
                'val_pixel_loss_unweighted',
                'val_ssim_loss_unweighted',
                'val_perceptual_loss_unweighted',
                'val_gan_loss_unweighted',
                # Validation weighted losses
                'val_pixel_loss_weighted',
                'val_ssim_loss_weighted',
                'val_perceptual_loss_weighted',
                'val_gan_loss_weighted',
                'learning_rate'
            ])

    def update_csv(self, metrics_dict):
        """Append a row of metrics to the CSV file."""
        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                metrics_dict.get('epoch', ''),
                metrics_dict.get('g_loss_total', ''),
                metrics_dict.get('d_loss', ''),
                metrics_dict.get('total_train_loss', ''),
                metrics_dict.get('val_loss', ''),
                # Training unweighted losses
                metrics_dict.get('pixel_loss_unweighted', ''),
                metrics_dict.get('ssim_loss_unweighted', ''),
                metrics_dict.get('perceptual_loss_unweighted', ''),
                metrics_dict.get('gan_loss_unweighted', ''),
                # Training weighted losses
                metrics_dict.get('pixel_loss_weighted', ''),
                metrics_dict.get('ssim_loss_weighted', ''),
                metrics_dict.get('perceptual_loss_weighted', ''),
                metrics_dict.get('gan_loss_weighted', ''),
                # Validation unweighted losses
                metrics_dict.get('val_pixel_loss_unweighted', ''),
                metrics_dict.get('val_ssim_loss_unweighted', ''),
                metrics_dict.get('val_perceptual_loss_unweighted', ''),
                metrics_dict.get('val_gan_loss_unweighted', ''),
                # Validation weighted losses
                metrics_dict.get('val_pixel_loss_weighted', ''),
                metrics_dict.get('val_ssim_loss_weighted', ''),
                metrics_dict.get('val_perceptual_loss_weighted', ''),
                metrics_dict.get('val_gan_loss_weighted', ''),
                metrics_dict.get('learning_rate', '')
            ])

            # Update the metrics history for tracking
            for key, value in metrics_dict.items():
                if key in self.metrics_history and value != '':
                    if key not in self.metrics_history or len(self.metrics_history[key]) == 0:
                        self.metrics_history[key] = [value]  # Initialize if empty
                    else:
                        self.metrics_history[key].append(value)
    
    def update_loss_plots(self):
        """
        Generate and update the loss plots based on current metrics.
        Creates three plots:
        1. Training losses (G loss and D loss)
        2. Validation loss (if available)
        3. Total training loss vs validation loss (to see overall convergence)
        """
        # Create the plots directory if it doesn't exist
        plots_dir = self.exp_dir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if we have any data to plot
        if len(self.metrics_history['epoch']) == 0:
            return
        
        # 1. Training losses plot
        plt.figure(figsize=(12, 6))
        epochs = self.metrics_history['epoch']
        
        # Plot generator loss
        if 'g_loss_total' in self.metrics_history and len(self.metrics_history['g_loss_total']) > 0:
            plt.plot(epochs, self.metrics_history['g_loss_total'], 'b-', label='Generator Loss')
        
        # Plot discriminator loss
        if 'd_loss' in self.metrics_history and len(self.metrics_history['d_loss']) > 0:
            plt.plot(epochs, self.metrics_history['d_loss'], 'r-', label='Discriminator Loss')
        
        # Plot validation loss on the same graph if available
        if 'val_loss' in self.metrics_history and len(self.metrics_history['val_loss']) > 0:
            # Make sure val_loss array is the same length as epochs
            val_epochs = epochs[:len(self.metrics_history['val_loss'])]
            plt.plot(val_epochs, self.metrics_history['val_loss'], 'g-', label='Validation Loss')
        
        plt.title('Training and Validation Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # If we're in early epochs, use appropriate y-axis scaling
        if len(epochs) < 5:
            plt.ylim(bottom=0)  # Start y-axis at 0
        
        # Save the figure, overwriting any existing file
        loss_plot_path = plots_dir / 'losses.png'
        plt.savefig(loss_plot_path, dpi=120, bbox_inches='tight')
        plt.close()
        
        # 2. Component losses plot (if we have more than one epoch)
        if len(epochs) > 1:
            plt.figure(figsize=(12, 6))
            
            # Only plot enabled loss components
            component_losses = [
                ('pixel_loss_weighted', 'Pixel Loss (L1)', 'b-'),
                ('perceptual_loss_weighted', 'Perceptual Loss', 'r-'),
                ('ssim_loss_weighted', 'SSIM Loss', 'g-'),
                ('gan_loss_weighted', 'GAN Loss', 'm-')
            ]
            
            for key, label, style in component_losses:
                if key in self.metrics_history and any(v > 0 for v in self.metrics_history[key]):
                    plt.plot(epochs, self.metrics_history[key], style, label=label)
            
            plt.title('Loss Components (Weighted)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save the components plot
            components_plot_path = plots_dir / 'loss_components.png'
            plt.savefig(components_plot_path, dpi=120, bbox_inches='tight')
            plt.close()
            
            # 3. Plot total training loss vs validation loss
            plt.figure(figsize=(12, 6))
            
            # Calculate and plot total weighted loss (sum of all weighted loss components)
            # Initialize with zeros
            total_loss = np.zeros(len(epochs))
            
            # Add each enabled component
            if 'pixel_loss_weighted' in self.metrics_history:
                total_loss += np.array(self.metrics_history['pixel_loss_weighted'])
            if 'ssim_loss_weighted' in self.metrics_history:
                total_loss += np.array(self.metrics_history['ssim_loss_weighted'])
            if 'gan_loss_weighted' in self.metrics_history:
                total_loss += np.array(self.metrics_history['gan_loss_weighted'])
            if 'perceptual_loss_weighted' in self.metrics_history:
                total_loss += np.array(self.metrics_history['perceptual_loss_weighted'])
                
            # Plot total training loss
            plt.plot(epochs, total_loss, 'b-', label='Total Loss (Combined Components)')
            
            # Plot validation loss on same graph for comparison
            if 'val_loss' in self.metrics_history and len(self.metrics_history['val_loss']) > 0:
                val_epochs = epochs[:len(self.metrics_history['val_loss'])]
                plt.plot(val_epochs, self.metrics_history['val_loss'], 'r-', label='Validation Loss')
            
            plt.title('Total Training Loss vs Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save the combined plot
            total_loss_path = plots_dir / 'total_vs_validation_loss.png'
            plt.savefig(total_loss_path, dpi=120, bbox_inches='tight')
            plt.close()
            
        # Log the plot update
        with open(self.log_dir / 'plot_log.txt', 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp}: Updated loss plots at epoch {epochs[-1]}\n")

    def setup_data(self):
        """Setup datasets and dataloaders."""
        print("Setting up datasets...")

        # Get crop padding from config
        crop_padding = self.config.get('preprocessing', {}).get('crop_padding', 10)

        # Get approved participants CSV path from config
        approved_csv_path = self.config.get('data', {}).get('approved_csv_path')
        if approved_csv_path:
            print(f"Using approved participants from: {approved_csv_path}")
            
        # Get augmentation config
        aug_config = self.config.get('augmentation', {})
        print(f"Augmentation enabled: {aug_config.get('enabled', True)}")

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
            circle_crop=self.use_circle_crop,
            augment=aug_config.get('enabled', True),  # Use enabled setting from config
            aug_config=aug_config  # Pass full augmentation config
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
            circle_crop=self.use_circle_crop,
            aug_config=aug_config  # Pass config even though augment=False (for consistency)
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
        """Train for one epoch with memory optimizations."""
        self.generator.train()
        self.discriminator.train()
        
        # Use empty_cache at the start of each epoch to clear leftover memory
        # This helps prevent memory fragmentation
        if torch.cuda.is_available():
            print(f"GPU memory before cache clearing: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            torch.cuda.empty_cache()

        # Initialize loss tracking for this epoch - only track what's needed
        self.epoch_losses = {
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

        total_g_loss = 0
        total_d_loss = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for i, (hsi, octa, _) in enumerate(pbar):
            batch_size = hsi.size(0)

            # Move data to device
            hsi = hsi.to(self.device)
            octa = octa.to(self.device)

            # Train discriminator
            if self.loss_components['adversarial_enabled']:
                self.optimizer_D.zero_grad(set_to_none=True)  # More memory efficient

                # Pre-allocate labels only once and reuse
                # This avoids repeatedly allocating new tensors
                if not hasattr(self, 'real_label') or self.real_label.size(0) != batch_size:
                    # Use 1.0 for real, 0.0 for fake labels regardless of loss function
                    self.real_label = torch.ones(batch_size, 1, 30, 30, device=self.device)
                    self.fake_label = torch.zeros(batch_size, 1, 30, 30, device=self.device)

                # Generate fake image (with mixed precision if enabled)
                if self.use_mixed_precision:
                    # Use new autocast syntax
                    with torch.cuda.amp.autocast():
                        fake_octa = self.generator(hsi)
                        
                        # Real loss
                        real_output = self.discriminator(octa)
                        d_real_loss = self.criterion_gan(real_output, self.real_label)
                        
                        # Fake loss
                        fake_output = self.discriminator(fake_octa.detach())  # detach to avoid gradient flow
                        d_fake_loss = self.criterion_gan(fake_output, self.fake_label)
                        
                        # Combined D loss
                        d_loss = (d_real_loss + d_fake_loss) * 0.5

                    # Scale the gradients for mixed precision training
                    self.scaler.scale(d_loss).backward()
                    
                    # Apply gradient clipping to prevent exploding gradients
                    self.scaler.unscale_(self.optimizer_D)
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=self.config['gradient_clip'])
                    
                    # Update weights with scaled gradients
                    self.scaler.step(self.optimizer_D)
                    self.scaler.update()
                else:
                    # Standard precision training (original implementation)
                    fake_octa = self.generator(hsi)
                    
                    # Real loss
                    real_output = self.discriminator(octa)
                    d_real_loss = self.criterion_gan(real_output, self.real_label)
                    
                    # Fake loss
                    fake_output = self.discriminator(fake_octa.detach())  # detach to avoid gradient flow
                    d_fake_loss = self.criterion_gan(fake_output, self.fake_label)
                    
                    # Combined D loss
                    d_loss = (d_real_loss + d_fake_loss) * 0.5
                    
                    d_loss.backward()
                    
                    # Apply gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=self.config['gradient_clip'])
                    
                    self.optimizer_D.step()
                
                # Free memory
                del real_output, fake_output, d_real_loss, d_fake_loss
            else:
                # Skip discriminator training if adversarial loss is disabled
                d_loss = torch.tensor(0.0, device=self.device)
                # Still need to generate fake OCTA for the generator training
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        fake_octa = self.generator(hsi)
                else:
                    fake_octa = self.generator(hsi)

            # Train generator
            self.optimizer_G.zero_grad(set_to_none=True)  # More memory efficient

            # Apply mixed precision for generator training if enabled
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    # Compute fake discriminator output
                    fake_output = self.discriminator(fake_octa)
                    
                    # Calculate individual loss components only if enabled
                    if self.loss_components['adversarial_enabled']:
                        g_gan_loss = self.criterion_gan(fake_output, self.real_label)
                        g_gan_loss_weighted = g_gan_loss * self.config['lambda_adv']
                    else:
                        g_gan_loss = g_gan_loss_weighted = torch.tensor(0.0, device=self.device)
                    
                    if self.loss_components['pixel_enabled']:
                        g_pixel_loss_unweighted = self.criterion_pixel(fake_octa, octa)
                        g_pixel_loss = g_pixel_loss_unweighted * self.config['lambda_pixel']
                    else:
                        g_pixel_loss_unweighted = g_pixel_loss = torch.tensor(0.0, device=self.device)
                    
                    if self.loss_components['perceptual_enabled']:
                        g_perceptual_loss_unweighted = self.criterion_perceptual(fake_octa, octa)
                        g_perceptual_loss = g_perceptual_loss_unweighted * self.config['lambda_perceptual']
                    else:
                        g_perceptual_loss_unweighted = g_perceptual_loss = torch.tensor(0.0, device=self.device)
                    
                    if self.loss_components['ssim_enabled']:
                        g_ssim_loss_unweighted = self.criterion_ssim(fake_octa, octa)
                        g_ssim_loss = g_ssim_loss_unweighted * self.config['lambda_ssim']
                    else:
                        g_ssim_loss_unweighted = g_ssim_loss = torch.tensor(0.0, device=self.device)
                    
                    # Combined G loss
                    g_loss = g_gan_loss_weighted + g_pixel_loss + g_perceptual_loss + g_ssim_loss
                
                # Scale gradients for numerical stability with FP16
                self.scaler.scale(g_loss).backward()
                
                # Unscale for gradient clipping
                self.scaler.unscale_(self.optimizer_G)
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=self.config['gradient_clip'])
                
                # Update with scaled gradients
                self.scaler.step(self.optimizer_G)
                self.scaler.update()
            else:
                # Standard precision (original implementation)
                fake_output = self.discriminator(fake_octa)
                
                # Calculate individual loss components only if enabled (memory efficient)
                if self.loss_components['adversarial_enabled']:
                    g_gan_loss = self.criterion_gan(fake_output, self.real_label)
                    g_gan_loss_weighted = g_gan_loss * self.config['lambda_adv']
                else:
                    g_gan_loss = g_gan_loss_weighted = torch.tensor(0.0, device=self.device)
                
                if self.loss_components['pixel_enabled']:
                    g_pixel_loss_unweighted = self.criterion_pixel(fake_octa, octa)
                    g_pixel_loss = g_pixel_loss_unweighted * self.config['lambda_pixel']
                else:
                    g_pixel_loss_unweighted = g_pixel_loss = torch.tensor(0.0, device=self.device)
                
                if self.loss_components['perceptual_enabled']:
                    g_perceptual_loss_unweighted = self.criterion_perceptual(fake_octa, octa)
                    g_perceptual_loss = g_perceptual_loss_unweighted * self.config['lambda_perceptual']
                else:
                    g_perceptual_loss_unweighted = g_perceptual_loss = torch.tensor(0.0, device=self.device)
                
                if self.loss_components['ssim_enabled']:
                    g_ssim_loss_unweighted = self.criterion_ssim(fake_octa, octa)
                    g_ssim_loss = g_ssim_loss_unweighted * self.config['lambda_ssim']
                else:
                    g_ssim_loss_unweighted = g_ssim_loss = torch.tensor(0.0, device=self.device)
                
                # Combined G loss
                g_loss = g_gan_loss_weighted + g_pixel_loss + g_perceptual_loss + g_ssim_loss
                
                g_loss.backward()
                
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=self.config['gradient_clip'])
                
                self.optimizer_G.step()
            
            # Explicitly free memory
            if not self.loss_components['adversarial_enabled']:
                del fake_output
                
            # Empty CUDA cache periodically if configured
            empty_cache_freq = self.config.get('hardware', {}).get('empty_cache_freq', 0)
            if empty_cache_freq > 0 and (i + 1) % empty_cache_freq == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Update statistics
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

            # Store unweighted losses for recording
            self.epoch_losses['g_loss'].append(g_loss.item())
            self.epoch_losses['d_loss'].append(d_loss.item())
            self.epoch_losses['gan_loss'].append(
                g_gan_loss.item() if self.loss_components['adversarial_enabled'] else 0.0)
            self.epoch_losses['pixel_loss'].append(
                g_pixel_loss_unweighted.item() if self.loss_components['pixel_enabled'] else 0.0)
            self.epoch_losses['perceptual_loss'].append(
                g_perceptual_loss_unweighted.item() if self.loss_components['perceptual_enabled'] else 0.0)
            self.epoch_losses['ssim_loss'].append(
                g_ssim_loss_unweighted.item() if self.loss_components['ssim_enabled'] else 0.0)

            # Store weighted losses for recording
            self.epoch_losses['gan_loss_weighted'].append(
                g_gan_loss_weighted.item() if self.loss_components['adversarial_enabled'] else 0.0)
            self.epoch_losses['pixel_loss_weighted'].append(
                g_pixel_loss.item() if self.loss_components['pixel_enabled'] else 0.0)
            self.epoch_losses['perceptual_loss_weighted'].append(
                g_perceptual_loss.item() if self.loss_components['perceptual_enabled'] else 0.0)
            self.epoch_losses['ssim_loss_weighted'].append(
                g_ssim_loss.item() if self.loss_components['ssim_enabled'] else 0.0)

            # Print interval logging (optional, for terminal feedback only)
            if i % self.config['logging']['print_interval'] == 0:
                pbar.set_postfix({
                    'G_loss': g_loss.item(),
                    'D_loss': d_loss.item()
                })
                
        # Clear CUDA cache at the end of each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Calculate average losses for the epoch
        avg_g_loss = total_g_loss / len(self.train_loader)
        avg_d_loss = total_d_loss / len(self.train_loader)

        # Calculate averages of all loss components
        avg_gan_loss = sum(self.epoch_losses['gan_loss']) / len(self.epoch_losses['gan_loss']) if self.epoch_losses[
            'gan_loss'] else 0
        avg_pixel_loss = sum(self.epoch_losses['pixel_loss']) / len(self.epoch_losses['pixel_loss']) if \
        self.epoch_losses['pixel_loss'] else 0
        avg_perceptual_loss = sum(self.epoch_losses['perceptual_loss']) / len(self.epoch_losses['perceptual_loss']) if \
        self.epoch_losses['perceptual_loss'] else 0
        avg_ssim_loss = sum(self.epoch_losses['ssim_loss']) / len(self.epoch_losses['ssim_loss']) if self.epoch_losses[
            'ssim_loss'] else 0

        # Calculate averages of weighted loss components
        avg_gan_loss_weighted = sum(self.epoch_losses['gan_loss_weighted']) / len(
            self.epoch_losses['gan_loss_weighted']) if self.epoch_losses['gan_loss_weighted'] else 0
        avg_pixel_loss_weighted = sum(self.epoch_losses['pixel_loss_weighted']) / len(
            self.epoch_losses['pixel_loss_weighted']) if self.epoch_losses['pixel_loss_weighted'] else 0
        avg_perceptual_loss_weighted = sum(self.epoch_losses['perceptual_loss_weighted']) / len(
            self.epoch_losses['perceptual_loss_weighted']) if self.epoch_losses['perceptual_loss_weighted'] else 0
        avg_ssim_loss_weighted = sum(self.epoch_losses['ssim_loss_weighted']) / len(
            self.epoch_losses['ssim_loss_weighted']) if self.epoch_losses['ssim_loss_weighted'] else 0

        # Calculate total training loss (sum of all weighted components)
        total_train_loss = avg_gan_loss_weighted + avg_pixel_loss_weighted + avg_perceptual_loss_weighted + avg_ssim_loss_weighted

        # Log metrics to json/text files
        training_metrics = {
            'g_loss_total': avg_g_loss,
            'd_loss': avg_d_loss,
            'gan_loss_unweighted': avg_gan_loss,
            'pixel_loss_unweighted': avg_pixel_loss,
            'perceptual_loss_unweighted': avg_perceptual_loss,
            'ssim_loss_unweighted': avg_ssim_loss,
            'gan_loss_weighted': avg_gan_loss_weighted,
            'pixel_loss_weighted': avg_pixel_loss_weighted,
            'perceptual_loss_weighted': avg_perceptual_loss_weighted,
            'ssim_loss_weighted': avg_ssim_loss_weighted,
            'total_train_loss': total_train_loss,
            'learning_rate': self.optimizer_G.param_groups[0]['lr']
        }
        
        log_metrics(training_metrics, self.log_dir, epoch, is_training=True)

        # Update the CSV with this epoch's metrics (validation loss will be added later if available)
        metrics_dict = {
            'epoch': epoch,
            'g_loss_total': avg_g_loss,
            'd_loss': avg_d_loss,
            'total_train_loss': total_train_loss,
            'val_loss': '',  # Will be updated when validation is run
            # Training unweighted losses
            'pixel_loss_unweighted': avg_pixel_loss,
            'ssim_loss_unweighted': avg_ssim_loss,
            'perceptual_loss_unweighted': avg_perceptual_loss,
            'gan_loss_unweighted': avg_gan_loss,
            # Training weighted losses
            'pixel_loss_weighted': avg_pixel_loss_weighted,
            'ssim_loss_weighted': avg_ssim_loss_weighted,
            'perceptual_loss_weighted': avg_perceptual_loss_weighted,
            'gan_loss_weighted': avg_gan_loss_weighted,
            # Validation unweighted losses (will be updated when validation is run)
            'val_pixel_loss_unweighted': '',
            'val_ssim_loss_unweighted': '',
            'val_perceptual_loss_unweighted': '',
            'val_gan_loss_unweighted': '',
            # Validation weighted losses (will be updated when validation is run)
            'val_pixel_loss_weighted': '',
            'val_ssim_loss_weighted': '',
            'val_perceptual_loss_weighted': '',
            'val_gan_loss_weighted': '',
            'learning_rate': self.optimizer_G.param_groups[0]['lr']
        }
        self.update_csv(metrics_dict)

        return avg_g_loss, avg_d_loss

    def validate(self, epoch: int):
        """Run validation with memory optimizations."""
        self.generator.eval()
        self.discriminator.eval()
        
        # Clear CUDA cache before validation
        if torch.cuda.is_available() and self.config.get('memory_optimization', {}).get('empty_cache_after_checkpoint', False):
            print(f"GPU memory before validation cache clearing: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            torch.cuda.empty_cache()

        total_val_loss = 0
        total_val_pixel_loss = 0
        total_val_ssim_loss = 0
        total_val_perceptual_loss = 0
        total_val_gan_loss = 0
        
        # Standard precision (no mixed precision) like in mini test
        with torch.no_grad():
            for hsi, octa, _ in self.val_loader:
                # Move to device just-in-time to reduce GPU memory usage with non-blocking transfer
                hsi = hsi.to(self.device, non_blocking=True)  # Non-blocking transfer
                octa = octa.to(self.device, non_blocking=True)

                # For validation with mixed precision, we need to be careful about type mismatches
                if self.use_mixed_precision:
                    # First generate the fake output with autocast
                    with torch.cuda.amp.autocast():
                        fake_octa = self.generator(hsi)
                    
                    # Then cast back to float32 for loss calculation to avoid type mismatches
                    fake_octa = fake_octa.float()
                    
                    # Now calculate losses in float32 precision
                    val_pixel_loss = self.criterion_pixel(fake_octa, octa) if self.loss_components['pixel_enabled'] else 0
                    val_ssim_loss = self.criterion_ssim(fake_octa, octa) if self.loss_components['ssim_enabled'] else 0
                    val_perceptual_loss = self.criterion_perceptual(fake_octa, octa) if self.loss_components['perceptual_enabled'] else 0
                else:
                    # Standard precision validation
                    fake_octa = self.generator(hsi)
                    
                    # Calculate losses
                    val_pixel_loss = self.criterion_pixel(fake_octa, octa) if self.loss_components['pixel_enabled'] else 0
                    val_ssim_loss = self.criterion_ssim(fake_octa, octa) if self.loss_components['ssim_enabled'] else 0
                    val_perceptual_loss = self.criterion_perceptual(fake_octa, octa) if self.loss_components['perceptual_enabled'] else 0
                
                # Compute GAN loss if needed
                val_gan_loss = 0
                if self.loss_components['adversarial_enabled']:
                    # Create real labels for validation samples
                    batch_size = hsi.size(0)
                    real_label = torch.ones(batch_size, 1, 30, 30, device=self.device)
                    
                    # For mixed precision, ensure everything is float32 for the discriminator
                    if self.use_mixed_precision:
                        # Make sure fake_octa is float32 for the discriminator
                        # This avoids the "Input type (struct c10::Half) and bias type (float)" error
                        fake_output = self.discriminator(fake_octa.float())
                    else:
                        # Standard precision
                        fake_output = self.discriminator(fake_octa)
                        
                    val_gan_loss = self.criterion_gan(fake_output, real_label)
                
                # Combine losses with the same weights used in training
                val_pixel_loss_weighted = val_pixel_loss * self.config['lambda_pixel'] if self.loss_components['pixel_enabled'] else 0
                val_ssim_loss_weighted = val_ssim_loss * self.config['lambda_ssim'] if self.loss_components['ssim_enabled'] else 0
                val_perceptual_loss_weighted = val_perceptual_loss * self.config['lambda_perceptual'] if self.loss_components['perceptual_enabled'] else 0
                val_gan_loss_weighted = val_gan_loss * self.config['lambda_adv'] if self.loss_components['adversarial_enabled'] else 0
                
                # Total validation loss is the sum of all weighted components
                val_loss = val_pixel_loss_weighted + val_ssim_loss_weighted + val_perceptual_loss_weighted + val_gan_loss_weighted

                # Accumulate losses (convert to Python scalar to free GPU memory)
                total_val_loss += val_loss.item()
                total_val_pixel_loss += val_pixel_loss.item() if isinstance(val_pixel_loss, torch.Tensor) else val_pixel_loss
                total_val_ssim_loss += val_ssim_loss.item() if isinstance(val_ssim_loss, torch.Tensor) else val_ssim_loss
                total_val_perceptual_loss += val_perceptual_loss.item() if isinstance(val_perceptual_loss, torch.Tensor) else val_perceptual_loss
                total_val_gan_loss += val_gan_loss.item() if isinstance(val_gan_loss, torch.Tensor) else val_gan_loss
                
                # Clear variables to free memory
                del hsi, octa, fake_octa, val_loss
                
                # Manually trigger garbage collection if needed
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Clear unused memory
                    
        # Empty cache after validation if configured
        if torch.cuda.is_available() and self.config.get('memory_optimization', {}).get('empty_cache_after_validation', False):
            torch.cuda.empty_cache()

            # Calculate average losses
            avg_val_loss = total_val_loss / len(self.val_loader)
            avg_val_pixel_loss = total_val_pixel_loss / len(self.val_loader)
            avg_val_ssim_loss = total_val_ssim_loss / len(self.val_loader)
            avg_val_perceptual_loss = total_val_perceptual_loss / len(self.val_loader)
            avg_val_gan_loss = total_val_gan_loss / len(self.val_loader)
            
            # Log validation metrics to json/text files
            validation_metrics = {
                'val_loss': avg_val_loss,
                'val_pixel_loss_unweighted': avg_val_pixel_loss,
                'val_ssim_loss_unweighted': avg_val_ssim_loss,
                'val_perceptual_loss_unweighted': avg_val_perceptual_loss,
                'val_gan_loss_unweighted': avg_val_gan_loss,
                'val_pixel_loss_weighted': avg_val_pixel_loss * self.config['lambda_pixel'] if self.loss_components['pixel_enabled'] else 0,
                'val_ssim_loss_weighted': avg_val_ssim_loss * self.config['lambda_ssim'] if self.loss_components['ssim_enabled'] else 0,
                'val_perceptual_loss_weighted': avg_val_perceptual_loss * self.config['lambda_perceptual'] if self.loss_components['perceptual_enabled'] else 0,
                'val_gan_loss_weighted': avg_val_gan_loss * self.config['lambda_adv'] if self.loss_components['adversarial_enabled'] else 0
            }
            log_metrics(validation_metrics, self.log_dir, epoch, is_training=False)

            # Update metrics history
            for key, value in validation_metrics.items():
                if key not in self.metrics_history:
                    self.metrics_history[key] = []
                
                # If the list is shorter than expected, append the value instead of updating
                if len(self.metrics_history[key]) < len(self.metrics_history['epoch']):
                    self.metrics_history[key].append(value)
                else:
                    # Otherwise update the last element
                    self.metrics_history[key][-1] = value

            # Update the CSV file
            with open(self.csv_path, 'r') as csvfile:
                rows = list(csv.reader(csvfile))

            # Find the row for the current epoch (should be the last row)
            for i in range(len(rows) - 1, 0, -1):
                if rows[i][0] == str(epoch):
                    # Update validation metrics columns - note the column indices are based on the new CSV structure
                    rows[i][4] = str(avg_val_loss)  # Total val loss
                    
                    # Validation unweighted losses
                    rows[i][13] = str(avg_val_pixel_loss)  # Unweighted val pixel loss
                    rows[i][14] = str(avg_val_ssim_loss)  # Unweighted val SSIM loss
                    rows[i][15] = str(avg_val_perceptual_loss)  # Unweighted val perceptual loss
                    rows[i][16] = str(avg_val_gan_loss)  # Unweighted val GAN loss
                    
                    # Validation weighted losses
                    val_pixel_loss_weighted = avg_val_pixel_loss * self.config['lambda_pixel'] if self.loss_components['pixel_enabled'] else 0
                    val_ssim_loss_weighted = avg_val_ssim_loss * self.config['lambda_ssim'] if self.loss_components['ssim_enabled'] else 0
                    val_perceptual_loss_weighted = avg_val_perceptual_loss * self.config['lambda_perceptual'] if self.loss_components['perceptual_enabled'] else 0
                    val_gan_loss_weighted = avg_val_gan_loss * self.config['lambda_adv'] if self.loss_components['adversarial_enabled'] else 0
                    
                    rows[i][17] = str(val_pixel_loss_weighted)  # Weighted val pixel loss
                    rows[i][18] = str(val_ssim_loss_weighted)  # Weighted val SSIM loss
                    rows[i][19] = str(val_perceptual_loss_weighted)  # Weighted val perceptual loss
                    rows[i][20] = str(val_gan_loss_weighted)  # Weighted val GAN loss
                    break

            # Write the updated rows back to the CSV file
            with open(self.csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(rows)

            return avg_val_loss

    def generate_training_summary(self, training_time_seconds, reason="final"):
        """
        Generate a comprehensive training summary text file with all important information.

        Args:
            training_time_seconds: Total training time in seconds
            reason: The reason for generating the summary ("best_model", "final", etc.)

        Returns:
            Path to the generated summary file
        """
        # Convert training time to hours, minutes, seconds
        hours, remainder = divmod(training_time_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        # Create a unique filename based on the reason
        if reason == "final":
            summary_path = self.exp_dir / 'training_summary.txt'
        else:
            summary_path = self.exp_dir / f'training_summary_{reason}.txt'

        print(f"\nGenerating training summary at: {summary_path}")

        with open(summary_path, 'w') as f:
            # Write header
            f.write("=" * 80 + "\n")
            f.write(f"TRAINING SUMMARY: {self.exp_id} ({reason})\n")
            f.write("=" * 80 + "\n\n")

            # Write experiment information
            f.write("EXPERIMENT INFORMATION\n")
            f.write("-" * 50 + "\n")
            f.write(f"Experiment ID: {self.exp_id}\n")
            f.write(f"Start time: {self.start_time_str}\n")
            f.write(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Training time so far: {int(hours)}h {int(minutes)}m {int(seconds)}s\n")
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
            gen_trainable_params = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
            disc_params = sum(p.numel() for p in self.discriminator.parameters())
            disc_trainable_params = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
            
            f.write(f"Generator parameters: {gen_params:,} (trainable: {gen_trainable_params:,})\n")
            f.write(f"Discriminator parameters: {disc_params:,} (trainable: {disc_trainable_params:,})\n")
            f.write(f"Total parameters: {gen_params + disc_params:,}\n\n")
            
            # Include paths to the saved model structure files
            f.write(f"Generator structure file: {self.exp_dir / 'generator_structure.txt'}\n")
            f.write(f"Discriminator structure file: {self.exp_dir / 'discriminator_structure.txt'}\n\n")

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

            # Write training status
            f.write("TRAINING STATUS\n")
            f.write("-" * 50 + "\n")
            f.write(f"Completed epochs: {self.completed_epochs}\n")
            if hasattr(self, 'early_stopped') and self.early_stopped:
                f.write("Training stopped early: Yes\n")
            f.write(f"Best validation loss: {self.best_val_loss:.6f}\n")
            f.write(f"Current learning rate: {self.optimizer_G.param_groups[0]['lr']:.6f}\n\n")

            # Include current metrics if available
            if self.metrics_history['epoch']:
                latest_idx = len(self.metrics_history['epoch']) - 1
                f.write("CURRENT LOSS METRICS\n")
                f.write("-" * 50 + "\n")
                f.write(f"Generator loss: {self.metrics_history['g_loss_total'][latest_idx]:.6f}\n")
                f.write(f"Discriminator loss: {self.metrics_history['d_loss'][latest_idx]:.6f}\n\n")

                f.write("Unweighted loss components:\n")
                f.write(f"  Pixel loss: {self.metrics_history['pixel_loss_unweighted'][latest_idx]:.6f}\n")
                f.write(f"  Perceptual loss: {self.metrics_history['perceptual_loss_unweighted'][latest_idx]:.6f}\n")
                f.write(f"  SSIM loss: {self.metrics_history['ssim_loss_unweighted'][latest_idx]:.6f}\n")
                f.write(f"  GAN loss: {self.metrics_history['gan_loss_unweighted'][latest_idx]:.6f}\n\n")

                f.write("Weighted loss components:\n")
                f.write(f"  Pixel loss: {self.metrics_history['pixel_loss_weighted'][latest_idx]:.6f}\n")
                f.write(f"  Perceptual loss: {self.metrics_history['perceptual_loss_weighted'][latest_idx]:.6f}\n")
                f.write(f"  SSIM loss: {self.metrics_history['ssim_loss_weighted'][latest_idx]:.6f}\n")
                f.write(f"  GAN loss: {self.metrics_history['gan_loss_weighted'][latest_idx]:.6f}\n\n")

            # Write output directories
            f.write("OUTPUT DIRECTORIES\n")
            f.write("-" * 50 + "\n")
            f.write(f"Base directory: {self.exp_dir}\n")
            f.write(f"Checkpoints: {self.config['output']['checkpoint_dir']}\n")
            f.write(f"Logs: {self.log_dir}\n")
            f.write(f"CSV metrics: {self.csv_path}\n")
            f.write(
                f"Visual samples: {self.config['output'].get('visualization_dir', self.exp_dir / 'visual_samples')}\n\n")

            # Write loss component information
            f.write("LOSS COMPONENTS\n")
            f.write("-" * 50 + "\n")
            for component, enabled in self.loss_components.items():
                component_name = component.replace('_enabled', '')
                f.write(f"{component_name.capitalize()}: {'Enabled' if enabled else 'Disabled'}\n")
                if enabled:
                    weight_name = f"lambda_{component_name}"
                    if weight_name in self.config:
                        f.write(f"  - Weight ({weight_name}): {self.config[weight_name]}\n")
            f.write("\n")

            # Write recommendations for evaluation
            if reason == "best_model" or reason == "final":
                f.write("NEXT STEPS\n")
                f.write("-" * 50 + "\n")
                f.write("To evaluate this model, run:\n")
                f.write(f"python evaluation-script.py --config eval_config.json --exp_id {self.exp_id}_eval\n\n")
                f.write("Make sure your eval_config.json points to the best model checkpoint at:\n")
                f.write(f"{self.config['output']['checkpoint_dir'] / 'best_model.pth'}\n\n")

            # Write footer
            f.write("=" * 80 + "\n")
            f.write(f"End of training summary ({reason})\n")
            f.write("=" * 80 + "\n")

        print(f"Training summary saved to: {summary_path}")
        return summary_path

    def train(self, start_epoch=0):
        """Main training loop."""
        self.setup_data()

        # Record start time if it wasn't already done
        if not hasattr(self, 'start_time'):
            self.start_time = time.time()
            self.start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        self.completed_epochs = start_epoch
        self.early_stopped = False

        print(f"Starting training for {self.config['num_epochs']} epochs")
        print(f"Checkpoints will be saved to {self.config['output']['checkpoint_dir']}")
        print(f"Training metrics will be logged to {self.csv_path}")
        print(f"Detailed logs will be saved to {self.log_dir}")

        # Create a directory for visual samples
        vis_dir = self.exp_dir / 'visual_samples'
        vis_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(start_epoch, self.config['num_epochs']):
            start_time = time.time()
            val_loss = None

            # Train for one epoch
            train_g_loss, train_d_loss = self.train_epoch(epoch)
            self.completed_epochs = epoch + 1

            # Visualize samples at configured interval
            save_images_interval = self.config.get('logging', {}).get('save_images_interval',
                                                                      5)  # Default to 5 if not specified
            save_images_start = self.config.get('logging', {}).get('save_images_interval_start',
                                                                   0)  # Default to 0 if not specified
            # Get number of visualization samples from config or default to 3
            num_vis_samples = self.config.get('logging', {}).get('num_visualization_samples', 3)

            if epoch >= save_images_start and epoch % save_images_interval == 0:
                save_sample_visualizations(
                    generator=self.generator,
                    val_loader=self.val_loader,
                    device=self.device,
                    epoch=epoch,
                    output_dir=vis_dir,
                    log_dir=self.log_dir,
                    num_samples=num_vis_samples
                )

            # Validate
            if epoch % self.config['validate_interval'] == 0:
                val_loss = self.validate(epoch)

                # Check for new best model
                if val_loss < self.best_val_loss:
                    prev_best = self.best_val_loss
                    self.best_val_loss = val_loss

                    # Convert Path objects to strings for serialization
                    serializable_config = json.loads(json.dumps(self.config, cls=PathEncoder))

                    # Save checkpoint data
                    checkpoint_data = {
                        'epoch': epoch,
                        'exp_id': self.exp_id,
                        'generator_state_dict': self.generator.state_dict(),
                        'discriminator_state_dict': self.discriminator.state_dict(),
                        'optimizer_G_state_dict': self.optimizer_G.state_dict(),
                        'optimizer_D_state_dict': self.optimizer_D.state_dict(),
                        'val_loss': val_loss,
                        'config': serializable_config,
                        'circle_crop': self.use_circle_crop,
                        'metrics_history': self.metrics_history,
                        'loss_components': self.loss_components
                    }

                    # First save as best_model.pth
                    best_model_path = str(self.config['output']['checkpoint_dir'] / 'best_model.pth')
                    save_checkpoint(checkpoint_data, best_model_path)

                    # Then save as epoch-specific version
                    epoch_checkpoint_path = str(
                        self.config['output']['checkpoint_dir'] / f'checkpoint_epoch_{epoch}.pth')
                    save_checkpoint(checkpoint_data, epoch_checkpoint_path)

                    # Also save as latest_best.pth for easy reference
                    latest_best_path = str(self.config['output']['checkpoint_dir'] / 'latest_best.pth')
                    save_checkpoint(checkpoint_data, latest_best_path)

                    # Generate training summary when a new best model is saved
                    current_training_time = time.time() - self.start_time
                    improvement = ((prev_best - val_loss) / prev_best) * 100 if prev_best != float('inf') else 0
                    summary_reason = f"best_model_epoch_{epoch}_loss_{val_loss:.6f}_imp_{improvement:.2f}pct"
                    self.generate_training_summary(current_training_time, reason=summary_reason)

                    print(f"\nNew best model! Validation loss: {val_loss:.6f} (improved by {improvement:.2f}%)")
                    print(f"Training summary saved for best model at epoch {epoch}")

            # Save regular checkpoint
            if epoch % self.config['save_interval'] == 0:
                # Convert Path objects to strings for serialization
                serializable_config = json.loads(json.dumps(self.config, cls=PathEncoder))

                # Use the most recent val_loss if available, otherwise use best_val_loss
                current_val_loss = val_loss if val_loss is not None else self.best_val_loss

                save_checkpoint({
                    'epoch': epoch,
                    'exp_id': self.exp_id,
                    'generator_state_dict': self.generator.state_dict(),
                    'discriminator_state_dict': self.discriminator.state_dict(),
                    'optimizer_G_state_dict': self.optimizer_G.state_dict(),
                    'optimizer_D_state_dict': self.optimizer_D.state_dict(),
                    'val_loss': current_val_loss,  # Updated line
                    'config': serializable_config,
                    'circle_crop': self.use_circle_crop,
                    'metrics_history': self.metrics_history,
                    'loss_components': self.loss_components
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
            if isinstance(self.scheduler_G, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # For ReduceLROnPlateau, we need validation loss
                if val_loss is not None:
                    self.scheduler_G.step(val_loss)
                    self.scheduler_D.step(val_loss)
                else:
                    # If no validation was run this epoch, use best val loss
                    self.scheduler_G.step(self.best_val_loss)
                    self.scheduler_D.step(self.best_val_loss)
            else:
                # For other schedulers like StepLR, just step without arguments
                self.scheduler_G.step()
                self.scheduler_D.step()

            # Generate loss plots after each epoch
            self.update_loss_plots()

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

        # Generate and save final training summary
        summary_path = self.generate_training_summary(total_training_time, reason="final")

        # Generate final plots with the most complete data
        self.update_loss_plots()
        
        # Also create the full detailed plots from visualization_utils for reference
        # This will include the total training loss vs validation loss plot
        save_loss_plots(self.metrics_history, self.exp_dir / 'plots', self.log_dir)
        
        # Save any missing files log
        HSI_OCTA_Dataset_Cropped.save_missing_files_log(self.log_dir)

        # Final cleanup
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Model checkpoints saved in: {self.config['output']['checkpoint_dir']}")
        print(f"Training metrics saved to: {self.csv_path}")
        print(f"Training summary saved to: {summary_path}")
        print(f"Plots saved in: {self.exp_dir / 'plots'}")
        print(f"Detailed logs saved in: {self.log_dir}")
        print(f"All experiment files saved in: {self.exp_dir}")



if __name__ == '__main__':
    # Create argument parser correctly
    parser = argparse.ArgumentParser(description='Train HSI to OCTA translation model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config JSON file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint for resuming training')
    parser.add_argument('--exp_id', type=str, default=None,
                        help='Experiment ID (will default to timestamp if not provided)')
    parser.add_argument('--loss_weights', nargs=4, type=float,
                        help='Override loss weights in format: pixel perceptual ssim adv')

    args = parser.parse_args()

    # Create trainer instance
    trainer = Trainer(config_path=args.config, exp_id=args.exp_id)

    # Override loss weights if specified
    if args.loss_weights:
        trainer.config['lambda_pixel'] = args.loss_weights[0]
        trainer.config['lambda_perceptual'] = args.loss_weights[1]
        trainer.config['lambda_ssim'] = args.loss_weights[2]
        trainer.config['lambda_adv'] = args.loss_weights[3]
        print(f"\nOverriding loss weights with custom values:")
        print(f"Pixel loss (L1): {trainer.config['lambda_pixel']}")
        print(f"Perceptual loss: {trainer.config['lambda_perceptual']}")
        print(f"SSIM loss: {trainer.config['lambda_ssim']}")
        print(f"Adversarial loss: {trainer.config['lambda_adv']}")

    # Check for resume in command line args (priority) or config
    resume_path = args.resume
    if not resume_path:
        # Check if resume is specified in config
        resume_config = trainer.config.get('resume', {})
        resume_enabled = resume_config.get('enabled', False)
        resume_path = resume_config.get('checkpoint_path', None)
        if not (resume_enabled and resume_path):
            resume_path = None

    try:
        # Resume training if a checkpoint path is available
        if resume_path:
            print(f"Resuming training from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=trainer.device)
            trainer.generator.load_state_dict(checkpoint['generator_state_dict'])
            trainer.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            trainer.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            trainer.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            trainer.best_val_loss = checkpoint.get('val_loss', float('inf'))

            # Load metrics history if available
            if 'metrics_history' in checkpoint:
                trainer.metrics_history = checkpoint['metrics_history']
                print(f"Loaded metrics history with {len(trainer.metrics_history['epoch'])} previous epochs")

                # Update the CSV file with previously recorded metrics
                if os.path.exists(trainer.csv_path):
                    # Backup existing CSV first
                    backup_path = trainer.csv_path.with_suffix('.bak')
                    shutil.copy2(trainer.csv_path, backup_path)
                    print(f"Backed up existing CSV to {backup_path}")

                # Rewrite the CSV with the loaded metrics history
                with open(trainer.csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # Write header
                    writer.writerow([
                        'epoch',
                        'g_loss_total',
                        'd_loss',
                        'total_train_loss',
                        'val_loss',
                        # Training unweighted losses
                        'pixel_loss_unweighted',
                        'ssim_loss_unweighted',
                        'perceptual_loss_unweighted',
                        'gan_loss_unweighted',
                        # Training weighted losses
                        'pixel_loss_weighted',
                        'ssim_loss_weighted',
                        'perceptual_loss_weighted',
                        'gan_loss_weighted',
                        # Validation unweighted losses
                        'val_pixel_loss_unweighted',
                        'val_ssim_loss_unweighted',
                        'val_perceptual_loss_unweighted',
                        'val_gan_loss_unweighted',
                        # Validation weighted losses
                        'val_pixel_loss_weighted',
                        'val_ssim_loss_weighted',
                        'val_perceptual_loss_weighted',
                        'val_gan_loss_weighted',
                        'learning_rate'
                    ])

                    # Write data for each epoch
                    for i in range(len(trainer.metrics_history['epoch'])):
                        # Get total training loss (compute if not available)
                        if 'total_train_loss' in trainer.metrics_history and i < len(trainer.metrics_history['total_train_loss']):
                            total_train_loss = trainer.metrics_history['total_train_loss'][i]
                        else:
                            # Calculate from component losses if available
                            total_train_loss = 0
                            if 'gan_loss_weighted' in trainer.metrics_history and i < len(trainer.metrics_history['gan_loss_weighted']):
                                total_train_loss += trainer.metrics_history['gan_loss_weighted'][i]
                            if 'pixel_loss_weighted' in trainer.metrics_history and i < len(trainer.metrics_history['pixel_loss_weighted']):
                                total_train_loss += trainer.metrics_history['pixel_loss_weighted'][i]
                            if 'perceptual_loss_weighted' in trainer.metrics_history and i < len(trainer.metrics_history['perceptual_loss_weighted']):
                                total_train_loss += trainer.metrics_history['perceptual_loss_weighted'][i]
                            if 'ssim_loss_weighted' in trainer.metrics_history and i < len(trainer.metrics_history['ssim_loss_weighted']):
                                total_train_loss += trainer.metrics_history['ssim_loss_weighted'][i]
                        
                        # Helper function to safely get values from metrics history
                        def get_metric(key, default=''):
                            if key in trainer.metrics_history and i < len(trainer.metrics_history[key]):
                                return trainer.metrics_history[key][i]
                            return default
                        
                        # Handle possible key naming differences in old checkpoints
                        val_pixel_loss_unweighted = get_metric('val_pixel_loss_unweighted')
                        if val_pixel_loss_unweighted == '' and 'val_pixel_loss' in trainer.metrics_history and i < len(trainer.metrics_history['val_pixel_loss']):
                            val_pixel_loss_unweighted = trainer.metrics_history['val_pixel_loss'][i]
                        
                        val_ssim_loss_unweighted = get_metric('val_ssim_loss_unweighted')
                        if val_ssim_loss_unweighted == '' and 'val_ssim_loss' in trainer.metrics_history and i < len(trainer.metrics_history['val_ssim_loss']):
                            val_ssim_loss_unweighted = trainer.metrics_history['val_ssim_loss'][i]
                            
                        val_perceptual_loss_unweighted = get_metric('val_perceptual_loss_unweighted')
                        if val_perceptual_loss_unweighted == '' and 'val_perceptual_loss' in trainer.metrics_history and i < len(trainer.metrics_history['val_perceptual_loss']):
                            val_perceptual_loss_unweighted = trainer.metrics_history['val_perceptual_loss'][i]
                            
                        val_gan_loss_unweighted = get_metric('val_gan_loss_unweighted')
                        if val_gan_loss_unweighted == '' and 'val_gan_loss' in trainer.metrics_history and i < len(trainer.metrics_history['val_gan_loss']):
                            val_gan_loss_unweighted = trainer.metrics_history['val_gan_loss'][i]
                        
                        # Calculate weighted validation losses if they're not already in history
                        val_pixel_loss_weighted = get_metric('val_pixel_loss_weighted')
                        if val_pixel_loss_weighted == '' and val_pixel_loss_unweighted != '':
                            try:
                                val_pixel_loss_weighted = float(val_pixel_loss_unweighted) * trainer.config['lambda_pixel'] if trainer.loss_components['pixel_enabled'] else 0
                            except:
                                val_pixel_loss_weighted = ''
                        
                        val_ssim_loss_weighted = get_metric('val_ssim_loss_weighted')
                        if val_ssim_loss_weighted == '' and val_ssim_loss_unweighted != '':
                            try:
                                val_ssim_loss_weighted = float(val_ssim_loss_unweighted) * trainer.config['lambda_ssim'] if trainer.loss_components['ssim_enabled'] else 0
                            except:
                                val_ssim_loss_weighted = ''
                                
                        val_perceptual_loss_weighted = get_metric('val_perceptual_loss_weighted')
                        if val_perceptual_loss_weighted == '' and val_perceptual_loss_unweighted != '':
                            try:
                                val_perceptual_loss_weighted = float(val_perceptual_loss_unweighted) * trainer.config['lambda_perceptual'] if trainer.loss_components['perceptual_enabled'] else 0
                            except:
                                val_perceptual_loss_weighted = ''
                                
                        val_gan_loss_weighted = get_metric('val_gan_loss_weighted')
                        if val_gan_loss_weighted == '' and val_gan_loss_unweighted != '':
                            try:
                                val_gan_loss_weighted = float(val_gan_loss_unweighted) * trainer.config['lambda_adv'] if trainer.loss_components['adversarial_enabled'] else 0
                            except:
                                val_gan_loss_weighted = ''
                        
                        writer.writerow([
                            get_metric('epoch'),
                            get_metric('g_loss_total'),
                            get_metric('d_loss'),
                            total_train_loss,
                            get_metric('val_loss'),
                            # Training unweighted losses
                            get_metric('pixel_loss_unweighted'),
                            get_metric('ssim_loss_unweighted'),
                            get_metric('perceptual_loss_unweighted'),
                            get_metric('gan_loss_unweighted'),
                            # Training weighted losses
                            get_metric('pixel_loss_weighted'),
                            get_metric('ssim_loss_weighted'),
                            get_metric('perceptual_loss_weighted'),
                            get_metric('gan_loss_weighted'),
                            # Validation unweighted losses
                            val_pixel_loss_unweighted,
                            val_ssim_loss_unweighted,
                            val_perceptual_loss_unweighted,
                            val_gan_loss_unweighted,
                            # Validation weighted losses
                            val_pixel_loss_weighted,
                            val_ssim_loss_weighted,
                            val_perceptual_loss_weighted,
                            val_gan_loss_weighted,
                            get_metric('learning_rate')
                        ])

                print(f"Restored metrics CSV file from checkpoint data")

            # Load loss components if available
            if 'loss_components' in checkpoint:
                trainer.loss_components = checkpoint['loss_components']
                print("\nLoaded loss component settings from checkpoint:")
                for component, enabled in trainer.loss_components.items():
                    component_name = component.replace('_enabled', '')
                    print(f"  - {component_name}: {'Enabled' if enabled else 'Disabled'}")
            else:
                print("\nNo loss component settings found in checkpoint, using defaults")

            # If resuming, we can use the experiment ID from the checkpoint
            if 'exp_id' in checkpoint and not args.exp_id:
                trainer.exp_id = checkpoint['exp_id']
                print(f"Using experiment ID from checkpoint: {trainer.exp_id}")

            # Check if the checkpoint was trained with circle cropping
            if 'circle_crop' in checkpoint:
                saved_crop = checkpoint['circle_crop']
                if saved_crop != trainer.use_circle_crop:
                    print(f"WARNING: Checkpoint was trained with circle_crop={saved_crop}, "
                          f"but current setting is circle_crop={trainer.use_circle_crop}")

            start_epoch = checkpoint['epoch'] + 1
            print(f"Starting training from epoch {start_epoch}/{trainer.config['num_epochs']}")

            # Start training from the checkpoint epoch
            trainer.train(start_epoch=start_epoch)
        else:
            # Start training from scratch
            trainer.train(start_epoch=0)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save missing files log
        if hasattr(trainer, 'log_dir'):
            output_dir = trainer.log_dir
        else:
            output_dir = Path(trainer.config['output']['base_dir']) / f"{trainer.exp_id}" / 'logs'
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created fallback log directory: {output_dir}")
        HSI_OCTA_Dataset_Cropped.save_missing_files_log(output_dir)
    except Exception as e:
        print(f"\nError occurred during training: {str(e)}")
        # Save missing files log
        if hasattr(trainer, 'log_dir'):
            output_dir = trainer.log_dir
        else:
            output_dir = Path(trainer.config['output']['base_dir']) / f"{trainer.exp_id}" / 'logs'
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created fallback log directory: {output_dir}")
        HSI_OCTA_Dataset_Cropped.save_missing_files_log(output_dir)
        raise