import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import csv
import json
from datetime import datetime


def select_representative_wavelengths(total_wavelengths=31, min_wavelength=450, max_wavelength=905):
    """
    Select representative wavelengths across the spectral range,
    considering every third wavelength from the original 91 wavelengths.

    Args:
        total_wavelengths (int): Total number of wavelengths (31 from every 3rd of 91)
        min_wavelength (int): Minimum wavelength in nm
        max_wavelength (int): Maximum wavelength in nm

    Returns:
        dict: Mapping of color channels to wavelength indices
    """
    # Generate wavelength array for the 31 wavelengths
    # Mapping back to the original 91 wavelengths range
    full_wavelengths = np.linspace(min_wavelength, max_wavelength, 91)
    sampled_wavelengths = full_wavelengths[::3][:total_wavelengths]

    # Define key wavelengths for human vision
    wavelength_map = {
        'red': 660,  # Deep red
        'green': 555,  # Peak green perception
        'blue': 475  # Blue-green border
    }

    # Find indices closest to these wavelengths in the sampled wavelengths
    wavelength_indices = {}
    for color, target_wavelength in wavelength_map.items():
        wavelength_indices[color] = np.argmin(np.abs(sampled_wavelengths - target_wavelength))

    return wavelength_indices


def normalize_wavelength_image(wavelength_image):
    """
    Normalize an image for display, handling potential outliers.

    Args:
        wavelength_image (torch.Tensor or np.ndarray): Single wavelength image

    Returns:
        torch.Tensor: Normalized image
    """
    import numpy as np
    import torch

    # Convert to numpy if it's a tensor
    if torch.is_tensor(wavelength_image):
        img_np = wavelength_image.squeeze().cpu().numpy()
    else:
        img_np = np.squeeze(wavelength_image)

    # Clip to remove extreme outliers (99th percentile)
    lower = np.percentile(img_np, 1)
    upper = np.percentile(img_np, 99)

    # Clip and normalize
    img_np = np.clip(img_np, lower, upper)

    # Min-max normalization
    if upper > lower:
        img_np = (img_np - lower) / (upper - lower)

    return torch.from_numpy(img_np).unsqueeze(0)


def save_sample_visualizations(generator, val_loader, device, epoch, output_dir, log_dir=None, num_samples=3):
    """
    Generate and save sample visualizations during training with memory optimizations.

    Args:
        generator (nn.Module): The generator model
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to run the model on
        epoch (int): Current training epoch
        output_dir (Path): Directory to save visualizations
        log_dir (Path, optional): Directory to save log files
        num_samples (int, optional): Number of samples to visualize
    """
    print(f"\n==== Visualization Debug ====")
    print(f"Epoch: {epoch}")
    print(f"Requested samples: {num_samples}")
    print(f"Validation loader batch size: {val_loader.batch_size}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    print(f"Output directory: {output_dir}")
    
    generator.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

    # Use mixed precision for visualization generation, but with updated API
    with torch.no_grad():
        # Use standard precision for visualization to avoid type mismatches
        # Create a data iterator to get samples
        val_iterator = iter(val_loader)
        
        print("Collecting samples for visualization...")
        
        # Collect enough samples
        hsi_samples = []
        octa_samples = []
        patient_id_samples = []
        
        # Collect requested number of samples, potentially from multiple batches
        samples_needed = num_samples
        batch_count = 0
        
        while samples_needed > 0:
            try:
                hsi_batch, octa_batch, patient_ids = next(val_iterator)
                batch_count += 1
                
                print(f"  Batch {batch_count}: {len(hsi_batch)} samples, patient IDs: {patient_ids}")
                
                # Take as many samples as needed from this batch
                samples_to_take = min(samples_needed, len(hsi_batch))
                
                hsi_samples.append(hsi_batch[:samples_to_take])
                octa_samples.append(octa_batch[:samples_to_take])
                patient_id_samples.extend(patient_ids[:samples_to_take])
                
                print(f"  - Added {samples_to_take} samples, total now: {len(patient_id_samples)}")
                
                samples_needed -= samples_to_take
                print(f"  - Still need {samples_needed} more samples")
                
                if samples_needed <= 0:
                    print(f"  - Collected all {num_samples} requested samples")
                    break
                    
            except StopIteration:
                # If we run out of batches, restart the iterator
                print(f"  - Reached end of validation dataset after {batch_count} batches")
                val_iterator = iter(val_loader)
                # If we couldn't get enough samples, just use what we have
                if samples_needed > 0:
                    print(f"WARNING: Could only collect {num_samples - samples_needed} samples for visualization instead of {num_samples}")
                break
        
        # Concatenate all collected samples
        if len(hsi_samples) > 0:
            hsi_batch = torch.cat(hsi_samples, dim=0)
            octa_batch = torch.cat(octa_samples, dim=0)
            
            # Make sure we have the right number of samples
            num_samples = len(hsi_batch)
            print(f"Final number of samples for visualization: {num_samples}")
            print(f"Patient IDs to visualize: {patient_id_samples}")
        else:
            print("ERROR: No samples could be collected for visualization")
            return

        # Create a patient-specific output directory
        patient_vis_dir = output_dir / f"epoch_{epoch}"
        patient_vis_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created patient-specific visualization directory: {patient_vis_dir}")

        # List to track all visualization paths
        all_vis_paths = []

        for i in range(num_samples):
            # Get a single sample - process one at a time to save memory
            hsi = hsi_batch[i:i+1].to(device, non_blocking=True)
            octa = octa_batch[i:i+1].to(device, non_blocking=True)
            patient_id = patient_id_samples[i]

            # Clean the patient ID to make it safe for filenames
            safe_patient_id = patient_id.replace('/', '_').replace('\\', '_')

            # Generate fake OCTA image
            fake_octa = generator(hsi)

            # Denormalize images to [0, 1] range for visualization
            hsi_norm = (hsi + 1) / 2  # Assuming normalization was [-1, 1]
            octa_norm = (octa + 1) / 2
            fake_octa_norm = (fake_octa + 1) / 2

            # Process one by one, move to CPU, and free GPU memory
            hsi_np = hsi_norm.cpu().squeeze().numpy()
            del hsi, hsi_norm  # Free memory
            
            octa_np = octa_norm.cpu().squeeze().numpy()
            del octa, octa_norm  # Free memory
            
            fake_octa_np = fake_octa_norm.cpu().squeeze().numpy()
            del fake_octa, fake_octa_norm  # Free memory

            # Select representative wavelength indices
            wavelength_indices = select_representative_wavelengths()

            # Create RGB-like representation of HSI efficiently
            rgb_channels = []
            for color in ['red', 'green', 'blue']:
                # Process one channel at a time to save memory
                channel = hsi_np[wavelength_indices[color]]
                normalized = normalize_wavelength_image(channel).squeeze().cpu().numpy()
                rgb_channels.append(normalized)
                del channel  # Free memory but keep normalized for stacking
            
            # Stack channels, avoiding interim large tensors
            rgb_hsi = np.stack(rgb_channels)
            del rgb_channels  # Free memory
            
            # Transpose to HWC for plotting
            rgb_hsi = np.transpose(rgb_hsi, (1, 2, 0))
            
            # Create a new figure for this patient
            plt.figure(figsize=(15, 5))
            
            # Plot HSI input
            plt.subplot(1, 3, 1)
            plt.imshow(rgb_hsi)
            plt.title(f"HSI Input (RGB-like)\n{patient_id}")
            plt.axis('off')
            
            # Plot generated OCTA
            plt.subplot(1, 3, 2)
            plt.imshow(fake_octa_np, cmap='gray')
            plt.title(f"Generated OCTA")
            plt.axis('off')
            
            # Plot real OCTA
            plt.subplot(1, 3, 3)
            plt.imshow(octa_np, cmap='gray')
            plt.title(f"Real OCTA")
            plt.axis('off')
            
            # Save this patient's visualization
            plt.tight_layout()
            patient_vis_path = patient_vis_dir / f'sample_{safe_patient_id}.png'
            plt.savefig(patient_vis_path, bbox_inches='tight', dpi=100)
            plt.close()
            
            print(f"Saved visualization for patient {patient_id} to {patient_vis_path}")
            all_vis_paths.append(patient_vis_path)
            
            # After saving, we can clear memory
            del rgb_hsi, fake_octa_np, octa_np
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Also create the combined visualization
        plt.figure(figsize=(15, 5 * num_samples))
        
        for i in range(num_samples):
            # Load the individual patient images we just saved
            patient_img = plt.imread(str(all_vis_paths[i]))
            
            # Add it to the combined figure
            plt.subplot(num_samples, 1, i+1)
            plt.imshow(patient_img)
            plt.axis('off')
            plt.title(f"Patient: {patient_id_samples[i]}")
        
        # Save the combined visualization
        plt.tight_layout()
        vis_path = output_dir / f'epoch_{epoch}_samples.png'
        plt.savefig(vis_path, bbox_inches='tight', dpi=100)
        plt.close('all')  # Close all figures to free memory
        
        # Clean up remaining arrays to fully free memory
        # At this point, it's safe to delete these as the figure has been saved
        del hsi_batch, octa_batch, patient_ids
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache one final time
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Log the visualization path if log_dir is provided
        if log_dir is not None:
            with open(log_dir / 'visualization_log.txt', 'a') as f:
                f.write(f"Epoch {epoch}: Saved visualization to {vis_path}\n")
        
        print(f"Saved visualization for epoch {epoch} to {vis_path}")
        print(f"==== End Visualization Debug ====\n")


def save_loss_plots(metrics_history, output_dir, log_dir=None):
    """
    Generate and save plots for training and validation losses.
    
    Args:
        metrics_history (dict): Dictionary containing loss metrics over time
        output_dir (Path): Directory to save the plots
        log_dir (Path, optional): Directory to save log files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot training losses
    plt.figure(figsize=(12, 8))
    plt.plot(metrics_history['epoch'], metrics_history['g_loss_total'], label='Generator Loss')
    plt.plot(metrics_history['epoch'], metrics_history['d_loss'], label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    train_loss_path = output_dir / 'training_losses.png'
    plt.savefig(train_loss_path)
    plt.close()
    
    # Plot validation loss
    if 'val_loss' in metrics_history and metrics_history['val_loss']:
        plt.figure(figsize=(12, 8))
        
        # Get actual epochs where validation was performed
        # Filter out empty values (validation not performed)
        val_loss_values = [v for v in metrics_history['val_loss'] if v != '']
        
        # Find which epochs have validation data (typically every N epochs)
        epochs = metrics_history['epoch']
        val_epochs = []
        val_indices = []
        
        for i, epoch in enumerate(epochs):
            # Check if we have validation data for this epoch
            if i < len(metrics_history['val_loss']) and metrics_history['val_loss'][i] != '':
                val_epochs.append(epoch)
                val_indices.append(i)
        
        # Plot validation loss at correct epochs
        plt.plot(val_epochs, val_loss_values, 'ro-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        val_loss_path = output_dir / 'validation_loss.png'
        plt.savefig(val_loss_path)
        plt.close()
    
    # Plot total training loss (combined components) vs validation loss
    plt.figure(figsize=(12, 8))
    
    # Calculate and plot total weighted loss (sum of all weighted loss components)
    if all(key in metrics_history for key in ['gan_loss_weighted', 'pixel_loss_weighted', 'ssim_loss_weighted']):
        # Create arrays with zeros for any missing values
        epochs = metrics_history['epoch']
        total_loss = np.zeros(len(epochs))
        
        # Add each component
        if 'pixel_loss_weighted' in metrics_history:
            total_loss += np.array(metrics_history['pixel_loss_weighted'])
        if 'ssim_loss_weighted' in metrics_history:
            total_loss += np.array(metrics_history['ssim_loss_weighted'])
        if 'gan_loss_weighted' in metrics_history:
            total_loss += np.array(metrics_history['gan_loss_weighted'])
        if 'perceptual_loss_weighted' in metrics_history:
            total_loss += np.array(metrics_history['perceptual_loss_weighted'])
            
        # Plot total training loss
        plt.plot(epochs, total_loss, 'b-', label='Total Training Loss (Combined Components)')
        
        # Plot validation loss on same graph for comparison
        if 'val_loss' in metrics_history and metrics_history['val_loss']:
            # Get actual epochs where validation was performed
            val_loss_values = []
            val_epochs = []
            
            for i, epoch in enumerate(epochs):
                if i < len(metrics_history['val_loss']) and metrics_history['val_loss'][i] != '':
                    val_epochs.append(epoch)
                    val_loss_values.append(metrics_history['val_loss'][i])
            
            if val_epochs and val_loss_values:
                plt.plot(val_epochs, val_loss_values, 'ro-', label='Validation Loss', markersize=4)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Total Training Loss vs Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the combined plot
        combined_loss_path = output_dir / 'total_vs_validation_loss.png'
        plt.savefig(combined_loss_path)
        plt.close()
        
        # Log the new plot
        if log_dir is not None:
            with open(log_dir / 'plot_log.txt', 'a') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{timestamp}: Saved total vs validation loss plot to {combined_loss_path}\n")
    
    # Plot component losses
    plt.figure(figsize=(12, 8))
    
    # Only plot enabled loss components (non-zero values)
    component_losses = [
        ('pixel_loss_unweighted', 'Pixel Loss (L1)'),
        ('perceptual_loss_unweighted', 'Perceptual Loss'),
        ('ssim_loss_unweighted', 'SSIM Loss'),
        ('gan_loss_unweighted', 'GAN Loss')
    ]
    
    for key, label in component_losses:
        if key in metrics_history and any(v > 0 for v in metrics_history[key]):
            plt.plot(metrics_history['epoch'], metrics_history[key], label=label)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Loss Components (Unweighted)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    components_path = output_dir / 'loss_components.png'
    plt.savefig(components_path)
    plt.close()
    
    # Plot learning rate
    if 'learning_rate' in metrics_history and metrics_history['learning_rate']:
        plt.figure(figsize=(12, 8))
        plt.plot(metrics_history['epoch'], metrics_history['learning_rate'], label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for better visualization
        lr_path = output_dir / 'learning_rate.png'
        plt.savefig(lr_path)
        plt.close()
    
    # Log the plot paths if log_dir is provided
    if log_dir is not None:
        with open(log_dir / 'plot_log.txt', 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp}: Saved training loss plot to {train_loss_path}\n")
            if 'val_loss' in metrics_history and metrics_history['val_loss']:
                f.write(f"{timestamp}: Saved validation loss plot to {val_loss_path}\n")
            f.write(f"{timestamp}: Saved loss components plot to {components_path}\n")
            if 'learning_rate' in metrics_history and metrics_history['learning_rate']:
                f.write(f"{timestamp}: Saved learning rate plot to {lr_path}\n")


def log_metrics(metrics, log_dir, epoch, is_training=True):
    """
    Log metrics to a JSON file for each epoch.
    
    Args:
        metrics (dict): Dictionary of metrics to log
        log_dir (Path): Directory to save log files
        epoch (int): Current epoch number
        is_training (bool): Whether the metrics are from training or validation
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine the log filename based on whether it's training or validation
    if is_training:
        log_file = log_dir / 'training_metrics.json'
        prefix = 'train'
    else:
        log_file = log_dir / 'validation_metrics.json'
        prefix = 'val'
    
    # Load existing log file if it exists
    if log_file.exists():
        with open(log_file, 'r') as f:
            try:
                log_data = json.load(f)
            except json.JSONDecodeError:
                log_data = {}
    else:
        log_data = {}
    
    # Create epoch entry if it doesn't exist
    if str(epoch) not in log_data:
        log_data[str(epoch)] = {}
    
    # Add metrics to the epoch entry
    for key, value in metrics.items():
        # Convert tensors to floats if needed
        if isinstance(value, torch.Tensor):
            value = value.item()
        log_data[str(epoch)][f"{prefix}_{key}"] = value
    
    # Save updated log file
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    # Also save as text for easier reading
    text_log_file = log_file.with_suffix('.txt')
    with open(text_log_file, 'a') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Epoch {epoch} ({timestamp}):\n")
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            f.write(f"  {prefix}_{key}: {value:.6f}\n")
        f.write("\n")


def save_image_grid(img_tensor, filepath):
    """
    Save a grid of images to a file.

    Args:
        img_tensor (torch.Tensor): Image tensor, shape [num_images, C, H, W]
        filepath (Path): Path to save the image
    """
    # Create grid of images
    grid = vutils.make_grid(img_tensor, normalize=True, scale_each=True)

    # Convert to numpy and plot
    plt.figure(figsize=(10, 5))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
    plt.close()