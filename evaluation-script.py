import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pandas as pd
import seaborn as sns
import json
from datetime import datetime
import os
import shutil

from base import (
    HSI_OCTA_Dataset, Generator, Discriminator,
    PerceptualLoss, SSIMLoss, TrainingConfig
)
from config_utils import load_config
from visualization_utils import (
    select_representative_wavelengths,
    normalize_wavelength_image
)


# Custom JSON encoder to handle Path objects
class PathEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


class Evaluator:
    def __init__(self, config_path: str, exp_id: str = None):
        """Initialize the evaluator with a config file."""
        # Load configuration
        self.config = load_config(config_path)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Get checkpoint path from config
        checkpoint_path = self.config.get('evaluation', {}).get('checkpoint_path')
        if not checkpoint_path:
            raise ValueError("Missing checkpoint_path in configuration")

        # Initialize model
        self.generator = Generator().to(self.device)

        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.epoch = checkpoint.get('epoch', 'unknown')

        # Set experiment ID
        if exp_id:
            self.exp_id = exp_id
        elif 'exp_id' in checkpoint:
            # Extract exp_id from checkpoint if available
            self.exp_id = checkpoint['exp_id']
            print(f"Using experiment ID from checkpoint: {self.exp_id}")
        else:
            # Use experiment ID from config or generate a timestamp-based one
            self.exp_id = self.config.get('evaluation', {}).get('exp_id',
                                                                datetime.now().strftime("%Y%m%d_%H%M%S"))
            print(f"Using experiment ID: {self.exp_id}")

        # Set output base directory
        self.output_base_dir = Path(self.config.get('evaluation', {}).get('output_dir', './output'))

        # Create experiment directory structure
        self.exp_dir = self.output_base_dir / f"experiment_{self.exp_id}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using experiment directory: {self.exp_dir}")

        # Create evaluation directory within experiment directory
        self.eval_dir = self.exp_dir / 'evaluation'
        self.eval_dir.mkdir(exist_ok=True)

        # Set model to evaluation mode
        self.generator.eval()

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Initialize metrics dictionary
        self.metrics = {
            'patient_id': [],
            'psnr': [],
            'ssim': [],
            'mse': [],
            'mae': []
        }

        # Copy checkpoint and config file to the experiment directory for completeness
        checkpoint_filename = Path(checkpoint_path).name
        config_filename = Path(config_path).name

        shutil.copy(checkpoint_path, self.exp_dir / checkpoint_filename)
        shutil.copy(config_path, self.exp_dir / config_filename)
        print(f"Copied checkpoint and config to experiment directory")

    def setup_data(self):
        """Setup the test dataset and dataloader."""

        data_dir = self.config.get('evaluation', {}).get('data_dir')
        print(data_dir)
        if not data_dir:
            raise ValueError("Missing data_dir in configuration")

        print(f"Setting up test dataset from {data_dir}")

        # Get batch size from config or use default
        batch_size = self.config.get('evaluation', {}).get('batch_size', 1)

        # Create test dataset
        self.test_dataset = HSI_OCTA_Dataset(
            data_dir=data_dir,
            transform=self.transform,
            split='test',
            target_size=self.config.get('data', {}).get('target_size', 500),
            augment=False
        )

        print(f"Test dataset size: {len(self.test_dataset)} samples")

        # Create dataloader
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.get('hardware', {}).get('num_workers', 4),
            pin_memory=self.config.get('hardware', {}).get('pin_memory', True)
        )

    def denormalize(self, tensor):
        """Denormalize the tensor from [-1,1] to [0,1] range."""
        return (tensor + 1) / 2

    def compute_metrics(self, fake_octa: torch.Tensor, real_octa: torch.Tensor) -> dict:
        """Compute various image quality metrics."""
        # Denormalize tensors
        fake_octa = self.denormalize(fake_octa)
        real_octa = self.denormalize(real_octa)

        # Convert to numpy arrays
        fake_np = fake_octa.cpu().numpy().squeeze()
        real_np = real_octa.cpu().numpy().squeeze()

        # Compute metrics
        psnr_value = psnr(real_np, fake_np, data_range=1.0)
        ssim_value = ssim(real_np, fake_np, data_range=1.0)
        mse_value = np.mean((real_np - fake_np) ** 2)
        mae_value = np.mean(np.abs(real_np - fake_np))

        return {
            'psnr': psnr_value,
            'ssim': ssim_value,
            'mse': mse_value,
            'mae': mae_value
        }

    def save_comparison(self, hsi: torch.Tensor, fake_octa: torch.Tensor, real_octa: torch.Tensor,
                        patient_id: str, save_dir: Path):
        """Save a visual comparison between HSI, generated, and real OCTA images."""
        import numpy as np

        # Denormalize tensors
        hsi = self.denormalize(hsi)
        fake_octa = self.denormalize(fake_octa)
        real_octa = self.denormalize(real_octa)

        # Select representative wavelength indices
        wavelength_indices = select_representative_wavelengths()

        # Create RGB-like representation of HSI
        hsi_np = hsi.squeeze().cpu().numpy()

        # Select wavelength indices for RGB-like representation
        rgb_hsi = np.stack([
            hsi_np[wavelength_indices['red']],  # Red channel
            hsi_np[wavelength_indices['green']],  # Green channel
            hsi_np[wavelength_indices['blue']]  # Blue channel
        ])

        # Normalize each channel
        rgb_hsi = np.stack([
            normalize_wavelength_image(rgb_hsi[0]).squeeze(),
            normalize_wavelength_image(rgb_hsi[1]).squeeze(),
            normalize_wavelength_image(rgb_hsi[2]).squeeze()
        ])

        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Plot HSI in RGB-like representation
        hsi_rgb = np.transpose(rgb_hsi, (1, 2, 0))  # Convert to HWC
        ax1.imshow(hsi_rgb)
        ax1.set_title('HSI (RGB-like)')
        ax1.axis('off')

        # Plot generated OCTA
        ax2.imshow(fake_octa.cpu().squeeze(), cmap='gray')
        ax2.set_title('Generated OCTA')
        ax2.axis('off')

        # Plot real OCTA
        ax3.imshow(real_octa.cpu().squeeze(), cmap='gray')
        ax3.set_title('Real OCTA')
        ax3.axis('off')

        # Save figure in the experiment's visualization directory
        plt.tight_layout()
        plt.savefig(save_dir / f'{patient_id}_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def evaluate(self):
        """Run full evaluation and save results."""
        # Setup data
        self.setup_data()

        # Create visualization directory
        vis_dir = self.eval_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)

        print(f"Starting evaluation for experiment {self.exp_id}...")
        with torch.no_grad():
            for hsi, real_octa, patient_id in tqdm(self.test_loader):
                # Move data to device
                hsi = hsi.to(self.device)
                real_octa = real_octa.to(self.device)

                # Generate fake OCTA
                fake_octa = self.generator(hsi)

                # Compute metrics
                batch_metrics = self.compute_metrics(fake_octa, real_octa)

                # Store metrics
                self.metrics['patient_id'].append(patient_id[0])
                for k, v in batch_metrics.items():
                    self.metrics[k].append(v)

                # Save visualization
                self.save_comparison(hsi[0], fake_octa[0], real_octa[0], patient_id[0], vis_dir)

        # Create results summary
        self.save_results()

        print("Evaluation completed!")

    def save_results(self):
        """Save evaluation results and generate summary visualizations."""
        # Convert metrics to DataFrame
        df = pd.DataFrame(self.metrics)

        # Save raw metrics
        df.to_csv(self.eval_dir / 'metrics.csv', index=False)

        # Calculate summary statistics
        summary = df.describe()
        summary.to_csv(self.eval_dir / 'summary_statistics.csv')

        # Create metric distributions plot
        plt.figure(figsize=(15, 10))
        for i, metric in enumerate(['psnr', 'ssim', 'mse', 'mae'], 1):
            plt.subplot(2, 2, i)
            sns.histplot(data=df, x=metric, kde=True)
            plt.title(f'{metric.upper()} Distribution')
        plt.tight_layout()
        plt.savefig(self.eval_dir / 'metric_distributions.png')
        plt.close()

        # Create visualization of best and worst cases
        self.save_extreme_cases(df)

        # Save configuration
        eval_config = {
            'experiment_id': self.exp_id,
            'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'checkpoint_epoch': self.epoch,
            'device': str(self.device),
            'num_test_samples': len(self.test_dataset),
            'config_file': self.config,
            'mean_metrics': {
                metric: float(np.mean(values))
                for metric, values in self.metrics.items()
                if metric != 'patient_id'
            }
        }

        with open(self.eval_dir / 'evaluation_config.json', 'w') as f:
            json.dump(eval_config, f, indent=4, cls=PathEncoder)

        # Print summary
        print("\nEvaluation Summary:")
        print("-" * 50)
        print(f"Experiment ID: {self.exp_id}")
        print(f"Model checkpoint epoch: {self.epoch}")
        print(f"Number of test samples: {len(self.test_dataset)}")
        print("-" * 50)
        for metric, values in self.metrics.items():
            if metric != 'patient_id':
                print(f"Mean {metric.upper()}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        print(f"Results saved to: {self.eval_dir}")

    def save_extreme_cases(self, df: pd.DataFrame):
        """Save visualizations of the best and worst cases based on SSIM."""
        # Find best and worst cases by SSIM
        best_idx = df['ssim'].idxmax()
        worst_idx = df['ssim'].idxmin()

        best_case = df.iloc[best_idx]
        worst_case = df.iloc[worst_idx]

        # Create summary file
        with open(self.eval_dir / 'extreme_cases.txt', 'w') as f:
            f.write("Best Case (by SSIM):\n")
            f.write(f"Patient ID: {best_case['patient_id']}\n")
            f.write(f"SSIM: {best_case['ssim']:.4f}\n")
            f.write(f"PSNR: {best_case['psnr']:.4f}\n")
            f.write(f"MSE: {best_case['mse']:.4f}\n")
            f.write(f"MAE: {best_case['mae']:.4f}\n\n")

            f.write("Worst Case (by SSIM):\n")
            f.write(f"Patient ID: {worst_case['patient_id']}\n")
            f.write(f"SSIM: {worst_case['ssim']:.4f}\n")
            f.write(f"PSNR: {worst_case['psnr']:.4f}\n")
            f.write(f"MSE: {worst_case['mse']:.4f}\n")
            f.write(f"MAE: {worst_case['mae']:.4f}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate HSI to OCTA translation model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to evaluation config JSON file')
    parser.add_argument('--exp_id', type=str, default=None,
                        help='Experiment ID (optional - will use ID from checkpoint or config if not provided)')

    args = parser.parse_args()

    try:
        # Run evaluation
        evaluator = Evaluator(args.config, args.exp_id)
        evaluator.evaluate()
    except Exception as e:
        print(f"\nError occurred during evaluation: {str(e)}")
        raise