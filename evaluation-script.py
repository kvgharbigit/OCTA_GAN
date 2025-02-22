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

from base import (
    HSI_OCTA_Dataset, Generator, Discriminator,
    PerceptualLoss, SSIMLoss, TrainingConfig
)

class Evaluator:
    def __init__(self, checkpoint_path: str):
        """Initialize the evaluator with a trained model checkpoint."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.generator = Generator().to(self.device)
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.epoch = checkpoint.get('epoch', 'unknown')
        
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
        
    def setup_data(self, data_dir: str, batch_size: int = 1):
        """Setup the test dataset and dataloader."""
        print(f"Setting up test dataset from {data_dir}")
        
        # Create test dataset
        self.test_dataset = HSI_OCTA_Dataset(
            data_dir=data_dir,
            transform=self.transform,
            split='test',
            target_size=500,
            augment=False
        )
        
        print(f"Test dataset size: {len(self.test_dataset)} samples")
        
        # Create dataloader
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
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
    
    def save_comparison(self, fake_octa: torch.Tensor, real_octa: torch.Tensor, 
                       patient_id: str, save_dir: Path):
        """Save a visual comparison between generated and real OCTA images."""
        # Denormalize tensors
        fake_octa = self.denormalize(fake_octa)
        real_octa = self.denormalize(real_octa)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot images
        ax1.imshow(fake_octa.cpu().squeeze(), cmap='gray')
        ax1.set_title('Generated OCTA')
        ax1.axis('off')
        
        ax2.imshow(real_octa.cpu().squeeze(), cmap='gray')
        ax2.set_title('Real OCTA')
        ax2.axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(save_dir / f'{patient_id}_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate(self, data_dir: str, results_dir: str):
        """Run full evaluation and save results."""
        # Setup data
        self.setup_data(data_dir)
        
        # Create results directory
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        vis_dir = results_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        print("Starting evaluation...")
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
                self.save_comparison(fake_octa[0], real_octa[0], patient_id[0], vis_dir)
        
        # Create results summary
        self.save_results(results_dir)
        
        print("Evaluation completed!")
    
    def save_results(self, results_dir: Path):
        """Save evaluation results and generate summary visualizations."""
        # Convert metrics to DataFrame
        df = pd.DataFrame(self.metrics)
        
        # Save raw metrics
        df.to_csv(results_dir / 'metrics.csv', index=False)
        
        # Calculate summary statistics
        summary = df.describe()
        summary.to_csv(results_dir / 'summary_statistics.csv')
        
        # Create metric distributions plot
        plt.figure(figsize=(15, 10))
        for i, metric in enumerate(['psnr', 'ssim', 'mse', 'mae'], 1):
            plt.subplot(2, 2, i)
            sns.histplot(data=df, x=metric, kde=True)
            plt.title(f'{metric.upper()} Distribution')
        plt.tight_layout()
        plt.savefig(results_dir / 'metric_distributions.png')
        plt.close()
        
        # Save configuration
        config = {
            'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'checkpoint_epoch': self.epoch,
            'device': str(self.device),
            'num_test_samples': len(self.test_dataset),
            'mean_metrics': {
                metric: float(np.mean(values)) 
                for metric, values in self.metrics.items() 
                if metric != 'patient_id'
            }
        }
        
        with open(results_dir / 'evaluation_config.json', 'w') as f:
            json.dump(config, f, indent=4)
        
        # Print summary
        print("\nEvaluation Summary:")
        print("-" * 50)
        for metric, values in self.metrics.items():
            if metric != 'patient_id':
                print(f"Mean {metric.upper()}: {np.mean(values):.4f} ± {np.std(values):.4f}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate HSI to OCTA translation model')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to test data directory')
    parser.add_argument('--results_dir', type=str, required=True,
                      help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = Evaluator(args.checkpoint)
    evaluator.evaluate(args.data_dir, args.results_dir)