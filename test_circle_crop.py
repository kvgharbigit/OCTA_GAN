import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import argparse

# Import our modules
from circle_crop_utils import crop_and_resize, detect_and_crop_circle
from hsi_octa_dataset_cropped import HSI_OCTA_Dataset_Cropped
from visualization_utils import select_representative_wavelengths, normalize_wavelength_image


def visualize_samples(dataset, num_samples=3, save_path=None, show_cropped=True):
    """
    Visualize a few samples from the dataset to compare original and cropped images.

    Args:
        dataset: Dataset to sample from
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization
        show_cropped: Whether to apply cropping to samples
    """
    # Create figure with subplots
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    for i in range(num_samples):
        # Get a sample
        hsi_img, octa_img, patient_id = dataset[i]

        # Denormalize if necessary
        if hsi_img.min() < 0:
            hsi_img = (hsi_img + 1) / 2
            octa_img = (octa_img + 1) / 2

        # Apply cropping if requested
        if show_cropped:
            hsi_img_cropped, octa_img_cropped = crop_and_resize(hsi_img, octa_img, target_size=500)

        # Create RGB representation of HSI
        wavelength_indices = select_representative_wavelengths()
        hsi_np = hsi_img.cpu().numpy()

        rgb_hsi = np.stack([
            hsi_np[wavelength_indices['red']],
            hsi_np[wavelength_indices['green']],
            hsi_np[wavelength_indices['blue']]
        ])

        rgb_hsi = np.stack([
            normalize_wavelength_image(rgb_hsi[0]).squeeze().numpy(),
            normalize_wavelength_image(rgb_hsi[1]).squeeze().numpy(),
            normalize_wavelength_image(rgb_hsi[2]).squeeze().numpy()
        ])

        hsi_rgb = np.transpose(rgb_hsi, (1, 2, 0))

        # Create cropped version if requested
        if show_cropped:
            cropped_hsi_np = hsi_img_cropped.cpu().numpy()
            cropped_rgb_hsi = np.stack([
                cropped_hsi_np[wavelength_indices['red']],
                cropped_hsi_np[wavelength_indices['green']],
                cropped_hsi_np[wavelength_indices['blue']]
            ])

            cropped_rgb_hsi = np.stack([
                normalize_wavelength_image(cropped_rgb_hsi[0]).squeeze().numpy(),
                normalize_wavelength_image(cropped_rgb_hsi[1]).squeeze().numpy(),
                normalize_wavelength_image(cropped_rgb_hsi[2]).squeeze().numpy()
            ])

            cropped_hsi_rgb = np.transpose(cropped_rgb_hsi, (1, 2, 0))

        # Plot original or cropped HSI
        if show_cropped:
            axes[i, 0].imshow(hsi_rgb)
            axes[i, 0].set_title(f'Original HSI - {patient_id}')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(cropped_hsi_rgb)
            axes[i, 1].set_title('Cropped HSI')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(octa_img_cropped.squeeze().cpu().numpy(), cmap='gray')
            axes[i, 2].set_title('Cropped OCTA')
            axes[i, 2].axis('off')
        else:
            axes[i, 0].imshow(hsi_rgb)
            axes[i, 0].set_title(f'HSI - {patient_id}')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(octa_img.squeeze().cpu().numpy(), cmap='gray')
            axes[i, 1].set_title('OCTA')
            axes[i, 1].axis('off')

            # Empty third subplot
            axes[i, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved visualization to {save_path}")

    plt.show()


def test_crop_and_resize():
    """
    Test the circle detection and cropping functions on a sample image.
    """
    # Create a simple test image
    size = 200
    channels = 31

    # Create a dark background with a bright circle
    test_image = torch.zeros(channels, size, size)

    # Draw a circle
    center = size // 2
    radius = size // 3

    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    mask = ((x - center) ** 2 + (y - center) ** 2) <= radius ** 2

    for c in range(channels):
        test_image[c][mask] = 1.0

    # Create a test OCTA image
    test_octa = torch.zeros(1, size, size)
    test_octa[0][mask] = 1.0

    # Apply cropping
    cropped_hsi, cropped_octa = crop_and_resize(test_image, test_octa, target_size=100)

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(test_image[0].numpy(), cmap='gray')
    axes[0, 0].set_title('Original HSI (1st channel)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(test_octa[0].numpy(), cmap='gray')
    axes[0, 1].set_title('Original OCTA')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(cropped_hsi[0].numpy(), cmap='gray')
    axes[1, 0].set_title('Cropped HSI (1st channel)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(cropped_octa[0].numpy(), cmap='gray')
    axes[1, 1].set_title('Cropped OCTA')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('synthetic_test.png', dpi=150)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Test circle detection and cropping')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--test_synthetic', action='store_true',
                        help='Run test on synthetic data')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of samples to visualize')
    parser.add_argument('--save_path', type=str, default="circle_crop_comparison.png",
                        help='Path to save the visualization')

    args = parser.parse_args()

    if args.test_synthetic:
        test_crop_and_resize()

    # Set up transform
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Create dataset without cropping
    original_dataset = HSI_OCTA_Dataset_Cropped(
        data_dir=args.data_dir,
        transform=transform,
        split='train',
        augment=False,
        circle_crop=False
    )

    print(f"Dataset size: {len(original_dataset)}")

    # Visualize samples
    visualize_samples(original_dataset, num_samples=args.num_samples, save_path=args.save_path)


if __name__ == "__main__":
    main()