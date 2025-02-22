import torch
import numpy as np
import h5py
import os
from PIL import Image
import tempfile
import unittest
from base import (
    HSI_OCTA_Dataset,
    Generator,
    Discriminator,
    SelfAttention,
    PerceptualLoss,
    SSIMLoss
)


class TestBaseComponents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create temporary test data"""
        cls.temp_dir = tempfile.mkdtemp()

        # Create sample HSI data
        cls.hsi_data = np.random.rand(64, 64, 100).astype(np.float32)
        cls.hsi_file = os.path.join(cls.temp_dir, 'test.h5')
        with h5py.File(cls.hsi_file, 'w') as f:
            f.create_dataset('hsi', data=cls.hsi_data)

        # Create sample OCTA data
        cls.octa_data = (np.random.rand(64, 64) * 255).astype(np.uint8)
        cls.octa_file = cls.hsi_file.replace('.h5', '.jpeg')
        Image.fromarray(cls.octa_data).save(cls.octa_file)

        # Set device
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {cls.device}")

    def test_dataset(self):
        """Test HSI_OCTA_Dataset functionality"""
        print("\nTesting Dataset...")

        # Test dataset initialization
        dataset = HSI_OCTA_Dataset(self.temp_dir, augment=True)
        self.assertEqual(len(dataset), 1)

        # Test data loading
        hsi, octa = dataset[0]
        self.assertEqual(hsi.shape, (100, 64, 64))  # (C, H, W)
        self.assertEqual(octa.shape, (1, 64, 64))  # (1, H, W)
        print("Dataset tests passed")

    def test_generator(self):
        """Test Generator architecture"""
        print("\nTesting Generator...")

        # Initialize generator
        generator = Generator(spectral_channels=100).to(self.device)

        # Test forward pass
        batch_size = 2
        input_tensor = torch.randn(batch_size, 100, 64, 64).to(self.device)
        output = generator(input_tensor)

        self.assertEqual(output.shape, (batch_size, 1, 64, 64))
        print("Generator tests passed")

    def test_discriminator(self):
        """Test Discriminator architecture"""
        print("\nTesting Discriminator...")

        # Initialize discriminator
        discriminator = Discriminator().to(self.device)

        # Test forward pass
        batch_size = 2
        input_tensor = torch.randn(batch_size, 1, 64, 64).to(self.device)
        output = discriminator(input_tensor)

        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], 1)
        print("Discriminator tests passed")

    def test_self_attention(self):
        """Test SelfAttention module"""
        print("\nTesting SelfAttention...")

        # Initialize self-attention
        attention = SelfAttention(64).to(self.device)

        # Test forward pass
        input_tensor = torch.randn(2, 64, 32, 32).to(self.device)
        output = attention(input_tensor)

        self.assertEqual(output.shape, input_tensor.shape)
        print("SelfAttention tests passed")

    def test_perceptual_loss(self):
        """Test PerceptualLoss module"""
        print("\nTesting PerceptualLoss...")

        # Initialize perceptual loss
        perceptual_loss = PerceptualLoss().to(self.device)

        # Test forward pass
        img1 = torch.randn(2, 1, 64, 64).to(self.device)
        img2 = torch.randn(2, 1, 64, 64).to(self.device)
        loss = perceptual_loss(img1, img2)

        self.assertTrue(isinstance(loss.item(), float))
        print("PerceptualLoss tests passed")

    def test_ssim_loss(self):
        """Test SSIMLoss module"""
        print("\nTesting SSIMLoss...")

        # Initialize SSIM loss
        ssim_loss = SSIMLoss().to(self.device)

        # Test forward pass
        img1 = torch.randn(2, 1, 64, 64).to(self.device)
        img2 = torch.randn(2, 1, 64, 64).to(self.device)
        loss = ssim_loss(img1, img2)

        self.assertTrue(isinstance(loss.item(), float))
        print("SSIMLoss tests passed")

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(cls.temp_dir)


if __name__ == '__main__':
    # Run tests with more detailed output
    unittest.main(verbosity=2)