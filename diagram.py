'''
# HSI to OCTA Translation Model Implementation

## Architecture Overview
```
Input HSI [B, 31, 500, 500]
      |
      v
[B, 1, 31, 500, 500] (Unsqueeze)
      |
      v
GENERATOR ENCODER
      |
    Conv3d_1 [B, 32, 16, 500, 500] -----> Skip1 [B, 32, 16, 500, 500]
      |
    Conv3d_2 [B, 64, 8, 500, 500]  -----> Skip2 [B, 64, 8, 500, 500]
      |
    Conv3d_3 [B, 128, 4, 500, 500]
      |
   MaxPool2d [B, 128, 500, 500]
      |
      v
GENERATOR DECODER
      |
    Cat+Red1 [B, 192, 500, 500] <----- Skip2.max() [B, 64, 500, 500]
      |
    Conv2d_1 [B, 128, 500, 500]
      |
    Cat+Red2 [B, 160, 500, 500] <----- Skip1.max() [B, 32, 500, 500]
      |
    Conv2d_2 [B, 128, 500, 500]
      |
    Conv2d_3 [B, 64, 500, 500]
      |
    Conv2d_4 [B, 32, 500, 500]
      |
    Conv2d_5 [B, 1, 500, 500]
      |
      v
Output OCTA [B, 1, 500, 500] --------> DISCRIMINATOR
                                          |
                                     Conv2d_1 [B, 64, 250, 250]
                                          |
                                     Conv2d_2 [B, 128, 125, 125]
                                          |
                                     Conv2d_3 [B, 256, 62, 62]
                                          |
                                     Conv2d_4 [B, 512, 31, 31]
                                          |
                                     Conv2d_5 [B, 1, 30, 30]
                                          |
                                          v
                                   Binary Output
                                   Real/Fake Score
```

## Data Processing & Training Pipeline
```
DATA PREPROCESSING & INPUT PIPELINE
=================================
Raw Data Files
     |
     |     +-----------------+
     +---->| HSI (h5)       |  91 wavelengths
     |     | [H, W, 91]     |----+
     |     +-----------------+    |   Take every 3rd wavelength
     |                            +-> [31, H, W]
     |     +-----------------+    |   Normalize & Resize
     +---->| OCTA (tiff)    |----+-> [1, 500, 500]
           | [H, W, 1]      |
           +-----------------+

TRAINING PIPELINE & LOSS CALCULATION
==================================
                                                   Real OCTA
                                                      ^
HSI Input                                            |
[B, 31, 500, 500]                                   |
     |                                              |
     v                                              |
  Generator ---------------------------------> Generated OCTA
     |                                         [B, 1, 500, 500]
     |                                              |
     |                                              v
     |                                        Discriminator
     |                                              |
     |                                              v
     |                                    Patch Output [30x30]
     |                                              |
     |                                              |
     |                  +-------------------------+  |
     |                  |      Loss Terms         | |
     +----------------->|                         | |
     |                  | 1. GAN Loss <----------+ |
Generated OCTA -+------>| 2. L1 Pixel Loss         |
                |       | 3. Perceptual Loss (VGG)  |
Real OCTA -----+------>| 4. SSIM Loss             |
                       +-------------------------+
                                |
                                v
                         Total Generator Loss
                       λ1*L1 + λ2*Perceptual +
                       λ3*SSIM + λ4*GAN

LOSS WEIGHT CONFIGURATION
========================
λ_pixel      = 100.0  (L1)
λ_perceptual = 10.0   (VGG)
λ_ssim       = 5.0    (SSIM)
λ_adv        = 1.0    (GAN)

DATA AUGMENTATION
================
+------------------------+
| - Random Horizontal    |
|   & Vertical Flip     |
| - Random Rotation     |
|   (±10 degrees)       |
| - Random Translation  |
|   (±10% x,y)         |
+------------------------+

OPTIMIZATION
===========
Adam Optimizer:
- Learning Rate: 0.0002
- Beta1: 0.5
- Beta2: 0.999
- Weight Decay: 1e-4

LR Schedule:
- Decay Start: Epoch 100
- Decay Factor: 0.1
- Decay Interval: 50 epochs
```


## Usage Instructions

1. Data Organization:
   - Place HSI data (.h5 files) and OCTA data (.tiff files) in patient-specific directories
   - File naming convention: *C1*.h5 for HSI and *RetinaAngiographyEnface*.tiff for OCTA

2. Training:
   - Configure hyperparameters in TrainingConfig class
   - Set data paths in test script
   - Run training script to begin training

3. Model Testing:
   - Use test_dataset() to verify data loading
   - Use test_models() to verify model architecture
   - Use test_training_loop() to verify training process

4. Checkpointing:
   - Models are saved every 10 epochs
   - Use save_checkpoint() and load_checkpoint() for model persistence

## Notes

- Model uses 3D convolutions to process spectral information
- Skip connections help preserve spatial details
- Multiple loss terms ensure both structural and perceptual quality
- PatchGAN discriminator provides localized real/fake predictions
- Extensive data augmentation helps prevent overfitting

'''