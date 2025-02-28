# HSI to OCTA Translation

## Overview

This project implements a deep learning-based translation between Hyperspectral Imaging (HSI) and Optical Coherence Tomography Angiography (OCTA) for retinal analysis. The model learns to generate OCTA images from HSI data, potentially enabling more accessible diagnostic tools for retinal conditions.

## Table of Contents
- [Theoretical Background](#theoretical-background)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Training Pipeline](#training-pipeline)
- [Visualization](#visualization)
- [Evaluation](#evaluation)
- [Configuration](#configuration)

## Theoretical Background

### Hyperspectral Imaging (HSI) and OCTA
HSI captures data across many wavelength bands, providing spectral information about the retina. Each HSI image contains 31 spectral channels (derived from original 91 wavelengths by taking every third wavelength), offering detailed tissue composition data.

OCTA is a non-invasive imaging technique that visualizes blood flow in the retina without dye injection. OCTA images highlight the vascular structure, which is critical for diagnosing conditions like diabetic retinopathy and age-related macular degeneration.

### Translation Approach
This project employs a generative adversarial network (GAN) approach to translate between these modalities:
- The generator learns to transform multi-channel HSI data into single-channel OCTA-like images
- The discriminator learns to distinguish between real and generated OCTA images
- Multiple loss terms (L1, perceptual, SSIM, adversarial) ensure structural and perceptual quality

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

Key architectural components:
1. **Generator**:
   - 3D convolutional encoder processes spectral dimension of HSI data
   - Skip connections preserve spatial information
   - 2D convolutional decoder converts features to OCTA-like output
   
2. **Discriminator**:
   - PatchGAN architecture for local texture evaluation
   - Outputs a 30×30 grid of real/fake predictions

3. **Loss Functions**:
   - L1 pixel-wise loss (λ=100.0)
   - Perceptual (VGG) loss (λ=10.0)
   - SSIM loss (λ=5.0)
   - GAN adversarial loss (λ=1.0)

## Project Structure

```
├── base.py                     # Core model definitions and dataset class
├── circle_crop_utils.py        # Circle detection and cropping utilities
├── config.json                 # Default configuration
├── config_utils.py             # Configuration loading and validation
├── debug-script.py             # Testing and debugging script
├── diagram.py                  # Architecture diagram
├── evaluation-script.py        # Evaluation script for trained models
├── hsi_octa_dataset_cropped.py # Extended dataset with circle cropping
├── test_circle_crop.py         # Testing circle cropping functionality
├── training-script.py          # Main training script
└── visualization_utils.py      # Visualization utilities
```

## Setup and Installation

### Requirements
- Python 3.8+
- PyTorch 1.8+
- torchvision
- numpy
- scipy
- scikit-image
- h5py
- PIL
- matplotlib
- tqdm
- pandas
- seaborn
- tensorboard

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/hsi-octa-translation.git
cd hsi-octa-translation

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Organization
Place HSI data (.h5 files) and OCTA data (.tiff files) in patient-specific directories:
```
DummyData/
├── patient1/
│   ├── *C1*.h5               # HSI data
│   └── *RetinaAngiographyEnface*.tiff  # OCTA data
├── patient2/
│   ├── *C1*.h5
│   └── *RetinaAngiographyEnface*.tiff
└── ...
```

### Training

#### Step-by-Step Training Guide

1. **Prepare your configuration file**:
   - Modify `config.json` to set data paths, hyperparameters, and output directories
   - Ensure your data directory structure is correct (see Data Organization section)

2. **Start a new training run**:
   ```bash
   python training-script.py --config config.json --exp_id my_experiment
   ```

3. **Enable or disable circle cropping**:
   - With circle cropping (better for circular retinal images):
     ```bash
     python training-script.py --config config.json --exp_id with_crop --circle_crop
     ```
   - Without circle cropping:
     ```bash
     python training-script.py --config config.json --exp_id no_crop --no_circle_crop
     ```

4. **Resume training from a checkpoint**:
   ```bash
   python training-script.py --config config.json --resume ./output/experiment_my_experiment/checkpoints/checkpoint_epoch_50.pth
   ```

5. **Monitor training progress**:
   - View TensorBoard logs:
     ```bash
     tensorboard --logdir ./output/experiment_my_experiment/tensorboard
     ```
   - Check visual samples in `./output/experiment_my_experiment/visual_samples`

#### Common Training Patterns

- **Short training run to test setup**:
  ```bash
  # Modify config.json to set num_epochs to 3
  python training-script.py --config config.json --exp_id test_run
  ```

- **Full training with early stopping**:
  ```bash
  # Ensure early_stopping is enabled in config.json
  python training-script.py --config config.json --exp_id full_run
  ```

### Evaluation

#### Step-by-Step Evaluation Guide

1. **Prepare your evaluation config**:
   - Modify `eval_config.json` to point to your trained model checkpoint
   - Set the data directory to your test dataset

2. **Run evaluation on a trained model**:
   ```bash
   python evaluation-script.py --config eval_config.json --exp_id my_evaluation
   ```

3. **Specify circle cropping option** (should match training):
   ```bash
   python evaluation-script.py --config eval_config.json --circle_crop
   ```
   or
   ```bash
   python evaluation-script.py --config eval_config.json --no_circle_crop
   ```

4. **View evaluation results**:
   - Metrics CSV: `./output/experiment_my_evaluation/evaluation/metrics.csv`
   - Summary statistics: `./output/experiment_my_evaluation/evaluation/summary_statistics.csv`
   - Visualizations: `./output/experiment_my_evaluation/evaluation/visualizations/`
   - Distribution plots: `./output/experiment_my_evaluation/evaluation/metric_distributions.png`

#### Example Complete Workflow

```bash
# 1. Train model with circle cropping
python training-script.py --config config.json --exp_id exp001 --circle_crop

# 2. Evaluate the trained model
python evaluation-script.py --config eval_config.json \
  --exp_id exp001_eval \
  --circle_crop
```

Be sure to modify `eval_config.json` to point to the best model checkpoint from training:
```json
{
  "evaluation": {
    "checkpoint_path": "./output/experiment_exp001/checkpoints/best_model.pth",
    ...
  }
}
```

### Circle Crop Testing
```bash
python test_circle_crop.py --data_dir /path/to/data --num_samples 5 --save_path results.png
```

## Model Details

### Generator
- **Encoder**: Three 3D convolutional layers that process spectral and spatial dimensions:
  - First layer: 1→32 channels, spectral dimension 31→16
  - Second layer: 32→64 channels, spectral dimension 16→8
  - Third layer: 64→128 channels, spectral dimension 8→4
  
- **Skip Connections**: Key features from earlier layers are preserved and passed to the decoder to maintain spatial information.

- **Decoder**: 2D convolutional layers that reconstruct the OCTA image:
  - Features from skip connections are integrated through concatenation
  - Channel reduction layers manage the increased channel dimensions
  - Multiple convolutional blocks progressively refine the output
  - Final layer outputs a single-channel OCTA-like image with tanh activation

### Discriminator (PatchGAN)
- Five convolutional layers with stride 2 to downsample
- Classifies 30×30 overlapping patches as real or fake
- Focuses on local texture details rather than global structure
- Provides more detailed gradient information to the generator

### Loss Functions
1. **L1 Loss**: Pixel-wise absolute difference, encourages structural similarity
2. **Perceptual Loss**: Compares VGG16 feature representations, promotes perceptual similarity
3. **SSIM Loss**: Evaluates structural similarity considering luminance, contrast, and structure
4. **Adversarial Loss**: Binary cross-entropy loss from the discriminator

## Preprocessing Pipeline

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
```

Key preprocessing steps:
1. **HSI Processing**:
   - Load .h5 files
   - Select every third wavelength (91→31 channels)
   - Normalize to [0,1] range
   - Resize to 500×500 spatial dimensions

2. **OCTA Processing**:
   - Load .tiff files
   - Convert to grayscale
   - Normalize to [0,1] range
   - Resize to 500×500 spatial dimensions

3. **Circle Cropping (Optional)**:
   - Detect circular field of view in the retinal images
   - Crop to this region with padding
   - Resize back to standard 500×500 dimensions

4. **Data Augmentation**:
   - Random horizontal and vertical flips
   - Random rotation (±10 degrees)
   - Random translation (±10% in x,y directions)

## Training Pipeline

```
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
```

### Training Process:
1. **Initialization**:
   - Load and split the dataset into train/validation/test
   - Initialize generator and discriminator with Xavier weights
   - Set up Adam optimizers with β1=0.5, β2=0.999
   
2. **Training Loop**:
   - For each batch:
     - Train discriminator on real and fake OCTA images
     - Train generator with combined loss functions
     - Apply gradient clipping
   - Apply learning rate scheduling
   - Save checkpoints at specified intervals
   - Log metrics to TensorBoard
   - Monitor validation loss for early stopping

### Optimization Settings:
- Adam optimizer with learning rate 0.0002
- Weight decay 1e-4
- Learning rate decay schedule:
  - Start decay at epoch 100
  - Decay factor 0.1
  - Apply every 50 epochs

## Visualization

The project includes several visualization utilities:
1. **Sample Visualizations**: During training, samples are generated periodically to visualize progress
2. **TensorBoard Logging**: Loss metrics and image samples are logged to TensorBoard
3. **Circle Crop Comparison**: Visualize the effect of circle detection and cropping
4. **HSI Rendering**: Convert multi-channel HSI to RGB-like representation for visualization

### HSI Visualization
HSI data (31 channels) is visualized by:
1. Selecting three representative wavelengths close to red (660nm), green (555nm), and blue (475nm)
2. Normalizing each channel to handle potential outliers
3. Combining into an RGB-like representation

## Evaluation

The evaluation script measures model performance using:
1. **PSNR** (Peak Signal-to-Noise Ratio): Measures pixel-level accuracy
2. **SSIM** (Structural Similarity Index): Measures structural similarity
3. **MSE** (Mean Squared Error): Measures pixel-level differences
4. **MAE** (Mean Absolute Error): Measures absolute pixel-level differences

Evaluation outputs:
- Per-patient metrics
- Summary statistics
- Best and worst case visualizations
- Metric distribution plots

## Configuration

The project uses JSON configuration files with the following key sections:

### Training Configuration (`config.json`)
```json
{
  "num_epochs": 200,
  "batch_size": 8,
  "learning_rate": 0.0002,
  "beta1": 0.5,
  "beta2": 0.999,
  "weight_decay": 1e-4,
  "lambda_pixel": 100.0,
  "lambda_perceptual": 10.0,
  "lambda_ssim": 5.0,
  "lambda_adv": 1.0,
  "data": {
    "data_dir": "./DummyData",
    "val_ratio": 0.15,
    "test_ratio": 0.4,
    "target_size": 500
  },
  "output": {
    "base_dir": "./output",
    "checkpoint_dir": "${base_dir}/checkpoints",
    "results_dir": "${base_dir}/results",
    "tensorboard_dir": "${base_dir}/tensorboard"
  },
  "early_stopping": {
    "enabled": true,
    "patience": 20,
    "min_delta": 0.001
  }
}
```

### Evaluation Configuration (`eval_config.json`)
```json
{
  "evaluation": {
    "checkpoint_path": "./output/experiment_xyz/checkpoints/best_model.pth",
    "data_dir": "./DummyData",
    "output_dir": "./output",
    "batch_size": 1
  },
  "data": {
    "target_size": 500
  },
  "hardware": {
    "num_workers": 4,
    "pin_memory": true
  }
}
```

## Citation

If you use this code or model in your research, please cite:

```
@misc{hsi-octa-translation,
  author = {Your Name},
  title = {HSI to OCTA Translation},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/hsi-octa-translation}}
}
```

## License

[MIT License](LICENSE)
