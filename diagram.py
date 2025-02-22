"""

Input Data Flow
==============
HSI (.h5)                 OCTA (.jpeg)
[B, H, W, S] →            [B, H, W] →
    ↓                         ↓
Preprocessing            Preprocessing
    ↓                         ↓
[B, S, H, W]            [B, 1, H, W]
    ↓                         ↓
Normalization           Normalization
[-1, 1] range           [-1, 1] range

Generator Architecture
=====================
                HSI Input
                [B, S, H, W]
                     ↓
        Add Channel Dimension
        [B, 1, H, W, S]
                     ↓
╔════════════════════════════════════╗
║           3D Encoder               ║
║ ┌─────────────────────────────┐   ║
║ │ Conv3D (1→32)               │   ║
║ │ [B, 32, H, W, S/2] ────┐    │   ║
║ │         ↓              │    │   ║
║ │ Conv3D (32→64)         │    │   ║
║ │ [B, 64, H, W, S/4] ─┐  │    │   ║
║ │         ↓           │  │    │   ║
║ │ Conv3D (64→128)     │  │    │   ║
║ │ [B, 128, H, W, S/8] │  │    │   ║
║ └─────────────────────┘  │    │   ║
╚═══════════════|══════════|════|═══╝
                ↓          │    │
        Remove Spectral    │    │
        [B, 128, H, W]    │    │
                ↓          │    │
╔═══════════════|══════════|════|═══╗
║           2D Decoder     │    │   ║
║ ┌─────────────────────┐  │    │   ║
║ │     Attention       │  │    │   ║
║ │ Conv2D + Skip ←─────┘  │    │   ║
║ │ [B, 128, H, W]         │    │   ║
║ │         ↓              │    │   ║
║ │     Attention          │    │   ║
║ │ Conv2D + Skip ←────────┘    │   ║
║ │ [B, 64, H, W]              │   ║
║ │         ↓                   │   ║
║ │ Conv2D + Skip ←─────────────┘   ║
║ │ [B, 32, H, W]                   ║
║ │         ↓                       ║
║ │ Conv2D (32→1)                   ║
║ │ [B, 1, H, W]                    ║
║ └─────────────────────────────────┘
╚════════════════════════════════════╝
                ↓
        Generated OCTA
        [B, 1, H, W]
                ↓
╔════════════════════════════════════╗
║         Discriminator              ║
║ ┌─────────────────────────────┐   ║
║ │ Conv2D (1→64)               │   ║
║ │         ↓                   │   ║
║ │ Conv2D (64→128)            │   ║
║ │         ↓                   │   ║
║ │ Conv2D (128→256)           │   ║
║ │         ↓                   │   ║
║ │ Conv2D (256→512)           │   ║
║ │         ↓                   │   ║
║ │ Conv2D (512→1)             │   ║
║ └─────────────────────────────┘   ║
╚════════════════════════════════════╝
                ↓
           Real/Fake
        [B, 1, H/16, W/16]

Loss Functions
=============
├── GAN Loss (BCE)
├── Pixel Loss (L1)
├── Perceptual Loss (VGG)
└── SSIM Loss

Where:
B = Batch size
H = Height
W = Width
S = Spectral channels
→ = Data flow
─ = Skip connection

"""