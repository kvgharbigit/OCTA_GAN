"""
Model size definitions for OCTA-GAN architecture.

This module defines various model size configurations that can be used
with the Generator and Discriminator classes to create models of different capacities.
Each size defines the number of filters at different stages of the network.
"""

# Model sizes for Generator
GENERATOR_SIZES = {
    # Existing size (as defined in base.py)
    "small": {
        "initial_filters": 16,
        "mid_filters": 32,
        "max_filters": 64,
        "decoder_mid_filters": 48
    },
    
    # Medium size (original definition from base.py)
    "medium": {
        "initial_filters": 32,
        "mid_filters": 64,
        "max_filters": 128,
        "decoder_mid_filters": 96
    },
    
    # Large size (original definition from base.py)
    "large": {
        "initial_filters": 64,
        "mid_filters": 128,
        "max_filters": 320,
        "decoder_mid_filters": 192
    },
    
    # Extra large size (new)
    "xlarge": {
        "initial_filters": 96,
        "mid_filters": 192,
        "max_filters": 384,
        "decoder_mid_filters": 256
    }
}

# Model sizes for Discriminator
DISCRIMINATOR_SIZES = {
    # Existing size (as defined in base.py)
    "small": {
        "initial_filters": 32,
        "mid_filters": 48,
        "large_filters": 96,
        "max_filters": 128
    },
    
    # Medium size (based on base.py)
    "medium": {
        "initial_filters": 64,
        "mid_filters": 96,
        "large_filters": 192,
        "max_filters": 256
    },
    
    # Large size (based on base.py)
    "large": {
        "initial_filters": 96,
        "mid_filters": 128,
        "large_filters": 256,
        "max_filters": 512
    },
    
    # Extra large size (new)
    "xlarge": {
        "initial_filters": 128,
        "mid_filters": 192,
        "large_filters": 384,
        "max_filters": 768
    }
}

def get_generator_size(size_name):
    """
    Get generator filter sizes for a specified model size.
    
    Args:
        size_name: Name of the model size ("small", "medium", "large", "xlarge")
        
    Returns:
        Dictionary containing filter sizes for the generator
        
    Raises:
        ValueError: If the size_name is not recognized
    """
    if size_name not in GENERATOR_SIZES:
        valid_sizes = list(GENERATOR_SIZES.keys())
        raise ValueError(f"Invalid generator size: {size_name}. Choose from {valid_sizes}")
    
    return GENERATOR_SIZES[size_name]

def get_discriminator_size(size_name):
    """
    Get discriminator filter sizes for a specified model size.
    
    Args:
        size_name: Name of the model size ("small", "medium", "large", "xlarge")
        
    Returns:
        Dictionary containing filter sizes for the discriminator
        
    Raises:
        ValueError: If the size_name is not recognized
    """
    if size_name not in DISCRIMINATOR_SIZES:
        valid_sizes = list(DISCRIMINATOR_SIZES.keys())
        raise ValueError(f"Invalid discriminator size: {size_name}. Choose from {valid_sizes}")
    
    return DISCRIMINATOR_SIZES[size_name]

# Create template configurations for different model sizes
CONFIG_TEMPLATES = {
    "small": {
        "model": {
            "size": "small",
            "spectral_channels": 31,
            "generator": {
                "initial_filters": GENERATOR_SIZES["small"]["initial_filters"],
                "max_filters": GENERATOR_SIZES["small"]["max_filters"]
            },
            "discriminator": {
                "initial_filters": DISCRIMINATOR_SIZES["small"]["initial_filters"],
                "max_filters": DISCRIMINATOR_SIZES["small"]["max_filters"]
            }
        }
    },
    "medium": {
        "model": {
            "size": "medium",
            "spectral_channels": 31,
            "generator": {
                "initial_filters": GENERATOR_SIZES["medium"]["initial_filters"],
                "max_filters": GENERATOR_SIZES["medium"]["max_filters"]
            },
            "discriminator": {
                "initial_filters": DISCRIMINATOR_SIZES["medium"]["initial_filters"],
                "max_filters": DISCRIMINATOR_SIZES["medium"]["max_filters"]
            }
        }
    },
    "large": {
        "model": {
            "size": "large",
            "spectral_channels": 31,
            "generator": {
                "initial_filters": GENERATOR_SIZES["large"]["initial_filters"],
                "max_filters": GENERATOR_SIZES["large"]["max_filters"]
            },
            "discriminator": {
                "initial_filters": DISCRIMINATOR_SIZES["large"]["initial_filters"],
                "max_filters": DISCRIMINATOR_SIZES["large"]["max_filters"]
            }
        }
    },
    "xlarge": {
        "model": {
            "size": "xlarge",
            "spectral_channels": 31,
            "generator": {
                "initial_filters": GENERATOR_SIZES["xlarge"]["initial_filters"],
                "max_filters": GENERATOR_SIZES["xlarge"]["max_filters"]
            },
            "discriminator": {
                "initial_filters": DISCRIMINATOR_SIZES["xlarge"]["initial_filters"],
                "max_filters": DISCRIMINATOR_SIZES["xlarge"]["max_filters"]
            }
        }
    }
}

def get_config_template(size_name):
    """
    Get a configuration template for a specific model size.
    
    Args:
        size_name: Name of the model size ("small", "medium", "large", "xlarge")
        
    Returns:
        Dictionary containing configuration template for the specified size
        
    Raises:
        ValueError: If the size_name is not recognized
    """
    if size_name not in CONFIG_TEMPLATES:
        valid_sizes = list(CONFIG_TEMPLATES.keys())
        raise ValueError(f"Invalid config template size: {size_name}. Choose from {valid_sizes}")
    
    return CONFIG_TEMPLATES[size_name]


# Approximate resource requirements for different model sizes
# These are estimates for training with batch_size=2 and target_size=500
RESOURCE_ESTIMATES = {
    "small": {
        "cpu_ram": "~4-6 GB",
        "gpu_vram": "~4-5 GB",
        "gpu_recommendation": "GTX 1080 or better",
        "training_time": "~10-12 hours for 150 epochs on modern GPU"
    },
    "medium": {
        "cpu_ram": "~8-10 GB",
        "gpu_vram": "~6-8 GB",
        "gpu_recommendation": "GTX 1080Ti or better",
        "training_time": "~16-20 hours for 150 epochs on modern GPU"
    },
    "large": {
        "cpu_ram": "~12-16 GB",
        "gpu_vram": "~10-12 GB",
        "gpu_recommendation": "RTX 2080Ti or better",
        "training_time": "~24-30 hours for 150 epochs on modern GPU"
    },
    "xlarge": {
        "cpu_ram": "~16-24 GB",
        "gpu_vram": "~16-20 GB",
        "gpu_recommendation": "RTX 3090 or better",
        "training_time": "~36-48 hours for 150 epochs on modern GPU"
    }
}

def get_resource_estimates(size_name):
    """
    Get estimated resource requirements for a specific model size.
    
    Args:
        size_name: Name of the model size ("small", "medium", "large", "xlarge")
        
    Returns:
        Dictionary containing resource estimates for CPU RAM, GPU VRAM, 
        recommended GPU, and estimated training time
        
    Raises:
        ValueError: If the size_name is not recognized
    """
    if size_name not in RESOURCE_ESTIMATES:
        valid_sizes = list(RESOURCE_ESTIMATES.keys())
        raise ValueError(f"Invalid size for resource estimates: {size_name}. Choose from {valid_sizes}")
    
    return RESOURCE_ESTIMATES[size_name]