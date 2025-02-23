from pathlib import Path
import json
import re


def resolve_path_variables(config_dict, parent_key=''):
    """Recursively resolve ${variable} references in configuration paths."""
    for key, value in config_dict.items():
        if isinstance(value, dict):
            resolve_path_variables(value, f"{parent_key}/{key}" if parent_key else key)
        elif isinstance(value, str) and '${' in value:
            variables = re.findall(r'\${([^}]+)}', value)
            resolved_value = value

            for var in variables:
                var_parts = var.split('/')
                var_value = config_dict
                for part in var_parts:
                    if part in var_value:
                        var_value = var_value[part]
                    else:
                        raise ValueError(f"Unable to resolve variable ${{{var}}} in {parent_key}/{key}")

                resolved_value = resolved_value.replace(f"${{{var}}}", str(var_value))

            config_dict[key] = resolved_value


def load_config(config_path: str) -> dict:
    """Load and process configuration file."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Remove comment entries
    config = {k: v for k, v in config.items() if not k.startswith('//')}

    # Resolve path variables
    resolve_path_variables(config)

    # Convert paths to Path objects
    for section in ['data', 'output']:
        if section in config:
            for key, value in config[section].items():
                if isinstance(value, str) and not key.endswith('_ratio'):
                    config[section][key] = Path(value)

    return config


def setup_directories(config: dict) -> None:
    """Create all required output directories from config."""
    output_config = config.get('output', {})

    for dir_path in output_config.values():
        if isinstance(dir_path, Path):
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")


def validate_directories(config: dict) -> None:
    """Validate that all required directories exist."""
    # Check data directory
    data_config = config.get('data', {})
    data_dir = data_config.get('data_dir')

    if not data_dir:
        raise ValueError("Missing required data directory")
    if not isinstance(data_dir, Path):
        raise ValueError(f"Invalid path type for data_dir: {type(data_dir)}")
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    # Check output directories
    output_config = config.get('output', {})
    required_output_dirs = ['checkpoint_dir', 'results_dir', 'tensorboard_dir']

    for dir_name in required_output_dirs:
        dir_path = output_config.get(dir_name)
        if not dir_path:
            raise ValueError(f"Missing required output directory: {dir_name}")
        if not isinstance(dir_path, Path):
            raise ValueError(f"Invalid path type for {dir_name}: {type(dir_path)}")