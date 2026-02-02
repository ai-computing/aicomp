"""
Model configuration module.
Uses centralized config.py for model specifications.
"""

from config import get_model_config_dict, MODEL_CONFIGS


def get_model_config(model_type: str, precision: int, heterogeneous: bool = True, pareto: bool = False):
    """
    Returns the model configuration for the given model type and precision.
    
    Args:
        model_type: Model identifier (e.g., "llama1b", "llama70b")
        precision: Training precision (e.g., 16 for FP16)
        heterogeneous: Whether to use heterogeneous cluster
        pareto: Whether this is a pareto experiment
    
    Returns:
        tuple: (model_config dict, global_batch_size, experiment_name)
    """
    # Get model config from centralized config.py
    model_config, gbs = get_model_config_dict(model_type, precision)
    
    # Build experiment name
    exp_name = model_type
    if heterogeneous:
        exp_name = exp_name + "_heterogeneous"
    if pareto:
        exp_name = exp_name + "_pareto"
    
    return model_config, gbs, exp_name


def get_supported_models():
    """
    Returns list of supported model types.
    """
    return list(MODEL_CONFIGS.keys())
