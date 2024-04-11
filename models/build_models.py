"""
A simplistic model builder for building models 
from scratch or from checkpoints
"""

from models.architectures.baseline import BaseGPT
from models.architectures.profiler import ProfilerGPT
from models.architectures.shared_fnn_head import SharedFNNHead
from models.architectures.shared_fnn_lora import SharedFNNLora


MODEL_CLASSES = {
    "baseline": BaseGPT,
    "profiler": ProfilerGPT,
    "sharedfnn": SharedFNNHead,
    "sharedfnnlora": SharedFNNLora,
}


def build_model(cfg=None, model_checkpoint=None):
    """
    Build model from scratch or load model from checkpoint
    Args:
        cfg: model configuration
        model_checkpoint: model weight dict
    Returns:
        model: model instance
    """
    # check if model is loaded
    if model_checkpoint is not None:
        # load model with correct architecture
        model = MODEL_CLASSES[model_checkpoint["config"]["arch"]["model"]](
            cfg=model_checkpoint["config"]
        )

        # load model weights
        model.load_state_dict(model_checkpoint["model"])
        model.eval()

        return model

    else:
        # build model from scratch
        model = MODEL_CLASSES[cfg["model"]](cfg=cfg)

        return model
