"""SLAM-ASR: An Embarrassingly Simple Approach for LLM with Strong ASR Capacity."""

__version__ = "0.1.0"

try:
    from slam_asr.models.slam_model import SLAM_ASR, SLAMASRConfig
    from slam_asr.models.projector import SpeechProjector, LinearProjector, MLPProjector
    from slam_asr.models.speech_encoder import SpeechEncoderWrapper

    __all__ = [
        "SLAM_ASR",
        "SLAMASRConfig",
        "SpeechProjector",
        "LinearProjector",
        "MLPProjector",
        "SpeechEncoderWrapper",
    ]
except ImportError:
    # PyTorch not yet installed in active environment
    __all__ = []
