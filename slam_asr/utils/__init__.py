from slam_asr.utils.metrics import compute_wer, compute_cer, normalize_text_librispeech
from slam_asr.utils.device_utils import (
    get_device_specs,
    get_recommended_hyperparameters,
    resolve_auto_hyperparameters,
    format_device_summary,
)

try:
    from slam_asr.utils.scheduler import get_warmup_constant_scheduler, get_warmup_cosine_scheduler
    __all__ = [
        "compute_wer",
        "compute_cer",
        "normalize_text_librispeech",
        "get_warmup_constant_scheduler",
        "get_warmup_cosine_scheduler",
        "get_device_specs",
        "get_recommended_hyperparameters",
        "resolve_auto_hyperparameters",
        "format_device_summary",
    ]
except ImportError:
    __all__ = [
        "compute_wer",
        "compute_cer",
        "normalize_text_librispeech",
        "get_device_specs",
        "get_recommended_hyperparameters",
        "resolve_auto_hyperparameters",
        "format_device_summary",
    ]
