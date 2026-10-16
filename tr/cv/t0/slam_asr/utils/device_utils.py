"""Hardware-aware hyperparameter resolution and optimization for SLAM-ASR.

Automatically inspects the runtime environment (GPU model, VRAM capacity,
compute capability, and CPU cores) to configure optimal training and evaluation
hyperparameters: precision (fp16 / bf16), 4-bit quantization, batch size,
gradient accumulation, and worker threads.
"""

import os
from typing import Any, Dict, Optional, Tuple
try:
    import torch
except ImportError:
    torch = None


def get_device_specs(device_idx: int = 0) -> Dict[str, Any]:
    """Inspect and return current device specifications."""
    specs: Dict[str, Any] = {
        "device_type": "cpu",
        "device_name": "CPU",
        "vram_gb": 0.0,
        "compute_capability": None,
        "supports_bf16": False,
        "supports_fp16": False,
        "cpu_count": os.cpu_count() or 2,
    }

    try:
        sys_ram_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        specs["ram_gb"] = round(sys_ram_bytes / (1024 ** 3), 2)
    except (AttributeError, ValueError):
        specs["ram_gb"] = 0.0

    if torch is not None and torch.cuda.is_available():
        specs["device_type"] = "cuda"
        specs["device_name"] = torch.cuda.get_device_name(device_idx)
        total_bytes = torch.cuda.get_device_properties(device_idx).total_memory
        specs["vram_gb"] = round(total_bytes / (1024 ** 3), 2)

        major, minor = torch.cuda.get_device_capability(device_idx)
        specs["compute_capability"] = (major, minor)
        # Ampere (sm_80), Ada (sm_89), Hopper (sm_90) and newer natively support bfloat16
        specs["supports_bf16"] = major >= 8
        # Volta (sm_70), Turing (sm_75) and newer support float16 Tensor Cores
        specs["supports_fp16"] = major >= 7
    elif torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        specs["device_type"] = "mps"
        specs["device_name"] = "Apple Silicon (MPS)"
        specs["vram_gb"] = specs["ram_gb"]
        specs["supports_fp16"] = True

    return specs


def get_recommended_hyperparameters(
    vram_gb: Optional[float] = None,
    compute_capability: Optional[Tuple[int, int]] = None,
    supports_bf16: Optional[bool] = None,
    device_type: Optional[str] = None,
    cpu_count: Optional[int] = None,
    is_vicuna: bool = True,
) -> Dict[str, Any]:
    """Calculate optimal hyperparameters based on device specifications.

    Target effective batch size is 16-24 for stable ASR convergence.
    """
    if vram_gb is None or device_type is None:
        specs = get_device_specs()
        vram_gb = specs["vram_gb"] if vram_gb is None else vram_gb
        compute_capability = specs["compute_capability"] if compute_capability is None else compute_capability
        supports_bf16 = specs["supports_bf16"] if supports_bf16 is None else supports_bf16
        device_type = specs["device_type"] if device_type is None else device_type
        cpu_count = specs["cpu_count"] if cpu_count is None else cpu_count

    cpu_count = cpu_count or (os.cpu_count() or 2)
    optimal_workers = min(4, max(1, cpu_count // 2))

    if device_type == "cuda":
        # Tier 1: Ultra-high VRAM (>= 68 GB: e.g. H100 80GB, A100 80GB)
        if vram_gb >= 68.0:
            return {
                "profile": "Ultra-High VRAM (H100 / A100 80GB)",
                "torch_dtype": "bfloat16" if supports_bf16 else "float16",
                "load_in_4bit": False,
                "load_in_8bit": False,
                "batch_size": 16,
                "gradient_accumulation_steps": 1,
                "eval_batch_size": 16,
                "fp16": not supports_bf16,
                "bf16": bool(supports_bf16),
                "num_workers": optimal_workers,
                "notes": "Full 16-bit precision (Paper Setup). Maximum throughput without quantization.",
            }

        # Tier 2: High VRAM (34 GB - 68 GB: e.g. A100 40GB)
        if vram_gb >= 34.0:
            return {
                "profile": "High VRAM (A100 40GB)",
                "torch_dtype": "bfloat16" if supports_bf16 else "float16",
                "load_in_4bit": False,
                "load_in_8bit": False,
                "batch_size": 8,
                "gradient_accumulation_steps": 2,
                "eval_batch_size": 8,
                "fp16": not supports_bf16,
                "bf16": bool(supports_bf16),
                "num_workers": optimal_workers,
                "notes": "Full 16-bit precision (Paper Setup). Fits unquantized Vicuna-7B with batch size 8.",
            }

        # Tier 3: Mid VRAM (18 GB - 34 GB: e.g. L4 24GB, RTX 3090/4090 24GB)
        if vram_gb >= 18.0:
            return {
                "profile": "Mid VRAM (L4 24GB / RTX 4090)",
                "torch_dtype": "bfloat16" if supports_bf16 else "float16",
                "load_in_4bit": True,
                "load_in_8bit": False,
                "batch_size": 8,
                "gradient_accumulation_steps": 2,
                "eval_batch_size": 8,
                "fp16": not supports_bf16,
                "bf16": bool(supports_bf16),
                "num_workers": optimal_workers,
                "notes": "4-bit NF4 with native bfloat16 compute. High throughput and large safety margin.",
            }

        # Tier 4: Standard VRAM (10 GB - 18 GB: e.g. T4 16GB, RTX 3080/4080 12-16GB, V100 16GB)
        if vram_gb >= 10.0:
            use_bf16 = bool(supports_bf16)
            return {
                "profile": "Standard VRAM (Tesla T4 16GB / V100)",
                "torch_dtype": "bfloat16" if use_bf16 else "float16",
                "load_in_4bit": True,
                "load_in_8bit": False,
                "batch_size": 4,
                "gradient_accumulation_steps": 4,
                "eval_batch_size": 4,
                "fp16": not use_bf16,
                "bf16": use_bf16,
                "num_workers": min(2, optimal_workers),
                "notes": "4-bit NF4 with float16 compute. Optimized for 16GB VRAM constraint.",
            }

        # Tier 5: Low VRAM GPU (< 10 GB)
        return {
            "profile": "Low VRAM GPU (< 10GB)",
            "torch_dtype": "float16",
            "load_in_4bit": True,
            "load_in_8bit": False,
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "eval_batch_size": 2,
            "fp16": True,
            "bf16": False,
            "num_workers": 1,
            "notes": "Minimal 4-bit footprint with batch size 2.",
        }

    # Apple Silicon (MPS)
    if device_type == "mps":
        return {
            "profile": "Apple Silicon (MPS Unified Memory)",
            "torch_dtype": "float16",
            "load_in_4bit": False,
            "load_in_8bit": False,
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "eval_batch_size": 2,
            "fp16": False,
            "bf16": False,
            "num_workers": 0,
            "notes": "Native float16 on Apple Silicon MPS. 4-bit quantization disabled (requires CUDA).",
        }

    # CPU Environment
    return {
        "profile": "CPU Environment",
        "torch_dtype": "float32",
        "load_in_4bit": False,
        "load_in_8bit": False,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "eval_batch_size": 2,
        "fp16": False,
        "bf16": False,
        "num_workers": 0,
        "notes": "Standard float32 on CPU.",
    }


def resolve_auto_hyperparameters(
    model_cfg: Dict[str, Any],
    data_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    cli_args: Optional[Any] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Inspect configs and replace 'auto' values with hardware-optimal settings."""
    specs = get_device_specs()
    rec = get_recommended_hyperparameters(
        vram_gb=specs["vram_gb"],
        compute_capability=specs["compute_capability"],
        supports_bf16=specs["supports_bf16"],
        device_type=specs["device_type"],
        cpu_count=specs["cpu_count"],
    )

    auto_device_enabled = (
        getattr(cli_args, "auto_device", False)
        or train_cfg.get("auto_device", False)
        or any(v == "auto" for v in list(model_cfg.values()) + list(data_cfg.values()) + list(train_cfg.values()))
    )

    # 1. Model precision & quantization
    if model_cfg.get("torch_dtype") == "auto" or auto_device_enabled:
        if model_cfg.get("torch_dtype") in (None, "auto"):
            model_cfg["torch_dtype"] = rec["torch_dtype"]

    if model_cfg.get("load_in_4bit") == "auto" or auto_device_enabled:
        if model_cfg.get("load_in_4bit") in (None, "auto"):
            model_cfg["load_in_4bit"] = rec["load_in_4bit"]

    if model_cfg.get("load_in_8bit") == "auto":
        model_cfg["load_in_8bit"] = rec["load_in_8bit"]

    # 2. Data loader
    cli_batch_size = getattr(cli_args, "batch_size", None) if cli_args else None
    if cli_batch_size is not None:
        data_cfg["batch_size"] = cli_batch_size
    elif data_cfg.get("batch_size") in (None, "auto") or auto_device_enabled:
        data_cfg["batch_size"] = rec["batch_size"]

    if data_cfg.get("num_workers") in (None, "auto") or auto_device_enabled:
        data_cfg["num_workers"] = rec["num_workers"]

    # 3. Training arguments
    if train_cfg.get("gradient_accumulation_steps") in (None, "auto") or auto_device_enabled:
        train_cfg["gradient_accumulation_steps"] = rec["gradient_accumulation_steps"]

    if train_cfg.get("fp16") in (None, "auto") or auto_device_enabled:
        train_cfg["fp16"] = rec["fp16"]

    if train_cfg.get("bf16") in (None, "auto") or auto_device_enabled:
        train_cfg["bf16"] = rec["bf16"]

    if verbose:
        banner = format_device_summary(specs, rec, model_cfg, data_cfg, train_cfg)
        print(banner)

    return model_cfg, data_cfg, train_cfg


def format_device_summary(
    specs: Dict[str, Any],
    rec: Dict[str, Any],
    model_cfg: Dict[str, Any],
    data_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
) -> str:
    """Format a clean, readable ASCII summary banner of hardware and hyperparameters."""
    dev_str = f"{specs['device_name']} ({specs['vram_gb']} GB VRAM)" if specs["device_type"] == "cuda" else specs["device_name"]
    cap_str = f", Compute sm_{specs['compute_capability'][0]}{specs['compute_capability'][1]}" if specs["compute_capability"] else ""
    eff_bs = data_cfg.get("batch_size", 4) * train_cfg.get("gradient_accumulation_steps", 1)

    lines = [
        "=" * 70,
        f"[*] HARDWARE-ADAPTIVE HYPERPARAMETER PROFILER",
        f"    Device Detected:      {dev_str}{cap_str}",
        f"    Hardware Profile:     {rec['profile']}",
        f"    Profile Strategy:     {rec['notes']}",
        "-" * 70,
        f"    Active Settings:",
        f"      • torch_dtype:       {model_cfg.get('torch_dtype')}",
        f"      • load_in_4bit:      {model_cfg.get('load_in_4bit')} (NF4)",
        f"      • batch_size:        {data_cfg.get('batch_size')}",
        f"      • grad_accumulation: {train_cfg.get('gradient_accumulation_steps')}  --> Effective Batch: {eff_bs}",
        f"      • mixed_precision:   bf16={train_cfg.get('bf16')}, fp16={train_cfg.get('fp16')}",
        f"      • num_workers:       {data_cfg.get('num_workers')}",
        "=" * 70,
    ]
    return "\n".join(lines)
