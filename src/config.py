from dotenv import load_dotenv
import os
import torch

load_dotenv(override=True)

DTYPE_ALIASES = {
    "fp16": "float16",
    "float16": "float16",
    "f16": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fb16": "bfloat16",
    "fp32": "float32",
    "float32": "float32",
    "f32": "float32",
    "float": "float32",
    "fp64": "float64",
    "float64": "float64",
    "f64": "float64",
    "double": "float64",
    "float8_e4m3fn": "float8_e4m3fn",
    "float8_e5m2": "float8_e5m2",
    "float8_e4m3fnuz": "float8_e4m3fnuz",
    "float8_e5m2fnuz": "float8_e5m2fnuz",
}


def parse_dtype(dtype_str: str) -> torch.dtype:
    mapped = DTYPE_ALIASES.get(dtype_str, dtype_str)
    dtype = getattr(torch, mapped, None)
    if dtype is None:
        raise ValueError(
            f"Unknown dtype: '{dtype_str}'. "
            f"Supported: {', '.join(sorted(set(DTYPE_ALIASES.values())))}"
        )
    return dtype


PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise ValueError("PROJECT_ROOT environment variable not set")


STORAGE_PATH = os.getenv(
    "STORAGE_PATH",
    os.path.join(PROJECT_ROOT, "storage")
)


EVALUATION_RESULTS_PATH = os.getenv(
    "EVALUATION_RESULTS_PATH",
    os.path.join(
        PROJECT_ROOT,
        "storage",
        "evaluation_results"
    )
)


TRAINED_MODELS_PATH = os.getenv(
    "TRAINED_MODELS_PATH",
    os.path.join(
        PROJECT_ROOT,
        "storage",
        "trained_models"
    )
)


TRAINED_AUTOENCODERS_PATH = os.getenv(
    "TRAINED_AUTOENCODERS_PATH",
    os.path.join(
        PROJECT_ROOT,
        "storage",
        "trained_autoencoders"
    )
)
