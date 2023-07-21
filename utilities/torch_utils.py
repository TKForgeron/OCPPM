import platform
import random

import cpuinfo
import numpy as np
import psutil
import torch
import torch_geometric

from loan_application_experiment.models.geometric_models import GraphModel


# pass this to the dataloaders to guarantee reproducibility
def seed_worker(worker_id: int, state: int) -> None:
    worker_seed = torch.initial_seed() % state
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def count_parameters(model: GraphModel) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def __add_cores_to_cpu_info(cpu_info_str: str, num_cores: int) -> str:
    # Split the CPU information string using "@" as the delimiter
    parts = cpu_info_str.split("@")

    # Insert the "(4x)" substring between "@" and what comes after
    updated_cpu_info_str = f"{parts[0]}@{parts[1]} ({num_cores}x)"

    return updated_cpu_info_str


def _get_cpu_info() -> str:
    # Get the raw CPU information
    cpu_info = cpuinfo.get_cpu_info()
    # Extract the CPU model name from the raw information
    cpu_model = cpu_info.get("brand_raw", "Unknown CPU")

    num_cores = psutil.cpu_count()
    updated_cpu_info_str = __add_cores_to_cpu_info(str(cpu_model), num_cores)
    return updated_cpu_info_str


def _get_byte_size(bytes, suffix="B") -> str:
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def print_system_info(gpu_info: bool = True) -> None:
    print(f"CPU: {_get_cpu_info()}")
    print(f"Total CPU memory: {_get_byte_size(psutil.virtual_memory().total)}")
    print(f"Available CPU memory: {_get_byte_size(psutil.virtual_memory().available)}")
    if gpu_info:
        import GPUtil

        gpu = GPUtil.getGPUs()[0]
        print(f"GPU: {gpu.name}")
        print(f"Total GPU memory: {gpu.memoryTotal}MB")
        print(f"Available GPU memory: {gpu.memoryFree}MB")
    print(f"Platform: {platform.platform()}")


def print_torch_info() -> None:
    print(f"Torch version: {torch.__version__}")
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"Torch geometric version: {torch_geometric.__version__}")
