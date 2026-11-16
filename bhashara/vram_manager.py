"""
Bhashara VRAM Manager & Janitor
Enforces strict one-model-at-a-time GPU allocation with automatic CPU fallback.
Protects consumer GPUs (6 GB VRAM) from CUDA Out-Of-Memory (OOM) failures.
"""

import gc
import logging
from contextlib import contextmanager
from typing import Generator, Optional, Tuple

try:
    import torch  # type: ignore
except ImportError:
    torch = None

from bhashara.config import MIN_FREE_VRAM_GB

logger = logging.getLogger("bhashara.vram")


class VRAMJanitor:
    """Manages VRAM allocations, GPU cache eviction, and CPU fallback."""

    @staticmethod
    def get_vram_info() -> Tuple[float, float, float]:
        """
        Returns (allocated_gb, reserved_gb, free_gb).
        If CUDA is unavailable, returns (0.0, 0.0, 0.0).
        """
        if torch is None or not torch.cuda.is_available():
            return 0.0, 0.0, 0.0
        
        try:
            device = torch.cuda.current_device()
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            allocated_bytes = torch.cuda.memory_allocated(device)
            reserved_bytes = torch.cuda.memory_reserved(device)
            
            allocated_gb = allocated_bytes / (1024 ** 3)
            reserved_gb = reserved_bytes / (1024 ** 3)
            free_gb = free_bytes / (1024 ** 3)
            return allocated_gb, reserved_gb, free_gb
        except Exception as e:
            logger.warning(f"Could not query CUDA memory info: {e}")
            return 0.0, 0.0, 0.0

    @classmethod
    def get_peak_vram_gb(cls) -> float:
        """Returns peak allocated CUDA VRAM in GB."""
        if torch is not None and torch.cuda.is_available():
            try:
                return torch.cuda.max_memory_allocated() / (1024 ** 3)
            except Exception:
                return 0.0
        return 0.0

    @classmethod
    def reset_peak_stats(cls) -> None:
        """Resets peak CUDA memory tracking stats."""
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

    @classmethod
    def clean_vram(cls, stage_name: str = "general") -> None:
        """
        Strictly flushes all unreferenced tensors and PyTorch memory pools.
        Must be called between every heavy model stage.
        """
        pre_alloc, pre_res, pre_free = cls.get_vram_info()
        
        # 1. Force Python Garbage Collection
        gc.collect()
        
        # 2. Flush CUDA allocator
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            
        post_alloc, post_res, post_free = cls.get_vram_info()
        logger.debug(
            f"[VRAM Janitor] Post-{stage_name} Cleanup: "
            f"Free VRAM {pre_free:.2f}GB -> {post_free:.2f}GB (Allocated: {post_alloc:.2f}GB)"
        )

    @classmethod
    def determine_device(
        cls, stage_name: str, required_vram_gb: float = 1.8
    ) -> str:
        """
        Determines whether to allocate to CUDA or fallback to CPU based on available VRAM.
        Logs an explicit warning if CPU fallback is triggered.
        """
        if torch is None or not torch.cuda.is_available():
            logger.info(f"[{stage_name}] No CUDA device found. Running on CPU.")
            return "cpu"
            
        # Clean before measuring
        cls.clean_vram(f"{stage_name}_preflight")
        _, _, free_gb = cls.get_vram_info()
        
        if free_gb < required_vram_gb:
            logger.warning(
                f"[{stage_name}] VRAM WARNING: Only {free_gb:.2f} GB free, "
                f"requires {required_vram_gb:.2f} GB. Moving stage to CPU automatically!"
            )
            return "cpu"
            
        logger.info(f"[{stage_name}] Allocating on GPU (CUDA). Available VRAM: {free_gb:.2f} GB.")
        return "cuda"

    @classmethod
    @contextmanager
    def stage_context(
        cls, stage_name: str, required_vram_gb: float = 1.8
    ) -> Generator[str, None, None]:
        """
        Context manager that assigns the safe device and guarantees VRAM cleanup upon exit.
        Usage:
            with VRAMJanitor.stage_context("Demucs", required_vram_gb=1.5) as device:
                model = load_demucs(device)
                model.run(...)
                del model
        """
        device = cls.determine_device(stage_name, required_vram_gb)
        try:
            yield device
        finally:
            cls.clean_vram(stage_name)
