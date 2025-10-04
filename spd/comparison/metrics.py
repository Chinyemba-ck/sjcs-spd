"""Metrics tracking for clustering comparison interface."""

import time
import psutil
import torch
from typing import Dict, Any, Optional, Tuple
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class ComputeMetrics:
    """Track compute metrics for clustering operations."""
    
    wall_time: float = 0.0
    cpu_percent: float = 0.0
    memory_mb_used: float = 0.0
    memory_mb_peak: float = 0.0
    theoretical_flops: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for display."""
        return {
            "Wall Time (s)": f"{self.wall_time:.3f}",
            "CPU Usage (%)": f"{self.cpu_percent:.1f}",
            "Memory Used (MB)": f"{self.memory_mb_used:.1f}",
            "Memory Peak (MB)": f"{self.memory_mb_peak:.1f}",
            "Theoretical FLOPs": f"{self.theoretical_flops:,}"
        }


class MetricsTracker:
    """Track timing and compute metrics for clustering operations."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.peak_memory = None
        self.cpu_samples = []
    
    @contextmanager
    def track(self):
        """Context manager to track metrics during an operation."""
        # Reset and start tracking
        self.reset()
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        
        # Start CPU sampling in background
        import threading
        stop_sampling = threading.Event()
        
        def sample_cpu():
            while not stop_sampling.is_set():
                try:
                    cpu = self.process.cpu_percent(interval=0.1)
                    self.cpu_samples.append(cpu)
                    mem = self.process.memory_info().rss / 1024 / 1024
                    self.peak_memory = max(self.peak_memory, mem)
                except:
                    pass
                time.sleep(0.1)
        
        cpu_thread = threading.Thread(target=sample_cpu)
        cpu_thread.start()
        
        try:
            yield self
        finally:
            # Stop tracking
            self.end_time = time.time()
            stop_sampling.set()
            cpu_thread.join(timeout=1)
    
    def get_metrics(self) -> ComputeMetrics:
        """Get the collected metrics."""
        if self.start_time is None or self.end_time is None:
            return ComputeMetrics()
        
        wall_time = self.end_time - self.start_time
        cpu_percent = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0
        memory_used = self.process.memory_info().rss / 1024 / 1024 - self.start_memory
        
        return ComputeMetrics(
            wall_time=wall_time,
            cpu_percent=cpu_percent,
            memory_mb_used=memory_used,
            memory_mb_peak=self.peak_memory - self.start_memory
        )


def estimate_mdl_flops(n_components: int, n_samples: int, n_iterations: int) -> int:
    """Estimate theoretical FLOPs for MDL clustering.
    
    Main operations:
    1. Coactivation matrix: O(n_components^2 * n_samples)
    2. MDL cost computation per iteration: O(n_components^2)
    3. Merge operations: O(n_iterations * n_components)
    
    Args:
        n_components: Number of components being clustered
        n_samples: Number of activation samples
        n_iterations: Number of merge iterations
        
    Returns:
        Estimated FLOPs count
    """
    # Coactivation matrix computation
    coact_flops = n_components * n_components * n_samples
    
    # MDL cost computation (per iteration, checking all pairs)
    mdl_flops = n_iterations * n_components * n_components
    
    # Merge operations
    merge_flops = n_iterations * n_components
    
    return coact_flops + mdl_flops + merge_flops


def compare_metrics(metrics1: Optional[ComputeMetrics], metrics2: Optional[ComputeMetrics]) -> Dict[str, str]:
    """Compare two sets of metrics.
    
    Args:
        metrics1: First set of metrics (e.g., custom clustering)
        metrics2: Second set of metrics (e.g., MDL clustering)
        
    Returns:
        Dictionary with comparison results
    """
    if metrics1 is None and metrics2 is None:
        return {"Status": "No metrics available"}
    
    if metrics1 is None:
        return {"Status": "Only method 2 has metrics", **metrics2.to_dict()}
    
    if metrics2 is None:
        return {"Status": "Only method 1 has metrics", **metrics1.to_dict()}
    
    # Calculate relative differences
    time_diff = (metrics2.wall_time - metrics1.wall_time) / metrics1.wall_time * 100 if metrics1.wall_time > 0 else 0
    memory_diff = (metrics2.memory_mb_peak - metrics1.memory_mb_peak) / metrics1.memory_mb_peak * 100 if metrics1.memory_mb_peak > 0 else 0
    
    return {
        "Time Difference": f"{time_diff:+.1f}%",
        "Memory Difference": f"{memory_diff:+.1f}%",
        "Method 1 Time": f"{metrics1.wall_time:.3f}s",
        "Method 2 Time": f"{metrics2.wall_time:.3f}s",
        "Method 1 Memory": f"{metrics1.memory_mb_peak:.1f}MB",
        "Method 2 Memory": f"{metrics2.memory_mb_peak:.1f}MB",
    }