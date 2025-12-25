"""Profiling and timing utilities for performance analysis."""

import time
import contextlib
from typing import Dict, Optional
from collections import defaultdict
import numpy as np


class Timer:
    """Simple timer context manager for profiling code sections.
    
    Example:
        >>> timer = Timer()
        >>> with timer("data_loading"):
        ...     data = load_data()
        >>> with timer("training"):
        ...     train_model()
        >>> timer.print_summary()
    """
    
    def __init__(self):
        self.times: Dict[str, list] = defaultdict(list)
        self.current_start: Optional[float] = None
        self.current_name: Optional[str] = None
    
    @contextlib.contextmanager
    def __call__(self, name: str):
        """Time a code block."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.times[name].append(elapsed)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a named section."""
        if name not in self.times:
            return {}
        
        times = self.times[name]
        return {
            'count': len(times),
            'total': sum(times),
            'mean': np.mean(times),
            'std': np.std(times),
            'min': min(times),
            'max': max(times),
        }
    
    def print_summary(self):
        """Print timing summary for all sections."""
        if not self.times:
            print("No timing data collected")
            return
        
        print("\n" + "="*70)
        print("TIMING SUMMARY")
        print("="*70)
        
        # Calculate total time
        total_time = sum(sum(times) for times in self.times.values())
        
        # Print each section
        for name in sorted(self.times.keys()):
            stats = self.get_stats(name)
            section_total = stats['total']
            percentage = (section_total / total_time * 100) if total_time > 0 else 0
            
            print(f"\n{name}:")
            print(f"  Count:   {stats['count']}")
            print(f"  Total:   {section_total:.3f}s ({percentage:.1f}%)")
            print(f"  Mean:    {stats['mean']:.3f}s ± {stats['std']:.3f}s")
            print(f"  Range:   [{stats['min']:.3f}s, {stats['max']:.3f}s]")
        
        print(f"\nTotal time: {total_time:.3f}s")
        print("="*70)
    
    def reset(self):
        """Clear all timing data."""
        self.times.clear()


class EpochTimer:
    """Specialized timer for training epochs with detailed breakdown.
    
    Tracks:
    - Data loading time
    - Forward pass time
    - Backward pass time
    - Optimizer step time
    - Metric computation time
    
    Example:
        >>> timer = EpochTimer()
        >>> timer.start_epoch()
        >>> 
        >>> for batch in dataloader:
        ...     timer.start_batch()
        ...     
        ...     with timer.time_section('data_to_device'):
        ...         batch = batch.to(device)
        ...     
        ...     with timer.time_section('forward'):
        ...         output = model(batch)
        ...     
        ...     with timer.time_section('backward'):
        ...         loss.backward()
        ...     
        ...     with timer.time_section('optimizer'):
        ...         optimizer.step()
        ...     
        ...     timer.end_batch()
        >>> 
        >>> timer.end_epoch()
        >>> timer.print_epoch_summary()
    """
    
    def __init__(self):
        self.epoch_start: Optional[float] = None
        self.batch_start: Optional[float] = None
        self.section_times: Dict[str, list] = defaultdict(list)
        self.batch_times: list = []
        self.epoch_times: list = []
    
    def start_epoch(self):
        """Mark the start of an epoch."""
        self.epoch_start = time.perf_counter()
    
    def end_epoch(self):
        """Mark the end of an epoch."""
        if self.epoch_start is not None:
            elapsed = time.perf_counter() - self.epoch_start
            self.epoch_times.append(elapsed)
            self.epoch_start = None
    
    def start_batch(self):
        """Mark the start of a batch."""
        self.batch_start = time.perf_counter()
    
    def end_batch(self):
        """Mark the end of a batch."""
        if self.batch_start is not None:
            elapsed = time.perf_counter() - self.batch_start
            self.batch_times.append(elapsed)
            self.batch_start = None
    
    @contextlib.contextmanager
    def time_section(self, name: str):
        """Time a specific section within a batch."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.section_times[name].append(elapsed)
    
    def print_epoch_summary(self):
        """Print detailed timing breakdown for the epoch."""
        if not self.batch_times:
            print("No timing data collected")
            return
        
        print("\n" + "="*70)
        print("EPOCH TIMING BREAKDOWN")
        print("="*70)
        
        # Batch statistics
        print(f"\nBatches:")
        print(f"  Count:        {len(self.batch_times)}")
        print(f"  Mean time:    {np.mean(self.batch_times):.3f}s")
        print(f"  Std:          {np.std(self.batch_times):.3f}s")
        print(f"  Throughput:   {1.0/np.mean(self.batch_times):.2f} batches/s")
        
        # Section breakdown
        if self.section_times:
            print(f"\nSection breakdown:")
            total_section_time = sum(sum(times) for times in self.section_times.values())
            
            for name in sorted(self.section_times.keys()):
                times = self.section_times[name]
                section_total = sum(times)
                percentage = (section_total / total_section_time * 100) if total_section_time > 0 else 0
                
                print(f"\n  {name}:")
                print(f"    Total:   {section_total:.3f}s ({percentage:.1f}%)")
                print(f"    Mean:    {np.mean(times):.4f}s")
                print(f"    Per batch: {section_total/len(self.batch_times):.4f}s")
        
        # Epoch statistics
        if self.epoch_times:
            print(f"\nEpoch statistics:")
            print(f"  Completed epochs: {len(self.epoch_times)}")
            print(f"  Mean epoch time:  {np.mean(self.epoch_times):.2f}s")
            print(f"  Last epoch time:  {self.epoch_times[-1]:.2f}s")
        
        print("="*70)
    
    def reset(self):
        """Clear all timing data."""
        self.section_times.clear()
        self.batch_times.clear()
        self.epoch_times.clear()
        self.epoch_start = None
        self.batch_start = None


def profile_dataloader(dataloader, num_batches: int = 10, device: str = 'cpu'):
    """Profile dataloader performance.
    
    Args:
        dataloader: PyTorch DataLoader to profile
        num_batches: Number of batches to profile
        device: Device to transfer data to
        
    Returns:
        Dictionary with profiling statistics
    """
    import torch
    
    print(f"\nProfiling dataloader ({num_batches} batches)...")
    
    timer = Timer()
    batch_sizes = []
    
    device_obj = torch.device(device)
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        # Time data loading
        with timer("data_loading"):
            pass  # Already loaded by iterator
        
        # Time device transfer
        with timer("device_transfer"):
            if isinstance(batch, (tuple, list)):
                batch = [b.to(device_obj) if hasattr(b, 'to') else b for b in batch]
            else:
                batch = batch.to(device_obj)
        
        # Track batch size
        if isinstance(batch, (tuple, list)):
            batch_sizes.append(batch[0].shape[0] if hasattr(batch[0], 'shape') else 1)
        else:
            batch_sizes.append(batch.shape[0] if hasattr(batch, 'shape') else 1)
    
    # Calculate statistics
    results = {
        'batches_profiled': num_batches,
        'mean_batch_size': np.mean(batch_sizes),
        'data_loading': timer.get_stats('data_loading'),
        'device_transfer': timer.get_stats('device_transfer'),
    }
    
    # Print summary
    print(f"\nDataLoader Profile:")
    print(f"  Batches: {results['batches_profiled']}")
    print(f"  Mean batch size: {results['mean_batch_size']:.1f}")
    print(f"  Data loading: {results['data_loading']['mean']:.4f}s/batch")
    print(f"  Device transfer: {results['device_transfer']['mean']:.4f}s/batch")
    print(f"  Total: {results['data_loading']['mean'] + results['device_transfer']['mean']:.4f}s/batch")
    print(f"  Throughput: {1.0/(results['data_loading']['mean'] + results['device_transfer']['mean']):.2f} batches/s")
    
    return results


def compare_device_performance(model, sample_input, devices: list = ['cpu', 'mps'], num_iterations: int = 100):
    """Compare model performance across different devices.
    
    Args:
        model: PyTorch model to test
        sample_input: Sample input tensor
        devices: List of device names to test
        num_iterations: Number of forward passes to run
        
    Returns:
        Dictionary with performance comparison
    """
    import torch
    
    print(f"\nComparing device performance ({num_iterations} iterations)...")
    
    results = {}
    
    for device_name in devices:
        try:
            device = torch.device(device_name)
            
            # Move model and input to device
            model_copy = model.to(device)
            input_copy = sample_input.to(device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = model_copy(input_copy)
            
            # Time forward passes
            times = []
            with torch.no_grad():
                for _ in range(num_iterations):
                    start = time.perf_counter()
                    _ = model_copy(input_copy)
                    elapsed = time.perf_counter() - start
                    times.append(elapsed)
            
            results[device_name] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'throughput': 1.0 / np.mean(times),
            }
            
            print(f"\n{device_name}:")
            print(f"  Mean: {results[device_name]['mean']*1000:.2f}ms")
            print(f"  Std:  {results[device_name]['std']*1000:.2f}ms")
            print(f"  Throughput: {results[device_name]['throughput']:.1f} inferences/s")
            
        except Exception as e:
            print(f"\n{device_name}: Failed ({str(e)})")
            results[device_name] = None
    
    # Find best device
    valid_devices = {k: v for k, v in results.items() if v is not None}
    if valid_devices:
        best_device = min(valid_devices.keys(), key=lambda k: valid_devices[k]['mean'])
        print(f"\nBest device: {best_device}")
        
        # Show speedup comparison
        if len(valid_devices) > 1:
            print(f"\nSpeedup comparison (vs {best_device}):")
            for device_name, stats in valid_devices.items():
                if device_name != best_device:
                    speedup = valid_devices[best_device]['mean'] / stats['mean']
                    print(f"  {device_name}: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
    
    return results
