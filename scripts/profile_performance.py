#!/usr/bin/env python3
"""Profile training performance to identify bottlenecks.

This script helps diagnose performance issues by profiling:
- DataLoader performance
- Device (CPU vs MPS) performance
- Training loop breakdown
- Batch processing times

Usage:
    python scripts/profile_performance.py --model mlp
    python scripts/profile_performance.py --model mlp --compare-devices
    python scripts/profile_performance.py --model mlp --profile-epochs 3
"""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
# sys.path.insert(0, str(Path(__file__).parent.parent))  # No longer needed with proper __init__.py

from neuro_foundation.data.activity_map_dataset import get_dataloaders
from neuro_foundation.models.baseline_nn import get_model
from src.neuro_foundation.utils.profiling import (
    Timer, 
    EpochTimer, 
    profile_dataloader,
    compare_device_performance
)


def main():
    parser = argparse.ArgumentParser(description="Profile training performance")
    
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn'],
                        help='Model architecture')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for testing')
    parser.add_argument('--profile-batches', type=int, default=20,
                        help='Number of batches to profile for dataloader')
    parser.add_argument('--profile-epochs', type=int, default=0,
                        help='Number of epochs to profile (0 = skip)')
    parser.add_argument('--compare-devices', action='store_true',
                        help='Compare performance across CPU and MPS')
    parser.add_argument('--device', type=str, default='mps',
                        help='Device to use for profiling')
    
    args = parser.parse_args()
    
    print("="*70)
    print("PERFORMANCE PROFILING")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        processed_dir='data/02_processed',
    )
    print(f"✓ Loaded {len(train_loader.dataset)} training samples")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Batches per epoch: {len(train_loader)}")
    
    # Create model
    print(f"\nCreating {args.model.upper()} model...")
    model = get_model(
        model_type=args.model,
        input_dim=268,
        output_shape=(79, 43),
        dropout=0.35,
    )
    print(f"✓ Model created")
    
    # 1. Profile DataLoader
    print("\n" + "="*70)
    print("1. DATALOADER PROFILING")
    print("="*70)
    
    dataloader_stats = profile_dataloader(
        train_loader,
        num_batches=args.profile_batches,
        device=args.device,
    )
    
    # 2. Compare devices
    if args.compare_devices:
        print("\n" + "="*70)
        print("2. DEVICE COMPARISON")
        print("="*70)
        
        # Get sample batch
        sample_batch = next(iter(train_loader))
        sample_input = sample_batch[0][:1]  # Single sample
        
        device_stats = compare_device_performance(
            model,
            sample_input,
            devices=['cpu', 'mps'],
            num_iterations=100,
        )
    
    # 3. Profile training epochs
    if args.profile_epochs > 0:
        print("\n" + "="*70)
        print(f"3. TRAINING LOOP PROFILING ({args.profile_epochs} epochs)")
        print("="*70)
        
        device_obj = torch.device(args.device)
        model = model.to(device_obj)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        criterion = torch.nn.MSELoss()
        
        epoch_timer = EpochTimer()
        
        for epoch in range(args.profile_epochs):
            print(f"\nEpoch {epoch + 1}/{args.profile_epochs}...")
            
            epoch_timer.start_epoch()
            model.train()
            
            for batch_idx, (features, targets, metadata) in enumerate(train_loader):
                epoch_timer.start_batch()
                
                # Data transfer
                with epoch_timer.time_section('data_to_device'):
                    features = features.to(device_obj)
                    targets = targets.to(device_obj)
                
                # Forward pass
                with epoch_timer.time_section('forward'):
                    predictions = model(features)
                    loss = criterion(predictions, targets)
                
                # Backward pass
                with epoch_timer.time_section('backward'):
                    optimizer.zero_grad()
                    loss.backward()
                
                # Optimizer step
                with epoch_timer.time_section('optimizer'):
                    optimizer.step()
                
                epoch_timer.end_batch()
            
            epoch_timer.end_epoch()
        
        epoch_timer.print_epoch_summary()
    
    # Final summary
    print("\n" + "="*70)
    print("PROFILING COMPLETE")
    print("="*70)
    
    print("\nKey findings:")
    
    # DataLoader findings
    total_data_time = dataloader_stats['data_loading']['mean'] + dataloader_stats['device_transfer']['mean']
    print(f"\n1. DataLoader:")
    print(f"   - Time per batch: {total_data_time*1000:.2f}ms")
    print(f"   - Throughput: {1.0/total_data_time:.1f} batches/s")
    
    if total_data_time > 0.1:
        print(f"   ⚠️  Data loading is slow (>100ms/batch)")
        print(f"      Consider: reducing batch complexity or checking disk I/O")
    else:
        print(f"   ✓ Data loading is fast")
    
    # Device findings
    if args.compare_devices and device_stats:
        print(f"\n2. Device Performance:")
        valid_devices = {k: v for k, v in device_stats.items() if v is not None}
        if valid_devices:
            best = min(valid_devices.keys(), key=lambda k: valid_devices[k]['mean'])
            print(f"   - Best device: {best}")
            for device_name, stats in valid_devices.items():
                print(f"   - {device_name}: {stats['mean']*1000:.2f}ms/inference")
    
    # Training findings
    if args.profile_epochs > 0:
        print(f"\n3. Training Loop:")
        batch_time = sum(epoch_timer.batch_times) / len(epoch_timer.batch_times)
        samples_per_sec = args.batch_size / batch_time
        print(f"   - Time per batch: {batch_time*1000:.2f}ms")
        print(f"   - Throughput: {samples_per_sec:.1f} samples/s")
        
        epoch_time = sum(epoch_timer.epoch_times) / len(epoch_timer.epoch_times)
        print(f"   - Time per epoch: {epoch_time:.2f}s")
        print(f"   - Estimated time for 100 epochs: {epoch_time*100/60:.1f} minutes")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
