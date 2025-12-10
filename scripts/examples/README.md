# Example Scripts

This folder contains example scripts demonstrating how to use the data loading and processing utilities.

## Available Examples

### Data Loading Examples

- **`example_load_cached.py`** - Demonstrates NPZ vs CSV loading performance
  - Benchmarks loading speed comparison (NPZ is ~1.3-1.6x faster)
  - Shows how to load molecules, behavior, and activity maps
  - Example usage: `python scripts/examples/example_load_cached.py`

- **`example_load_by_cid.py`** - Shows CID-based activity map loading
  - Load all maps for a specific CID
  - Get averaged maps across repetitions
  - Find CIDs with multiple maps
  - Batch loading examples
  - Example usage: `python scripts/examples/example_load_by_cid.py`

- **`example_stimuli_metadata.py`** - Demonstrates stimuli metadata usage
  - Load experimental metadata (CID, Rep, Name, Conditions)
  - Search by molecule name
  - Filter by CID
  - Link stimuli to activity maps
  - Example usage: `python scripts/examples/example_stimuli_metadata.py`

## Quick Start

All examples are self-contained and can be run directly:

```bash
# Run from project root
python scripts/examples/example_load_cached.py
python scripts/examples/example_load_by_cid.py
python scripts/examples/example_stimuli_metadata.py
```

## Requirements

- Data must be downloaded first: `python scripts/load_all_data.py`
- All examples load from `data/01_raw/` directory

## See Also

- **Main Scripts**: `scripts/` - Pipeline scripts (load_all_data.py, preprocess.py, etc.)
- **Exploration Tools**: `scripts/exploration/` - Interactive data inspection tools
