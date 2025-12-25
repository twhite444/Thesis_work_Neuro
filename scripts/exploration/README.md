# Exploration Tools

This folder contains interactive tools for exploring and visualizing the neuroimaging data.

## Available Tools

### `inspect_activity_map.py`

Interactive CLI tool for inspecting individual activity maps with visualization capabilities.

#### Features
- List all available activity maps
- View statistics for specific CIDs or files
- Interactive visualization with matplotlib popup windows
- Optional image saving
- Zero values masked as NaN for cleaner brain region visualization

#### Usage

```bash
# List all available maps with statistics
python scripts/exploration/inspect_activity_map.py --list-all

# Show statistics for a specific CID
python scripts/exploration/inspect_activity_map.py --cid 180

# Show statistics AND interactive visualization (popup window)
python scripts/exploration/inspect_activity_map.py --cid 180 --show-images

# Show visualization AND save to file
python scripts/exploration/inspect_activity_map.py --cid 180 --show-images --save-images

# Inspect a specific map file
python scripts/exploration/inspect_activity_map.py --filename 180_0.csv
```

#### Command-Line Options

- `--list-all` - List all available activity maps with CID counts
- `--cid <CID>` - Inspect all maps for a specific CID
- `--filename <FILE>` - Inspect a specific activity map CSV file
- `--show-images` - Display interactive visualization popup (requires --cid)
- `--save-images` - Save visualization to PNG file (requires --show-images)
- `--data-dir <DIR>` - Specify data directory (default: data/01_raw)

#### Examples

```bash
# Explore CID 7991 (valeric acid) - has 11 maps!
python scripts/exploration/inspect_activity_map.py --cid 7991 --show-images

# Quick stats for acetone
python scripts/exploration/inspect_activity_map.py --cid 180

# Find CIDs with multiple maps
python scripts/exploration/inspect_activity_map.py --list-all
```

#### Visualization Features

- **Grid Layout**: Multiple maps displayed in 3-column grid
- **Color Scaling**: Consistent vmin/vmax across all maps for a CID
- **Coverage Stats**: Shows % coverage and mean activity per map
- **Clean Display**: Zero values (non-brain regions) shown as white/transparent
- **Interactive**: Popup window stays open until you close it
- **High Quality**: Saves at 150 DPI when using --save-images

## Adding New Exploration Tools

Feel free to add new exploration scripts to this folder:

```python
#!/usr/bin/env python3
"""Brief description of your exploration tool."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.olfactory_modeling.data.pyrfume_loader import ...

# Your exploration code here
```

## See Also

- **Main Scripts**: `scripts/` - Pipeline scripts for data processing
- **Examples**: `scripts/examples/` - Usage examples and tutorials
