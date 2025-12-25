# API Reference & Documentation

Complete reference for all modules, functions, and classes in the Neuro Foundation project.

---

## 📦 **Module Overview**

```
src/olfactory_modeling/
├── data/
│   ├── interfaces.py       # Data interfaces and types
│   └── pyrfume_loader.py   # Data loading utilities
└── pipeline/
    ├── preprocess.py       # Feature extraction
    ├── feature_select.py   # Feature selection
    ├── train_linear.py     # Linear model training
    └── activity_maps.py    # Activity map processing
```

---

## 🗂️ **Data Module**

### `src.olfactory_modeling.data.pyrfume_loader`

Handles all data loading from Pyrfume database with dual CSV/NPZ format support.

#### Main Loader Class

```python
class PyrfumeLoader:
    """Main class for loading data from Pyrfume.
    
    Attributes:
        output_dir (str): Directory to save downloaded data
        
    Methods:
        load_molecules() -> pd.DataFrame
        load_behavior() -> pd.DataFrame  
        load_stimuli() -> pd.DataFrame
        load_activity_maps(save_individual_csvs=True) -> None
    """
```

**Usage Example:**
```python
from src.olfactory_modeling.data.pyrfume_loader import PyrfumeLoader

# Initialize loader
loader = PyrfumeLoader(output_dir='data/01_raw')

# Download data
molecules = loader.load_molecules()  # 287 unique molecules
behavior = loader.load_behavior()    # 405 activity map entries
stimuli = loader.load_stimuli()      # 432 stimuli with metadata
loader.load_activity_maps()          # 405 brain activity maps

# Data is saved as both CSV (human-readable) and NPZ (fast loading)
```

#### Helper Functions - CSV Loading

```python
def load_molecules_csv(data_dir: str = "data/01_raw") -> pd.DataFrame:
    """Load molecules from CSV file.
    
    Returns DataFrame with columns:
    - CID: PubChem Compound ID
    - IsomericSMILES: Molecular structure
    - MolecularWeight: Molecular weight
    - IUPACName: IUPAC chemical name
    - name: Common name
    """

def load_behavior_csv(data_dir: str = "data/01_raw") -> pd.DataFrame:
    """Load behavior data from CSV file.
    
    Returns DataFrame with columns:
    - Stimulus: CID of the stimulus
    - Activity Map Path: Path to activity map CSV
    """

def load_stimuli_csv(data_dir: str = "data/01_raw") -> pd.DataFrame:
    """Load stimuli metadata from CSV file.
    
    Returns DataFrame with columns:
    - CID: Compound ID
    - Rep: Repetition number
    - Name: Stimulus name
    - Conditions: Experimental conditions
    - SourceFile: Original data file
    """
```

#### Helper Functions - NPZ Loading (Faster!)

```python
def load_molecules_npz(data_dir: str = "data/01_raw") -> pd.DataFrame:
    """Load molecules from NPZ file (1.3-1.6x faster than CSV)."""

def load_behavior_npz(data_dir: str = "data/01_raw") -> pd.DataFrame:
    """Load behavior data from NPZ file (1.3-1.6x faster than CSV)."""

def load_stimuli_npz(data_dir: str = "data/01_raw") -> pd.DataFrame:
    """Load stimuli metadata from NPZ file (1.3-1.6x faster than CSV)."""
```

#### Activity Map Helper Functions

```python
def load_activity_maps_npz(data_dir: str = "data/01_raw") -> dict:
    """Load all activity maps from NPZ file.
    
    Returns:
        dict with keys: 'maps', 'cids', 'filenames'
        - maps: numpy array (405, 79, 43)
        - cids: numpy array of CIDs
        - filenames: numpy array of filenames
    """

def load_activity_maps_as_arrays(data_dir: str = "data/01_raw") -> Tuple[List[np.ndarray], np.ndarray]:
    """Load activity maps as list of arrays.
    
    Returns:
        Tuple of (maps, cids)
        - maps: List of 405 numpy arrays (79, 43)
        - cids: numpy array of CIDs
    """

def load_activity_maps_by_cid(cid: int, data_dir: str = "data/01_raw") -> List[np.ndarray]:
    """Load all activity maps for a specific CID.
    
    Args:
        cid: PubChem Compound ID
        data_dir: Data directory
        
    Returns:
        List of numpy arrays, one per repetition
        Empty list if CID not found
        
    Example:
        maps = load_activity_maps_by_cid(180)  # Get all maps for acetone
        print(f"Found {len(maps)} maps")  # 3 maps
    """

def load_activity_map_by_cid_averaged(cid: int, data_dir: str = "data/01_raw") -> Optional[np.ndarray]:
    """Load averaged activity map for a CID.
    
    Args:
        cid: PubChem Compound ID
        data_dir: Data directory
        
    Returns:
        Averaged numpy array across all repetitions
        None if CID not found
        
    Example:
        avg_map = load_activity_map_by_cid_averaged(180)
        coverage = (avg_map > 0).mean() * 100  # 27.5% active
    """
```

**Performance Tips:**
- Use NPZ functions for repeated loading (1.3-1.6x faster)
- Use CSV functions for one-time inspection
- Use `load_activity_maps_by_cid` for efficient CID filtering

---

## ⚙️ **Pipeline Module**

### `src.olfactory_modeling.pipeline.preprocess`

Feature extraction using Mordred molecular descriptors.

```python
def featurize_and_standardize(
    molecules: pd.DataFrame,
    output_dir: str = "data/02_processed"
) -> pd.DataFrame:
    """Extract Mordred features and standardize.
    
    Args:
        molecules: DataFrame with 'IsomericSMILES' and 'CID' columns
        output_dir: Directory to save processed data
        
    Returns:
        DataFrame with molecular descriptors (287 x 1187 features)
        
    Process:
        1. Compute 1826 Mordred descriptors
        2. Remove invariant features
        3. Standardize (mean=0, std=1)
        4. Save as cleaned_data.csv
        
    Example:
        from src.olfactory_modeling.data.pyrfume_loader import load_molecules_npz
        from src.olfactory_modeling.pipeline.preprocess import featurize_and_standardize
        
        molecules = load_molecules_npz()
        features = featurize_and_standardize(molecules)
        print(features.shape)  # (287, 1187)
    """
```

**Output Files:**
- `cleaned_data.csv`: Standardized features
- `scaler_stats.json`: Mean and std for each feature

---

### `src.olfactory_modeling.pipeline.feature_select`

Variance-based feature selection.

```python
def select_features(
    df: pd.DataFrame,
    threshold: float = 1.0,
    output_dir: str = "data/02_processed"
) -> pd.DataFrame:
    """Select features based on variance threshold.
    
    Args:
        df: Standardized feature DataFrame
        threshold: Minimum variance (default: 1.0 for standardized data)
        output_dir: Directory to save selected features
        
    Returns:
        DataFrame with high-variance features only
        
    Process:
        1. Compute variance for each feature
        2. Keep features with variance >= threshold
        3. Save as selected_features.csv
        4. Save metadata as feature_select_meta.json
        
    Example:
        features = pd.read_csv('data/02_processed/cleaned_data.csv')
        selected = select_features(features, threshold=1.0)
        print(f"Kept {len(selected.columns)} features")
    """
```

**Output Files:**
- `selected_features.csv`: Filtered features
- `feature_select_meta.json`: Selection metadata

---

### `src.olfactory_modeling.pipeline.train_linear`

Linear regression model training.

```python
def train_linear_model(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str = "data/02_processed"
) -> Tuple[Any, pd.DataFrame, pd.Series]:
    """Train linear regression model.
    
    Args:
        X: Feature matrix
        y: Target values
        output_dir: Directory to save model outputs
        
    Returns:
        Tuple of (model, predictions_df, coefficients_series)
        
    Process:
        1. Fit scikit-learn LinearRegression
        2. Generate predictions
        3. Save coefficients
        4. Save predictions
        
    Example:
        X = pd.read_csv('data/02_processed/selected_features.csv', index_col=0)
        y = pd.Series([...])  # Your target values
        
        model, preds, coefs = train_linear_model(X, y)
        print(f"R² score: {model.score(X, y):.3f}")
    """
```

**Output Files:**
- `model_coefficients.csv`: Feature coefficients
- `predictions.csv`: Model predictions

---

### `src.olfactory_modeling.pipeline.activity_maps`

Brain activity map processing with masking and averaging.

#### Data Structures

```python
@dataclass
class ActivityMapRecord:
    """Container for activity map data.
    
    Attributes:
        cid (int): PubChem Compound ID
        map (np.ndarray): Activity map array (79, 43)
    """
```

#### Core Functions

```python
def load_directory_csv(path: str) -> pd.DataFrame:
    """Load behavior directory CSV and derive CID column.
    
    Expects columns: Stimulus, Activity Map Path
    Adds CID column extracted from Stimulus
    Filters out negative CIDs (natural mixtures)
    """

def load_activity_maps(
    directory_df: pd.DataFrame,
    data_dir: str = 'data/01_raw'
) -> List[ActivityMapRecord]:
    """Load activity maps from local CSV files.
    
    Args:
        directory_df: DataFrame from load_directory_csv()
        data_dir: Directory containing activity_maps_csv/ folder
        
    Returns:
        List of ActivityMapRecord objects
        
    Note: Converts NaN to 0 during loading
    """

def compute_global_mask(
    records: List[ActivityMapRecord],
    coverage_threshold: float
) -> np.ndarray:
    """Compute global brain mask based on coverage.
    
    Args:
        records: List of activity map records
        coverage_threshold: Fraction of maps required (0-1)
        
    Returns:
        Boolean mask (79, 43) indicating valid brain regions
        
    Process:
        1. Count non-zero pixels across all maps
        2. Require coverage_threshold fraction of maps
        3. Apply morphological operations (erosion + dilation)
        4. Keep regions >= 100 pixels
        
    Example:
        # Require 50% of maps to have activity
        mask = compute_global_mask(records, coverage_threshold=0.5)
        print(f"Mask covers {mask.sum()} pixels")
    """

def apply_mask(
    records: List[ActivityMapRecord],
    mask: np.ndarray
) -> List[ActivityMapRecord]:
    """Apply global mask to all activity maps.
    
    Sets pixels outside mask to 0.
    """

def average_by_cid(
    records: List[ActivityMapRecord]
) -> Tuple[List[np.ndarray], List[int]]:
    """Average maps by CID to handle multiple repetitions.
    
    Returns:
        Tuple of (averaged_maps, cids)
        - One averaged map per unique CID
        - CIDs in corresponding order
    """

def pipeline_load_and_mask(
    directory_csv: str,
    data_dir: str = 'data/01_raw',
    coverage_threshold: float = 1.0,
    output_dir: str = 'data/02_processed',
    verbose: bool = False
) -> Tuple[List[np.ndarray], List[int], np.ndarray]:
    """Complete pipeline: load, mask, average, and visualize.
    
    Args:
        directory_csv: Path to behavior CSV
        data_dir: Directory with activity_maps_csv/
        coverage_threshold: Coverage threshold (0-1)
        output_dir: Output directory
        verbose: Print debug info
        
    Returns:
        Tuple of (averaged_maps, cids, mask)
        
    Output Files:
        - global_mask.png: Refined brain mask
        - coverage_counts.png: Heatmap of coverage
        - coverage_histogram.png: Distribution
        - masked_averaged_example.png: Example map
        - masked_averaged_gallery.png: Gallery of 6 maps
        
    Example:
        maps, cids, mask = pipeline_load_and_mask(
            directory_csv='data/01_raw/behavior_data.csv',
            data_dir='data/01_raw',
            coverage_threshold=0.5,
            output_dir='data/02_processed'
        )
        print(f"Got {len(maps)} averaged maps")
        print(f"Mask covers {mask.sum()}/{mask.size} pixels")
    """
```

**Visualization Notes:**
- Zero values displayed as NaN (white/transparent) in images
- Original data arrays keep zeros intact for analysis
- Coverage counts show number of maps at each pixel

---

## 🔧 **Scripts API**

### Main Pipeline Scripts

#### `scripts/load_all_data.py`
```bash
# Download all data from Pyrfume
python scripts/load_all_data.py

# Skip activity maps (large download)
python scripts/load_all_data.py --skip-activity-maps
```

#### `scripts/preprocess.py`
```bash
# Use cached data (default, fast NPZ loading)
python scripts/preprocess.py

# Force fresh download
python scripts/preprocess.py --force-download

# Custom directories
python scripts/preprocess.py --data-dir /path/to/data --output-dir /path/to/output
```

#### `scripts/run_activity_maps.py`
```bash
# Default (50% coverage threshold)
python scripts/run_activity_maps.py

# Custom threshold
python scripts/run_activity_maps.py --coverage-threshold 0.3

# Verbose output
python scripts/run_activity_maps.py --verbose
```

#### `scripts/select_features.py`
```bash
# Default variance threshold (1.0)
python scripts/select_features.py

# Custom threshold
python scripts/select_features.py --threshold 0.5
```

#### `scripts/train_linear.py`
```bash
python scripts/train_linear.py --input-csv data/02_processed/selected_features.csv
```

---

### Exploration Tools

#### `scripts/exploration/inspect_activity_map.py`

Interactive tool for exploring activity maps.

```bash
# List all available maps
python scripts/exploration/inspect_activity_map.py --list-all

# Show statistics for a CID
python scripts/exploration/inspect_activity_map.py --cid 180

# Show interactive visualization (popup window)
python scripts/exploration/inspect_activity_map.py --cid 180 --show-images

# Show visualization AND save to file
python scripts/exploration/inspect_activity_map.py --cid 180 --show-images --save-images

# Inspect specific file
python scripts/exploration/inspect_activity_map.py --filename 180_0.csv
```

**Command-Line Options:**
- `--list-all`: List all maps with CID counts
- `--cid <CID>`: Inspect maps for specific CID
- `--filename <FILE>`: Inspect specific map file
- `--show-images`: Display interactive matplotlib window
- `--save-images`: Save visualization to PNG
- `--data-dir <DIR>`: Specify data directory

---

### Example Scripts

#### `scripts/examples/example_load_by_cid.py`
Demonstrates CID-based activity map loading with examples of:
- Single vs multiple maps
- Averaged maps
- Batch loading
- Finding CIDs with most maps

#### `scripts/examples/example_load_cached.py`
Benchmarks NPZ vs CSV loading performance:
- Shows 1.3-1.6x speedup with NPZ
- Demonstrates all helper functions
- Usage examples for each data type

#### `scripts/examples/example_stimuli_metadata.py`
Shows how to use stimuli metadata:
- Search by molecule name
- Filter by CID
- Link stimuli to activity maps
- Understand experimental conditions

---

## 📊 **Data Schemas**

### Molecules DataFrame
```
Columns: CID, IsomericSMILES, MolecularWeight, IUPACName, name
Shape: (287, 5)
Example:
    CID  IsomericSMILES  MolecularWeight  IUPACName        name
    180  CC(=O)C         58.08            propan-2-one     acetone
```

### Behavior DataFrame
```
Columns: Stimulus, Activity Map Path
Shape: (405, 2)
Example:
    Stimulus  Activity Map Path
    180       csvs/180_0.csv
    180       csvs/180_1.csv
```

### Stimuli DataFrame
```
Columns: CID, Rep, Name, Conditions, SourceFile
Shape: (432, 5)
Example:
    CID  Rep  Name     Conditions  SourceFile
    180  0    acetone  control     data1.csv
    180  1    acetone  control     data1.csv
```

### Activity Maps
```
Format: numpy array or CSV
Shape: (79, 43) pixels per map
Values: Float (z-scores of activity)
Zero: No activity/outside brain
Non-zero: Brain activity level
```

### Cleaned Features
```
Format: pandas DataFrame
Shape: (287, 1187)
Index: CID
Columns: Mordred descriptor names
Values: Standardized (mean=0, std=1)
```

---

## 🎯 **Common Usage Patterns**

### Pattern 1: Fast Data Loading
```python
from src.olfactory_modeling.data.pyrfume_loader import (
    load_molecules_npz,
    load_behavior_npz,
    load_activity_maps_by_cid
)

# Fast loading with NPZ
molecules = load_molecules_npz()
behavior = load_behavior_npz()

# Get maps for specific molecule
maps = load_activity_maps_by_cid(180)  # acetone
```

### Pattern 2: Feature Pipeline
```python
from src.olfactory_modeling.data.pyrfume_loader import load_molecules_npz
from src.olfactory_modeling.pipeline.preprocess import featurize_and_standardize
from src.olfactory_modeling.pipeline.feature_select import select_features

# Load data
molecules = load_molecules_npz()

# Extract features
features = featurize_and_standardize(molecules)

# Select high-variance features
selected = select_features(features, threshold=1.0)
```

### Pattern 3: Activity Map Analysis
```python
from src.olfactory_modeling.pipeline.activity_maps import pipeline_load_and_mask

# Complete pipeline
maps, cids, mask = pipeline_load_and_mask(
    directory_csv='data/01_raw/behavior_data.csv',
    coverage_threshold=0.5
)

# Analyze results
print(f"Processed {len(maps)} unique molecules")
print(f"Mask covers {mask.sum()} brain pixels")
```

### Pattern 4: CID-based Analysis
```python
from src.olfactory_modeling.data.pyrfume_loader import (
    load_molecules_npz,
    load_activity_maps_by_cid,
    load_activity_map_by_cid_averaged
)

# Get molecule info
molecules = load_molecules_npz()
acetone = molecules[molecules['CID'] == 180].iloc[0]

# Get all repetitions
maps = load_activity_maps_by_cid(180)
print(f"Found {len(maps)} repetitions")

# Get averaged map
avg_map = load_activity_map_by_cid_averaged(180)
coverage = (avg_map > 0).mean() * 100
print(f"Coverage: {coverage:.1f}%")
```

---

## ⚠️ **Important Notes**

### Data Consistency
- CIDs may have 1-11 activity maps (repetitions)
- Use averaged maps for single representation
- Negative CIDs are natural mixtures (filtered out)
- 287 unique molecules, 405 total activity maps

### Performance
- NPZ loading is 1.3-1.6x faster than CSV
- Activity maps are ~10MB uncompressed, ~3MB compressed
- Use `load_activity_maps_by_cid` for efficient filtering

### Visualization
- Zero values displayed as NaN (transparent) in plots
- Original data keeps zeros for computational analysis
- Interactive plots block until user closes window
- Use `--save-images` flag to save visualizations

### Common Pitfalls
1. Don't call `load_all_data.py` repeatedly (data already cached)
2. Use NPZ functions for repeated loading
3. Remember to close interactive matplotlib windows
4. Check that `activity_maps_csv/` folder exists before processing

---

**Last Updated**: December 10, 2025
**Version**: 1.0.0
