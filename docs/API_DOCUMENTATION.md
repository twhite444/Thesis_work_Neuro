# API Reference & Documentation

Complete reference for all modules, functions, and classes in the Neuro Foundation project.

---

## 📦 **Module Overview**

```
src/neuro_foundation/
├── data/
│   ├── interfaces.py         # Data interfaces and types
│   ├── pyrfume_loader.py     # Data loading utilities
│   ├── molecular_graphs.py   # Graph generation and loading
│   └── graph_viz.py          # Graph visualization tools
└── pipeline/
    ├── preprocess.py         # Feature extraction (Mordred)
    ├── feature_select.py     # Feature selection
    ├── train_linear.py       # Linear model training
    └── activity_maps.py      # Activity map processing
```

---

## 🗂️ **Data Module**

### `src.neuro_foundation.data.pyrfume_loader`

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
from src.neuro_foundation.data.pyrfume_loader import PyrfumeLoader

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

## 🔬 **Molecular Graph Module**

### `src.neuro_foundation.data.molecular_graphs`

Generate and load graph-based molecular representations for use with Graph Neural Networks.

#### Core Functions

```python
def smiles_to_graph(
    smiles: str,
    include_edge_features: bool = True
) -> Optional[dict]:
    """Convert SMILES string to graph representation.
    
    Args:
        smiles: SMILES molecular structure string
        include_edge_features: Include bond features (default: True)
        
    Returns:
        Dictionary with:
        - 'node_features': np.ndarray (num_nodes, 137)
            - Atomic number (100 dims, one-hot)
            - Degree (11 dims, one-hot, 0-10)
            - Formal charge (5 dims, one-hot, -2 to +2)
            - Hybridization (5 dims, one-hot)
            - Aromaticity (1 dim, boolean)
            - Num hydrogens (5 dims, one-hot, 0-4)
            - Radical electrons (5 dims, one-hot, 0-4)
            - In ring (1 dim, boolean)
            - Chirality (4 dims, one-hot)
        - 'edge_index': np.ndarray (2, num_edges*2)
            - Bidirectional edge connectivity
        - 'edge_attr': np.ndarray (num_edges*2, 10) [if included]
            - Bond type (4 dims, one-hot: single/double/triple/aromatic)
            - Conjugation (1 dim, boolean)
            - In ring (1 dim, boolean)
            - Stereo (4 dims, one-hot)
        - 'num_nodes': int
        - 'num_edges': int (bidirectional count)
        
    Example:
        graph = smiles_to_graph('CC(=O)O')  # Acetic acid
        print(f"Nodes: {graph['num_nodes']}, Edges: {graph['num_edges']}")
        print(f"Node features: {graph['node_features'].shape}")  # (4, 137)
    """

def generate_all_graphs(
    molecules: pd.DataFrame,
    output_dir: str = 'data/01_raw',
    include_edge_features: bool = True
) -> dict:
    """Generate graphs for all molecules in DataFrame.
    
    Args:
        molecules: DataFrame with 'CID' and 'IsomericSMILES' columns
        output_dir: Directory to save molecular_graphs.npz
        include_edge_features: Include bond features
        
    Returns:
        Dictionary with:
        - 'cids': np.ndarray of CIDs
        - 'node_features_list': List of node feature arrays
        - 'edge_index_list': List of edge index arrays
        - 'edge_attr_list': List of edge feature arrays [if included]
        - 'num_nodes': np.ndarray of node counts
        - 'num_edges': np.ndarray of edge counts
        - 'valid_mask': Boolean array indicating successful conversions
        
    Example:
        from src.neuro_foundation.data.pyrfume_loader import load_molecules_npz
        molecules = load_molecules_npz()
        graph_data = generate_all_graphs(molecules)
        print(f"Generated {graph_data['valid_mask'].sum()} graphs")
    """

def load_graph_data(data_dir: str = 'data/01_raw') -> dict:
    """Load pre-generated molecular graphs from NPZ file.
    
    Args:
        data_dir: Directory containing molecular_graphs.npz
        
    Returns:
        Dictionary with same structure as generate_all_graphs()
        
    Example:
        graph_data = load_graph_data()
        print(f"Loaded {len(graph_data['cids'])} molecules")
    """

def get_graph_by_cid(cid: int, graph_data: dict) -> Optional[dict]:
    """Extract graph for specific CID from graph_data.
    
    Args:
        cid: PubChem Compound ID
        graph_data: Output from load_graph_data()
        
    Returns:
        Dictionary with graph data for single molecule, or None if not found
        
    Example:
        graph_data = load_graph_data()
        acetone = get_graph_by_cid(180, graph_data)
        if acetone:
            print(f"Acetone: {acetone['num_nodes']} atoms, {acetone['num_edges']} bonds")
    """

def print_graph_summary(graph_data: dict) -> None:
    """Print statistical summary of graph dataset.
    
    Args:
        graph_data: Output from load_graph_data()
        
    Prints:
        - Total molecules
        - Valid molecules
        - Node statistics (mean, std, range)
        - Edge statistics (mean, std, range)
        - Feature dimensions
    """
```

**Graph Feature Details:**

**Node Features (137 dimensions):**
1. **Atomic Number** (100 dims): One-hot encoding for elements 1-100
2. **Degree** (11 dims): One-hot encoding for 0-10 neighbors
3. **Formal Charge** (5 dims): One-hot for charges -2, -1, 0, +1, +2
4. **Hybridization** (5 dims): SP, SP2, SP3, SP3D, SP3D2
5. **Aromaticity** (1 dim): Boolean aromatic flag
6. **Num Hydrogens** (5 dims): One-hot for 0-4 hydrogens
7. **Radical Electrons** (5 dims): One-hot for 0-4 radicals
8. **In Ring** (1 dim): Boolean ring membership
9. **Chirality** (4 dims): R, S, unspecified, other

**Edge Features (10 dimensions):**
1. **Bond Type** (4 dims): Single, double, triple, aromatic
2. **Conjugation** (1 dim): Boolean conjugated flag
3. **In Ring** (1 dim): Boolean ring membership
4. **Stereo** (4 dims): E, Z, cis, trans configurations

---

## 🎨 **Graph Visualization Module**

### `src.neuro_foundation.data.graph_viz`

Advanced visualization tools for molecular structures and graphs.

#### Main Visualization Functions

```python
def visualize_molecular_graph(
    cid: int,
    graph_data: dict,
    molecules_df: Optional[pd.DataFrame] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    show_atom_indices: bool = False,
    mode: str = 'detailed',
    projection: str = '2d',
    figsize: Optional[tuple] = None
) -> None:
    """Visualize a molecular graph with structure and statistics.
    
    Args:
        cid: Compound ID
        graph_data: Dictionary from load_graph_data()
        molecules_df: Optional DataFrame with molecule info
        save_path: Path to save PNG (default: viz/molecules/)
        show: Display interactive window
        show_atom_indices: Show atom numbering
        mode: 'simple' (molecule only) or 'detailed' (molecule + info)
        projection: '2d' (flat) or '3d' (ball-and-stick)
        figsize: Figure size (auto-determined if None)
        
    Example:
        from src.neuro_foundation.data.molecular_graphs import load_graph_data
        from src.neuro_foundation.data.graph_viz import visualize_molecular_graph
        from src.neuro_foundation.data.pyrfume_loader import load_molecules_npz
        
        graph_data = load_graph_data()
        molecules = load_molecules_npz()
        
        # Detailed 2D view
        visualize_molecular_graph(180, graph_data, molecules, mode='detailed')
        
        # Simple 3D view
        visualize_molecular_graph(180, graph_data, molecules, 
                                 mode='simple', projection='3d')
    """

def visualize_molecule_3d_pymol(
    smiles: str,
    output_path: str,
    width: int = 1200,
    height: int = 1200,
    style: str = 'sticks',
    ray_trace: bool = True
) -> bool:
    """Create high-quality 3D visualization using PyMOL.
    
    Args:
        smiles: SMILES string
        output_path: Path to save PNG file
        width: Image width in pixels
        height: Image height in pixels
        style: 'sticks', 'spheres', 'lines', 'surface', 'cartoon'
        ray_trace: Use photorealistic ray tracing
        
    Returns:
        True if successful, False otherwise
        
    Requirements:
        PyMOL (conda install -c conda-forge pymol-open-source)
        
    Example:
        success = visualize_molecule_3d_pymol(
            'CC(=O)O',
            'viz/molecules/acetic_acid.png',
            style='sticks',
            ray_trace=True
        )
    """

def visualize_multiple_graphs(
    cids: List[int],
    graph_data: dict,
    molecules_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[Union[str, Path]] = None,
    max_per_row: int = 3
) -> None:
    """Visualize multiple molecular graphs in a grid layout.
    
    Args:
        cids: List of compound IDs
        graph_data: Dictionary from load_graph_data()
        molecules_df: Optional DataFrame with molecule info
        output_dir: Directory to save visualizations (default: viz/molecules/)
        max_per_row: Maximum molecules per row
        
    Creates:
        - gallery.png: Grid view of all molecules
        - Individual CID_{cid}.png files for each molecule
        
    Example:
        visualize_multiple_graphs(
            cids=[180, 240, 7991],
            graph_data=graph_data,
            molecules_df=molecules,
            max_per_row=3
        )
    """

def compare_molecule_and_graph(
    cid: int,
    graph_data: dict,
    molecules_df: Optional[pd.DataFrame] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    figsize: tuple = (14, 6)
) -> None:
    """Create side-by-side comparison of structure and graph.
    
    Args:
        cid: Compound ID
        graph_data: Dictionary from load_graph_data()
        molecules_df: Optional DataFrame with molecule info
        save_path: Path to save comparison image
        show: Display interactive window
        figsize: Figure size
        
    Creates:
        Left panel: Molecular structure with atom indices
        Right panel: Graph connectivity matrix (adjacency matrix)
        
    Example:
        compare_molecule_and_graph(
            180, graph_data, molecules,
            save_path='viz/molecules/acetone_comparison.png'
        )
    """

def draw_molecule_from_smiles(
    smiles: str,
    title: Optional[str] = None,
    size: tuple = (400, 400),
    show_atom_indices: bool = False,
    projection: str = '2d'
) -> Optional[object]:
    """Draw molecule from SMILES string.
    
    Args:
        smiles: SMILES string
        title: Optional title
        size: Image size (width, height)
        show_atom_indices: Show atom numbering
        projection: '2d' or '3d'
        
    Returns:
        PIL Image object or None if invalid SMILES
        
    Example:
        img = draw_molecule_from_smiles('CC(=O)O', size=(500, 500))
        if img:
            img.save('molecule.png')
    """
```

**PyMOL Visualization Styles:**
- **sticks**: Ball-and-stick representation (default)
- **spheres**: Space-filling CPK model
- **lines**: Wireframe bonds only
- **surface**: Molecular surface
- **cartoon**: Ribbon diagram (for proteins)

**Visualization Output:**
- Default directory: `viz/molecules/`
- High-resolution PNG (2000x2000 for PyMOL, 300 DPI)
- CPK color scheme (C=gray, O=red, N=blue, etc.)
- Automatic directory creation

---

## ⚙️ **Pipeline Module**

### `src.neuro_foundation.pipeline.preprocess`

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
        from src.neuro_foundation.data.pyrfume_loader import load_molecules_npz
        from src.neuro_foundation.pipeline.preprocess import featurize_and_standardize
        
        molecules = load_molecules_npz()
        features = featurize_and_standardize(molecules)
        print(features.shape)  # (287, 1187)
    """
```

**Output Files:**
- `cleaned_data.csv`: Standardized features
- `scaler_stats.json`: Mean and std for each feature

---

### `src.neuro_foundation.pipeline.feature_select`

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

### `src.neuro_foundation.pipeline.train_linear`

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

### `src.neuro_foundation.pipeline.activity_maps`

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
from src.neuro_foundation.data.pyrfume_loader import (
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
from src.neuro_foundation.data.pyrfume_loader import load_molecules_npz
from src.neuro_foundation.pipeline.preprocess import featurize_and_standardize
from src.neuro_foundation.pipeline.feature_select import select_features

# Load data
molecules = load_molecules_npz()

# Extract features
features = featurize_and_standardize(molecules)

# Select high-variance features
selected = select_features(features, threshold=1.0)
```

### Pattern 3: Activity Map Analysis
```python
from src.neuro_foundation.pipeline.activity_maps import pipeline_load_and_mask

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
from src.neuro_foundation.data.pyrfume_loader import (
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

### Pattern 5: Graph-Based Analysis
```python
from src.neuro_foundation.data.molecular_graphs import load_graph_data, get_graph_by_cid
from src.neuro_foundation.data.graph_viz import visualize_molecular_graph
from src.neuro_foundation.data.pyrfume_loader import load_molecules_npz

# Load graph data
graph_data = load_graph_data()
molecules = load_molecules_npz()

# Get specific molecule's graph
acetone_graph = get_graph_by_cid(180, graph_data)
print(f"Acetone: {acetone_graph['num_nodes']} atoms")
print(f"Node features: {acetone_graph['node_features'].shape}")
print(f"Edge index: {acetone_graph['edge_index'].shape}")

# Visualize
visualize_molecular_graph(
    180, graph_data, molecules,
    mode='detailed',
    projection='3d',
    save_path='viz/molecules/acetone_3d.png'
)
```

### Pattern 6: Batch Graph Processing
```python
from src.neuro_foundation.data.molecular_graphs import load_graph_data
from src.neuro_foundation.data.graph_viz import visualize_multiple_graphs
from src.neuro_foundation.data.pyrfume_loader import load_molecules_npz

# Load data
graph_data = load_graph_data()
molecules = load_molecules_npz()

# Get subset of CIDs
cids = molecules['CID'].head(10).tolist()

# Create gallery
visualize_multiple_graphs(
    cids, graph_data, molecules,
    output_dir='viz/molecules',
    max_per_row=5
)

# Iterate over all graphs
for cid in cids:
    graph = get_graph_by_cid(cid, graph_data)
    if graph:
        print(f"CID {cid}: {graph['num_nodes']} nodes, {graph['num_edges']} edges")
```

---

## ⚠️ **Important Notes**

### Data Consistency
- CIDs may have 1-11 activity maps (repetitions)
- Use averaged maps for single representation
- Negative CIDs are natural mixtures (filtered out)
- 287 unique molecules, 405 total activity maps
- All 287 molecules have valid molecular graphs

### Performance
- NPZ loading is 1.3-1.6x faster than CSV
- Activity maps are ~10MB uncompressed, ~3MB compressed
- Use `load_activity_maps_by_cid` for efficient filtering
- Molecular graphs are compact (~60 KB for all 287 molecules)
- Graph generation is one-time (pre-compute and save)

### Molecular Graphs
- Node features: 137 dimensions per atom
- Edge features: 10 dimensions per bond
- Bidirectional edges (if A-B bond exists, both A→B and B→A are stored)
- All graphs successfully generated from SMILES
- Compatible with PyTorch Geometric and DGL

### Visualization
- Zero values displayed as NaN (transparent) in plots
- Original data keeps zeros for computational analysis
- Interactive plots block until user closes window
- Use `--save-images` flag to save visualizations
- PyMOL creates ultra-high resolution (2000x2000) images
- 3D visualizations require PyMOL installation
- 2D visualizations only require RDKit

### Common Pitfalls
1. Don't call `load_all_data.py` repeatedly (data already cached)
2. Use NPZ functions for repeated loading
3. Remember to close interactive matplotlib windows
4. Check that `activity_maps_csv/` folder exists before processing
5. PyMOL visualizations require conda installation (not pip)
6. Graph edge_index uses bidirectional edges (count accordingly)

---

**Last Updated**: December 13, 2025
**Version**: 2.0.0
