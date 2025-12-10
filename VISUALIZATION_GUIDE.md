# Molecular Graph Visualization Guide

This guide covers all the visualization options available for molecular graphs.

---

## 📊 STATIC VISUALIZATIONS (PNG Images)

### Function: `load_graph_by_cid()`

Creates static PNG images of molecules with graph information.

### Parameters:
- `cid`: PubChem Compound ID
- `mode`: `"simple"` or `"detailed"`
- `projection`: `"2d"` or `"3d"`
- `show_image`: Display the image (default: True)
- `save_image`: Save to file (default: False)

### Visualization Modes:

#### 1. Simple 2D
Clean RDKit molecular drawing with basic info.
```python
load_graph_by_cid(1183, mode='simple', projection='2d')
```

#### 2. Simple 3D (PyMOL)
High-quality ray-traced sphere rendering
- Resolution: 2000×2000 pixels
- Ray tracing with specular reflections
- Publication-quality output
- Settings: sphere_quality=4, stick_quality=30, spec_reflect=0.3
```python
load_graph_by_cid(1183, mode='simple', projection='3d')
```

#### 3. Detailed 2D
Molecule drawing + comprehensive graph information panel
- Molecular properties (name, MW, IUPAC)
- Graph structure (nodes, edges, degree)
- Feature dimensions and groups
```python
load_graph_by_cid(440917, mode='detailed', projection='2d')
```

#### 4. Detailed 3D
Matplotlib 3D ball-and-stick + graph information panel
```python
load_graph_by_cid(440917, mode='detailed', projection='3d')
```

---

## 🌐 INTERACTIVE VISUALIZATIONS (Browser)

### Function: `visualize_molecule_interactive()`

Creates interactive 3D visualizations using py3Dmol that open in your web browser.

### Features:
- **Rotate**: Click and drag to rotate the molecule
- **Zoom**: Scroll to zoom in/out
- **Pan**: Right-click and drag to pan
- **Web-based**: Opens in your default browser
- **Shareable**: Save and share HTML files

### Parameters:
- `cid`: PubChem Compound ID
- `style`: Rendering style (see below)
- `width`: Viewer width in pixels (default: 800)
- `height`: Viewer height in pixels (default: 600)
- `open_browser`: Auto-open in browser (default: True)

### Available Styles:

#### `stick` (default)
Ball-and-stick representation showing bonds as cylinders
```python
visualize_molecule_interactive(1183, style='stick')
```

#### `sphere`
Space-filling spheres (CPK representation)
```python
visualize_molecule_interactive(440917, style='sphere')
```

#### `line`
Wireframe showing only bonds
```python
visualize_molecule_interactive(180, style='line')
```

#### `cartoon`
Simplified cartoon representation
```python
visualize_molecule_interactive(5281515, style='cartoon')
```

#### `cross`
Atom positions marked with crosses
```python
visualize_molecule_interactive(439250, style='cross')
```

---

## 💡 Usage Examples

### Quick visualization
```python
from src.neuro_foundation.data.molecular_graphs import load_graph_by_cid

# Show vanillin in 3D PyMOL quality
load_graph_by_cid(1183, mode='simple', projection='3d')
```

### Save without displaying
```python
# Save limonene to file without showing
load_graph_by_cid(440917, show_image=False, save_image=True, 
                  mode='detailed', projection='2d')
```

### Interactive browser viewer
```python
from src.neuro_foundation.data.molecular_graphs import visualize_molecule_interactive

# Open interactive 3D viewer in browser
visualize_molecule_interactive(1183, style='sphere')

# Create HTML without auto-opening
visualize_molecule_interactive(440917, style='stick', open_browser=False)
```

---

## 📁 Output Locations

All visualizations are saved to `data/01_raw/`:

- **Static images**: `molecular_graph_CID_{cid}.png`
- **Interactive HTML**: `molecule_interactive_CID_{cid}.html`

---

## 🎨 Choosing the Right Visualization

| Use Case | Recommendation |
|----------|---------------|
| Publication figures | Simple 3D (PyMOL) |
| Quick structure check | Simple 2D |
| Graph analysis | Detailed 2D or 3D |
| Teaching/presentation | Interactive (browser) |
| Exploratory analysis | Interactive (browser) |
| Batch processing | Simple 2D (fastest) |

---

## 🔧 Dependencies

- **RDKit**: Required for all visualizations
- **PyMOL**: Required for high-quality 3D static images
- **py3Dmol**: Required for interactive browser visualizations
- **Matplotlib**: Required for detailed mode visualizations

Install with:
```bash
conda install -c conda-forge rdkit pymol-open-source
pip install py3Dmol
```
