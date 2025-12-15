"""Molecular graph visualization utilities.

This module provides functions to visualize molecular graphs as images,
similar to the activity map visualizations. Useful for inspecting graph
structures and understanding the featurization.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union, List
import os


try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Install with: conda install -c conda-forge rdkit")

try:
    import pymol
    from pymol import cmd
    PYMOL_AVAILABLE = True
except ImportError:
    PYMOL_AVAILABLE = False
    print("Warning: PyMOL not available. Install with: conda install -c conda-forge pymol-open-source")

try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False
    print("Warning: py3Dmol not available. Install with: pip install py3Dmol")

from .molecular_graphs import get_graph_by_cid


# Atomic radii (van der Waals) in Angstroms
ATOMIC_RADII = {
    1: 1.20,   # H
    6: 1.70,   # C
    7: 1.55,   # N
    8: 1.52,   # O
    9: 1.47,   # F
    15: 1.80,  # P
    16: 1.80,  # S
    17: 1.75,  # Cl
    35: 1.85,  # Br
    53: 1.98,  # I
}

# CPK colors for atoms
ATOMIC_COLORS = {
    1: '#FFFFFF',  # H - white
    6: '#909090',  # C - gray
    7: '#3050F8',  # N - blue
    8: '#FF0D0D',  # O - red
    9: '#90E050',  # F - green
    15: '#FF8000', # P - orange
    16: '#FFFF30', # S - yellow
    17: '#1FF01F', # Cl - green
    35: '#A62929', # Br - dark red
    53: '#940094', # I - purple
}


def draw_molecule_3d(
    smiles: str,
    ax: plt.Axes,
    title: Optional[str] = None,
    show_atom_indices: bool = False
) -> bool:
    """Draw a molecule in true 3D ball-and-stick style.
    
    Args:
        smiles: SMILES string
        ax: 3D matplotlib axes
        title: Optional title for the plot
        show_atom_indices: Whether to show atom indices
        
    Returns:
        True if successful, False otherwise
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for molecule visualization")
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    
    # Add hydrogens
    mol = Chem.AddHs(mol)
    
    # Generate 3D conformer
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception as e:
        print(f"3D embedding failed: {e}")
        return False
    
    # Get conformer
    conf = mol.GetConformer()
    
    # Get atom positions
    positions = []
    atom_numbers = []
    atom_colors = []
    atom_sizes = []
    
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        positions.append([pos.x, pos.y, pos.z])
        
        atomic_num = atom.GetAtomicNum()
        atom_numbers.append(atomic_num)
        atom_colors.append(ATOMIC_COLORS.get(atomic_num, '#FF1493'))  # Default: hot pink
        
        # Scale radii for visualization (multiply by 200 for good sphere size)
        radius = ATOMIC_RADII.get(atomic_num, 1.5)
        atom_sizes.append(radius * 200)
    
    positions = np.array(positions)
    
    # Draw bonds as cylinders (lines with thickness)
    for bond in mol.GetBonds():
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()
        
        pos1 = positions[idx1]
        pos2 = positions[idx2]
        
        # Draw bond as line
        ax.plot([pos1[0], pos2[0]], 
               [pos1[1], pos2[1]], 
               [pos1[2], pos2[2]], 
               color='gray', linewidth=2, alpha=0.6, zorder=1)
    
    # Draw atoms as spheres
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
              c=atom_colors, s=atom_sizes, alpha=0.95, edgecolors='black', 
              linewidths=0.5, zorder=2)
    
    # Add atom indices if requested
    if show_atom_indices:
        for idx, pos in enumerate(positions):
            ax.text(pos[0], pos[1], pos[2], str(idx), 
                   fontsize=8, color='black', weight='bold',
                   ha='center', va='center', zorder=3)
    
    # Set title
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    # Set labels
    ax.set_xlabel('X (Å)', fontsize=9)
    ax.set_ylabel('Y (Å)', fontsize=9)
    ax.set_zlabel('Z (Å)', fontsize=9)
    
    # Equal aspect ratio for all axes
    max_range = np.array([positions[:, 0].max()-positions[:, 0].min(),
                         positions[:, 1].max()-positions[:, 1].min(),
                         positions[:, 2].max()-positions[:, 2].min()]).max() / 2.0
    
    mid_x = (positions[:, 0].max()+positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max()+positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max()+positions[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Grid and background
    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    return True


def visualize_molecule_3d_pymol(
    smiles: str,
    output_path: str,
    width: int = 1200,
    height: int = 1200,
    style: str = 'spheres',
    ray_trace: bool = True
) -> bool:
    """Create high-quality 3D molecular visualization using PyMOL and save as PNG.
    
    Args:
        smiles: SMILES string
        output_path: Path to save PNG file
        width: Image width in pixels
        height: Image height in pixels
        style: Visualization style - 'sticks', 'spheres', 'cartoon', 'lines', 'surface'
        ray_trace: If True, use ray tracing for photorealistic rendering
        
    Returns:
        True if successful, False otherwise
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for molecule visualization")
    if not PYMOL_AVAILABLE:
        raise ImportError("PyMOL is required for 3D visualization. Install with: conda install -c conda-forge pymol-open-source")
    
    # Convert SMILES to 3D structure with RDKit
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception as e:
        print(f"3D embedding failed: {e}")
        return False
    
    # Save to temporary SDF file for PyMOL
    import tempfile
    fd, sdf_path = tempfile.mkstemp(suffix='.sdf')
    os.close(fd)
    
    try:
        # Write molecule to SDF
        writer = Chem.SDWriter(sdf_path)
        writer.write(mol)
        writer.close()
        
        # Initialize PyMOL in quiet/headless mode
        import __main__
        __main__.pymol_argv = ['pymol', '-c']  # -c for no GUI
        pymol.finish_launching()
        
        # Load molecule
        cmd.load(sdf_path, 'molecule')
        
        # Set visualization style
        cmd.hide('everything', 'molecule')
        if style == 'sticks':
            cmd.show('sticks', 'molecule')
            cmd.set('stick_radius', 0.15)
            cmd.set('stick_ball', 1)
            cmd.set('stick_ball_ratio', 0.4)
        elif style == 'spheres':
            cmd.show('spheres', 'molecule')
            cmd.set('sphere_scale', 0.3)
        elif style == 'lines':
            cmd.show('lines', 'molecule')
        elif style == 'surface':
            cmd.show('surface', 'molecule')
            cmd.set('surface_quality', 2)
        elif style == 'cartoon':
            cmd.show('cartoon', 'molecule')
        else:
            # Default: ball and stick
            cmd.show('sticks', 'molecule')
            cmd.show('spheres', 'molecule')
            cmd.set('stick_radius', 0.15)
            cmd.set('sphere_scale', 0.25)
        
        # Color by element (CPK coloring)
        cmd.util.cbag('molecule')  # Carbon = gray, others = standard colors
        
        # Set background color
        cmd.bg_color('white')
        
        # Center and zoom
        cmd.center('molecule')
        cmd.zoom('molecule', buffer=2.0)
        
        # Set view angle for nice 3D perspective
        cmd.turn('x', 20)
        cmd.turn('y', 30)
        
        # High-quality ray tracing settings
        cmd.set('sphere_quality', 4)       # High quality spheres
        cmd.set('stick_quality', 30)       # High quality sticks
        cmd.set('ray_texture', 1)          # Matte texture
        cmd.set('spec_reflect', 0.3)       # Specular reflection
        cmd.set('spec_power', 100)         # Specular power (shininess)
        cmd.set('antialias', 2)            # Antialiasing
        cmd.set('hash_max', 300)           # Ray tracing quality
        
        # Render at ultra-high resolution (2000x2000) with ray tracing
        cmd.ray(2000, 2000)
        cmd.png(output_path, width=2000, height=2000, dpi=300, ray=0)  # ray=0 since we already ray traced
        
        # Clean up PyMOL
        cmd.delete('all')
        cmd.reinitialize()
        
        return True
        
    except Exception as e:
        print(f"PyMOL rendering failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up temp file
        if os.path.exists(sdf_path):
            os.remove(sdf_path)


def visualize_molecule_3d_interactive(
    smiles: str,
    output_path: str,
    width: int = 800,
    height: int = 600,
    style: str = 'stick'
) -> bool:
    """Create interactive 3D molecular visualization using py3Dmol and save as HTML.
    
    This creates an interactive viewer that opens in your web browser where you can
    rotate, zoom, and explore the molecule in 3D.
    
    Args:
        smiles: SMILES string
        output_path: Path to save HTML file
        width: Viewer width in pixels
        height: Viewer height in pixels
        style: Visualization style - 'stick', 'sphere', 'cartoon', 'line', 'cross'
        
    Returns:
        True if successful, False otherwise
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for molecule visualization")
    if not PY3DMOL_AVAILABLE:
        raise ImportError("py3Dmol is required for interactive visualization. Install with: pip install py3Dmol")
    
    # Convert SMILES to 3D structure with RDKit
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception as e:
        print(f"Failed to generate 3D coordinates: {e}")
        return False
    
    # Convert to MOL block for py3Dmol
    mol_block = Chem.MolToMolBlock(mol)
    
    # Create py3Dmol viewer
    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(mol_block, 'mol')
    viewer.setStyle({style: {}})
    viewer.setBackgroundColor('white')
    viewer.zoomTo()
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>3D Molecular Viewer</title>
    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
</head>
<body>
    <h2 style="text-align: center;">Interactive 3D Molecular Viewer</h2>
    <p style="text-align: center;">Drag to rotate • Scroll to zoom • Right-click to pan</p>
    <div style="text-align: center;">
        {viewer._make_html()}
    </div>
    <div style="text-align: center; margin-top: 20px;">
        <p><strong>SMILES:</strong> {smiles}</p>
    </div>
</body>
</html>"""
    
    # Save HTML file
    with open(output_path, 'w') as f:
        f.write(html)
    
    return True


def draw_molecule_from_smiles(
    smiles: str,
    title: Optional[str] = None,
    size: tuple = (400, 400),
    show_atom_indices: bool = False,
    projection: str = '2d'
) -> Optional[object]:
    """Draw a molecule from SMILES string.
    
    Args:
        smiles: SMILES string
        title: Optional title for the image
        size: Image size (width, height)
        show_atom_indices: Whether to show atom indices
        projection: '2d' for 2D coordinates or '3d' for 3D conformer
        
    Returns:
        PIL Image or None if invalid SMILES
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for molecule visualization")
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Add explicit hydrogens for complete structure
    mol = Chem.AddHs(mol)
    
    if projection == '3d':
        # Generate 3D conformer
        try:
            # Embed 3D structure
            AllChem.EmbedMolecule(mol, randomSeed=42)
            # Optimize geometry with MMFF force field
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception as e:
            # If 3D embedding fails, fall back to 2D
            print(f"3D embedding failed, falling back to 2D: {e}")
            AllChem.Compute2DCoords(mol)
    else:
        # Generate 2D coordinates
        AllChem.Compute2DCoords(mol)
    
    # Draw molecule
    if show_atom_indices:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
    
    img = Draw.MolToImage(mol, size=size)
    
    return img


def visualize_molecular_graph(
    cid: int,
    graph_data: dict,
    molecules_df=None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    show_atom_indices: bool = False,
    mode: str = 'detailed',
    projection: str = '2d',
    figsize: Optional[tuple] = None
) -> None:
    """Visualize a molecular graph with structure and optional statistics.
    
    Creates a visualization showing molecular structure with optional detailed information
    including graph statistics, molecular properties, and feature summaries.
    
    Args:
        cid: Compound ID
        graph_data: Dictionary containing graph data from load_graph_data
        molecules_df: Optional DataFrame with molecule info (SMILES, name, MW, etc.)
        save_path: Path to save the visualization (PNG). If None and show=False, displays interactively.
        show: If True, display interactive visualization window. If save_path is also provided, saves the image.
        show_atom_indices: Whether to show atom indices on structure
        mode: 'simple' (just molecule) or 'detailed' (molecule + comprehensive info)
        projection: '2d' (2D layout) or '3d' (3D conformer with optimized geometry)
        figsize: Figure size (width, height). Auto-determined if None based on mode.
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for molecule visualization")
    
    # Get graph for this CID
    graph = get_graph_by_cid(cid, graph_data)
    if graph is None:
        print(f"No graph found for CID {cid}")
        return
    
    # Get molecule info if available
    smiles = None
    mol_name = None
    mol_weight = None
    iupac_name = None
    
    if molecules_df is not None:
        mol_row = molecules_df[molecules_df['CID'] == cid]
        if len(mol_row) > 0:
            row = mol_row.iloc[0]
            smiles = row.get('IsomericSMILES', row.get('SMILES', None))
            mol_name = row.get('name', row.get('Name', None))
            mol_weight = row.get('MolecularWeight', None)
            iupac_name = row.get('IUPACName', None)
    
    # Auto-determine figure size based on mode
    if figsize is None:
        if mode == 'simple':
            figsize = (8, 8)
        else:
            figsize = (14, 8)
    
    # ========================================================================
    # SIMPLE MODE: Just the molecule
    # ========================================================================
    if mode == 'simple':
        if projection == '3d':
            # PyMOL 3D visualization to PNG
            if smiles:
                # Create output path - use viz/molecules/ directory
                if not save_path:
                    viz_dir = os.path.join('viz', 'molecules')
                    os.makedirs(viz_dir, exist_ok=True)
                    save_path = os.path.join(viz_dir, f'CID_{cid}_3d.png')
                
                # Title for display
                title = f'CID {cid}'
                if mol_name:
                    title += f' - {mol_name}'
                
                try:
                    success = visualize_molecule_3d_pymol(
                        smiles,
                        output_path=str(save_path),
                        width=1200,
                        height=1200,
                        style='sticks',
                        ray_trace=True
                    )
                    
                    if success:
                        print("✨ 3D visualization created with PyMOL!")
                        print(f"   Saved to: {save_path}")
                        if show:
                            # Open image viewer
                            import subprocess
                            subprocess.run(['open', save_path])
                        return
                    else:
                        print("PyMOL rendering failed, falling back to matplotlib 3D")
                        
                except Exception as e:
                    print(f"PyMOL failed ({e}), falling back to matplotlib 3D")
                
                # Fallback to matplotlib 3D
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111, projection='3d')
                
                title_text = f'CID {cid}'
                if mol_name:
                    title_text += f' - {mol_name}'
                title_text += ' (3D Ball-and-Stick)'
                
                success = draw_molecule_3d(smiles, ax, title=title_text, 
                                          show_atom_indices=show_atom_indices)
                if not success:
                    ax.text(0.5, 0.5, 0.5, 'Failed to generate 3D structure', 
                           ha='center', va='center', fontsize=14)
            else:
                print(f"No SMILES available for CID {cid}")
                return
        else:
            # 2D flat image visualization
            fig, ax = plt.subplots(figsize=figsize)
            
            if smiles:
                img = draw_molecule_from_smiles(smiles, size=(800, 800), 
                                               show_atom_indices=show_atom_indices,
                                               projection=projection)
                if img:
                    ax.imshow(img)
                    ax.axis('off')
                    
                    # Title with CID and name
                    title = f'CID {cid}'
                    if mol_name:
                        title += f' - {mol_name}'
                    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
                else:
                    ax.text(0.5, 0.5, 'Invalid SMILES', ha='center', va='center', fontsize=14)
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, f'CID {cid}\nNo SMILES available', 
                       ha='center', va='center', fontsize=14)
                ax.axis('off')
    
    # ========================================================================
    # DETAILED MODE: Molecule + comprehensive information
    # ========================================================================
    else:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.25)
        
        # --------------------------------------------------------------------
        # Left: Molecule structure
        # --------------------------------------------------------------------
        if projection == '3d':
            # Real 3D ball-and-stick visualization
            ax_mol = fig.add_subplot(gs[0, 0], projection='3d')
            
            if smiles:
                # Title with CID and name
                title = f'CID {cid}'
                if mol_name:
                    title += f' - {mol_name}'
                title += ' (3D Ball-and-Stick)'
                
                success = draw_molecule_3d(smiles, ax_mol, title=title, 
                                          show_atom_indices=show_atom_indices)
                if not success:
                    ax_mol.text(0.5, 0.5, 0.5, 'Failed to generate 3D structure', 
                               ha='center', va='center')
            else:
                ax_mol.text(0.5, 0.5, 0.5, 'No SMILES available', 
                           ha='center', va='center')
        else:
            # 2D flat image visualization
            ax_mol = fig.add_subplot(gs[0, 0])
            
            if smiles:
                img = draw_molecule_from_smiles(smiles, size=(700, 700), 
                                               show_atom_indices=show_atom_indices,
                                               projection=projection)
                if img:
                    ax_mol.imshow(img)
                    ax_mol.axis('off')
                    
                    # Title with CID and name
                    title = f'CID {cid}'
                    if mol_name:
                        title += f' - {mol_name}'
                    ax_mol.set_title(title, fontsize=14, fontweight='bold', pad=10)
                else:
                    ax_mol.text(0.5, 0.5, 'Invalid SMILES', ha='center', va='center')
                    ax_mol.axis('off')
            else:
                ax_mol.text(0.5, 0.5, 'No SMILES available', ha='center', va='center')
                ax_mol.axis('off')
        
        # --------------------------------------------------------------------
        # Right: Comprehensive information panel
        # --------------------------------------------------------------------
        ax_info = fig.add_subplot(gs[0, 1])
        ax_info.axis('off')
        
        # Build information text
        info_lines = []
        
        # Molecular Properties Section
        info_lines.append("╔═══ MOLECULAR PROPERTIES ═══╗")
        if mol_name:
            # Wrap long names
            name_display = mol_name if len(mol_name) < 30 else mol_name[:27] + '...'
            info_lines.append(f"Name: {name_display}")
        if mol_weight:
            info_lines.append(f"MW: {mol_weight:.2f} g/mol")
        if iupac_name and iupac_name != mol_name:
            # Show first part of IUPAC name if different from common name
            iupac_display = iupac_name if len(iupac_name) < 35 else iupac_name[:32] + '...'
            info_lines.append(f"IUPAC: {iupac_display}")
        
        info_lines.append("")
        
        # Graph Structure Section
        info_lines.append("╔═══ GRAPH STRUCTURE ═══╗")
        info_lines.append(f"Nodes (atoms): {graph['num_nodes']}")
        info_lines.append(f"Edges (bonds): {graph['num_edges']}")
        info_lines.append(f"Avg degree: {2 * graph['num_edges'] / graph['num_nodes']:.2f}")
        info_lines.append("")
        
        # Node Features Section
        info_lines.append("╔═══ NODE FEATURES ═══╗")
        info_lines.append("Dimensions: 137")
        info_lines.append(f"Shape: {graph['node_features'].shape}")
        info_lines.append("")
        info_lines.append("Feature Groups:")
        info_lines.append("  • Atomic # (100 dims)")
        info_lines.append("  • Degree (11 dims)")
        info_lines.append("  • Charge (5 dims)")
        info_lines.append("  • Hybridization (5 dims)")
        info_lines.append("  • Aromaticity (1 dim)")
        info_lines.append("  • # Hydrogens (5 dims)")
        info_lines.append("  • Radicals (5 dims)")
        info_lines.append("  • In Ring (1 dim)")
        info_lines.append("  • Chirality (4 dims)")
        info_lines.append("")
        
        # Edge Information Section
        info_lines.append("╔═══ EDGE INFORMATION ═══╗")
        info_lines.append(f"Edge Index: {graph['edge_index'].shape}")
        info_lines.append("  (bidirectional)")
        
        if 'edge_attr' in graph:
            info_lines.append(f"Edge Features: {graph['edge_attr'].shape[1]} dims")
            info_lines.append("")
            info_lines.append("Feature Groups:")
            info_lines.append("  • Bond type (4 dims)")
            info_lines.append("  • Conjugation (1 dim)")
            info_lines.append("  • In Ring (1 dim)")
            info_lines.append("  • Stereo (4 dims)")
        else:
            info_lines.append("Edge Features: None")
        
        # Combine all info
        info_text = '\n'.join(info_lines)
        
        # Display info with nice formatting
        ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, pad=1))
    
    plt.tight_layout()
    
    # Save and/or show
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    if show:
        plt.show()
    elif not save_path:
        # If neither save nor show specified, default to showing
        plt.show()
    
    plt.close()


def visualize_multiple_graphs(
    cids: List[int],
    graph_data: dict,
    molecules_df=None,
    output_dir: Optional[Union[str, Path]] = None,
    max_per_row: int = 3
) -> None:
    """Visualize multiple molecular graphs in a grid layout.
    
    Args:
        cids: List of compound IDs to visualize
        graph_data: Dictionary containing graph data from load_graph_data
        molecules_df: Optional DataFrame with molecule info
        output_dir: Directory to save individual visualizations
        max_per_row: Maximum molecules per row in grid
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for molecule visualization")
    
    # Calculate grid dimensions
    num_mols = len(cids)
    n_cols = min(max_per_row, num_mols)
    n_rows = (num_mols + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if num_mols == 1:
        axes = np.array([axes])
    axes = axes.flatten() if num_mols > 1 else axes
    
    for idx, cid in enumerate(cids):
        ax = axes[idx] if num_mols > 1 else axes
        
        # Get SMILES
        smiles = None
        if molecules_df is not None:
            mol_row = molecules_df[molecules_df['CID'] == cid]
            if len(mol_row) > 0:
                smiles = mol_row.iloc[0].get('IsomericSMILES', mol_row.iloc[0].get('SMILES', None))
        
        # Get graph stats
        graph = get_graph_by_cid(cid, graph_data)
        
        if smiles and graph:
            img = draw_molecule_from_smiles(smiles, size=(300, 300))
            if img:
                ax.imshow(img)
                ax.set_title(f'CID {cid}\n{graph["num_nodes"]} atoms, {graph["num_edges"]} bonds',
                           fontsize=10)
            else:
                ax.text(0.5, 0.5, f'CID {cid}\nInvalid SMILES', ha='center', va='center')
        else:
            ax.text(0.5, 0.5, f'CID {cid}\nNo data', ha='center', va='center')
        
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_mols, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save or show
    if output_dir:
        # Use viz/molecules/ directory for galleries
        viz_dir = os.path.join('viz', 'molecules')
        os.makedirs(viz_dir, exist_ok=True)
        output_path = Path(viz_dir) / 'gallery.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved gallery to {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Also save individual visualizations if output_dir provided
    if output_dir:
        viz_dir = os.path.join('viz', 'molecules')
        os.makedirs(viz_dir, exist_ok=True)
        for cid in cids:
            output_path = Path(viz_dir) / f'CID_{cid}.png'
            visualize_molecular_graph(cid, graph_data, molecules_df, output_path)


def compare_molecule_and_graph(
    cid: int,
    graph_data: dict,
    molecules_df=None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    figsize: tuple = (14, 6)
) -> None:
    """Create side-by-side comparison of molecule structure and graph representation.
    
    Args:
        cid: Compound ID
        graph_data: Dictionary containing graph data
        molecules_df: Optional DataFrame with molecule info
        save_path: Path to save the visualization. If None and show=False, displays interactively.
        show: If True, display interactive visualization window. If save_path is also provided, saves the image.
        figsize: Figure size
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for molecule visualization")
    
    # Get data
    graph = get_graph_by_cid(cid, graph_data)
    if graph is None:
        print(f"No graph found for CID {cid}")
        return
    
    smiles = None
    if molecules_df is not None:
        mol_row = molecules_df[molecules_df['CID'] == cid]
        if len(mol_row) > 0:
            smiles = mol_row.iloc[0].get('IsomericSMILES', mol_row.iloc[0].get('SMILES', None))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left: molecule with atom indices
    if smiles:
        img = draw_molecule_from_smiles(smiles, size=(500, 500), show_atom_indices=True)
        if img:
            ax1.imshow(img)
            ax1.set_title(f'Molecular Structure with Atom Indices\nCID {cid}', fontsize=12, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'Invalid SMILES', ha='center', va='center')
    else:
        ax1.text(0.5, 0.5, 'No SMILES available', ha='center', va='center')
    ax1.axis('off')
    
    # Right: graph connectivity visualization
    # Create adjacency matrix from edge_index
    edge_index = graph['edge_index']
    num_nodes = graph['num_nodes']
    
    # Build adjacency matrix
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        adj_matrix[src, dst] = 1
    
    # Plot adjacency matrix
    im = ax2.imshow(adj_matrix, cmap='Blues', interpolation='nearest')
    ax2.set_title(f'Graph Connectivity Matrix\n{num_nodes} nodes, {graph["num_edges"]} edges', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Atom Index', fontsize=10)
    ax2.set_ylabel('Atom Index', fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Connected', fontsize=9)
    
    plt.tight_layout()
    
    # Save and/or show
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    
    if show:
        plt.show()
    elif not save_path:
        # If neither save nor show specified, default to showing
        plt.show()
    
    plt.close()


def print_graph_summary(graph_data: dict) -> None:
    """Print summary statistics for all graphs in dataset.
    
    Args:
        graph_data: Dictionary containing graph data from load_graph_data
    """
    print("=" * 70)
    print("Molecular Graph Dataset Summary")
    print("=" * 70)
    print()
    
    num_molecules = len(graph_data['cids'])
    print(f"Total molecules: {num_molecules}")
    print(f"Valid molecules: {graph_data['valid_mask'].sum()}")
    print()
    
    # Node statistics
    num_nodes = graph_data['num_nodes']
    print("Nodes (atoms) per molecule:")
    print(f"  Mean: {num_nodes.mean():.1f} ± {num_nodes.std():.1f}")
    print(f"  Range: [{num_nodes.min()}, {num_nodes.max()}]")
    print(f"  Total: {num_nodes.sum():,}")
    print()
    
    # Edge statistics
    num_edges = graph_data['num_edges']
    print("Edges (bonds) per molecule:")
    print(f"  Mean: {num_edges.mean():.1f} ± {num_edges.std():.1f}")
    print(f"  Range: [{num_edges.min()}, {num_edges.max()}]")
    print(f"  Total: {num_edges.sum():,}")
    print()
    
    # Feature dimensions
    sample_idx = 0
    sample_features = graph_data['node_features_list'][sample_idx]
    print(f"Node feature dimensions: {sample_features.shape[1]}")
    
    if 'edge_attr_list' in graph_data:
        sample_edge_features = graph_data['edge_attr_list'][sample_idx]
        print(f"Edge feature dimensions: {sample_edge_features.shape[1]}")
    else:
        print("Edge features: Not included")
    
    print()
    print("=" * 70)
