#!/usr/bin/env python3
"""Example: Using stimuli metadata to get detailed information about each stimulus."""
import sys
from pathlib import Path

# Add project root to path (go up 2 levels from scripts/examples/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.neuro_foundation.data.pyrfume_loader import (
    load_stimuli_npz,
    load_activity_maps_by_cid,
)


def main():
    print("=" * 70)
    print("Stimuli Metadata Examples")
    print("=" * 70)
    
    # Load stimuli metadata
    stimuli = load_stimuli_npz()
    
    print("\n1. Overview:")
    print(f"   Total stimuli: {len(stimuli)}")
    print(f"   Unique CIDs: {stimuli['CID'].nunique()}")
    print(f"   With experimental conditions: {(stimuli['Conditions'] != '').sum()}")
    
    # Show some examples
    print("\n2. Sample stimuli (showing molecule-based, not mixtures):")
    molecule_stimuli = stimuli[stimuli['CID'] > 0].head(10)
    print(molecule_stimuli[['Stimulus', 'CID', 'Rep', 'Name']].to_string())
    
    # Show natural mixtures (negative CIDs)
    print("\n3. Natural mixtures (negative CIDs):")
    mixtures = stimuli[stimuli['CID'] < 0].head(10)
    print(mixtures[['Stimulus', 'CID', 'Name']].to_string())
    
    # Find stimuli with multiple repetitions
    print("\n4. Stimuli with multiple repetitions:")
    rep_counts = stimuli.groupby('CID')['Rep'].max()
    multi_rep = rep_counts[rep_counts > 0].sort_values(ascending=False).head(10)
    print("\n   CIDs with most repetitions:")
    for cid, max_rep in multi_rep.items():
        stim_info = stimuli[stimuli['CID'] == cid].iloc[0]
        name = stim_info['Name']
        print(f"   - CID {cid} ({name}): {int(max_rep + 1)} repetitions")
    
    # Link stimuli to activity maps
    print("\n5. Linking stimuli to activity maps:")
    example_cid = 180
    stim_for_cid = stimuli[stimuli['CID'] == example_cid]
    maps = load_activity_maps_by_cid(example_cid)
    
    if len(stim_for_cid) > 0:
        print(f"\n   CID {example_cid}:")
        print(f"   Name: {stim_for_cid.iloc[0]['Name']}")
        print(f"   Stimuli entries: {len(stim_for_cid)}")
        print(f"   Activity maps: {len(maps)}")
        print("\n   Stimulus IDs for this CID:")
        for _, row in stim_for_cid.iterrows():
            print(f"   - {row['Stimulus']}: Rep {row['Rep']}, Source: {row['SourceFile']}")
    
    # Show how to filter by name
    print("\n6. Search stimuli by name (case-insensitive):")
    search_term = "apple"
    matching = stimuli[stimuli['Name'].str.contains(search_term, case=False, na=False)]
    print(f"\n   Found {len(matching)} stimuli containing '{search_term}':")
    print(matching[['Stimulus', 'CID', 'Name']].head(10).to_string())
    
    # Show unique odorant names
    print("\n7. Statistics:")
    print(f"   Total stimuli entries: {len(stimuli)}")
    print(f"   Unique odorant names: {stimuli['Name'].nunique()}")
    print(f"   Pure molecules (CID > 0): {(stimuli['CID'] > 0).sum()}")
    print(f"   Natural mixtures (CID < 0): {(stimuli['CID'] < 0).sum()}")
    
    print("\n" + "=" * 70)
    print("Usage Tips:")
    print("=" * 70)
    print("""
    # Load stimuli metadata
    stimuli = load_stimuli_npz()
    
    # Find all stimuli for a specific CID
    acetone_stimuli = stimuli[stimuli['CID'] == 180]
    
    # Search by name
    apple_stimuli = stimuli[stimuli['Name'].str.contains('apple', case=False)]
    
    # Get stimulus details for a specific stimulus ID
    stim_details = stimuli[stimuli['Stimulus'] == '1031_0']
    """)


if __name__ == "__main__":
    main()
