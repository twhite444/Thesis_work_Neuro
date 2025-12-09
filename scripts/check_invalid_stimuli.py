"""Check which stimuli have invalid CIDs."""

import pandas as pd

behavior = pd.read_csv('data/00_raw/behavior_data.csv')
molecules = pd.read_csv('data/00_raw/molecules_raw.csv')

print('=== STIMULUS VALIDATION ===')
print(f'Total stimuli in behavior_data.csv: {len(behavior)}')
print(f'Total molecules in molecules_raw.csv: {len(molecules)}')
print()

# Extract CID from stimulus names
behavior['CID'] = behavior['Stimulus'].str.split('_').str[0]

# Check which are valid (CID is a COLUMN in molecules, not index)
molecules_cids = set(molecules['CID'].astype(str))
behavior['valid'] = behavior['CID'].isin(molecules_cids)

valid_stimuli = behavior[behavior['valid']]
invalid_stimuli = behavior[~behavior['valid']]

print(f'✅ Valid stimuli (CID in molecules): {len(valid_stimuli)}')
print(f'❌ Invalid stimuli (CID NOT in molecules): {len(invalid_stimuli)}')
print()

if len(invalid_stimuli) > 0:
    print(f'Invalid stimuli (likely controls/blanks):')
    invalid_cids = invalid_stimuli['CID'].value_counts()
    for cid, count in list(invalid_cids.items())[:30]:
        print(f'  CID {cid}: {count} stimulus/stimuli')
    if len(invalid_cids) > 30:
        print(f'  ... and {len(invalid_cids) - 30} more')
print()

print(f'✅ Expected result: {len(valid_stimuli)} valid stimulus presentations')
print(f'   Thesis expected: 405 presentations')
print()

# Check unique valid CIDs
valid_unique_cids = valid_stimuli['CID'].nunique()
print(f'✅ Valid unique CIDs: {valid_unique_cids} (should be 287)')
