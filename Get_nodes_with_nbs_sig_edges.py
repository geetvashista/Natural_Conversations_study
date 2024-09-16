import numpy as np
import mne
import pandas as pd


def find_sig_edged_nodes(array_2d):
    locations = []
    for row_index, row in enumerate(array_2d):
        for col_index, element in enumerate(row):
            if element > 0:
                locations.append((row_index, col_index))
    return locations


def remove_second_occurrence_mirrored_pairs(tuple_list):
    seen = set()
    mirrored_seen = set()
    result = []

    for tpl in tuple_list:
        mirror = (tpl[1], tpl[0])

        if mirror in seen:
            # If mirror is in seen, it's the second occurrence
            if mirror not in mirrored_seen:
                # Add the mirror to mirrored_seen to avoid future removals
                mirrored_seen.add(mirror)
        else:
            # Add the current tuple to the seen set
            seen.add(tpl)
            result.append(tpl)

    return result


# Load labels
labels = mne.read_labels_from_annot("fsaverage", "HCPMMP1", "both")
# DON'T FORGET TO GET RID OF THE FIRST TWO LABELS
del labels[0]
del labels[0]

# Get the label names in order
names_org = [labels[i].name for i in range(360)]
names = np.array(names_org)
del labels

# Load data array
alpha_participant = np.load(r'C:\Users\em17531\Desktop\Google_data\nbs_stats\nbs_participant_alpha.npy')

# Find sig edges
edges = find_sig_edged_nodes(alpha_participant)
edges = remove_second_occurrence_mirrored_pairs(edges)
edges = [list(i) for i in edges]

edges_with_names = []
for i in edges:
    temp = []
    for k in i:
        temp.append(names[k])
    edges_with_names.append(temp)

# Write to array/Dataframe and save if wanted
array = np.array(edges_with_names)
df = pd.DataFrame(array)
df.to_csv('Beta_sig_edge_names.csv')
