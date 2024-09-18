import mne
import os
import pandas as pd
import numpy as np


def get_sig_index(metric):
    sig_index = []
    vals = []
    for index, val in enumerate(metric):
        if val < 0.05:
            sig_index.append(index)
            vals.append(val)

    return sig_index, vals


def retun_sig_names(sig_index):
    sig_names = []
    for i in sig_index:
        sig_names.append(names[i])
    return sig_names

def build_dataframe_of_sig_areas(df):
    metrics = ['strength', 'betweenness', 'eigenvector', 'clustering']

    results = []
    for i in metrics:
        sig_index, vals = get_sig_index(df[i])
        sig_names = retun_sig_names(sig_index)
        df_temp = pd.DataFrame({'Name': sig_names, 'P val': vals})
        results.append(df_temp)
    return results


# __main__
# Load needed files
names = np.load(r'C:\Users\em17531\PycharmProjects\pythonProject2\HCP_names.npy')

    # Coded to rebuild names from scratch
# labels = mne.read_labels_from_annot("fsaverage", "HCPMMP1", "both")
# # DON'T FORGET TO GET RID OF THE FIRST TWO LABELS
# del labels[0]
# del labels[0]
# # Get the label names in order
# names_org = [labels[i].name for i in range(360)]
# names = np.array(names_org)
# del labels

# Build data df
base_dir = r'C:\Users\em17531\Desktop\Google_data\Graph_stats\Theta'
band_name = '\Theta'
strength = np.load(os.path.join(base_dir + band_name + '_strength_fdc.npy'))
betweenness = np.load(os.path.join(base_dir + band_name + '_betweenness_fdc.npy'))
eigenvector = np.load(os.path.join(base_dir + band_name + '_eigenvector_fdc.npy'))
clustering = np.load(os.path.join(base_dir + band_name + '_clustering_fdc.npy'))

df = pd.DataFrame({'strength': strength,
                   'betweenness': betweenness,
                   'eigenvector': eigenvector,
                   'clustering': clustering
                   })

# clean up
del base_dir
del band_name
del strength
del betweenness
del eigenvector
del clustering

# Get results
results_alpha = build_dataframe_of_sig_areas(df)

strength_df = results_alpha[0]
betweenness_df = results_alpha[1]
eigenvector_df = results_alpha[2]
clustering_df = results_alpha[3]
del results_alpha

# saving if desired
strength_df.to_csv('strength_df')
betweenness_df.to_csv('betweenness_df')
eigenvector_df.to_csv('eigenvector_df')
clustering_df.to_csv('clustering_df')
