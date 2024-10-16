for k in data:
    for j in k:
        data[j, k] == k + 1 - min(j)



def get_file(master_dir, band, task):
    temp = []
    for filename in os.listdir(master_dir):
        if '-epo' in filename:
            if task in filename:
                if band in filename:
                    data_as_epo = mne.read_epochs(os.path.join(master_dir, filename))
                    data = data_as_epo.get_data()
                    temp.append(data)
    return  temp





# Cal stats for google data
import bct
import os
import numpy as np

bands = ['Theta/',
        'Alpha/',
        'Beta/',
        'Gamma/']

def get_data(base_con, base_rep, bands):
    con_array_master = []
    for band in bands:
        con_array = []
        for filename in os.listdir(os.path.join(base_con, band)):
            temp_array = np.load(os.path.join(base_con, band, filename))
            temp_array = np.mean(temp_array, axis=0)
            con_array.append(temp_array)
        con_array_master.append(con_array)

    rep_array_master = []
    for band in bands:
        rep_array = []
        for filename in os.listdir(os.path.join(base_rep, band)):
            temp_array = np.load(os.path.join(base_rep, band, filename))
            temp_array = np.mean(temp_array, axis=0)
            rep_array.append(temp_array)
        rep_array_master.append(rep_array)

    return con_array_master, rep_array_master

# Interviewer task
base_con = '/home/students/Ddrive_2TB/Geet/Paul_project/Connectivity/adj_matrices/interviewer_conversation/'
base_rep = '/home/students/Ddrive_2TB/Geet/Paul_project/Connectivity/adj_matrices/interviewer_repetition/'

con_array_master, rep_array_master = get_data(base_con, base_rep, bands)

# Theta
saving_out = '/home/students/Ddrive_2TB/Geet/Paul_project/Connectivity/Stats/Interviewer_trigger/Theta/'

p, adj, _ = bct.nbs_bct(np.array(con_array_master[0]), np.array(rep_array_master[0]), 3.2, k=5000, seed=2022)
np.save(saving_out + 'adj', adj)
np.save(saving_out + 'P', p)

# Alpha
saving_out = '/home/students/Ddrive_2TB/Geet/Paul_project/Connectivity/Stats/Interviewer_trigger/Alpha/'

p, adj, _ = bct.nbs_bct(np.array(con_array_master[1]), np.array(rep_array_master[1]), 3.2, k=5000, seed=2022)
np.save(saving_out + 'adj', adj)
np.save(saving_out + 'P', p)

# Beta
saving_out = '/home/students/Ddrive_2TB/Geet/Paul_project/Connectivity/Stats/Interviewer_trigger/Beta'

p, adj, _ = bct.nbs_bct(np.array(con_array_master[2]), np.array(rep_array_master[2]), 3.2, k=5000, seed=2022)
np.save(saving_out + 'adj', adj)
np.save(saving_out + 'P', p)

# Gamma
saving_out = '/home/students/Ddrive_2TB/Geet/Paul_project/Connectivity/Stats/Interviewer_trigger/Gamma'

p, adj, _ = bct.nbs_bct(np.array(con_array_master[3]), np.array(rep_array_master[3]), 3.2, k=5000, seed=2022)
np.save(saving_out + 'adj', adj)
np.save(saving_out + 'P', p)
