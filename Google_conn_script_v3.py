import dyconnmap
import os
import time
import numpy as np
import bct
start = time.time()

def get_all_data(dir):
    ':returns: a list of np arrays'
    participant_arrays = []
    for file in os.listdir(dir):
        participant_arrays.append(np.load(os.path.join(dir + file)))
    return  participant_arrays

def calculate_adj_matrix_participant(dir_root, saving_root):
    for i in bands:
        print('\n' + 'Being band : ' + i + '\n')
        dir_target = dir_root + i
        saving_dir = saving_root + i
        master_data_array = get_all_data(dir_target)
        sub = 0
        for participant in master_data_array:
            temp = []
            for instance in participant:
                data = instance[:, 10:160]
                print('calculating connectivity')
                adj_matrix = dyconnmap.fc.wpli(data, fs=200, fb=[7, 12])
                print('calculating complete')
                temp.append(adj_matrix)
            temp_participant_adj = np.array(temp)
            np.save(saving_dir + 'sub_' + str(sub) + '_adj_matrix', temp_participant_adj)
            sub = sub + 1
        print('Band : ' + i + ' now complete' + '\n')

def calculate_adj_matrix_interviewer(dir_root, saving_root):
    for i in bands:
        print('\n' + 'Being band : ' + i + '\n')
        dir_target = dir_root + i
        saving_dir = saving_root + i
        master_data_array = get_all_data(dir_target)
        sub = 0
        for participant in master_data_array:
            temp = []
            for instance in participant:
                data = instance[:, 161:310]
                print('calculating connectivity')
                adj_matrix = dyconnmap.fc.wpli(data, fs=200, fb=[7, 12])
                print('calculating complete')
                temp.append(adj_matrix)
            temp_participant_adj = np.array(temp)
            np.save(saving_dir + 'sub_' + str(sub) + '_adj_matrix', temp_participant_adj)
            sub = sub + 1
        print('Band : ' + i + ' now complete' + '\n')


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

# __main__
bands = ['Theta/',
        'Alpha/',
        'Beta/',
        'Gamma/']

# Set the base file paths. This will determine the TASK.
dir_root = '/home/students/Ddrive_2TB/Geet/Paul_project/New_connectivity_raw_source_arrays/Tasks/participant_conversation/'   # Don't forget that last '/'!!
saving_root = '/home/students/Ddrive_2TB/Geet/Paul_project/New_connectivity_raw_source_arrays/adj_matrices/participant_conversation/'  # Don't forget that last '/'!!
calculate_adj_matrix_participant(dir_root=dir_root, saving_root=saving_root)

dir_root = '/home/students/Ddrive_2TB/Geet/Paul_project/New_connectivity_raw_source_arrays/Tasks/participant_repetition/'   # Don't forget that last '/'!!
saving_root = '/home/students/Ddrive_2TB/Geet/Paul_project/New_connectivity_raw_source_arrays/adj_matrices/participant_repetition/'  # Don't forget that last '/'!!
calculate_adj_matrix_participant(dir_root=dir_root, saving_root=saving_root)

    # Participant task
base_con = '/home/students/Ddrive_2TB/Geet/Paul_project/New_connectivity_raw_source_arrays/adj_matrices/participant_conversation/'
base_rep = '/home/students/Ddrive_2TB/Geet/Paul_project/New_connectivity_raw_source_arrays/adj_matrices/participant_repetition/'

con_array_master, rep_array_master = get_data(base_con, base_rep, bands)

# Theta
saving_out = '/home/students/Ddrive_2TB/Geet/Paul_project/New_connectivity_raw_source_arrays/Stats/Participant_trigger/Theta/'

p, adj, _ = bct.nbs_bct(np.array(con_array_master[0]).T, np.array(rep_array_master[0]).T, 3.2, k=5000, seed=2022)
np.save(saving_out + 'adj', adj)
np.save(saving_out + 'P', p)

# Alpha
saving_out = '/home/students/Ddrive_2TB/Geet/Paul_project/New_connectivity_raw_source_arrays/Stats/Participant_trigger/Alpha/'

p, adj, _ = bct.nbs_bct(np.array(con_array_master[1]).T, np.array(rep_array_master[1]).T, 3.2, k=5000, seed=2022)
np.save(saving_out + 'adj', adj)
np.save(saving_out + 'P', p)

# Beta
saving_out = '/home/students/Ddrive_2TB/Geet/Paul_project/New_connectivity_raw_source_arrays/Stats/Participant_trigger/Beta/'

p, adj, _ = bct.nbs_bct(np.array(con_array_master[2]).T, np.array(rep_array_master[2]).T, 3.2, k=5000, seed=2022)
np.save(saving_out + 'adj', adj)
np.save(saving_out + 'P', p)

# Gamma
saving_out = '/home/students/Ddrive_2TB/Geet/Paul_project/New_connectivity_raw_source_arrays/Stats/Participant_trigger/Gamma/'

p, adj, _ = bct.nbs_bct(np.array(con_array_master[3]).T, np.array(rep_array_master[3]).T, 3.2, k=5000, seed=2022)
np.save(saving_out + 'adj', adj)
np.save(saving_out + 'P', p)

# print('\n' + "EXECUTION TIME: " + str(time.time() - start) + " sec")

### ------------------------------ Interviewer version of the task ---------------------------------------- ###

# Set the base file paths. This will determine the TASK.
dir_root = '/home/students/Ddrive_2TB/Geet/Paul_project/New_connectivity_raw_source_arrays/Tasks/interviewer_conversation/'   # Don't forget that last '/'!!
saving_root = '/home/students/Ddrive_2TB/Geet/Paul_project/New_connectivity_raw_source_arrays/adj_matrices/interviewer_conversation/'  # Don't forget that last '/'!!
calculate_adj_matrix_participant(dir_root=dir_root, saving_root=saving_root)

dir_root = '/home/students/Ddrive_2TB/Geet/Paul_project/New_connectivity_raw_source_arrays/Tasks/interviewer_repetition/'   # Don't forget that last '/'!!
saving_root = '/home/students/Ddrive_2TB/Geet/Paul_project/New_connectivity_raw_source_arrays/adj_matrices/interviewer_repetition/'  # Don't forget that last '/'!!
calculate_adj_matrix_participant(dir_root=dir_root, saving_root=saving_root)

    # Interviewer task
base_con = '/home/students/Ddrive_2TB/Geet/Paul_project/New_connectivity_raw_source_arrays/adj_matrices/interviewer_conversation/'
base_rep = '/home/students/Ddrive_2TB/Geet/Paul_project/New_connectivity_raw_source_arrays/adj_matrices/interviewer_repetition/'

con_array_master, rep_array_master = get_data(base_con, base_rep, bands)

# Theta
saving_out = '/home/students/Ddrive_2TB/Geet/Paul_project/New_connectivity_raw_source_arrays/Stats/Interviewer_trigger/Theta/'

p, adj, _ = bct.nbs_bct(np.array(con_array_master[0]).T, np.array(rep_array_master[0]).T, 3.2, k=5000, seed=2022)
np.save(saving_out + 'adj', adj)
np.save(saving_out + 'P', p)

# Alpha
saving_out = '/home/students/Ddrive_2TB/Geet/Paul_project/New_connectivity_raw_source_arrays/Stats/Interviewer_trigger/Alpha/'

p, adj, _ = bct.nbs_bct(np.array(con_array_master[1]).T, np.array(rep_array_master[1]).T, 3.2, k=5000, seed=2022)
np.save(saving_out + 'adj', adj)
np.save(saving_out + 'P', p)

# Beta
saving_out = '/home/students/Ddrive_2TB/Geet/Paul_project/New_connectivity_raw_source_arrays/Stats/Interviewer_trigger/Beta/'

p, adj, _ = bct.nbs_bct(np.array(con_array_master[2]).T, np.array(rep_array_master[2]).T, 3.2, k=5000, seed=2022)
np.save(saving_out + 'adj', adj)
np.save(saving_out + 'P', p)

# Gamma
saving_out = '/home/students/Ddrive_2TB/Geet/Paul_project/New_connectivity_raw_source_arrays/Stats/Interviewer_trigger/Gamma/'

p, adj, _ = bct.nbs_bct(np.array(con_array_master[3]).T, np.array(rep_array_master[3]).T, 3.2, k=5000, seed=2022)
np.save(saving_out + 'adj', adj)
np.save(saving_out + 'P', p)

print('\n' + "EXECUTION TIME: " + str(time.time() - start) + " sec")
