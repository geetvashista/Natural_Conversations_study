import dyconnmap
import os
import time
import numpy as np
start = time.time()

def get_all_data(dir):
    ':returns: a list of np arrays'
    participant_arrays = []
    for file in os.listdir(dir):
        participant_arrays.append(np.load(os.path.join(dir + file)))
    return  participant_arrays

def calculate_adj_matrix(dir_root, saving_root):
    for i in band:
        dir_target = dir_root + i
        saving_dir = saving_root + i
        master_data_array = get_all_data(dir_target)
        sub = 0
        for participant in master_data_array:
            temp = []
            for instance in participant:
                data = instance[:, 160:310]
                print('calculating connectivity')
                adj_matrix = dyconnmap.fc.wpli(data, fs=200, fb=[7, 12])
                print('calculating complete')
                temp.append(adj_matrix)
            temp_participant_adj = np.array(temp)
            np.save(saving_dir + 'sub_' + str(sub) + '_adj_matrix', temp_participant_adj)
            sub = sub + 1

# __main__
band = ['Theta/',
        'Alpha/',
        'Beta/',
        'Gamma/']

# tasks = ['participant_conversation/',
#          'interviewer_repetition/',
#          'participant_repetition/',
#          'interviewer_conversation/']

# Set the base file paths. This will determine the TASK.
dir_root = '/home/students/Ddrive_2TB/Geet/Paul_project/Connectivity/Tasks/interviewer_conversation/'   # Don't forget that last '/'!!
saving_root = '/home/students/Ddrive_2TB/Geet/Paul_project/Connectivity/adj_matrices/interviewer_conversation'  # Don't forget that last '/'!!
calculate_adj_matrix(dir_root=dir_root, saving_root=saving_root)

print('\n' + "EXECUTION TIME: " + str(time.time() - start) + " sec")


    # This version of the function does all tasks as well:

# def calculate_adj_matrix(dir_root, saving_root):
#     for k in tasks:
#         dir_with_task = dir_root + k
#         saving_with_task = saving_root + k
#         for i in band:
#             dir_target = dir_with_task + i
#             saving_dir = saving_with_task + i
#             master_data_array = get_all_data(dir_target)
#             sub = 0
#             for participant in master_data_array:
#                 temp = []
#                 for instance in participant:
#                     data = instance[:, 160:310]
#                     print('calculating connectivity')
#                     adj_matrix = dyconnmap.fc.wpli(data, fs=200, fb=[7, 12])
#                     print('calculating complete')
#                     temp.append(adj_matrix)
#                 temp_participant_adj = np.array(temp)
#                 np.save(saving_dir + 'sub_' + str(sub) + '_adj_matrix', temp_participant_adj)
#                 sub = sub + 1


# CAN'T WORK!!!! --> each task has to be cropped differently

