import mne
import dyconnmap
import os
import time
import numpy as np

band = ['theta',
        'alpha',
        'beta',
        'gamma']

tasks = ['participant_conversation',
         'interviewer_repetition',
         'participant_repetition',
         'da',
         'ba',
         'interviewer_conversation']

master_dir = '/media/sahib/macos/root/LCMV_output_19092024'
save_dir = '/media/sahib/Mangor_2TB/Geet/Tasks/interviewer_repetition/Gamma/' # don't forget the / on the end!

def get_and_save_data(master_dir, band, task, save_dir):
    sub = 0
    for filename in os.listdir(master_dir):
        if '-epo' in filename:
            if task in filename:
                if band in filename:
                    data_as_epo = mne.read_epochs(os.path.join(master_dir, filename))
                    data = data_as_epo.get_data()
                    np.save(save_dir + task + '_' + band + '_sub_' + str(sub), data)
                    sub = sub + 1

get_and_save_data(master_dir=master_dir, band=band[3], task=tasks[1], save_dir=save_dir)
