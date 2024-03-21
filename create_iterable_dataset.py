import os
import numpy as np
import torch
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pickle

def target_coords_to_onehot_smoothed_lowres(target_coords_by_cam, img_height=260, img_width=344, sigma=5, scale=True, downsample_factor=2):
    # target_coords_by_cam.shape
    num_joints, num_cords = target_coords_by_cam.shape
    
    if downsample_factor is not None:
        target_coords_by_cam_onehot = torch.zeros(num_joints, int((img_height + img_width)/downsample_factor))
    else:
        target_coords_by_cam_onehot = torch.zeros(num_joints, img_height + img_width)
    # target_coords_by_cam_onehot.shape # joints x vec_length
    
    # joint_idx = 0
    # set the target coordinates to 1 from the target coordinates
    for joint_idx in range(num_joints):
        y_coord = int(target_coords_by_cam[joint_idx, 0])
        x_coord = int(target_coords_by_cam[joint_idx, 1])
    
        y_coord_one_hot = torch.zeros(img_height)
        y_coord_one_hot[y_coord] = 1

        x_coord_one_hot = torch.zeros(img_width)
        x_coord_one_hot[x_coord] = 1

        # smooth the one hot encoding
        y_coord_one_hot_smoothed = gaussian_filter1d(y_coord_one_hot, sigma=sigma)
        x_coord_one_hot_smoothed = gaussian_filter1d(x_coord_one_hot, sigma=sigma)
        if scale:
            y_coord_one_hot_smoothed = y_coord_one_hot_smoothed / np.max(y_coord_one_hot_smoothed)
            x_coord_one_hot_smoothed = x_coord_one_hot_smoothed / np.max(x_coord_one_hot_smoothed)

        # plt.plot(y_coord_one_hot, label='original', color='b', linestyle='', marker='.')
        # plt.plot(y_coord_one_hot_smoothed, label='smoothed', color='r')
        # plt.show()
        if downsample_factor is not None:
            y_coord_one_hot_smoothed = y_coord_one_hot_smoothed[::downsample_factor]
            x_coord_one_hot_smoothed = x_coord_one_hot_smoothed[::downsample_factor]
                
            # plt.plot(y_coord_one_hot_smoothed, label='downsampled', color='g')
            # plt.show()
            # y_coord_one_hot_smoothed.shape
            # plt.plot(x_coord_one_hot_smoothed, label='downsampled', color='g')
            # plt.show()
            # x_coord_one_hot_smoothed.shape
            xy_coords_one_hot_smoothed = torch.concat([torch.tensor(y_coord_one_hot_smoothed), torch.tensor(x_coord_one_hot_smoothed)])
            # xy_coords_one_hot_smoothed.shape

            target_coords_by_cam_onehot[joint_idx, :] = xy_coords_one_hot_smoothed

    # make flatt layout
    target_coords_by_cam_onehot_flatt = target_coords_by_cam_onehot.view(-1)
    
    return target_coords_by_cam_onehot_flatt


project_path = './'
model_path = project_path + './model/train/'
event_data_path = project_path + './data/dhp19_samples/'

# paramters of the training data
img_width = 344
img_height = 260
downsample_factor = 2
xy_coord_vec_length = int((img_width + img_height)/downsample_factor)

# Create Dataset instance
cam_idxs = [1,2] # camera index to train on 1,2 are frontal views
seq_length = 8 # number of event frames per sequence to be shown to the SDNN
joint_idxs = [7, 8] # joint indices to train on
num_joints = len(joint_idxs)

files = os.listdir(event_data_path)
files = [f for f in files if f.endswith('pt')]

cam_id = cam_idxs[0]
# for cam_id in cam_idxs:

input_frames = []
target_vectors = []
sessions = []
subjects = []
movs = []

# file = files[0]
# loop throug files
for i, file in enumerate(files):
    session = int(file.split('_')[1][7:])
    subject = int(file.split('_')[0][1:])
    mov = int(file.split('_')[2][3:])    
    data_dict = torch.load(event_data_path + file) # load file
    # loop over all time steps in the file and select input frame and target joints for cam N
    # t=0
    for t in range(seq_length):
        # select and tranform input frame
        act_input_tensor = data_dict['input_tensor'].to_dense()[0,:,:,cam_id,t].numpy().astype(np.float32) # take frame t of cam N, reshape to [260, 340]
        # act_input_tensor.shape
        # plt.imshow(act_input_tensor[:,:])
        # plt.colorbar()
        # select and tranform target coords
        target_coords_abs_by_cam = data_dict['target_coords_abs'][joint_idxs,:,cam_id,t] # take target joints for time t of cam N
        # target_coords_abs_by_cam.shape # joints, cam
        act_targets_1hot = target_coords_to_onehot_smoothed_lowres(target_coords_abs_by_cam)
        
        # save input frame, target_1hot, session, subject and mov
        input_frames.append(csr_matrix(act_input_tensor))  # save input frame as sparse matrix
        target_vectors.append(act_targets_1hot.numpy().astype(np.float32)) # [0] 
        sessions.append(session)
        subjects.append(subject)
        movs.append(mov)

# store the data in a dictionary
data_dict = {'input_frames': input_frames, 'target_vectors': target_vectors, 'sessions': sessions, 'subjects': subjects, 'movs': movs}
# save the dictionary as a .pkl file
with open(event_data_path + 'dhp19_data_subject1_cam' + str(cam_id) + '.pkl', 'wb') as f:
    pickle.dump(data_dict, f)

