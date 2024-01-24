from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter1d
import numpy as np
import torch
import os
# import matplotlib.pyplot as plt


class DummyDataset(Dataset):
    """

    """
    def __init__(self, num_samples):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return np.random.rand(344,266, 1, 1), np.random.rand(604)  



# target_coords_by_cam = target_coords_abs_by_cam 
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


class DHP19NetDataset(Dataset):
    """
    Dataset class for the DHP19 dataset
    Returns image and ground truth value (1hot smoothed encoded)
    when the object is indexed.

    Parameters
    ----------
    path : str
        Path of the dataset folder.
    joint_idx : list
        Joint indices the model was trained on
    cam_id : int
        Camera index the model was trained on and which should be used for inference
    num_time_steps : int
        Number of event frames per sequence per file (usually the number of frame the model was trained on)
    Usage
    -----

    >>> dataset = DHP19NetDataset(path='./../data/dhp19/', joint_idx=[7,8], cam_id=1, num_time_steps=8)
    >>> image, target_coords = dataeset[0]
    >>> num_samples = len(dataset)
    """
    def __init__(self, path, joint_idxs=[7,8], cam_id=1, num_time_steps=8):
        self.path = path
        self.joint_idxs = joint_idxs
        self.cam_id = cam_id
        self.num_time_steps = num_time_steps
        self.files = os.listdir(path)
        self.files = [f for f in self.files if f.endswith('pt')]
        self.input_frames = []
        self.target_vectors = []
        # self.session = []
        # self.subject = []
        # self.mov = []

        # loop over all files in the folder
        # file = files[0]
        for i, file in enumerate(self.files):
            session = int(file.split('_')[1][7:])
            subject = int(file.split('_')[0][1:])
            mov = int(file.split('_')[2][3:])    
            data_dict = torch.load(self.path + file) # load file
            # data_dict = torch.load(path + file) # load file

            # loop over all time steps in the file and select input frame and target joints for cam N
            # t=0
            for t in range(self.num_time_steps):
                # act_input_tensor = data_dict['input_tensor'][0].to_dense()[:,:,cam_id,t].numpy().astype(np.float32)[:,:, np.newaxis,np.newaxis] # take frame t of cam N, add axis for time dimension  
                # act_input_tensor.shape
                # select and tranform input frame
                self.input_frames.append(np.swapaxes(data_dict['input_tensor'][0].to_dense()[:,:,self.cam_id,t].numpy().astype(np.float32), 0, 1)[:,:, np.newaxis])  # take frame t of cam N
                # 344 x 260 x 1
                # 
                # select and tranform target coords
                # data_dict['target_coords_abs'].shape # joints, coords, cam, time
                target_coords_abs_by_cam = data_dict['target_coords_abs'][joint_idxs,:,cam_id,t] # take target joints for time t of cam N
                # target_coords_abs_by_cam.shape # joints, cam
                act_targets_1hot = target_coords_to_onehot_smoothed_lowres(target_coords_abs_by_cam)
                self.target_vectors.append(act_targets_1hot.numpy().astype(np.float32)) # [0] 
                # self.session.append(session)
                # self.subject.append(subject)
                # self.mov.append(mov)

    def __len__(self):
        return len(self.input_frames)

    def __getitem__(self, idx):
        return self.input_frames[idx], self.target_vectors[idx]
