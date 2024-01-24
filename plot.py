from matplotlib import pyplot as plt
import numpy as np

def plot_input_sample(input_frame, target_coords=None, title=None, path=None):
    """
    Plot a single sample from the dataset.
    Parameters
    ----------
    input_frame : torch.Tensor
        Input frame. (H x W) e.g (260 x 344)
    target_coords : torch.Tensor, optional
        Target coordinates. The default is None.
    title : str, optional
        Title of the plot. The default is None.
    path : str, optional
        Path to save the plot. The default is None.
    Returns
    -------
    None.
    
    """
    plt.imshow(input_frame, cmap='gray')
    if target_coords is not None:
        plt.plot(target_coords[1], target_coords[0], 'go', label='target')
        plt.legend()
    if title:
        plt.title(title)
    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()


def plot_target_coords_1hot(target, joint_idxs, img_height, img_width, downsample_factor, path=None):
    num_joints = len(joint_idxs)
    xy_coord_vec_length = int((img_width + img_height)/downsample_factor)

    # show target
    plt.figure(figsize=(20, 10))
    for joint in range(num_joints):
        coords_1hot = target[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
        coords_one_hot_y = coords_1hot[0:int(img_height / downsample_factor)]
        coords_one_hot_x = coords_1hot[int(img_height / downsample_factor):]
        act_target_y = np.where(coords_one_hot_y == coords_one_hot_y.max())[0][0]
        act_target_x = np.where(coords_one_hot_x == coords_one_hot_x.max())[0][0]

        plt.subplot(num_joints, 2, (2*joint)+1)
        plt.plot(coords_one_hot_y)
        plt.vlines(act_target_y, ymin=0, ymax=1, colors='g', linestyles='dashed', label='target')
        plt.ylabel('y target')
        plt.title('Joint ' + str(joint_idxs[joint]))
        
        plt.subplot(num_joints, 2, (2*joint)+2)
        plt.plot(coords_one_hot_x)
        plt.vlines(act_target_x, ymin=0, ymax=1, colors='g', linestyles='dashed', label='target')
        plt.ylabel('x target')
    
    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()


def plot_output_vs_target(model_output, target, downsample_factor=2, img_height=260, img_width=344, time_step=None, filename=None):
    xy_coord_vec_length = int((img_width + img_height)/downsample_factor)
    
    # get parts of the output data by coordinate and joint
    joint = 0    
    coords_1hot = model_output[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
    coords_one_hot_y1 = coords_1hot[0:int(img_height / downsample_factor)]
    coords_one_hot_x1 = coords_1hot[int(img_height / downsample_factor):]
    
    coords_one_hot_y1.shape
    # get prediction from model output
    predicted_y1 = np.where(coords_one_hot_y1 == coords_one_hot_y1.max())[0][0]
    predicted_x1 = np.where(coords_one_hot_x1 == coords_one_hot_x1.max())[0][0]

    target_1hot = target[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
    target_one_hot_y1 = target_1hot[0:int(img_height / downsample_factor)]
    target_one_hot_x1 = target_1hot[int(img_height / downsample_factor):]
    
    target_one_hot_y1.shape
    # get prediction from target
    target_y1 = np.where(target_one_hot_y1 == target_one_hot_y1.max())[0][0]
    target_x1 = np.where(target_one_hot_x1 == target_one_hot_x1.max())[0][0]    
    
    joint = 1    
    coords_1hot = model_output[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
    coords_one_hot_y2 = coords_1hot[0:int(img_height / downsample_factor)]
    coords_one_hot_x2 = coords_1hot[int(img_height / downsample_factor):]
    # get prediction from model output
    predicted_y2 = np.where(coords_one_hot_y2 == coords_one_hot_y2.max())[0][0]
    predicted_x2 = np.where(coords_one_hot_x2 == coords_one_hot_x2.max())[0][0]
        
    target_1hot = target[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
    target_one_hot_y2 = target_1hot[0:int(img_height / downsample_factor)]
    target_one_hot_x2 = target_1hot[int(img_height / downsample_factor):]
    # get prediction from target
    target_y2 = np.where(target_one_hot_y2 == target_one_hot_y2.max())[0][0]
    target_x2 = np.where(target_one_hot_x2 == target_one_hot_x2.max())[0][0]
    
    fig = plt.figure(figsize=(10, 10)) # create figure

    # self.input_frame = self.fig.add_subplot(111)
    y1 = plt.subplot2grid((2, 2), (0, 0))
    x1 = plt.subplot2grid((2, 2), (0, 1))
    y2 = plt.subplot2grid((2, 2), (1, 0))
    x2 = plt.subplot2grid((2, 2), (1, 1))

    y1.plot(coords_one_hot_y1)
    y1.plot(target_one_hot_y1)
    y1.vlines(target_y1, ymin=0, ymax=1, colors='g', linestyles='dashed', label='target')
    y1.vlines(predicted_y1, ymin=0, ymax=1, colors='r', linestyles='dashed', label='predicted')
    y1.set_ylabel('y')
    y1.set_title('Joint 1')

    x1.plot(coords_one_hot_x1)
    x1.plot(target_one_hot_x1)
    x1.vlines(target_x1, ymin=0, ymax=1, colors='g', linestyles='dashed', label='target')
    x1.vlines(predicted_x1, ymin=0, ymax=1, colors='r', linestyles='dashed', label='predicted')
    x1.set_ylabel('x')
    
    y2.plot(coords_one_hot_y2)
    y2.plot(target_one_hot_y2)
    y2.vlines(target_y2, ymin=0, ymax=1, colors='g', linestyles='dashed', label='target')
    y2.vlines(predicted_y2, ymin=0, ymax=1, colors='r', linestyles='dashed', label='predicted')
    y2.set_ylabel('y') 
    y2.set_title('Joint 2')
    
    x2.plot(coords_one_hot_x2)
    x2.plot(target_one_hot_x2)
    x2.vlines(target_x2, ymin=0, ymax=1, colors='g', linestyles='dashed', label='target')
    x2.vlines(predicted_x2, ymin=0, ymax=1, colors='r', linestyles='dashed', label='predicted')
    x2.set_ylabel('x')
    fig.tight_layout()
    if time_step is not None:
        fig.suptitle('Model output at time step ' + str(time_step))
    else:
        fig.suptitle('Model output')

    if filename is not None:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_input_vs_prediction_vs_target(input, model_output, target, downsample_factor=2, img_height=260, img_width=344, time_step=None, filename=None):
    xy_coord_vec_length = int((img_width + img_height)/downsample_factor)
    
    # get parts of the output data by coordinate and joint
    joint = 0    
    coords_1hot = model_output[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
    coords_one_hot_y1 = coords_1hot[0:int(img_height / downsample_factor)]
    coords_one_hot_x1 = coords_1hot[int(img_height / downsample_factor):]
    
    coords_one_hot_y1.shape
    # get prediction from model output
    predicted_y1 = np.where(coords_one_hot_y1 == coords_one_hot_y1.max())[0][0]
    predicted_x1 = np.where(coords_one_hot_x1 == coords_one_hot_x1.max())[0][0]
    predicted_y1 = predicted_y1 * downsample_factor
    predicted_x1 = predicted_x1 * downsample_factor

    target_1hot = target[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
    target_one_hot_y1 = target_1hot[0:int(img_height / downsample_factor)]
    target_one_hot_x1 = target_1hot[int(img_height / downsample_factor):]
    
    target_one_hot_y1.shape
    # get prediction from target
    target_y1 = np.where(target_one_hot_y1 == target_one_hot_y1.max())[0][0]
    target_x1 = np.where(target_one_hot_x1 == target_one_hot_x1.max())[0][0]    
    target_y1 = target_y1 * downsample_factor
    target_x1 = target_x1 * downsample_factor


    joint = 1    
    coords_1hot = model_output[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
    coords_one_hot_y2 = coords_1hot[0:int(img_height / downsample_factor)]
    coords_one_hot_x2 = coords_1hot[int(img_height / downsample_factor):]
    # get prediction from model output
    predicted_y2 = np.where(coords_one_hot_y2 == coords_one_hot_y2.max())[0][0]
    predicted_x2 = np.where(coords_one_hot_x2 == coords_one_hot_x2.max())[0][0]
    predicted_y2 = predicted_y2 * downsample_factor
    predicted_x2 = predicted_x2 * downsample_factor
        
    target_1hot = target[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
    target_one_hot_y2 = target_1hot[0:int(img_height / downsample_factor)]
    target_one_hot_x2 = target_1hot[int(img_height / downsample_factor):]
    # get prediction from target
    target_y2 = np.where(target_one_hot_y2 == target_one_hot_y2.max())[0][0]
    target_x2 = np.where(target_one_hot_x2 == target_one_hot_x2.max())[0][0]
    target_y2 = target_y2 * downsample_factor
    target_x2 = target_x2 * downsample_factor
        
    fig = plt.figure(figsize=(10,5)) # create figure
    joint1 = plt.subplot2grid((1, 2), (0, 0))
    joint2 = plt.subplot2grid((1, 2), (0, 1))

    joint1.imshow(np.swapaxes(input[:,:,0],0,1), cmap='gray')
    joint1.plot(target_x1, target_y1, 'go', label='target')
    joint1.plot(predicted_x1, predicted_y1, 'ro', label='predicted')

    joint2.imshow(np.swapaxes(input[:,:,0],0,1), cmap='gray')
    joint2.plot(target_x2, target_y2, 'go', label='target')
    joint2.plot(predicted_x2, predicted_y2, 'ro', label='predicted')

    fig.tight_layout()
    if time_step is not None:
        fig.suptitle('Model output at time step ' + str(time_step))
    else:
        fig.suptitle('Model output')

    if filename is not None:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def show_model_output(model_output, downsample_factor=2, img_height=260, img_width=344, time_step=None):
    xy_coord_vec_length = int((img_width + img_height)/downsample_factor)
        # get parts of the output data by coordinate and joint
    joint = 0    
    coords_1hot = model_output[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
    coords_one_hot_y1 = coords_1hot[0:int(img_height / downsample_factor)]
    coords_one_hot_x1 = coords_1hot[int(img_height / downsample_factor):]

    joint = 1    
    coords_1hot = model_output[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
    coords_one_hot_y2 = coords_1hot[0:int(img_height / downsample_factor)]
    coords_one_hot_x2 = coords_1hot[int(img_height / downsample_factor):]
    
    fig = plt.figure(figsize=(10, 10)) # create figure

    # self.input_frame = self.fig.add_subplot(111)
    y1 = plt.subplot2grid((2, 2), (0, 0))
    x1 = plt.subplot2grid((2, 2), (0, 1))
    y2 = plt.subplot2grid((2, 2), (1, 0))
    x2 = plt.subplot2grid((2, 2), (1, 1))
    y1.plot(coords_one_hot_y1)
    y1.set_ylabel('y')
    y1.set_title('Joint 1')
    x1.plot(coords_one_hot_x1)
    x1.set_ylabel('x')
    y2.plot(coords_one_hot_y2)
    y2.set_ylabel('y') 
    y2.set_title('Joint 2')
    x2.plot(coords_one_hot_x2)
    x2.set_ylabel('y')
    fig.tight_layout()
    if time_step is not None:
        fig.suptitle('Model output at time step ' + str(time_step))
    else:
        fig.suptitle('Model output')
    
    plt.show()
