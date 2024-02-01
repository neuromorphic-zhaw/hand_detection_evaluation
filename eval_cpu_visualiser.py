# run infercence our traine DHP19 model on Loihi?
import os
from lava.lib.dl import netx
import logging
import torch
import numpy as np
from lava.proc import io
# from plot import plot_input_sample, plot_output_vs_target, plot_input_vs_prediction_vs_target, show_model_output
import numpy as np
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg
from lava.utils.system import Loihi2
from torch.utils.data import Dataset
# from dataset import DHP19NetDataset
from utils import generate_movement_name_df, comp_dist_for_sample
from IPython.display import display, clear_output
#from plot import plot_input_vs_prediction_vs_target
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from torch.utils.data import Dataset, DataLoader

path = './data/dhp19_samples/'

class dhp19(Dataset):
    """
    Dataset class for the dhp19 dataset
    """
    def __init__(self, path):
        self.path = path
        self.files = os.listdir(path)
        self.files = [f for f in self.files if f.endswith('pt')]
        # files = [f for f in files if f.endswith('pt')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # file = files[0]
        file = self.files[idx]
        session = int(file.split('_')[1][7:])
        subject = int(file.split('_')[0][1:])
        mov = int(file.split('_')[2][3:])
        data_dict = torch.load(self.path + file)
        # data_dict = torch.load(path + file)

        return data_dict['input_tensor'].to_dense(), data_dict['target_coords_abs'], data_dict['target_coords_rel'], session, subject, mov


def get_prediction_from_output_vector(output, downsample_factor=2, img_height=260, img_width=344):

    xy_coord_vec_length = int((img_width + img_height)/downsample_factor)

    joint_idx = 0    
    coords_1hot = output[joint_idx*xy_coord_vec_length:(joint_idx+1)*xy_coord_vec_length]
    coords_one_hot_y1 = coords_1hot[0:int(img_height / downsample_factor)]
    coords_one_hot_x1 = coords_1hot[int(img_height / downsample_factor):]
    
    # plt.plot(coords_1hot)
    # plt.plot(coords_one_hot_y1)
    # plt.plot(coords_one_hot_x1)
    # get prediction from model output
    
    predicted_y1 = np.where(coords_one_hot_y1 == coords_one_hot_y1.max())[0][0]
    predicted_x1 = np.where(coords_one_hot_x1 == coords_one_hot_x1.max())[0][0]
    predicted_y1 = predicted_y1 * downsample_factor
    predicted_x1 = predicted_x1 * downsample_factor

    # get target and predicted coordinates for joint 2
    joint_idx = 1    
    coords_1hot = output[joint_idx*xy_coord_vec_length:(joint_idx+1)*xy_coord_vec_length]
    coords_one_hot_y2 = coords_1hot[0:int(img_height / downsample_factor)]
    coords_one_hot_x2 = coords_1hot[int(img_height / downsample_factor):]
    # get prediction from model output
    predicted_y2 = np.where(coords_one_hot_y2 == coords_one_hot_y2.max())[0][0]
    predicted_x2 = np.where(coords_one_hot_x2 == coords_one_hot_x2.max())[0][0]
    predicted_y2 = predicted_y2 * downsample_factor
    predicted_x2 = predicted_x2 * downsample_factor

    return predicted_x1, predicted_y1, predicted_x2, predicted_y2

if __name__ == '__main__':      
    # Check if Loihi2 compiker is available and import related modules.
    # Loihi2.preferred_partition = 'oheogulch'
    # loihi2_is_available = Loihi2.is_loihi2_available

    loihi2_is_available = False # Force CPU execution
    print("Loihi2 compiler is not available in this system. "
        "This tutorial will execute on CPU backend.")
    compression = io.encoder.Compression.DENSE
    system = 'cpu_noquand_input_without_buffer'
    
    # Set paths to model and data
    project_path = './'
    model_path = project_path + './model/train/'
    event_data_path = project_path + './data/dhp19_samples/'
    # paramters of the traininf data
    img_width = 344
    img_height = 260
    downsample_factor = 2
    xy_coord_vec_length = int((img_width + img_height)/downsample_factor)

    # Create Dataset instance
    batch_size = 8  # batch size
    learning_rate = 0.00005 # leaerning rate
    lam = 0.001 # lagrangian for event rate loss
    num_epochs = 30  # training epochs
    # steps  = [60, 120, 160] # learning rate reduction milestones
    cam_idxs = [1,2] # camera index to train on 1,2 are frontal views
    seq_length = 8 # number of event frames per sequence to be shown to the SDNN
    joint_idxs = [7, 8] # joint indices to train on
    num_joints = len(joint_idxs)
    
    # Load model
    model_name = 'sdnn_1hot_smoothed_scaled_lowres_rmsprop_relu_v2'
    # create experiment name
    experiment_name = model_name + \
                    '_epochs' + str(num_epochs) + \
                    '_lr' + str(learning_rate) + \
                    '_batchsize' + str(batch_size) + \
                    '_seq' + str(seq_length) + \
                    '_cam' + str(cam_idxs).replace(' ', '') + \
                    '_lam' + str(lam)

    act_model_path = model_path + experiment_name + '/'
    net = netx.hdf5.Network(net_config=act_model_path + 'model.net', skip_layers=1)
    
    print(net)
    len(net)
    
    #load/define current dataset
    num_time_steps_per_file = 8 # number of time steps per file
    complete_dataset = dhp19(event_data_path)
    num_test_samples = complete_dataset.__len__()
    print('Dataset loaded: ' + str(len(complete_dataset)*num_time_steps_per_file) + ' samples found')
    
    # get names of movements based on session and movement number
    movement_names_df = generate_movement_name_df()
    # input, target_cords_abs, target_coords_rel, session, subject, mov = complete_dataset[0]
    
    # input.shape # C x H x W, Cam, T
    # target_cords_abs.shape # Joint, coords, Cam, T
    # cam_id = cam_idxs[0]
   
    # plt.imshow(input[0,:,:,cam_id,0], cmap='gray')
    # plt.colorbar()


    # setup lava process modules
    quantize = netx.modules.Quantize(exp=6)  # convert to fixed point representation with 6 bit of fraction
    sender = io.injector.Injector(shape=net.inp.shape, buffer_size=128)
    encoder = io.encoder.DeltaEncoder(shape=net.inp.shape,
                                  vth=net.net_config['layer'][0]['neuron']['vThMant'],
                                  spike_exp=6,
                                  num_bits=8,
                                  compression=compression)
    
    sender.out_port.shape
    receiver = io.extractor.Extractor(shape=net.out.shape, buffer_size=128)
    dequantize = netx.modules.Dequantize(exp=net.spike_exp + 12, num_raw_bits=24)

    # connect modules
    sender.out_port.connect(encoder.a_in)
    encoder.s_out.connect(net.inp)
    net.out.connect(receiver.in_port)

    # setup run conditions
    # num_files = len(complete_dataset)
    num_files = 2
    num_steps = num_files * num_time_steps_per_file
    run_condition = RunSteps(num_steps=num_steps, blocking=False)
    
    exception_proc_model_map = {io.encoder.DeltaEncoder: io.encoder.PyDeltaEncoderModelDense}
    run_config = Loihi2SimCfg(select_tag='fixed_pt',
                            exception_proc_model_map=exception_proc_model_map)

    sender._log_config.level = logging.WARN
    sender.run(condition=run_condition, run_cfg=run_config)
    
    # recodrd prediction error per movement
    eval_df = movement_names_df
    
    eval_df['fame count'] = 0
    eval_df['sum dist joint 1'] = 0
    eval_df['sum dist joint 2'] = 0

    input_list = []
    output_list = []
    target_x1_list = []
    target_y1_list = []
    target_x2_list = []
    target_y2_list = []
    prediction_x1_list = []
    prediction_y1_list = []
    prediction_x2_list = []
    prediction_y2_list = []
    
    # t=2
    # f = 0

    cam_id = cam_idxs[0]
    
    for f in range(num_files): # loop over files
        print('f = ' + str(f))
        input, target_coords_abs, target_coords_rel, session, subject, mov = complete_dataset[f]
        print('Session: ' + str(session) + ', Subject: ' + str(subject) + ', Movement: ' + str(mov) + ', ' + movement_names_df.loc[(movement_names_df['session'] == session) & (movement_names_df['mov'] == mov), 'mov_string'].iloc[0])
        
        for t in range(num_time_steps_per_file): # loop over time steps
            print('t = ' + str(t))
            input_frame = input[:,:,:,cam_id,t]
            # input_frame.shape # C x H x W
            # Swap the first and second axes
            input_frame = input_frame.permute(2, 1, 0)
            # input_frame.shape # H x W x C
            # input_quantized = quantize(input)
            # sender.send(quantize(input))        # This sends the input frame to the Lava network
            sender.send(input_frame)        # This sends the input frame to the Lava network
            model_out = receiver.receive()  # This receives the output from the Lava network
            out_dequantized = dequantize(model_out)
            print(out_dequantized)
            # get targets and prediction
            target_coords = target_coords_abs[joint_idxs,:,cam_id,t]
            target_coords.shape # [2, 2] num_joints, coords 
            target_x1 = target_coords[0,1]
            target_y1 = target_coords[0,0]
            target_x2 = target_coords[1,1]
            target_y2 = target_coords[1,0]

            predicted_x1, predicted_y1, predicted_x2, predicted_y2 = get_prediction_from_output_vector(output=out_dequantized, downsample_factor=2, img_height=260, img_width=344)      

            plt.imshow(input_frame, cmap='gray')
            plt.scatter(target_y1, target_x1, c='g', marker='o', s=25, label='target joint 1')
            plt.scatter(target_y2, target_x2, c='g', marker='x', s=25, label='target joint 1')
            plt.scatter(predicted_y1, predicted_x1, c='r', marker='o', s=25, label='predicted joint 1')
            plt.scatter(predicted_y2, predicted_x2, c='r', marker='x', s=25, label='predicted joint 2')

            output_list.append(out_dequantized)

        # # save to lists
        # input_list.append(input)
        # target_x1_list.append(target_x1)
        # target_y1_list.append(target_y1)
        # target_x2_list.append(target_x2)
        # target_y2_list.append(target_y2)
        # prediction_x1_list.append(predicted_x1)
        # prediction_y1_list.append(predicted_y1)
        # prediction_x2_list.append(predicted_x2)
        # prediction_y2_list.append(predicted_y2)
              

        # # compute prediction error
        # dist1, dist2 = comp_dist_for_sample(out_dequantized, target, downsample_factor=2, img_height=260, img_width=344)
        # # print('Dist joint 1: ' + str(dist1) + ', Dist joint 2: ' + str(dist2))
         
        # # record prediction error for current movement
        # eval_df.loc[(movement_names_df['session'] == session) & (movement_names_df['mov'] == mov), 'fame count'] += 1
        # eval_df.loc[(movement_names_df['session'] == session) & (movement_names_df['mov'] == mov), 'sum dist joint 1'] += dist1
        # eval_df.loc[(movement_names_df['session'] == session) & (movement_names_df['mov'] == mov), 'sum dist joint 2'] += dist2
        
        # # show_model_output(out_dequantized, downsample_factor=2, img_height=260, img_width=344, time_step=t)   
        # # plot_output_vs_target(out_dequantized, target, downsample_factor=2, img_height=260, img_width=344, time_step=t, filename='plots/output_vs_target' + str(t) + '_' + system + '.png')
        # # plot_output_vs_target(out_dequantized, target, downsample_factor=2, img_height=260, img_width=344, time_step=t, filename=None)
        # # plot_input_vs_prediction_vs_target(input, out_dequantized, target, downsample_factor=2, img_height=260, img_width=344, time_step=t, filename='plots/input_vs_prediction_vs_target_' + str(t) + '_' + system + '.png')
        

    sender.wait()
    sender.stop()
    print('Done')

    # # compute mean prediction error per movement
    # eval_df['mean dist joint 1'] = eval_df['sum dist joint 1'] / eval_df['fame count']
    # eval_df['mean dist joint 2'] = eval_df['sum dist joint 2'] / eval_df['fame count']
    # print(eval_df)
    # eval_df.to_csv('eval_df_' + system + '.csv')

    # # plot input and prediction as annimation
    # # fig, ax = plt.subplots()
    # fig, ax  = plt.subplots(figsize=(10,5)) # create figure
    # joint1_img = ax.imshow(np.swapaxes(input_list[0][:,:,0],0,1), cmap='gray')
    # joint1_target = ax.scatter(target_x1_list[0], target_y1_list[0], c='g', s=25, label='target')
    # joint1_predicted = ax.scatter(prediction_x1_list[0], prediction_y1_list[0], c='r', s=25, label='predicted')
    

    # # joint1 = plt.subplot2grid((1, 2), (0, 0))
    # # joint2 = plt.subplot2grid((1, 2), (0, 1))

    # # joint1.imshow(np.swapaxes(input[:,:,0],0,1), cmap='gray')
    # # joint1.plot(target_x1, target_y1, 'go', label='target')
    # # joint1.plot(predicted_x1, predicted_y1, 'ro', label='predicted')

    # # joint2.imshow(np.swapaxes(input[:,:,0],0,1), cmap='gray')
    # # joint2.plot(target_x2, target_y2, 'go', label='target')
    # # joint2.plot(predicted_x2, predicted_y2, 'ro', label='predicted')

    # def update(frame):
    #     # for each frame, update the data stored on each artist.
    #     input = input_list[frame]
    #     target_x1 = target_x1_list[frame]
    #     target_y1 = target_y1_list[frame]
    #     predicted_x1 = prediction_x1_list[frame]
    #     predicted_y1 = prediction_y1_list[frame]
    #     # target_x2 = target_x2_list[frame]
    #     # target_y2 = target_y2_list[frame]
    #     # predicted_x2 = prediction_x2_list[frame]
    #     # predicted_y2 = prediction_y2_list[frame]

    #     joint1_img.set_data(np.swapaxes(input[:,:,0],0,1))
    #     joint1_target.set_offsets([target_x1, target_y1])
    #     joint1_predicted.set_offsets([predicted_x1, predicted_y1])

    #     return joint1_img, joint1_target, joint1_predicted
     

    # ani = animation.FuncAnimation(fig=fig, func=update, frames=num_steps, interval=200)
    # # plt.show()

    # # ani.save(filename="annimation.html", writer="html")
    # ani.save(filename="annimation.mp4", writer="ffmpeg")

    # fig, ax  = plt.subplots(figsize=(10,5)) # create figure

    # plt.plot(target_x1_list, '-g', label='target')
    # plt.plot(prediction_x1_list, '-r', label='predicted')
    # plt.legend()
    # plt.show()


    # plt.plot(target_y1_list, '-g', label='target')
    # plt.plot(prediction_y1_list, '-r', label='predicted')
    # plt.legend()
    # plt.show()


    # import pickle

    # result_dict = {'input_list': input_list,
    #                 'target_x1_list': target_x1_list,
    #                 'target_y1_list': target_y1_list,
    #                 'target_x2_list': target_x2_list,
    #                 'target_y2_list': target_y2_list,
    #                 'prediction_x1_list': prediction_x1_list,
    #                 'prediction_y1_list': prediction_y1_list,
    #                 'prediction_x2_list': prediction_x2_list,
    #                 'prediction_y2_list': prediction_y2_list}

    # with open('results.pkl', 'wb') as f:
    #     pickle.dump(result_dict, f)