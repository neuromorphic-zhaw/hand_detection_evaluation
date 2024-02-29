# run infercence our trained DHP19 model on Loihi?
from lava.lib.dl import netx
import logging
import numpy as np

from lava.proc import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg
from lava.utils.system import Loihi2
from utils import generate_movement_name_df, comp_dist_for_sample
import pickle


class DHP19pklDataset():
    def __init__(self, path):
        self.path = path
        # load data from pickle file
        with open(path, 'rb') as f:
            data_dict = pickle.load(f)
            
            self.input_frames = data_dict['input_frames']
            self.target_vectors = data_dict['target_vectors']
            self.sessions = data_dict['sessions']
            self.subjects = data_dict['subjects']
            self.movs = data_dict['movs']
    
    def __len__(self):
        return len(self.input_frames)
    
    def __getitem__(self, idx):
        input_frame = self.input_frames[idx].todense() # 260 x 344
        input_frame = np.expand_dims(input_frame, axis=-1).transpose(1,0,2) # 344 x 260 x 1

        return input_frame, self.target_vectors[idx], self.sessions[idx], self.subjects[idx], self.movs[idx]


def get_prediction_from_1hot_vector(vector_1hot, downsample_factor=2, img_height=260, img_width=344):
    """
    Extracts the predicted coordinates and maximum values from a one-hot vector.

    Args:
        vector_1hot (numpy.ndarray): The one-hot vector containing the predicted coordinates.
        downsample_factor (int, optional): The factor by which the image is downsampled. Defaults to 2.
        img_height (int, optional): The height of the image. Defaults to 260.
        img_width (int, optional): The width of the image. Defaults to 344.

    Returns:
        tuple: A tuple containing the maximum y-coordinate, maximum x-coordinate, and the one-hot vectors for each coordinate.
    """
   
    xy_coord_vec_length = int((img_width + img_height)/downsample_factor)
    
    # get parts of the output data by coordinate and joint
    joint = 0    
    coords_1hot = vector_1hot[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
    coords_one_hot_y1 = coords_1hot[0:int(img_height / downsample_factor)]
    coords_one_hot_x1 = coords_1hot[int(img_height / downsample_factor):]
    
    coords_one_hot_y1.shape
    # get max 
    max_y1 = np.where(coords_one_hot_y1 == coords_one_hot_y1.max())[0][0]
    max_x1 = np.where(coords_one_hot_x1 == coords_one_hot_x1.max())[0][0]
    
    joint = 1    
    coords_1hot = vector_1hot[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
    coords_one_hot_y2 = coords_1hot[0:int(img_height / downsample_factor)]
    coords_one_hot_x2 = coords_1hot[int(img_height / downsample_factor):]
    # get max
    max_y2 = np.where(coords_one_hot_y2 == coords_one_hot_y2.max())[0][0]
    max_x2 = np.where(coords_one_hot_x2 == coords_one_hot_x2.max())[0][0]
    
    return max_y1, max_x1, max_y2, max_x2, coords_one_hot_y1, coords_one_hot_x1, coords_one_hot_y2, coords_one_hot_x2


if __name__ == '__main__':      
    # Check if Loihi2 compiler is available and import related modules.
    # Loihi2.preferred_partition = 'oheogulch'
    # loihi2_is_available = Loihi2.is_loihi2_available

    loihi2_is_available = False # Force CPU execution
    print("Loihi2 compiler is not available in this system. "
        "This tutorial will execute on CPU backend.")
    compression = io.encoder.Compression.DENSE
    
    # Set paths to model and data
    project_path = './'
    model_path = project_path + '../model/train/'
    cam_id = 1
    
    system = 'cpu_seq64_last100_cam' + str(cam_id)
    event_data_path = project_path + '../data/dhp19_samples/' + 'dhp19_data_subject1_cam' + str(cam_id) + '.pkl'
    # paramters of the traininf data
    img_width = 344
    img_height = 260
    downsample_factor = 2
    xy_coord_vec_length = int((img_width + img_height)/downsample_factor)

    # Create Dataset instance
    batch_size = 2  # batch size
    learning_rate = 0.000005 # leaerning rate
    lam = 0.01 # lagrangian for event rate loss
    num_epochs = 60  # training epochs
    # steps  = [60, 120, 160] # learning rate reduction milestones
    cam_idxs = [1,2] # camera index to train on 1,2 are frontal views
    seq_length = 8 # number of event frames per sequence to be shown to the SDNN
    joint_idxs = [7, 8] # joint indices to train on
    num_joints = len(joint_idxs)
    seq_length = 64

    system = 'cpu' + \
        '_seq' + str(seq_length) + \
        '_cam' + str(cam_id)

    # Load model
    model_name = 'sdnn_1hot_smoothed_scaled_lowres_relu'
    # create experiment name
    experiment_name = model_name + \
                    '_epochs' + str(num_epochs) + \
                    '_lr' + str(learning_rate) + \
                    '_batchsize' + str(batch_size) + \
                    '_seq' + str(seq_length) + \
                    '_cam' + str(cam_idxs).replace(' ', '') + \
                    '_lam' + str(lam) + \
                    '_seq' + str(seq_length)

    act_model_path = model_path + experiment_name + '/'
    
    print('Loading net ' + experiment_name    )
    net = netx.hdf5.Network(net_config=act_model_path + 'model.net', skip_layers=1)

    # Load dataset    
    complete_dataset = DHP19pklDataset(path=event_data_path)
    print('Dataset loaded: ' + str(len(complete_dataset)) + ' samples found')
    

    # get names of movements based on session and movement number
    movement_names_df = generate_movement_name_df()

    # show sample from the dataset
    # input, target, session, subject, mov = complete_dataset[4000]   
    # input.shape
    # plt.imshow(np.swapaxes(input, 0, 1), cmap='gray')
    # plt.colorbar()
    # plt.savefig('doc/img/input_sample.png')
    # # # # print('Session: ' + str(session) + ', Subject: ' + str(subject) + ', Movement: ' + str(mov) + ', ' + movement_names_df.loc[(movement_names_df['session'] == session) & (movement_names_df['mov'] == mov), 'mov_string'].iloc[0])   

    # setup run conditions
    # num_steps = len(complete_dataset)
    num_steps = 100
    buffer_size = num_steps+1

    # setup lava process modules
    quantize = netx.modules.Quantize(exp=6)  # convert to fixed point representation with 6 bit of fraction
    sender = io.injector.Injector(shape=net.inp.shape, buffer_size=buffer_size)
    encoder = io.encoder.DeltaEncoder(shape=net.inp.shape,
                                  vth=net.net_config['layer'][0]['neuron']['vThMant'],
                                  spike_exp=6,
                                  num_bits=8,
                                  compression=compression)
    
    sender.out_port.shape
    receiver = io.extractor.Extractor(shape=net.out.shape, buffer_size=buffer_size)
    dequantize = netx.modules.Dequantize(exp=net.spike_exp + 12, num_raw_bits=24)

    # connect modules
    sender.out_port.connect(encoder.a_in)
    encoder.s_out.connect(net.inp)
    net.out.connect(receiver.in_port)

    # setup display of the encoded input
    encoder_output_extractor = io.extractor.Extractor(shape=net.inp.shape, buffer_size=buffer_size)
    encoder.s_out.connect(encoder_output_extractor.in_port)

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
    output_dequand_list = []
    output_raw_list = []
    # encoder_output_list = []
    target_list = []   

    for t in range(len(complete_dataset)-num_steps, len(complete_dataset)):
    # for t in range(num_steps):
        print('t = ' + str(t))

        input, target, session, subject, mov  = complete_dataset[t]
        sender.send(quantize(input))        # This sends the input frame to the Lava network
        # encoder_output = encoder_output_extractor.receive()           
        model_out = receiver.receive()  # This receives the output from the Lava network
        out_dequantized = dequantize(model_out)
        
        # compute prediction error
        dist1, dist2 = comp_dist_for_sample(out_dequantized, target, downsample_factor=2, img_height=260, img_width=344)
        # print('Dist joint 1: ' + str(dist1) + ', Dist joint 2: ' + str(dist2))
         
        # record prediction error for current movement
        eval_df.loc[(movement_names_df['session'] == session) & (movement_names_df['mov'] == mov), 'fame count'] += 1
        eval_df.loc[(movement_names_df['session'] == session) & (movement_names_df['mov'] == mov), 'sum dist joint 1'] += dist1
        eval_df.loc[(movement_names_df['session'] == session) & (movement_names_df['mov'] == mov), 'sum dist joint 2'] += dist2
        
        input_list.append(input)
        # encoder_output_list.append(encoder_output)
        output_dequand_list.append(out_dequantized)
        output_raw_list.append(model_out)
        target_list.append(target)

    sender.wait()
    sender.stop()
    print('Done')

    # compute mean prediction error per movement
    eval_df['mean dist joint 1'] = eval_df['sum dist joint 1'] / eval_df['fame count']
    eval_df['mean dist joint 2'] = eval_df['sum dist joint 2'] / eval_df['fame count']
    print(eval_df)
    eval_df.to_csv('eval_df_' + system + '.csv')

    # Create animation of model outputs
    target_y1, target_x1, target_y2, target_x2, target_one_hot_y1, target_one_hot_x1, target_one_hot_y2, target_one_hot_x2 = get_prediction_from_1hot_vector(target_list[13], downsample_factor=2, img_height=260, img_width=344)
    prediction_y1, prediction_x1, prediction_y2, prediction_x2, output_one_hot_y1, output_one_hot_x1, output_one_hot_y2, output_one_hot_x2 = get_prediction_from_1hot_vector(output_dequand_list[13], downsample_factor=2, img_height=260, img_width=344)
        
    # np.where(output_one_hot_x1 == output_one_hot_x1.max())[0][0]
    # prediction_x1
    # plt.plot(output_one_hot_x1)

    ylim = (-1, 1)

    fig, axs = plt.subplots(2, 2)
    y1_plot = axs[0, 0].plot(output_one_hot_y1)
    y1_target = axs[0,0].axvline(target_y1, color='g', ls='--')
    y1_predict = axs[0,0].axvline(prediction_y1, color='r', ls='--')
    # y1.vlines(predicted_y1, ymin=0, ymax=1, colors='r', linestyles='dashed', label='predicted')
    axs[0, 0].set_ylabel('y1')
    axs[0, 0].set_title('Joint 1')
    axs[0, 0].set_ylim(ylim)

    x1_plot = axs[0, 1].plot(output_one_hot_x1)
    x1_target = axs[0,1].axvline(target_x1, color='g', ls='--')
    x1_predict = axs[0,1].axvline(prediction_x1, color='r', ls='--')
    axs[0, 1].set_ylabel('x1')
    axs[0, 1].set_ylim(ylim)

    y2_plot = axs[1, 0].plot(output_one_hot_y2)
    y2_target = axs[1,0].axvline(target_y2, color='g', ls='--')
    y2_predict = axs[1,0].axvline(prediction_y2, color='r', ls='--')
    axs[1, 0].set_ylabel('y2')
    axs[1, 0].set_title('Joint 2')
    axs[1, 0].set_ylim(ylim)

    x2_plot = axs[1, 1].plot(output_one_hot_x2)
    x2_target = axs[1,1].axvline(target_x2, color='g', ls='--')
    x2_predict = axs[1,1].axvline(prediction_x2, color='r', ls='--')
    axs[1, 1].set_ylabel('x2')
    axs[1, 1].set_ylim(ylim)

    title = axs[0, 0].text(0.5,0.85, "frame 0", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, ha="center")
    fig.tight_layout()

    def update(frame):
            # for each frame, update the data stored on each artist.
        target_y1, target_x1, target_y2, target_x2, target_one_hot_y1, target_one_hot_x1, target_one_hot_y2, target_one_hot_x2 = \
            get_prediction_from_1hot_vector(target_list[frame], downsample_factor=2, img_height=260, img_width=344)
        prediction_y1, prediction_x1, prediction_y2, prediction_x2, output_one_hot_y1, output_one_hot_x1, output_one_hot_y2, output_one_hot_x2 = \
            get_prediction_from_1hot_vector(output_dequand_list[frame], downsample_factor=2, img_height=260, img_width=344)

        title.set_text('frame ' + str (frame))
        y1_plot[0].set_ydata(output_one_hot_y1)
        y1_target.set_xdata(target_y1)
        y1_predict.set_xdata(prediction_y1)
        
        x1_plot[0].set_ydata(output_one_hot_x1)
        x1_target.set_xdata(target_x1)
        x1_predict.set_xdata(prediction_x1)
        
        y2_plot[0].set_ydata(output_one_hot_y2)
        y2_target.set_xdata(target_y2)
        y2_predict.set_xdata(prediction_y2)

        x2_plot[0].set_ydata(output_one_hot_x2)
        x2_target.set_xdata(target_x2)
        x2_predict.set_xdata(prediction_x2)
        
        return y1_plot, x1_plot, y2_plot, x2_plot, y1_target, y1_predict

    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(output_dequand_list), interval=200)
    ani.save(filename='model_outputs_' + system + '.mp4', writer='ffmpeg')
    