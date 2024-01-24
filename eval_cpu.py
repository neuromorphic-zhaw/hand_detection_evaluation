# run infercence our traine DHP19 model on Loihi?
from lava.lib.dl import netx
import logging
import torch
import numpy as np
from dataset import DHP19NetDataset
from lava.proc import io
# from plot import plot_input_sample, plot_output_vs_target, plot_input_vs_prediction_vs_target, show_model_output
import numpy as np
import matplotlib.pyplot as plt
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg
from lava.utils.system import Loihi2
from dataset import DHP19NetDataset
from utils import generate_movement_name_df, comp_dist_for_sample



# def get_xy_coords_for_prediction_and_target(prediction, target, joint, downsample_factor=2, img_height=260, img_width=344):
#     xy_coord_vec_length = int((img_width + img_height)/downsample_factor) # 302

#     coords_1hot = prediction[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
#     coords_one_hot_y1 = coords_1hot[0:int(img_height / downsample_factor)]
#     coords_one_hot_x1 = coords_1hot[int(img_height / downsample_factor):]
    
#     # get prediction from model output
#     predicted_y1 = np.where(coords_one_hot_y1 == coords_one_hot_y1.max())[0][0]
#     predicted_x1 = np.where(coords_one_hot_x1 == coords_one_hot_x1.max())[0][0]
#     predicted_y1 = predicted_y1 * downsample_factor
#     predicted_x1 = predicted_x1 * downsample_factor

#     # get parts of the target data by coordinate and joint
#     target_1hot = target[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
#     target_one_hot_y1 = target_1hot[0:int(img_height / downsample_factor)]
#     target_one_hot_x1 = target_1hot[int(img_height / downsample_factor):]
    
#     # get prediction from target
#     target_y1 = np.where(target_one_hot_y1 == target_one_hot_y1.max())[0][0]
#     target_x1 = np.where(target_one_hot_x1 == target_one_hot_x1.max())[0][0]    
#     target_y1 = target_y1 * downsample_factor
#     target_x1 = target_x1 * downsample_factor

#     return predicted_y1, predicted_x1, target_y1, target_x1


# def comp_dist_for_sample(prediction, target, downsample_factor=2, img_height=260, img_width=344):
#     prediction.shape # (604,0)
#     target.shape # (604,0)
#     xy_coord_vec_length = int((img_width + img_height)/downsample_factor) # 302

#     # get xy coords of prediction and target per joint
#     predicted_y1, predicted_x1, target_y1, target_x1 = get_xy_coords_for_prediction_and_target(prediction, target, joint=0, downsample_factor=2, img_height=260, img_width=344)
#     predicted_y2, predicted_x2, target_y2, target_x2 = get_xy_coords_for_prediction_and_target(prediction, target, joint=1, downsample_factor=2, img_height=260, img_width=344)

#     # compute distance between prediction and target
#     dist1 = np.sqrt((predicted_y1 - target_y1)**2 + (predicted_x1 - target_x1)**2)
#     dist2 = np.sqrt((predicted_y2 - target_y2)**2 + (predicted_x2 - target_x2)**2)

#     return dist1, dist2



if __name__ == '__main__':      
    # Check if Loihi2 compiker is available and import related modules.
    # Loihi2.preferred_partition = 'oheogulch'
    # loihi2_is_available = Loihi2.is_loihi2_available

    loihi2_is_available = False # Force CPU execution
    print("Loihi2 compiler is not available in this system. "
        "This tutorial will execute on CPU backend.")
    compression = io.encoder.Compression.DENSE
    system = 'cpu_quand_input'
    
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
    
    # print(net)
    len(net)
    # net.inp.shape
    # net.out.shape
    # net.input_message_bits
    # net.output_message_bits
    # net.spike_exp
    # net.in_layer.output_message_bits
    # net.out_layer.neuron

    # # print('Loading net ' + experiment_name    )
    complete_dataset = DHP19NetDataset(path=event_data_path, joint_idxs=joint_idxs, cam_id=cam_idxs[0], num_time_steps=seq_length)
    print('Dataset loaded: ' + str(len(complete_dataset)) + ' samples found')
    
    # get names of movements based on session and movement number
    movement_names_df = generate_movement_name_df()

    # # show sample from the dataset
    # for i in range(0, 100):
    #     input, target, session, subject, mov = complete_dataset[i]
    #     print('Session: ' + str(session) + ', Subject: ' + str(subject) + ', Movement: ' + str(mov) + ', ' + movement_names_df.loc[(movement_names_df['session'] == session) & (movement_names_df['mov'] == mov), 'mov_string'].iloc[0])   
    # # plot_input_sample(np.swapaxes(input[:,:,0],0,1), target_coords=None, title='Input frame', path=None)
    
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
    num_steps = len(complete_dataset)
    # num_steps = 40
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

    for t in range(num_steps):
        print('t = ' + str(t))

        input, target, session, subject, mov  = complete_dataset[t]
        # print('Session: ' + str(session) + ', Subject: ' + str(subject) + ', Movement: ' + str(mov) + ', ' + movement_names_df.loc[(movement_names_df['session'] == session) & (movement_names_df['mov'] == mov), 'mov_string'].iloc[0])
        # input_quantized = quantize(input)
        sender.send(quantize(input))        # This sends the input frame to the Lava network
        # sender.send(input)        # This sends the input frame to the Lava network
        model_out = receiver.receive()  # This receives the output from the Lava network
        out_dequantized = dequantize(model_out)
        
        # compute prediction error
        dist1, dist2 = comp_dist_for_sample(out_dequantized, target, downsample_factor=2, img_height=260, img_width=344)
        # print('Dist joint 1: ' + str(dist1) + ', Dist joint 2: ' + str(dist2))
         
        # record prediction error for current movement
        eval_df.loc[(movement_names_df['session'] == session) & (movement_names_df['mov'] == mov), 'fame count'] += 1
        eval_df.loc[(movement_names_df['session'] == session) & (movement_names_df['mov'] == mov), 'sum dist joint 1'] += dist1
        eval_df.loc[(movement_names_df['session'] == session) & (movement_names_df['mov'] == mov), 'sum dist joint 2'] += dist2
        
        # show_model_output(out_dequantized, downsample_factor=2, img_height=260, img_width=344, time_step=t)   
        # plot_output_vs_target(out_dequantized, target, downsample_factor=2, img_height=260, img_width=344, time_step=t, filename='plots/output_vs_target' + str(t) + '_' + system + '.png')
        # plot_output_vs_target(out_dequantized, target, downsample_factor=2, img_height=260, img_width=344, time_step=t, filename=None)
        # plot_input_vs_prediction_vs_target(input, out_dequantized, target, downsample_factor=2, img_height=260, img_width=344, time_step=t, filename='plots/input_vs_prediction_vs_target_' + str(t) + '_' + system + '.png')
        
        

    sender.wait()
    sender.stop()
    print('Done')

    # compute mean prediction error per movement
    eval_df['mean dist joint 1'] = eval_df['sum dist joint 1'] / eval_df['fame count']
    eval_df['mean dist joint 2'] = eval_df['sum dist joint 2'] / eval_df['fame count']
    print(eval_df)
    eval_df.to_csv('eval_df_' + system + '.csv')
