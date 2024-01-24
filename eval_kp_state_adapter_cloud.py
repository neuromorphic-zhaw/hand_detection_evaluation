# run infercence our traine DHP19 model on Loihi?
import logging

import numpy as np
import matplotlib.pyplot as plt

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2HwCfg
from lava.proc import embedded_io as eio
from lava.proc import io
from lava.lib.dl import netx

# from plot import plot_input_sample, plot_output_vs_target, plot_input_vs_prediction_vs_target, show_model_output
from utils import generate_movement_name_df, comp_dist_for_sample

from dataset import DHP19NetDataset
from lava.utils.system import Loihi2


if __name__ == '__main__':      
    # Check if Loihi2 compiker is available and import related modules.
    Loihi2.preferred_partition = 'oheogulch'
    loihi2_is_available = Loihi2.is_loihi2_available
    # loihi2_is_available = False # Force CPU execution

    print(f'Running on loihi2')
    from lava.magma.compiler.subcompilers.nc.ncproc_compiler import CompilerOptions
    CompilerOptions.verbose = True
    # compression = io.encoder.Compression.DELTA_SPARSE_8
    compression = io.encoder.Compression.DENSE
    system = 'loihi2_dense_noquand_input'
    
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
    
    # print('Loading net ' + experiment_name    )
    complete_dataset = DHP19NetDataset(path=event_data_path, joint_idxs=joint_idxs, cam_id=cam_idxs[0], num_time_steps=seq_length)
    print('Dataset loaded: ' + str(len(complete_dataset)) + ' samples found')

     # get names of movements based on session and movement number
    movement_names_df = generate_movement_name_df()

    quantize = netx.modules.Quantize(exp=6)  # convert to fixed point representation with 6 bit of fraction
    sender = io.injector.Injector(shape=net.inp.shape, buffer_size=128)
    encoder = io.encoder.DeltaEncoder(shape=net.inp.shape,
                                  vth=net.net_config['layer'][0]['neuron']['vThMant'],
                                  spike_exp=0,
                                  num_bits=8,
                                  compression=compression)
    
    inp_adapter = eio.spike.PyToN3ConvAdapter(shape=encoder.s_out.shape,
                                              num_message_bits=16,
                                              spike_exp=net.spike_exp,
                                              compression=compression)
    
    #state_adapter = eio.state.ReadConv(shape=net.out.shape)
    state_adapter = eio.state.Read(shape=net.out.shape)

    receiver = io.extractor.Extractor(shape=net.out.shape, buffer_size=128)
    dequantize = netx.modules.Dequantize(exp=net.spike_exp+12, num_raw_bits=24)
    
    # connect modules
    sender.out_port.connect(encoder.a_in)
    encoder.s_out.connect(inp_adapter.inp)
    inp_adapter.out.connect(net.inp)
    state_adapter.connect_var(net.out_layer.neuron.sigma)
    state_adapter.out.connect(receiver.in_port)
    
    # setup run conditions
    num_steps = len(complete_dataset)
    # num_steps = 40

    run_condition = RunSteps(num_steps=num_steps, blocking=False)
    # exception_proc_model_map = {io.encoder.DeltaEncoder: io.encoder.PyDeltaEncoderModelSparse}
    exception_proc_model_map = {io.encoder.DeltaEncoder: io.encoder.PyDeltaEncoderModelDense}
    run_config = Loihi2HwCfg(exception_proc_model_map=exception_proc_model_map)
    
    sender._log_config.level = logging.WARN
    sender.run(condition=run_condition, run_cfg=run_config)
    
    # record prediction error per movement
    eval_df = movement_names_df
    eval_df['fame count'] = 0
    eval_df['sum dist joint 1'] = 0
    eval_df['sum dist joint 2'] = 0

    # t = 1
    for t in range(num_steps):
        print('t = ' + str(t))

        input, target, session, subject, mov  = complete_dataset[t]        # input_quantized = quantize(input)
        # print('Session: ' + str(session) + ', Subject: ' + str(subject) + ', Movement: ' + str(mov) + ', ' + movement_names_df.loc[(movement_names_df['session'] == session) & (movement_names_df['mov'] == mov), 'mov_string'].iloc[0])
        # input_quantized = quantize(input)       
        # sender.send(quantize(input))        # This sends the input frame to the Lava network
        sender.send(input)        # This sends the input frame to the Lava network

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
        # plot_input_vs_prediction_vs_target(input, out_dequantized, target, downsample_factor=2, img_height=260, img_width=344, time_step=t, filename='plots/input_vs_prediction_vs_target_' + str(t) + '_' + system + '.png')
        
    sender.wait()
    sender.stop()
    print('Done')

    # compute mean prediction error per movement
    eval_df['mean dist joint 1'] = eval_df['sum dist joint 1'] / eval_df['fame count']
    eval_df['mean dist joint 2'] = eval_df['sum dist joint 2'] / eval_df['fame count']
    print(eval_df)
    eval_df.to_csv('eval_df_' + system + '.csv')