import argparse
import torch
import numpy as np
import time
import os

import matplotlib as mp


from cadelac.learning.models.context_aware_delan import ContextAwareDeLaN
from cadelac.learning.data_scripts.replay_memory import PyTorchReplayMemory
from cadelac.learning.data_scripts.utils import init_env, load_dataset, plot_torques
from pathlib import Path


if __name__ == "__main__":

    # Read Command Line Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", nargs=1, type=int, required=False, default=[True, ], help="Training using CUDA.")
    parser.add_argument("-i", nargs=1, type=int, required=False, default=[0, ], help="Set the CUDA id.")
    parser.add_argument("-s", nargs=1, type=int, required=False, default=[0, ], help="Set the random seed")
    parser.add_argument("-r", nargs=1, type=int, required=False, default=[1, ], help="Render the figure")
    parser.add_argument("-l", nargs=1, type=int, required=False, default=[1, ], help="Load the DeLaN model")
    parser.add_argument("-m", nargs=1, type=int, required=False, default=[1, ], help="Save the DeLaN model")
    parser.add_argument("-f", nargs=1, type=int, required=False, default=[0, ], help="Learn full robot model")
    parser.add_argument("-e", "--encoder", type=str, required=False, default="lstm", 
                       help="History encoder type: 'lstm' (default), 'tcn', or 'none'")
    seed, cuda, render, load_model, save_model, full_model = init_env(parser.parse_args())
    
    # Get encoder type from args
    args = parser.parse_args()
    history_encoder = args.encoder.lower()
    if history_encoder not in ['lstm', 'tcn', 'none']:
        raise ValueError(f"Invalid encoder type: {history_encoder}. Must be 'lstm', 'tcn', or 'none'.")

    # Construct Hyperparameters:
    nn_id = "ContextAware" # Equivalent to DeLaN for hist_length = 0
    if nn_id == "ContextAware":
        nn_type = ContextAwareDeLaN

    # Read the dataset:
    dataset_use = 1.0
    minibatch = 1024
    loss_power = False

    n_dof = 7
    flag_normalize_tau = True
    sample_offset = 1
    save_checkpoint_model = True
    log_period = 50

    add_noise_to_load_data = True

    LEARNING_DIR = str(Path(__file__).resolve().parents[0])

    ## LSTM parameters
    hist_length = 15 if (nn_id == "ContextAware" and not full_model) else 0
    hist_labels = ['qp', 'qv', 'tau', 'diff_tau']
    n_lstm_input = n_dof * 3
    n_lstm_hidden = 10
    n_lstm_output = 10
    n_lstm_depth = 5

    if full_model == False:
        dataset_name = 'panda_mj_101_rand_envs_20_runs_50Hz_lqr'
    else:
        dataset_name = 'panda_mj_nominal_env_20_runs_50Hz_lqr'

    if full_model:
        test_label = []
        model_type_folder = 'full_model/panda/' + nn_id
    else:
        if hist_length > 0:
           test_label = []
        else:
            test_label = ['env_0_run_1','env_0_run_2']
        model_type_folder = 'res_model/panda/' + nn_id
    dataset_path = LEARNING_DIR + '/datasets/panda/' + dataset_name + '.pkl'

    train_data, test_data, divider, dt_mean = load_dataset(filename=dataset_path, test_label=test_label,
                                                        full_model=full_model, sample_offset=sample_offset,
                                                        dataset_use=dataset_use, 
                                                        n_dof=n_dof, hist_length=hist_length, 
                                                        hist_labels=hist_labels,
                                                        add_noise=add_noise_to_load_data)

    if hist_length == 0:
        train_labels, train_qp, train_qv, train_qa, train_tau = train_data
        test_labels, test_qp, test_qv, test_qa, test_tau, test_m, test_c, test_g = test_data
        n_enc_input = 1
        train_lstm_input = np.ones_like(train_qp)
        test_lstm_input = np.ones_like(test_qp)
    else:
        train_labels, train_qp, train_qv, train_qa, train_tau, \
                     tain_hist_qp, tain_hist_qv, tain_hist_tau, tain_hist_diff_tau_nom = train_data
        test_labels, test_qp, test_qv, test_qa, test_tau, test_m, test_c, test_g, \
                     test_hist_qp, test_hist_qv, test_hist_tau, test_hist_diff_tau_nom = test_data
        
        train_lstm_input = np.concatenate((tain_hist_qp, tain_hist_qv, tain_hist_diff_tau_nom), axis=-1)
        test_lstm_input = np.concatenate((test_hist_qp, test_hist_qv, test_hist_diff_tau_nom), axis=-1)
        n_enc_input = n_lstm_output


    print("\n\n################################################")
    print("Runs:")
    print("   Test Runs = {0}".format(test_labels))
    print("  Train Runs = {0}".format(train_labels))
    print("# Training Samples = {0:05d}".format(int(train_qp.shape[0])))
    print("")

    # Training Parameters:
    print("\n################################################")
    if full_model and not load_model:
        print("Training Deep Lagrangian Networks (DeLaN):")
    elif not load_model:
        print("Training Contextual DeLaN:")

    # Construct Hyperparameters:
    hyper = {
             'diagonal_epsilon': 0.1,
             'activation': 'Tanh',
             'net_arch_inertia': [30, 20],
             'net_arch_pot': [30, 20],
             'b_init': 1.e-4,
             'b_diag_init': 0.001,
             'w_init': 'xavier_normal',
             'gain_hidden': np.sqrt(2.),
             'gain_output': 0.1,
             'n_minibatch': minibatch,
             'learning_rate': 5.e-04,
             'weight_decay': 1.e-4,
             'init_tf': True,
             'n_enc_input': n_enc_input,
             'n_lstm_hidden': n_lstm_hidden,
             'n_lstm_input': n_lstm_input,
             'n_lstm_depth': n_lstm_depth,
             'hist_length': hist_length,
             'history_encoder': history_encoder,  # Store encoder type in hyper dict
             'act_ld': 'Softplus',
             'max_epoch': 1000
            }

    # Update model name to include encoder type
    model_name = 'epochs_' + str(hyper['max_epoch'])
    if add_noise_to_load_data:
        model_name += '_noise_'
    if hist_length > 0:
        model_name += f'_enc_{history_encoder}_'  # Include encoder type in model name
    model_name += dataset_name + '.torch'

    # Loading trained model for IROS2025
    if not full_model and load_model == 2:
        model_name = 'iros2025_epochs_3000_panda_mj_101_rand_envs_20_runs_50Hz_lqr.torch'

    # Load existing model parameters:
    if load_model:
        print(f'Loading model: {model_name}')
        load_file = LEARNING_DIR + f"/trained_models/{model_type_folder}/{model_name}"
        state = torch.load(load_file, map_location=torch.device('cpu'), weights_only=False)

        delan_model = nn_type(n_dof, **state['hyper'])
        delan_model.load_state_dict(state['state_dict'])
        delan_model = delan_model.cuda() if cuda else delan_model.cpu()

    else:
        # Construct DeLaN:
        delan_model = nn_type(n_dof, **hyper)
        delan_model = delan_model.cuda() if cuda else delan_model.cpu()

    # Generate & Initialize the Optimizer:
    optimizer = torch.optim.Adam(delan_model.parameters(),
                                 lr=hyper["learning_rate"],
                                 weight_decay=hyper["weight_decay"],
                                 amsgrad=False)

    # Generate Replay Memory:
    if not full_model:
        lstm_input_shape = (hist_length, n_lstm_input, ) if hist_length > 0 else (1, )
    else:
        train_lstm_input = train_lstm_input.reshape(train_qp.shape[0], -1)
        lstm_input_shape = (train_lstm_input.shape[-1],)

    mem_dim = ((n_dof, ), (n_dof, ), (n_dof, ), (n_dof, ), lstm_input_shape)
    mem = PyTorchReplayMemory(train_qp.shape[0], hyper["n_minibatch"], mem_dim, cuda)
    mem.add_samples([train_qp, train_qv, train_qa, train_tau, train_lstm_input])

    # Start Training Loop:
    t0_start = time.perf_counter()

    if flag_normalize_tau:
        norm_tau = torch.from_numpy(np.var(train_tau,axis=0))
    else:
        norm_tau = torch.ones(n_dof)
    norm_tau = norm_tau.cuda() if cuda else norm_tau.cpu()
    norm_tau_np = norm_tau.detach().cpu().numpy()

    init_param_dict = delan_model.state_dict()
    sum_param = 0.0
    lstm_param = 0.0
    for key in init_param_dict.keys():
        sum_param += init_param_dict[key].reshape(-1).shape[0]
        if 'lstm' in key:
            lstm_param += init_param_dict[key].reshape(-1).shape[0]
    print(f'Number of parameters {int(sum_param)} | LSTM {int(lstm_param)} | MLPs {int(sum_param - lstm_param)}')

    epoch_i = 0
    t_avg_epoch = 0.0
    while epoch_i < hyper['max_epoch'] and not load_model:
        l_mem_mean_inv_dyn, l_mem_var_inv_dyn = 0.0, 0.0
        l_mem_mean_dEdt, l_mem_var_dEdt = 0.0, 0.0
        l_mem, n_batches = 0.0, 0.0
        t0_epoch = time.perf_counter()

        if save_checkpoint_model:
            if epoch_i > 0 and (epoch_i % 500) == 0:
                print(f'Saving checkpoint model epoch: {epoch_i}')
                CHECKPOINT_DIR = LEARNING_DIR + f"/trained_models/{model_type_folder}/checkpoint/"
                if not os.path.exists(CHECKPOINT_DIR):
                    os.makedirs(CHECKPOINT_DIR)
                torch.save({"epoch": epoch_i,
                            "hyper": hyper,
                            "state_dict": delan_model.state_dict()},
                            CHECKPOINT_DIR + f"/{epoch_i}_{model_name}") 

        for q, qd, qdd, tau, lstm_input in mem:
            t0_batch = time.perf_counter()

            # Reset gradients:
            optimizer.zero_grad()

            # Compute the Rigid Body Dynamics Model:
            if hist_length == 0:
                tau_hat, dEdt_hat = delan_model(q, qd, qdd)
            else:
                tau_hat, dEdt_hat = delan_model(q, qd, qdd, lstm_input)

            # Compute the loss of the Euler-Lagrange Differential Equation:
            err_inv = torch.sum((tau_hat - tau) ** 2 / norm_tau, dim=1)
            l_mean_inv_dyn = torch.mean(err_inv)
            l_var_inv_dyn = torch.var(err_inv)

            # Compute the loss of the Power Conservation:
            dEdt = torch.matmul(qd.view(-1, n_dof, 1).transpose(dim0=1, dim1=2), tau.view(-1, n_dof, 1)).view(-1)
            err_dEdt = (dEdt_hat - dEdt) ** 2
            l_mean_dEdt = torch.mean(err_dEdt)
            l_var_dEdt = torch.var(err_dEdt)

            # Compute gradients & update the weights:
            if loss_power:
                loss = l_mean_inv_dyn + l_mean_dEdt
            else:
                loss = l_mean_inv_dyn
            loss.backward()
            optimizer.step()

            # Update internal data:
            n_batches += 1
            l_mem += loss.item()
            l_mem_mean_inv_dyn += l_mean_inv_dyn.item()
            l_mem_var_inv_dyn += l_var_inv_dyn.item()
            l_mem_mean_dEdt += l_mean_dEdt.item()
            l_mem_var_dEdt += l_var_dEdt.item()

            t_batch = time.perf_counter() - t0_batch

        # Update Epoch Loss & Computation Time:
        l_mem_mean_inv_dyn /= float(n_batches)
        l_mem_var_inv_dyn /= float(n_batches)
        l_mem_mean_dEdt /= float(n_batches)
        l_mem_var_dEdt /= float(n_batches)
        l_mem /= float(n_batches)
        epoch_i += 1

        t_epoch = time.perf_counter() - t0_epoch
        t_avg_epoch = t_avg_epoch + t_epoch

        if epoch_i == 1 or np.mod(epoch_i, log_period) == 0:
            print("Epoch {0:05d}: ".format(epoch_i), end=" ")
            print("Time = {0:05.1f}s".format(time.perf_counter() - t0_start), end=", ")
            print("Time/Epoch = {0:05.4f}s".format(t_avg_epoch / log_period), end=", ")
            print("Loss = {0:.3e}".format(l_mem), end=", ")
            print("Inv Dyn = {0:.3e} \u00B1 {1:.3e}".format(l_mem_mean_inv_dyn, 1.96 * np.sqrt(l_mem_var_inv_dyn)), end=", ")
            print("Power Con = {0:.3e} \u00B1 {1:.3e}".format(l_mem_mean_dEdt, 1.96 * np.sqrt(l_mem_var_dEdt)))
            t_avg_epoch = 0.0

    # Save the Model:
    if save_model and not load_model:
        folder_model = LEARNING_DIR + f"/trained_models/{model_type_folder}"
        if not os.path.isdir(folder_model):
            os.makedirs(folder_model)
        print(f'Saving model: {model_name}')
        torch.save({"epoch": epoch_i,
                    "hyper": hyper,
                    "state_dict": delan_model.state_dict()},
                    folder_model + f"/{model_name}")

    print("\n################################################")
    print("Evaluating DeLaN:")

    # Compute the inertial, centrifugal & gravitational torque using batched samples
    t0_batch = time.perf_counter()

    # Convert NumPy samples to torch:
    q = torch.from_numpy(test_qp).float().to(delan_model.device)
    qd = torch.from_numpy(test_qv).float().to(delan_model.device)
    qdd = torch.from_numpy(test_qa).float().to(delan_model.device)
    lstm_input = torch.from_numpy(test_lstm_input).float().to(delan_model.device)
    zeros = torch.zeros_like(q).float().to(delan_model.device)

    # Compute the torque decomposition:
    with torch.no_grad():
        if hist_length == 0:
            delan_g = delan_model.inv_dyn(q, zeros, zeros).cpu().numpy().squeeze()
            delan_c = delan_model.inv_dyn(q, qd, zeros).cpu().numpy().squeeze() - delan_g
            delan_m = delan_model.inv_dyn(q, zeros, qdd).cpu().numpy().squeeze() - delan_g
            delan_output = delan_model(q, qd, qdd)
        else:
            delan_g = delan_model.inv_dyn(q, zeros, zeros, lstm_input).cpu().numpy().squeeze()
            delan_c = delan_model.inv_dyn(q, qd, zeros, lstm_input).cpu().numpy().squeeze() - delan_g
            delan_m = delan_model.inv_dyn(q, zeros, qdd, lstm_input).cpu().numpy().squeeze() - delan_g
            delan_output = delan_model(q, qd, qdd, lstm_input)
        delan_tau = delan_output[0].cpu().numpy()
        delan_dEdt = delan_output[1].cpu().numpy()
    t_batch = (time.perf_counter() - t0_batch) / (3. * float(test_qp.shape[0]))

    # Compute Errors:
    test_dEdt = np.sum(test_tau * test_qv, axis=1)
    err_g = 1. / float(test_qp.shape[0]) * np.sum((delan_g - test_g) ** 2 / norm_tau_np)
    err_m = 1. / float(test_qp.shape[0]) * np.sum((delan_m - test_m) ** 2 / norm_tau_np)
    err_c = 1. / float(test_qp.shape[0]) * np.sum((delan_c - test_c) ** 2 / norm_tau_np)
    err_tau = 1. / float(test_qp.shape[0]) * np.sum((delan_tau - test_tau) ** 2 / norm_tau_np)
    err_dEdt = 1. / float(test_qp.shape[0]) * np.sum((delan_dEdt - test_dEdt) ** 2)

    print("\nPerformance:")
    print("                Torque MSE = {0:.3e}".format(err_tau))
    print("              Inertial MSE = {0:.3e}".format(err_m))
    print("Coriolis & Centrifugal MSE = {0:.3e}".format(err_c))
    print("         Gravitational MSE = {0:.3e}".format(err_g))
    print("    Power Conservation MSE = {0:.3e}".format(err_dEdt))
    print("      Comp Time per Sample = {0:.3e}s / {1:.1f}Hz".format(t_batch, 1./t_batch))

    fig_dir = str(LEARNING_DIR) + f"/figures/mpc_DeLaN_Performance/{model_type_folder}/{model_name}"
    plot_torques(test_tau, test_m, test_c, test_g, delan_tau, delan_m, delan_c, delan_g, test_labels, divider, fig_dir, render)