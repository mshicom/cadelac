import os
import mujoco

from cadelac.control.panda_sim import PandaSim, build_models
from cadelac.control.acados_mpc import AcadosMPC
from cadelac.control.casadi_model import RealtimeApprox
from cadelac.control.kf_state_force import KFStateFee
from cadelac.control.logger import Logger
from cadelac.control.pin_utils import *
from cadelac.control.reference_generator import ReferenceGenerator

import time
import copy
import numpy as np
import matplotlib.pyplot as plt

class PandaMPCSim(PandaSim):
    def __init__(self,
                tracking_mode = 'joint',
                use_viewer = False,
                random_init = False,
                expl_dyn = True,
                delan_model = None,
                realtime_approx = RealtimeApprox.NO_APPROX,
                sim_total_time = 100,
                RTI_mode = False,
                name_suffix = None,
                ref_type = 'Default',
                hist_length = 0,
                delan_model_inference = None,
                inference_suffix = None,
                sim_xml_handles = None,
                xml_handles = None,
            ):

        self.use_viewer = use_viewer
        self.tracking_mode = tracking_mode
        self.random_init = random_init
        self.expl_dyn = expl_dyn
        self.delan_model = delan_model
        self.realtime_approx = realtime_approx
        self.RTI_mode = RTI_mode
        self.ref_type = ref_type

        self.logger = Logger()

        self.rand_env = len(sim_xml_handles)
        # Get XML
        if (sim_xml_handles is None) or (xml_handles is None):
            self.rand_env = 100
            self.sim_xml_handles, self.xml_handles, self.original_xml = build_models(num_rand_envs=self.rand_env,
                                                                                    box_pos=None,
                                                                                    box_inertia_flag=True,
                                                                                    box_mass=None,
                                                                                    collision=False,
                                                                                    )
        else:
            self.rand_env = len(sim_xml_handles)
            self.sim_xml_handles = sim_xml_handles
            self.xml_handles = xml_handles

        if self.rand_env == 1:
            self.sim_xml_handles = [copy.copy(self.original_xml)]
            self.xml_handles = [copy.copy(self.original_xml)]

        self.env_id = 0
        # Setup Mujoco
        self.mj_model = mujoco.MjModel.from_xml_string(self.xml_handles[self.env_id].to_xml_string(), assets=self.xml_handles[self.env_id].get_assets())
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        mujoco.mj_step(self.mj_model, self.mj_data)

        self.sim_mj_model = mujoco.MjModel.from_xml_string(self.sim_xml_handles[self.env_id].to_xml_string(), assets=self.sim_xml_handles[self.env_id].get_assets())
        self.sim_mj_data = mujoco.MjData(self.sim_mj_model)
        mujoco.mj_resetDataKeyframe(self.sim_mj_model, self.sim_mj_data, 0)
        mujoco.mj_forward(self.sim_mj_model, self.sim_mj_data)

        # Get model ids
        self.joint_ids, self.ee_id, self.box_id = self.get_model_ids(self.mj_model, act_blacklist = ['actuator8'])
        self.nq = len(self.joint_ids)
        self.joint_ranges = self.get_joint_ranges(self.mj_model, self.joint_ids)

        self.ee_pos_home = self.get_ee_pos(self.sim_mj_data)
        self.q_home = self.get_joint_pos(self.sim_mj_data)

        self.last_time = time.time_ns()
        self.initial_time = time.time_ns()

        self.set_joint_properties()

        # MPC Parameters
        self.mpc_period = 0.02
        self.mpc_freq = 1.0 / self.mpc_period
        self.N_horizon = 12
        self.T_horizon = self.N_horizon * self.mpc_period
        self.sim_length = sim_total_time / self.joint_ctrl_period
        self.sim_total_time = float(sim_total_time)

        # Setup MPC
        self.mpc = AcadosMPC(N_horizon = self.N_horizon,
                       T_horizon = self.T_horizon,
                       expl_dyn = self.expl_dyn,
                       delan_model = self.delan_model,
                       realtime_approx = realtime_approx,
                       name_suffix='_' + str(os.getpid()) if name_suffix is None else name_suffix,
                       RTI_mode = RTI_mode,
                    )
        self.nx = self.mpc.nx
        self.nu = self.mpc.nu
        self.pin_ee_id = self.mpc.robot_model.model_pin.getFrameId('end_effector')

        # Create Casadi model to get nominal model using pinocchio
        self.cs_robot_model = self.mpc.robot_model

        # Create MPC only for inference of Delan
        self.inferece_delan = False
        self.delan_model_inference = delan_model_inference
        if self.delan_model_inference is not None:
            self.inferece_delan = True
            self.inference_mpc = AcadosMPC(N_horizon = self.N_horizon,
                       T_horizon = self.T_horizon,
                       expl_dyn = self.expl_dyn,
                       delan_model = self.delan_model_inference,
                       realtime_approx = realtime_approx,
                       name_suffix='_' + str(os.getpid()) if inference_suffix is None else inference_suffix,
                       RTI_mode = RTI_mode,
                    )
            self.inference_delan_cs_robot_model = self.inference_mpc.robot_model

        # Historical Data
        self.hist_length = hist_length

        # Init data
        self.init_exc_ref_ok = True
        self.viewer = None
        self.reset(seed=0)

    def get_mpc_state(self, state_sim):
        q = self.get_joint_pos(state_sim)
        qd = self.get_joint_vel(state_sim)
        x_mpc = np.concatenate((qd, q))
        return x_mpc

    def generate_mpc_ref(self, n_steps):
        current_time = n_steps * self.mpc_period
        q_ref = np.zeros((self.nq, self.N_horizon+1))
        qd_ref = np.zeros((self.nq, self.N_horizon+1))

        for i in range(self.N_horizon+1):
            ref_time = current_time + i*self.mpc_period
            q_single, qd_single = self.generate_joint_sin_ref(ref_time)
            q_ref[:,i] = q_single
            qd_ref[:,i] = qd_single

        x_ref = np.vstack((qd_ref,
                           q_ref))
        return x_ref, q_ref, qd_ref
    
    def get_mpc_ref(self, time_index):

        if self.ref_type == 'EXC':
            q_ref = np.hstack((
                self.exc_q_traj[:, time_index:min(time_index + self.N_horizon + 1, self.exc_q_traj.shape[1])],
                self.exc_q_traj[:, :max(0, time_index + self.N_horizon + 1 - self.exc_q_traj.shape[1])]
            ))

            qd_ref = np.hstack((
                self.exc_qd_traj[:, time_index:min(time_index + self.N_horizon + 1, self.exc_qd_traj.shape[1])],
                self.exc_qd_traj[:, :max(0, time_index + self.N_horizon + 1 - self.exc_qd_traj.shape[1])]
            ))

        elif self.ref_type == 'FULL_INF':
            q_ref = np.hstack((
                self.full_inf_q_traj[:, time_index:min(time_index + self.N_horizon + 1, self.full_inf_q_traj.shape[1])],
                self.full_inf_q_traj[:, :max(0, time_index + self.N_horizon + 1 - self.full_inf_q_traj.shape[1])]
            ))

            qd_ref = np.hstack((
                self.full_inf_qd_traj[:, time_index:min(time_index + self.N_horizon + 1, self.full_inf_qd_traj.shape[1])],
                self.full_inf_qd_traj[:, :max(0, time_index + self.N_horizon + 1 - self.full_inf_qd_traj.shape[1])]
            ))

        elif self.ref_type == 'PICK_AND_PLACE':
            q_ref = np.hstack((
                self.full_pp_q_traj[:, time_index:min(time_index + self.N_horizon + 1, self.full_pp_q_traj.shape[1])],
                self.full_pp_q_traj[:, :max(0, time_index + self.N_horizon + 1 - self.full_pp_q_traj.shape[1])]
            ))

            qd_ref = np.hstack((
                self.full_pp_qd_traj[:, time_index:min(time_index + self.N_horizon + 1, self.full_pp_qd_traj.shape[1])],
                self.full_pp_qd_traj[:, :max(0, time_index + self.N_horizon + 1 - self.full_pp_qd_traj.shape[1])]
            ))

        else:
            q_ref = self.q_ref_traj[:,time_index:(time_index+self.N_horizon+1)]
            qd_ref = self.qd_ref_traj[:,time_index:(time_index+self.N_horizon+1)]

        x_ref = np.vstack((qd_ref,
                           q_ref))
        return x_ref, q_ref, qd_ref

    ## Kalman Filter
    def update_state_kf(self, q, qd, tau):
        self.q_old_kf = q
        self.qd_old_kf = qd
        self.tau_old_kf = tau

    def init_kf_fee(self, q_init, qd_init, tau_init):
        self.kf_state_Fee = KFStateFee(self.mpc.robot_model.model_pin,
                                     self.mpc.robot_model.data_pin,
                                     self.joint_ctrl_period,
                                     q_init,
                                     qd_init,
                                     Fee_init=None,
                                     u_init=tau_init,
                                     )
        
        self.update_state_kf(np.copy(q_init), np.copy(qd_init), np.copy(tau_init))

    def log_kf_state_data(self, fext_kf, tau_kf):
        self.log_new_value('fext_kf_state', np.copy(fext_kf))
        self.log_new_value('tau_kf_state', np.copy(tau_kf))

    def step(self):

        ## Update MPC State
        x_mpc = self.get_mpc_state(self.sim_mj_data)
        enc_input = self.get_enc_input()
        self.enc_input = np.copy(enc_input)

        # Generate Reference
        init_ref_time = time.time()
        y_ref, self.q_pos_ref_horizon, self.q_vel_ref_horizon = self.get_mpc_ref(self.sim_steps)
        self.time_ref += time.time() - init_ref_time

        # Log current state
        init_log_time = time.time()
        self.log_new_data(self.sim_steps * self.joint_ctrl_period, qp_ref = self.q_pos_ref_horizon[:,0], qv_ref = self.q_vel_ref_horizon[:,0])
        self.time_log += time.time() - init_log_time

        # Update MPC reference
        torque_ref = np.zeros((self.mpc.nu, self.N_horizon))
        self.mpc.set_mpc_ref(y_ref, torque_ref)

        init_param_comp = time.time()

        ## Update Kalman Filter
        noise = 0*np.random.normal(0.0, 0.1, (x_mpc[:self.nq,]).shape)
        fee_est_state, tau_kf_state = self.kf_state_Fee.update(x_mpc[self.nq:,], x_mpc[:self.nq,]+noise, self.tau_old_kf)
        self.log_kf_state_data(fee_est_state, tau_kf_state)

        if self.delan_model == 'KF':
            param = tau_kf_state.reshape((-1,1))
        elif self.delan_model is not None:
            param = self.compute_delan_param(y_ref, self.sim_steps, self.solver_status, enc_input)
        self.time_param += time.time() - init_param_comp

        # # Update MPC Parameters
        if self.mpc.model.p.shape[0] > 0.0:
            self.mpc.set_mpc_param(param)

        # MPC Control
        self.mpc.iterate_solver_warm_start()
        start_solver_time = time.time()
        u_mpc = self.mpc.solver.solve_for_x0(x_mpc, False, False)
        solver_total_time = time.time() - start_solver_time
        self.time_solver += solver_total_time
        self.solver_status = self.mpc.solver.get_status()

        ## Update Logger
        qt = self.get_joint_pos(self.sim_mj_data)
        qdt = self.get_joint_vel(self.sim_mj_data)
        mj_pos = self.get_ee_pos(self.sim_mj_data)
        pin_pos = pin_get_link_pos(self.mpc.robot_model.model_pin,
                                    self.mpc.robot_model.data_pin,
                                    qt, self.pin_ee_id)
        pin_jac = pin_get_frame_jacobian(self.mpc.robot_model.model_pin,
                                self.mpc.robot_model.data_pin,
                                qt, self.pin_ee_id)
        jac_lin = np.zeros((3,7))
        mujoco.mj_jac(self.sim_mj_model, self.sim_mj_data, jac_lin, None, mj_pos, self.ee_id)
        mj_vel = self.get_ee_vel(self.sim_mj_data)
        pin_vel = pin_jac @ qdt

        if self.ref_type == 'FULL_INF':
            ee_pos_ref = self.full_inf_ee_pos_traj[:,self.sim_steps]
            ee_vel_ref = self.full_inf_ee_vel_traj[:,self.sim_steps]
        elif self.ref_type == 'PICK_AND_PLACE':
            ee_pos_ref = self.full_pp_ee_pos_traj[:,self.sim_steps]
            ee_vel_ref = self.full_pp_ee_vel_traj[:,self.sim_steps]
        else:
            ee_pos_ref = np.zeros(3)
            ee_vel_ref = np.zeros(3)

        u_horizon, xpred_horizon = self.mpc.get_mpc_solution_full_horizon()
        hist_data = np.concatenate((self.hist_q, self.hist_qd, self.hist_diff_tau_nom), axis=-1)
        self.logger.log_sim_data(time = self.sim_steps * self.joint_ctrl_period,
                                 q = self.get_joint_pos(self.sim_mj_data), qd = self.get_joint_vel(self.sim_mj_data),
                                 qdd = self.get_joint_acc(self.sim_mj_data), tau_ctrl = u_mpc,
                                 ee_pos = self.get_ee_pos(self.sim_mj_data), ee_vel = self.get_ee_vel(self.sim_mj_data),
                                 tau_kf = tau_kf_state, fext_kf = fee_est_state,
                                 u_horizon = u_horizon, xpred_horizon = xpred_horizon,
                                 env_id = self.env_id , enc_input = enc_input,
                                 mj_model = self.mj_model, mj_data = self.mj_data,
                                 sim_mj_model = self.sim_mj_model, sim_mj_data = self.sim_mj_data,
                                 model_pin = self.mpc.robot_model.model_pin, data_pin = self.mpc.robot_model.data_pin,
                                 mpc_cost = self.mpc.solver.get_cost(), mpc_times = self.mpc.get_solver_times(),
                                 qp_ref = self.q_pos_ref_horizon[:,0], qv_ref = self.q_vel_ref_horizon[:,0],
                                 ee_pos_ref = ee_pos_ref, ee_vel_ref = ee_vel_ref, hist_data=hist_data,
                                 )
        
        # Delan inference
        if self.inferece_delan:
            q_eval_delan = self.get_joint_pos(self.sim_mj_data).reshape((-1,1))
            qd_eval_delan = self.get_joint_vel(self.sim_mj_data).reshape((-1,1))
            qdd_eval_delan = self.get_joint_acc(self.sim_mj_data).reshape((-1,1))

            enc_input_eval_delan = self.delan_model_inference.eval_lstm_np(hist_data).reshape((-1,1))

            H_delan, tau_delan_c, tau_delan_g = self.inference_delan_cs_robot_model.delan_output_fn(q_eval_delan, qd_eval_delan, enc_input_eval_delan)

            delan_tau_diff = H_delan @ qdd_eval_delan + tau_delan_c + tau_delan_g

            self.logger.log_delan_data(np.array(delan_tau_diff))

        # Save data for historical data
        q_old = self.get_joint_pos(self.sim_mj_data)
        qd_old = self.get_joint_vel(self.sim_mj_data)

        # Save data for Kalman Filter
        self.update_state_kf(np.copy(q_old), np.copy(qd_old), np.copy(u_mpc))

        # Update simulation
        torque = u_mpc
        start_mj_sim_time = time.time()
        self.update_mj_sim(torque)
        self.time_mj_sim += time.time() - start_mj_sim_time

        ## Updated predicted data and hist data
        qdd_new = self.get_joint_acc(self.sim_mj_data)
        if self.hist_length > 0:
            self.update_hist_data(q_old, qd_old, qdd_new, torque)

        param_mpc_dyn = param if self.mpc.model.p.shape[0] > 0.0 else None
        self.log_pred_error()

        self.sim_steps += 1
        if self.viewer is not None:
            self.update_viewer()

    def log_pred_error(self):
        new_x_mpc = self.get_mpc_state(self.sim_mj_data)

        # Get prediction from previous opt
        u_horizon, xpred_horizon = self.mpc.get_mpc_solution_full_horizon()
        u_horizon = u_horizon.reshape(-1)
        xpred_next = xpred_horizon[:,1]
        xpred_horizon = xpred_horizon.reshape(-1)

        xpred_error = new_x_mpc - xpred_next
        self.log_new_value('xpred_next', xpred_next)
        self.log_new_value('xpred_error', xpred_error)
        self.log_new_value('xpred_horizon', xpred_horizon)
        self.log_new_value('u_horizon', u_horizon)

    def log_torque_mpc_dynamics(self, q_old, qd_old, qdd, tau, param_old = None):
        # tau_mpc_delan = tau
        x_dot_eval = np.concatenate((qdd, qd_old))
        x = np.concatenate((qd_old, q_old))
        full_dyn = self.mpc.robot_model.imp_forward_dynamics(x_dot_eval, x, tau, param_old)
        tau_mpc = np.array(full_dyn[:self.nq]).reshape(-1) + tau
        diff_tau_mpc = tau - tau_mpc
        self.log_new_value('tau_mpc', tau_mpc)
        self.log_new_value('diff_tau_mpc', diff_tau_mpc)


    def init_full_infinity_ref(self, ee_init, q0 = None):
        freq_traj = 0.25
        
        inf_params = {}
        inf_params['amp'] = np.array([0, 0.0, 0.1])
        inf_params['theta'] = 0/180*np.pi
        inf_params['freq'] = freq_traj
        inf_params['center_pos'] = ee_init
        inf_params['Ts'] = self.mpc_period

        Nrepeat = int(np.ceil(self.sim_total_time / (1.0 / freq_traj)))
        
        time_traj, q_traj, qd_traj, pos_traj, vel_traj = self.ref_gen.generate_full_infinity_joint(model_pin = self.mpc.robot_model.model_pin,
                                                                                                   data_pin = self.mpc.robot_model.data_pin,
                                                                                                   params = inf_params,
                                                                                                   q0 = q0,
                                                                                                   return_ee_traj = True,
                                                                                                   Nrepeat = Nrepeat)

        self.full_inf_time_traj = np.arange(self.sim_length + self.N_horizon + 10) * self.mpc_period
        self.full_inf_q_traj = q_traj.T
        self.full_inf_qd_traj = qd_traj.T
        self.full_inf_ee_pos_traj = pos_traj.T
        self.full_inf_ee_vel_traj = vel_traj.T

        compute_qd_using_diff = True
        if compute_qd_using_diff:
            qd_diff_traj = []
            traj_length = self.full_inf_q_traj.shape[1]

            for i in range(traj_length-1):
                qd_new = (self.full_inf_q_traj[:,i+1] - self.full_inf_q_traj[:,i]) / self.mpc_period
                qd_diff_traj.append(qd_new)
            qd_diff_traj.append(qd_new) # Hold last one
            self.full_inf_qd_traj = np.array(qd_diff_traj).T

    def init_pick_and_place_ref(self, ee_init, q0 = None):
        
        pp_params = {}
        # pp_params['amp'] = np.array([0, 0.5, 0.3])
        # pp_params['amp'] = np.array([0, 0.35, 0.4])
        # pp_params['freq'] = 0.2

        pp_params['amp'] = np.array([0, 0.35, 0.35])
        pp_params['freq'] = 0.2
        pp_params['theta'] = 90/180*np.pi
        pp_params['init_pos'] = ee_init
        pp_params['center_pos'] = ee_init + np.array([0.1, -0.15, -0.05])
        pp_params['Ts'] = self.mpc_period
        pp_params['phase_offset'] = 45/180*np.pi
        pp_params['grasp_time'] = 2
        pp_params['spline_height'] = 0.2
        pp_params['spline_time'] = 1.5
        pp_params['spline_x_diff'] = 0.0
        pp_params['spline_y_diff'] = 0.30

        full_time_traj, pp_q_traj, pp_qd_traj, pp_ee_pos_traj, pp_ee_vel_traj = self.ref_gen.compute_pick_and_place_traj(model_pin = self.mpc.robot_model.model_pin,
                                                                                                   data_pin = self.mpc.robot_model.data_pin,
                                                                                                   full_param = pp_params, q0 = q0)

        self.full_pp_time_traj = np.arange(self.sim_length + self.N_horizon + 10) * self.mpc_period
        self.full_pp_q_traj = pp_q_traj.T
        self.full_pp_qd_traj = pp_qd_traj.T
        self.full_pp_ee_pos_traj = pp_ee_pos_traj.T
        self.full_pp_ee_vel_traj = pp_ee_vel_traj.T

        compute_qd_using_diff = True
        if compute_qd_using_diff:
            qd_diff_traj = []
            traj_length = self.full_pp_q_traj.shape[1]
            for i in range(traj_length-1):
                qd_new = (self.full_pp_q_traj[:,i+1] - self.full_pp_q_traj[:,i]) / self.mpc_period
                qd_diff_traj.append(qd_new)
            qd_diff_traj.append(qd_new) # Hold last one
            self.full_pp_qd_traj = np.array(qd_diff_traj).T

    def init_mpc_sin_ref(self):
        self.q_sin_ref_amp = np.random.uniform(low=np.zeros((self.nq,1)), high=np.array([[1.5, 1.0, 1.5, 1.0, 1.5, 1.0, 1.5]]).T, size=(self.nq,1))
        self.q_sin_ref_freq = np.random.uniform(low=np.zeros((self.nq,1)), high=0.15*np.ones((self.nq,1)), size=(self.nq,1))
        self.q_sin_ref_phase = np.zeros_like(self.q_sin_ref_amp)
        self.q_sin_ref_offset = np.copy(self.q_init).reshape((self.nq,1))

    def generate_reference(self):
        self.time_traj = np.arange(self.sim_length+self.N_horizon+10)*self.mpc_period
        self.q_ref_traj = self.q_sin_ref_amp * np.sin(2 * np.pi * self.q_sin_ref_freq * self.time_traj + self.q_sin_ref_phase) + self.q_sin_ref_offset
        self.qd_ref_traj = 2 * np.pi * self.q_sin_ref_freq * self.q_sin_ref_amp * np.cos(2 * np.pi * self.q_sin_ref_freq * self.time_traj + self.q_sin_ref_phase)

    def init_hist_data(self):
        self.hist_q = np.tile(self.q_init, (self.hist_length, 1))
        self.hist_qd = np.zeros((self.hist_length, self.nq))
        self.hist_diff_tau_nom = np.zeros((self.hist_length, self.nq))

    def update_hist_data(self, q_old, qd_old, qdd, tau):
        # Roll data
        self.hist_q = np.roll(self.hist_q, -1, axis=0)
        self.hist_qd = np.roll(self.hist_qd, -1, axis=0)
        self.hist_diff_tau_nom = np.roll(self.hist_diff_tau_nom, -1, axis=0)

        # Get torque from nominal model
        tau_nom = self.mpc.robot_model.tau_nom_inv_dyn_fn(q_old, qd_old, qdd)
        tau_nom = np.array(tau_nom).squeeze()
        diff_nom_torque = tau - tau_nom

        # Update data
        self.hist_q[-1,:] = q_old
        self.hist_qd[-1,:] = qd_old
        self.hist_diff_tau_nom[-1,:] = diff_nom_torque

    def reset(self, run_name = 'run', seed=None, env_id = None):
        if seed is not None:
            np.random.seed(seed)
            
        if env_id is not None:
            self.env_id = env_id
            self.mj_model = mujoco.MjModel.from_xml_string(self.xml_handles[self.env_id].to_xml_string(), assets=self.xml_handles[self.env_id].get_assets())
            self.mj_data = mujoco.MjData(self.mj_model)
            self.sim_mj_model = mujoco.MjModel.from_xml_string(self.sim_xml_handles[self.env_id].to_xml_string(), assets=self.sim_xml_handles[self.env_id].get_assets())
            self.sim_mj_data = mujoco.MjData(self.sim_mj_model)

        elif self.rand_env > 1:
            self.env_id = np.random.randint(low=0, high=self.rand_env)
            self.mj_model = mujoco.MjModel.from_xml_string(self.xml_handles[self.env_id].to_xml_string(), assets=self.xml_handles[self.env_id].get_assets())
            self.mj_data = mujoco.MjData(self.mj_model)
            self.sim_mj_model = mujoco.MjModel.from_xml_string(self.sim_xml_handles[self.env_id].to_xml_string(), assets=self.sim_xml_handles[self.env_id].get_assets())
            self.sim_mj_data = mujoco.MjData(self.sim_mj_model)

        # Set initial pos
        if self.random_init:
            self.q_home = np.array([0, -0.4, 0, -2.4, 0, 2.2, -0.7853])
            self.q_init = np.copy(self.q_home) + np.random.uniform(
                low=-0.1, high=0.1, size=self.nq
            )
            self.q_init = np.clip(np.copy(self.q_init), self.joint_ranges[:,0], self.joint_ranges[:,1])
        else:
            self.q_home = np.array([0, -0.4, 0, -2.4, 0, 2.2, -0.7853])
            self.q_init = np.copy(self.q_home)
        self.mj_data.qpos[:] = np.copy(self.q_init)
        self.sim_mj_data.qpos[:] = np.copy(self.q_init)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        mujoco.mj_forward(self.sim_mj_model, self.sim_mj_data)

        # Init reference data
        self.init_logger(run_name)
        self.log_init_value('tau_mpc')
        self.log_init_value('diff_tau_mpc')
        self.log_init_value('xpred_next')
        self.log_init_value('xpred_error')
        self.log_init_value('xpred_horizon')
        self.log_init_value('u_horizon')

        init_xmpc = np.concatenate((np.zeros(self.nq), np.copy(self.q_init)))
        self.logger.reset_logger(run_name, init_xmpc)
        self.logger.set_np_ignored_keys(['run_name', 'labels', 'env_id'])


        # self.ref_type = 'Default'
        # self.ref_type = 'EXC'

        if self.ref_type == 'EXC':
            self.ref_gen = ReferenceGenerator()
            duration = 10
            self.exc_fourier_config, self.exc_robot_config = self.ref_gen.exc_ref_init_config(duration)

            run_fourier_config = copy.deepcopy(self.exc_fourier_config)
            run_robot_config = copy.deepcopy(self.exc_robot_config)
            init_pos = np.copy(self.q_init)
            init_vel = np.zeros_like(self.q_init)

            self.init_exc_ref(run_fourier_config,
                              run_robot_config,
                              init_pos,
                              init_vel)
        elif self.ref_type == 'FULL_INF':
            self.ref_gen = ReferenceGenerator()

            inf_q_init = np.copy(self.q_init)
            inf_ee_pos_init = np.copy(self.get_ee_pos(self.sim_mj_data))
            self.init_full_infinity_ref(inf_ee_pos_init, inf_q_init)

        elif self.ref_type == 'PICK_AND_PLACE':
            self.ref_gen = ReferenceGenerator()

            pp_q_init = np.copy(self.q_init)
            pp_ee_pos_init = np.copy(self.get_ee_pos(self.sim_mj_data))
            self.init_pick_and_place_ref(pp_ee_pos_init, pp_q_init)

        else:
            self.init_mpc_sin_ref()
            self.generate_reference()

        self.solver_status = 0
        self.time_solver = 0
        self.time_param = 0
        self.time_mj_sim = 0
        self.time_ref = 0
        self.time_log = 0
        self.time_nn_eval = 0

        if self.hist_length > 0:
            self.init_hist_data()

        # Init KF
        tf_q_init = np.copy(self.q_init)
        tf_qd_init = np.zeros_like(tf_q_init)
        tf_tau_init = np.array(pin.nonLinearEffects(self.mpc.robot_model.model_pin, self.mpc.robot_model.data_pin, tf_q_init, tf_qd_init))
        self.init_kf_fee(np.copy(tf_q_init), tf_qd_init, tf_tau_init)

        if self.use_viewer:
            if self.viewer:
                self.viewer.close()
            self.viewer = mujoco.viewer.launch_passive(self.sim_mj_model, self.sim_mj_data)
        else:
            self.viewer = None

        # Initialize solver
        x0 = self.get_mpc_state(self.sim_mj_data)
        u0 = np.zeros((self.nu,))
        self.mpc.init_solver(x0, u0)

        self.sim_steps = 0


    def compute_delan_param(self, yref, n_steps, solver_status, enc_input = np.zeros(1)):
        qpos = self.get_joint_pos(self.sim_mj_data)
        q_horizon = [qpos]

        # Use reference for the first iteration or previous unfesiable solution
        if n_steps == 0 or (solver_status not in [0, 2]):
            for j in range(1,self.N_horizon):
                q_horizon.append(yref[self.nq:,j])
        else:
            for j in range(2,self.N_horizon+1):
                q_horizon.append(self.mpc.solver.get(j, "x")[self.nq:])
        q_horizon = np.array(q_horizon)

        if enc_input.shape[0] > 1:
            enc_input_horizon = np.tile(enc_input, (self.N_horizon, 1))
        else:
            enc_input_horizon = None

        if self.realtime_approx == RealtimeApprox.EXTERNAL_EVAL:
            init_eval = time.time()
            l_tay, dldq_tay, V_tay, dVdq_tay = self.delan_model.get_jac_value_inertia_torch(q_horizon, enc_input_horizon)
            self.time_nn_eval += time.time() - init_eval

            l_tay_np = l_tay.cpu().detach().numpy()
            dldq_tay_np = dldq_tay.cpu().detach().numpy()
            V_tay_np = V_tay.cpu().detach().numpy()
            dVdq_tay_np = dVdq_tay.cpu().detach().numpy()

            param = q_horizon
            param = np.hstack((param, l_tay_np.reshape((self.N_horizon,-1), order='F')))
            param = np.hstack((param, dldq_tay_np.reshape((self.N_horizon,-1), order='F')))
            param = np.hstack((param, V_tay_np.reshape((self.N_horizon,-1), order='F')))
            param = np.hstack((param, dVdq_tay_np.reshape((self.N_horizon,-1), order='F')))

        else:
            param = enc_input_horizon

        return param

    def get_enc_input(self):
        if self.delan_model is None or self.delan_model == 'KF':
            enc_input = np.zeros(1)
        else:
            if self.hist_length == 0:
                enc_input = np.zeros(self.delan_model.n_enc_input)
                enc_input[self.env_id] = 1
            else:
                # Call lstm
                lstm_input = np.concatenate((self.hist_q, self.hist_qd, self.hist_diff_tau_nom), axis=-1)
                enc_input = self.delan_model.eval_lstm_np(lstm_input)
        return enc_input

    def plot_kf_est(self, logged_data = None):
        if logged_data is None:
            logged_data = self.logged_data
        time_traj = logged_data['t']
        fext_kf = logged_data['fext_kf']
        fext_kf_v0 = logged_data['fext_kf_v0']

        fig, axes = plt.subplots(3, 1)
        for cor in range(3):
            axes[cor].plot(time_traj, fext_kf[:,cor], label=f'Est')
            axes[cor].plot(time_traj, fext_kf_v0[:,cor], label=f'Est - v0')
            axes[cor].grid()
            if cor == 0:
                axes[cor].legend(loc="upper right")
        axes[0].set_title(f'Force Est KF')

    def plot_kf_state_est(self, logged_data = None):
        if logged_data is None:
            logged_data = self.logged_data
        time_traj = logged_data['t']
        fext_kf_state = logged_data['fext_kf_state']

        ncor = fext_kf_state.shape[1]
        fig, axes = plt.subplots(ncor, 1)
        for cor in range(ncor):
            axes[cor].plot(time_traj, fext_kf_state[:,cor], label=f'Est - State')
            axes[cor].grid()
            if cor == 0:
                axes[cor].legend(loc="upper right")
        axes[0].set_title(f'Force Est KF - Fee State')

    def plot_ee_pos(self, logged_data = None):
        if logged_data is None:
            logged_data = self.logger.logged_data
        time_traj = logged_data['t']
        ee_pos = logged_data['ee_pos']
        ee_pos_ref = logged_data['ee_pos_ref']

        labels = ['x', 'y', 'z']
        fig, axes = plt.subplots(3, 1)
        for cor in range(3):
            axes[cor].plot(time_traj, ee_pos[:,cor], label=f'Real - {labels[cor]}')
            axes[cor].plot(time_traj, ee_pos_ref[:,cor], '--r', label=f'Ref - {labels[cor]}')
            axes[cor].grid()
            if cor == 0:
                axes[cor].legend(loc="upper right")
        axes[0].set_title(f'End-effector Position')

        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(ee_pos[:,0], ee_pos[:,1], ee_pos[:,2], label='Figure-Eight Path')
        ax.plot(ee_pos_ref[0,0], ee_pos_ref[0,1], ee_pos_ref[0,2], '*', label='r0')
        ax.plot(ee_pos_ref[-1,0], ee_pos_ref[-1,1], ee_pos_ref[-1,2], '*', label='r1')
        ax.plot(ee_pos_ref[:,0], ee_pos_ref[:,1], ee_pos_ref[:,2], '--r', label='Ref')
        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([0.0, 0.9])
        ax.set_ylim([-0.25, 0.25])
        ax.set_zlim([0.0, 1.0])
        ax.set_title('End-Effector Trajectory - Full Infinity')
        ax.legend()

    def plot_ee_vel(self, logged_data = None):
        if logged_data is None:
            logged_data = self.logger.logged_data
        time_traj = logged_data['t']
        ee_vel = logged_data['ee_vel']
        ee_vel_ref = logged_data['ee_vel_ref']

        labels = ['x', 'y', 'z']
        fig, axes = plt.subplots(3, 1)
        for cor in range(3):
            axes[cor].plot(time_traj, ee_vel[:,cor], label=f'Real - {labels[cor]}')
            axes[cor].plot(time_traj, ee_vel_ref[:,cor], '--r', label=f'Ref - {labels[cor]}')
            axes[cor].grid()
            if cor == 0:
                axes[cor].legend(loc="upper right")
        axes[0].set_title(f'End-effector Velocity')

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run MPC simulation with optional side-by-side comparison')
    parser.add_argument('--compare', action='store_true', help='Enable side-by-side comparison mode')
    parser.add_argument('--lstm-model', type=str, default=None, help='Path to LSTM model checkpoint')
    parser.add_argument('--tcn-model', type=str, default=None, help='Path to TCN model checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--sim-time', type=float, default=10.0, help='Simulation time in seconds')
    parser.add_argument('--use-viewer', action='store_true', help='Enable MuJoCo viewer')
    args = parser.parse_args()
    
    if args.compare:
        # Side-by-side comparison mode
        if args.lstm_model is None or args.tcn_model is None:
            raise ValueError("Both --lstm-model and --tcn-model must be provided for comparison mode")
        
        import torch
        from cadelac.control.l4c_context_aware_delan import L4CContextAwareDeLaN
        
        # Load models
        print("Loading LSTM model...")
        lstm_checkpoint = torch.load(args.lstm_model, map_location='cpu', weights_only=False)
        lstm_delan = L4CContextAwareDeLaN(lstm_checkpoint, n_dof=7, n_enc_input=lstm_checkpoint['hyper']['n_enc_input'])
        
        print("Loading TCN model...")
        tcn_checkpoint = torch.load(args.tcn_model, map_location='cpu', weights_only=False)
        tcn_delan = L4CContextAwareDeLaN(tcn_checkpoint, n_dof=7, n_enc_input=tcn_checkpoint['hyper']['n_enc_input'])
        
        # Get hist_length from models
        lstm_hist_length = lstm_checkpoint['hyper'].get('hist_length', 0)
        tcn_hist_length = tcn_checkpoint['hyper'].get('hist_length', 0)
        
        # Run LSTM simulation
        print("\n" + "="*60)
        print("Running LSTM Model Simulation")
        print("="*60)
        panda_mpc_lstm = PandaMPCSim(
            use_viewer=args.use_viewer,
            expl_dyn=True,
            delan_model=lstm_delan,
            sim_total_time=args.sim_time,
            name_suffix='_lstm_comparison',
            hist_length=lstm_hist_length,
        )
        panda_mpc_lstm.reset(seed=args.seed)
        
        Nsim = int(args.sim_time / panda_mpc_lstm.joint_ctrl_period)
        init_time_lstm = time.time()
        for i in range(Nsim):
            panda_mpc_lstm.step()
        total_time_lstm = time.time() - init_time_lstm
        
        panda_mpc_lstm.convert_log_data_to_np()
        lstm_rms_error = np.sqrt(np.mean(np.square(panda_mpc_lstm.logged_data['qp_ref']-panda_mpc_lstm.logged_data['qp']),axis=0))
        lstm_avg_solver_time = panda_mpc_lstm.time_solver / Nsim
        
        # Run TCN simulation
        print("\n" + "="*60)
        print("Running TCN Model Simulation")
        print("="*60)
        panda_mpc_tcn = PandaMPCSim(
            use_viewer=args.use_viewer,
            expl_dyn=True,
            delan_model=tcn_delan,
            sim_total_time=args.sim_time,
            name_suffix='_tcn_comparison',
            hist_length=tcn_hist_length,
        )
        panda_mpc_tcn.reset(seed=args.seed)
        
        init_time_tcn = time.time()
        for i in range(Nsim):
            panda_mpc_tcn.step()
        total_time_tcn = time.time() - init_time_tcn
        
        panda_mpc_tcn.convert_log_data_to_np()
        tcn_rms_error = np.sqrt(np.mean(np.square(panda_mpc_tcn.logged_data['qp_ref']-panda_mpc_tcn.logged_data['qp']),axis=0))
        tcn_avg_solver_time = panda_mpc_tcn.time_solver / Nsim
        
        # Print comparison summary
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"\nSimulation Time: {args.sim_time}s")
        print(f"Random Seed: {args.seed}")
        print(f"\nLSTM Model:")
        print(f"  Total Time: {total_time_lstm:.3f}s")
        print(f"  Avg Step Time: {total_time_lstm/Nsim:.4f}s")
        print(f"  Avg Solver Time: {lstm_avg_solver_time:.4f}s")
        print(f"  RMS Joint Tracking Error: {lstm_rms_error}")
        print(f"  Mean RMS Error: {np.mean(lstm_rms_error):.6f} rad")
        
        print(f"\nTCN Model:")
        print(f"  Total Time: {total_time_tcn:.3f}s")
        print(f"  Avg Step Time: {total_time_tcn/Nsim:.4f}s")
        print(f"  Avg Solver Time: {tcn_avg_solver_time:.4f}s")
        print(f"  RMS Joint Tracking Error: {tcn_rms_error}")
        print(f"  Mean RMS Error: {np.mean(tcn_rms_error):.6f} rad")
        
        print(f"\nDifference (TCN - LSTM):")
        print(f"  Mean RMS Error Diff: {np.mean(tcn_rms_error) - np.mean(lstm_rms_error):.6f} rad")
        print(f"  Solver Time Diff: {tcn_avg_solver_time - lstm_avg_solver_time:.4f}s")
        
        # Save logs
        print(f"\nLogs saved with suffixes: '_lstm_comparison' and '_tcn_comparison'")
        
    else:
        # Original single simulation mode
        panda_mpc = PandaMPCSim(use_viewer=args.use_viewer,
                                expl_dyn=False,
                                )

        # Simulation Parameters
        Tsim = args.sim_time
        Nsim = int(Tsim / panda_mpc.joint_ctrl_period)

        init_time = time.time()
        # Simulate
        for i in range(Nsim):
            panda_mpc.step()

        print(f'Sim total time {time.time() - init_time} | Avg step time {(time.time() - init_time)/Nsim}')

        panda_mpc.convert_log_data_to_np()
        panda_mpc.plot_joint_pos()
        panda_mpc.plot_joint_vel()
        panda_mpc.plot_torque()
        panda_mpc.plot_ee_pos_3d()

        rms_error = np.sqrt(np.mean(np.square(panda_mpc.logged_data['qp_ref']-panda_mpc.logged_data['qp']),axis=0))
        print(f'rms_error {rms_error} | q_init {panda_mpc.q_init} | q_home {panda_mpc.q_home}')

    plt.show()