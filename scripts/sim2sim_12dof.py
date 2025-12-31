# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

import argparse
import ast
import os
import sys
import time

import mujoco
import mujoco_viewer
import numpy as np
import onnx
import onnxruntime as ort
from pynput import keyboard


def sanitize_variable_name(name):
    """将key名称转换为有效的Python变量名"""
    # 替换非法字符为下划线
    import re

    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    # 确保不以数字开头
    if name[0].isdigit():
        name = "_" + name
    return name.lower()  # 可选：转换为小写


def csv_str_to_list_advanced(csv_str, *, delimiter: str = ",", try_eval: bool = True):
    """
    将 CSV 字符串转换回列表，尝试识别更多数据类型

    Args:
        csv_str: CSV 格式的字符串
        delimiter: 分隔符，默认为逗号
        try_eval: 是否尝试使用 ast.literal_eval 解析复杂数据类型

    Returns:
        list: 包含转换后元素的列表
    """
    items = csv_str.split(delimiter)
    result = []

    for item in items:
        item = item.strip()

        if try_eval:
            # 尝试使用 ast.literal_eval 解析 Python 字面量
            try:
                evaluated = ast.literal_eval(item)
                result.append(evaluated)
                continue
            except (ValueError, SyntaxError):
                pass

        # 尝试转换为数字
        if item.isdigit():
            result.append(int(item))
        else:
            try:
                # 检查是否为浮点数
                float_val = float(item)
                # 检查是否真的是浮点数（不是整数）
                if float_val != int(float_val):
                    result.append(float_val)
                else:
                    result.append(int(float_val))
            except ValueError:
                # 保持为字符串
                result.append(item)

    return result


class SimToSimCfg:
    """Configuration class for sim2sim parameters.

    Must be kept consistent with the training configuration.
    """

    class sim:
        sim_duration = 100.0
        num_obs_per_step = 45
        actor_obs_history_length = 5
        dt = 0.001
        decimation = 20
        clip_actions = 18.0


class MujocoRunner:
    """
    Sim2Sim runner that loads a policy and a MuJoCo model
    to run real-time humanoid control simulation.

    Args:
        cfg (SimToSimCfg): Configuration object for simulation.
        policy_path (str): Path to the Onnx exported policy.
        model_path (str): Path to the MuJoCo XML model.
    """

    def __init__(self, cfg: SimToSimCfg, policy_path, model_path):
        self.cfg = cfg
        network_path = policy_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        # 获取所有关节名称
        self.joint_names = [self.model.joint(i).name for i in range(self.model.njnt)]
        print(self.joint_names)
        self.model.opt.timestep = self.cfg.sim.dt

        self.onnx_model = onnx.load(network_path)
        onnx.checker.check_model(self.onnx_model)
        self.policy = ort.InferenceSession(network_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer._render_every_frame = False
        self.init_variables()
        self.data.qpos[-self.num_action :] = self.default_joint_pos
        mujoco.mj_step(self.model, self.data)


    def init_variables(self) -> None:
        """Initialize simulation variables and joint index mappings."""
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt

        assert self.onnx_model.metadata_props
        print("Loading robot properties from onnx: ")
        for prop in self.onnx_model.metadata_props:
            attr_name = sanitize_variable_name(prop.key)
            attr_value = csv_str_to_list_advanced(prop.value)
            setattr(self, attr_name, attr_value)
        # 23dof
        # self.mujoco_to_isaac_idx = [0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22]
        # self.isaac_to_mujoco_idx = [0, 3, 7, 11, 15, 19, 1, 4, 8, 12, 16, 20, 2, 5, 9, 13, 17, 21, 6, 10, 14, 18, 22]
        # 12dof
        # import ipdb; ipdb.set_trace()
        mujoco_joint_names = ['l_leg_hip_yaw_joint', 'l_leg_hip_roll_joint', 'l_leg_hip_pitch_joint',
                                'l_leg_knee_joint', 'l_leg_ankle_pitch_joint', 'l_leg_ankle_roll_joint',
                                'r_leg_hip_yaw_joint', 'r_leg_hip_roll_joint', 'r_leg_hip_pitch_joint',
                                'r_leg_knee_joint', 'r_leg_ankle_pitch_joint', 'r_leg_ankle_roll_joint']
        isaaclab_joint_names = ['l_leg_hip_yaw_joint', 'r_leg_hip_yaw_joint', 
                                'l_leg_hip_roll_joint', 'r_leg_hip_roll_joint', 
                                'l_leg_hip_pitch_joint', 'r_leg_hip_pitch_joint', 
                                'l_leg_knee_joint', 'r_leg_knee_joint', 
                                'l_leg_ankle_pitch_joint', 'r_leg_ankle_pitch_joint', 
                                'l_leg_ankle_roll_joint', 'r_leg_ankle_roll_joint']
        self.isaac_to_mujoco_idx = [isaaclab_joint_names.index(name) for name in mujoco_joint_names]
        self.mujoco_to_isaac_idx = [mujoco_joint_names.index(name) for name in isaaclab_joint_names]
        # self.mujoco_to_isaac_idx = [0, 6, 12, 1, 7, 13, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22, 18, 23]
        # self.isaac_to_mujoco_idx = [0, 3, 6, 10, 14, 18, 1, 4, 7, 11, 15, 19, 2, 5, 8, 12, 16, 20, 22, 9, 13, 17, 21, 23]

        self.num_action = len(self.default_joint_pos)
        self.default_joint_pos = np.array(self.default_joint_pos)[self.isaac_to_mujoco_idx]
        self.joint_stiffness = np.array(self.joint_stiffness)[self.isaac_to_mujoco_idx]
        self.joint_damping = np.array(self.joint_damping)[self.isaac_to_mujoco_idx]
        self.action_scale = np.array(self.action_scale)[self.isaac_to_mujoco_idx]

        self.data.qpos[-self.num_action :] = self.default_joint_pos.copy()

        self.dof_pos = np.zeros(self.num_action)
        self.dof_vel = np.zeros(self.num_action)
        self.action = np.zeros(self.num_action)

        self.episode_length_buf = 0

        # Initial command vel
        self.command_vel = np.array([0.0, 0.0, 0.0])
        self.obs_history = np.zeros(
            (self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,), dtype=np.float32
        )

    def get_obs(self) -> np.ndarray:
        """
        Compute current observation vector from MuJoCo sensors and internal state.

        Returns:
            np.ndarray: Normalized and clipped observation history.
        """
        # import ipdb; ipdb.set_trace()
        self.dof_pos = self.data.qpos.astype(np.double)[-self.num_action :]
        self.dof_vel = self.data.qvel.astype(np.double)[-self.num_action :]

        obs = np.zeros((self.cfg.sim.num_obs_per_step,), dtype=np.float32)

        # Command velocity
        obs[0:3] = self.command_vel

        # Angular vel
        obs[3:6] = self.data.sensor("angular-velocity").data.astype(np.double)

        # Projected gravity
        obs[6:9] = self.quat_rotate_inverse(
            self.data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double), np.array([0, 0, -1])
        )

        # Dof pos
        obs[9 : 9 + self.num_action] = (self.dof_pos - self.default_joint_pos)[self.mujoco_to_isaac_idx]

        # Dof vel
        obs[9 + self.num_action : 9 + self.num_action * 2] = self.dof_vel[self.mujoco_to_isaac_idx]

        # Action
        obs[9 + self.num_action * 2 : 9 + self.num_action * 3] = self.action

        # Update observation history
        self.obs_history = np.roll(self.obs_history, shift=-self.cfg.sim.num_obs_per_step)
        self.obs_history[-self.cfg.sim.num_obs_per_step :] = obs.copy()
        # import ipdb; ipdb.set_trace()
        return self.obs_history

    def pd_control(self, target_q, q, init_q, target_dq, dq):
        """Calculates torques from position commands"""
        return (target_q + init_q - q) * self.joint_stiffness + (target_dq - dq) * self.joint_damping

    def run(self) -> None:
        """
        Run the simulation loop with keyboard-controlled commands.
        """
        self.setup_keyboard_listener()
        self.listener.start()

        while self.data.time < self.cfg.sim.sim_duration:
            self.obs_history = self.get_obs()
            obs = np.zeros([1, self.obs_history.size], dtype=np.float32)
            obs[0] = self.obs_history.copy()
            self.action[:] = self.policy.run(None, {"obs": obs})[0]
            # import ipdb; ipdb.set_trace()
            # print("Action:", self.action)
            self.action = np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)
            action = self.action.copy()[self.isaac_to_mujoco_idx]

            for sim_update in range(self.cfg.sim.decimation):
                step_start_time = time.time()

                q = self.data.qpos.astype(np.double)[-self.num_action :]
                dq = self.data.qvel.astype(np.double)[-self.num_action :]
                target_q = action * self.action_scale
                target_dq = np.zeros((self.num_action), dtype=np.double)
                tau = self.pd_control(target_q, q, self.default_joint_pos, target_dq, dq)  # Calc torques
                self.data.ctrl = tau

                mujoco.mj_step(self.model, self.data)
                self.viewer.render()

                elapsed = time.time() - step_start_time
                sleep_time = self.cfg.sim.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            self.episode_length_buf += 1

        self.listener.stop()
        self.viewer.close()

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Rotate a vector by the inverse of a quaternion.

        Args:
            q (np.ndarray): Quaternion (x, y, z, w) format.
            v (np.ndarray): Vector to rotate.

        Returns:
            np.ndarray: Rotated vector.
        """
        q_w = q[-1]
        q_vec = q[:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v) * 2.0

        return a - b + c

    def adjust_command_vel(self, idx: int, increment: float) -> None:
        """
        Adjust command velocity vector.

        Args:
            idx (int): Index of velocity component (0=x, 1=y, 2=yaw).
            increment (float): Value to increment.
        """
        # self.command_vel[idx] += increment
        # print(self.command_vel)
        # # self.command_vel[np.abs(self.command_vel) < 0.3] = 0
        # self.command_vel[idx] = np.clip(self.command_vel[idx], 0, 0.8)  # vel clip
        # # print(self.command_vel)
        self.command_vel[idx] += increment
        print(f"Command vel before clip: {self.command_vel}")
        
        # 设置一个小的阈值，避免微小数值导致的不稳定
        if abs(self.command_vel[idx]) < 0.01:
            self.command_vel[idx] = 0.0
        
        # 扩展速度范围以支持后退
        self.command_vel[idx] = np.clip(self.command_vel[idx], -0.8, 0.8)
        print(f"Command vel after clip: {self.command_vel}")

    def setup_keyboard_listener(self) -> None:
        """
        Set up keyboard event listener for user control input.
        """

        # def on_press(key):
        #     try:
        #         if key.char == "8":  # NumPad 8      x += 0.1
        #             self.adjust_command_vel(0, 0.1)
        #         elif key.char == "2":  # NumPad 2      x -= 0.1
        #             self.adjust_command_vel(0, -0.1)
        #         elif key.char == "4":  # NumPad 4      y -= 0.1
        #             self.adjust_command_vel(1, -0.1)
        #         elif key.char == "6":  # NumPad 6      y += 0.1
        #             self.adjust_command_vel(1, 0.1)
        #         elif key.char == "7":  # NumPad 7      yaw += 0.1
        #             self.adjust_command_vel(2, -0.1)
        #         elif key.char == "9":  # NumPad 9      yaw -= 0.1
        #             self.adjust_command_vel(2, 0.1)
        #     except AttributeError:
        #         pass

        # self.listener = keyboard.Listener(on_press=on_press)

        def on_press(key):
            try:
                # 支持主键盘数字键
                if hasattr(key, 'char') and key.char:
                    if key.char == "8":
                        print("Pressed main keyboard 8 - x += 0.1")
                        self.adjust_command_vel(0, 0.1)
                    elif key.char == "2":
                        print("Pressed main keyboard 2 - x -= 0.1")
                        self.adjust_command_vel(0, -0.1)
                    elif key.char == "4":
                        print("Pressed main keyboard 4 - y -= 0.1")
                        self.adjust_command_vel(1, -0.1)
                    elif key.char == "6":
                        print("Pressed main keyboard 6 - y += 0.1")
                        self.adjust_command_vel(1, 0.1)
                    elif key.char == "7":
                        print("Pressed main keyboard 7 - yaw += 0.1")
                        self.adjust_command_vel(2, -0.1)
                    elif key.char == "9":
                        print("Pressed main keyboard 9 - yaw -= 0.1")
                        self.adjust_command_vel(2, 0.1)
                # 支持数字键盘（通过虚拟键码）
                elif hasattr(key, 'vk'):
                    # 数字键盘按键的虚拟键码（Linux系统）
                    if key.vk in [83, 104]:  # NumPad 8
                        print("Pressed numpad 8 - x += 0.1")
                        self.adjust_command_vel(0, 0.1)
                    elif key.vk in [84, 98]:  # NumPad 2
                        print("Pressed numpad 2 - x -= 0.1")
                        self.adjust_command_vel(0, -0.1)
                    elif key.vk in [81, 100]:  # NumPad 4
                        print("Pressed numpad 4 - y -= 0.1")
                        self.adjust_command_vel(1, -0.1)
                    elif key.vk in [82, 102]:  # NumPad 6
                        print("Pressed numpad 6 - y += 0.1")
                        self.adjust_command_vel(1, 0.1)
                    elif key.vk in [79, 103]:  # NumPad 7
                        print("Pressed numpad 7 - yaw += 0.1")
                        self.adjust_command_vel(2, -0.1)
                    elif key.vk in [80, 105]:  # NumPad 9
                        print("Pressed numpad 9 - yaw -= 0.1")
                        self.adjust_command_vel(2, 0.1)
                # 打印按键信息用于调试
                print(f"Raw key event: {key}, char: {getattr(key, 'char', 'None')}, vk: {getattr(key, 'vk', 'None')}")
            except Exception as e:
                print(f"Error processing key: {e}, key: {key}")

        # 创建并启动键盘监听器，使用非阻塞模式
        self.listener = keyboard.Listener(on_press=on_press)
        self.listener.daemon = True  # 确保监听器线程在主程序退出时也会退出


if __name__ == "__main__":
    LEGGED_LAB_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    parser = argparse.ArgumentParser(description="Run sim2sim Mujoco controller.")
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Path to policy.onnx.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(
            LEGGED_LAB_ROOT_DIR, "NoetixE1-Lab/source/NoetixE1/NoetixE1/assets/robots/e1/mjcf/e1_12dof.xml"
        ),
        help="Path to model.xml",
    )
    parser.add_argument("--duration", type=float, default=100.0, help="Simulation duration in seconds")
    args = parser.parse_args()

    if not os.path.isfile(args.policy):
        print(f"[ERROR] Policy file not found: {args.policy}")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"[ERROR] MuJoCo model file not found: {args.model}")
        sys.exit(1)

    print(f"[INFO] Loaded policy: {args.policy}")
    print(f"[INFO] Loaded model: {args.model}")

    sim_cfg = SimToSimCfg()
    sim_cfg.sim.sim_duration = args.duration

    runner = MujocoRunner(
        cfg=sim_cfg,
        policy_path=args.policy,
        model_path=args.model,
    )
    runner.run()
