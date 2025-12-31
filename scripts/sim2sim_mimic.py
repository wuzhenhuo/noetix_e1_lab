# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

import argparse
import ast
import os
import sys
import time

import matplotlib.pyplot as plt
import mujoco
import mujoco_viewer
import numpy as np
import onnx
import onnxruntime as ort


def matrix_to_quaternion_simple(matrix):
    """
    简化的矩阵转四元数实现
    """
    matrix = np.array(matrix)
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]

    trace = m00 + m11 + m22

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


def subtract_frame_transforms_mujoco(quat_a, quat_b, init_to_world):
    """
    与IsaacLab中subtract_frame_transforms完全相同的实现（一维版本）
    计算从坐标系A到坐标系B的相对变换

    参数:
        pos_a: 坐标系A的位置 (3,)
        quat_a: 坐标系A的四元数 (4,) [w, x, y, z]格式
        pos_b: 坐标系B的位置 (3,)
        quat_b: 坐标系B的四元数 (4,) [w, x, y, z]格式

    返回:
        rel_pos: B相对于A的位置 (3,)
        rel_quat: B相对于A的旋转四元数 (4,) [w, x, y, z]格式
    """

    # 计算相对旋转: quat_B_to_A = quat_A^* ⊗ quat_B
    rel_quat = quaternion_multiply(matrix_to_quaternion_simple(init_to_world), quat_b)
    rel_quat = quaternion_multiply(quaternion_conjugate(quat_a), rel_quat)

    # 确保四元数归一化（与IsaacLab保持一致）
    rel_quat = rel_quat / np.linalg.norm(rel_quat)

    return rel_quat


def quaternion_conjugate(q):
    """四元数共轭: [w, x, y, z] -> [w, -x, -y, -z]"""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_multiply(q1, q2):
    """四元数乘法: q1 ⊗ q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])


def yaw_quat(q):
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


mujoco_joint_index = [
    "l_leg_hip_yaw_joint",
    "l_leg_hip_roll_joint",
    "l_leg_hip_pitch_joint",
    "l_leg_knee_joint",
    "l_leg_ankle_pitch_joint",
    "l_leg_ankle_roll_joint",
    "r_leg_hip_yaw_joint",
    "r_leg_hip_roll_joint",
    "r_leg_hip_pitch_joint",
    "r_leg_knee_joint",
    "r_leg_ankle_pitch_joint",
    "r_leg_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "l_arm_shoulder_pitch_joint",
    "l_arm_shoulder_roll_joint",
    "l_arm_shoulder_yaw_joint",
    "l_arm_elbow_pitch_joint",
    "l_arm_elbow_yaw_joint",
    "r_arm_shoulder_pitch_joint",
    "r_arm_shoulder_roll_joint",
    "r_arm_shoulder_yaw_joint",
    "r_arm_elbow_pitch_joint",
    "r_arm_elbow_yaw_joint",
]

isaacLab_joint_index = [
    "l_leg_hip_yaw_joint",
    "r_leg_hip_yaw_joint",
    "waist_yaw_joint",
    "l_leg_hip_roll_joint",
    "r_leg_hip_roll_joint",
    "waist_roll_joint",
    "l_leg_hip_pitch_joint",
    "r_leg_hip_pitch_joint",
    "l_arm_shoulder_pitch_joint",
    "r_arm_shoulder_pitch_joint",
    "l_leg_knee_joint",
    "r_leg_knee_joint",
    "l_arm_shoulder_roll_joint",
    "r_arm_shoulder_roll_joint",
    "l_leg_ankle_pitch_joint",
    "r_leg_ankle_pitch_joint",
    "l_arm_shoulder_yaw_joint",
    "r_arm_shoulder_yaw_joint",
    "l_leg_ankle_roll_joint",
    "r_leg_ankle_roll_joint",
    "l_arm_elbow_pitch_joint",
    "r_arm_elbow_pitch_joint",
    "l_arm_elbow_yaw_joint",
    "r_arm_elbow_yaw_joint",
]


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
        num_obs_per_step = 129
        actor_obs_history_length = 5
        dt = 0.005
        decimation = 4
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

    def __init__(self, cfg: SimToSimCfg, policy_path, model_path, motion_file):
        self.cfg = cfg
        network_path = policy_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.cfg.sim.dt
        # self.motion = MotionLoader(motion_file,[0])
        self.motion = np.load(motion_file)

        body_name = "waist_roll_link"
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if self.body_id == -1:
            raise ValueError(f"Body {body_name} not found in model")

        self.motion_joint_pos = self.motion["joint_pos"]
        self.motion_joint_vel = self.motion["joint_vel"]
        self.motion_quat = self.motion["body_quat_w"][:, 6, :]
        self.onnx_model = onnx.load(network_path)
        onnx.checker.check_model(self.onnx_model)
        self.policy = ort.InferenceSession(network_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer._render_every_frame = False
        mujoco.mj_step(self.model, self.data)
        self.init_variables()

        self.ref_tau = []
        self.real_tau = []

    def init_variables(self) -> None:
        """Initialize simulation variables and joint index mappings."""
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt

        assert self.onnx_model.metadata_props
        print("Loading robot properties from onnx: ")
        for prop in self.onnx_model.metadata_props:
            attr_name = sanitize_variable_name(prop.key)
            attr_value = csv_str_to_list_advanced(prop.value)
            setattr(self, attr_name, attr_value)

        self.mujoco_to_isaac_idx = [mujoco_joint_index.index(name) for name in isaacLab_joint_index]
        self.isaac_to_mujoco_idx = [isaacLab_joint_index.index(name) for name in mujoco_joint_index]

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

        self.obs_history = np.zeros(
            (self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,), dtype=np.float32
        )

    def get_obs(self, timestep) -> np.ndarray:
        """
        Compute current observation vector from MuJoCo sensors and internal state.

        Returns:
            np.ndarray: Normalized and clipped observation history.
        """
        if timestep < 2:

            ref_motion_quat = self.motion_quat[timestep]
            yaw_motion_quat = yaw_quat(ref_motion_quat)
            yaw_motion_matrix = np.zeros(9)
            mujoco.mju_quat2Mat(yaw_motion_matrix, yaw_motion_quat)
            yaw_motion_matrix = yaw_motion_matrix.reshape(3, 3)

            robot_quat = self.data.xquat[self.body_id]
            yaw_robot_quat = yaw_quat(robot_quat)
            yaw_robot_matrix = np.zeros(9)
            mujoco.mju_quat2Mat(yaw_robot_matrix, yaw_robot_quat)
            yaw_robot_matrix = yaw_robot_matrix.reshape(3, 3)

            self.init_to_world = yaw_robot_matrix @ yaw_motion_matrix.T

        self.dof_pos = self.data.qpos.astype(np.double)[-self.num_action :]
        self.dof_vel = self.data.qvel.astype(np.double)[-self.num_action :]

        ref_motion_quat = self.motion_quat[timestep]
        quaternion = self.data.xquat[self.body_id]
        anchor_quat = subtract_frame_transforms_mujoco(quaternion, ref_motion_quat, self.init_to_world)
        anchor_ori = np.zeros(9)
        mujoco.mju_quat2Mat(anchor_ori, anchor_quat)
        anchor_ori = anchor_ori.reshape(3, 3)[:, :2]
        anchor_ori = anchor_ori.reshape(
            -1,
        )

        obs = np.zeros((self.cfg.sim.num_obs_per_step,), dtype=np.float32)

        # command
        obs[0 : self.num_action] = self.motion_joint_pos[timestep, :]
        obs[self.num_action : 2 * self.num_action] = self.motion_joint_vel[timestep, :]
        # Motion_anchor_ori
        obs[2 * self.num_action : 6 + 2 * self.num_action] = anchor_ori
        # Angular vel
        obs[6 + 2 * self.num_action : 9 + 2 * self.num_action] = self.data.sensor("angular-velocity").data.astype(
            np.double
        )
        # Dof pos
        obs[9 + 2 * self.num_action : 9 + 3 * self.num_action] = (self.dof_pos - self.default_joint_pos)[
            self.mujoco_to_isaac_idx
        ]

        # Dof vel
        obs[9 + 3 * self.num_action : 9 + 4 * self.num_action] = self.dof_vel[self.mujoco_to_isaac_idx]

        # Action
        obs[9 + 4 * self.num_action : 9 + 5 * self.num_action] = self.action

        # Update observation history
        self.obs_history = np.roll(self.obs_history, shift=-self.cfg.sim.num_obs_per_step)
        self.obs_history[-self.cfg.sim.num_obs_per_step :] = obs.copy()

        return self.obs_history

    def pd_control(self, target_q, q, init_q, target_dq, dq):
        """Calculates torques from position commands"""
        return (target_q + init_q - q) * self.joint_stiffness + (target_dq - dq) * self.joint_damping

    def run(self) -> None:
        """
        Run the simulation loop with keyboard-controlled commands.
        """
        timestep = 0
        while self.data.time < self.cfg.sim.sim_duration:
            max_timestep = self.motion_joint_pos.shape[0]
            if timestep >= max_timestep:
                timestep = 0
            self.obs_history = self.get_obs(timestep)
            obs = np.zeros([1, self.obs_history.size], dtype=np.float32)
            obs[0] = self.obs_history.copy()
            self.action[:] = self.policy.run(None, {"obs": obs})[0]
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
            timestep += 1

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


if __name__ == "__main__":
    LEGGED_LAB_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    parser = argparse.ArgumentParser(description="Run sim2sim Mujoco controller.")
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Path to policy.onnx.",
    )
    parser.add_argument("--motion_file", type=str, required=True, help="动作文件路径")
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(
            LEGGED_LAB_ROOT_DIR, "NoetixE1-Lab/source/NoetixE1/NoetixE1/assets/robots/e1/mjcf/e1_24dof.xml"
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

    runner = MujocoRunner(cfg=sim_cfg, policy_path=args.policy, model_path=args.model, motion_file=args.motion_file)
    runner.run()
