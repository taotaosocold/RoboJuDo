import logging
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as sRot

from robojudo.environment import Environment, env_registry
from robojudo.utils.rotation import TransformAlignment

logger = logging.getLogger(__name__)


@env_registry.register
class CasbotRealEnv(Environment):
    def __init__(self, cfg_env, device="cpu"):
        # Lazy ROS2 imports — only available on the real robot
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from sensor_msgs.msg import Imu, JointState

        self._rclpy = rclpy
        self._JointState = JointState
        self._Imu = Imu

        self.enabled: bool = cfg_env.act
        super().__init__(cfg_env=cfg_env, device=device)

        self.hl_cfg = cfg_env.hl
        self._dof_idx = cfg_env.joint2motor_idx
        self._waist_yaw_idx = self.joint_names.index("waist_yaw_joint")

        if self._dof_idx is not None and len(self._dof_idx) != self.num_dofs:
            raise ValueError("joint2motor_idx length must match num_dofs")

        # ---- name mapping: env (Casbot_25DoF) ↔ hl_motion (leg_l1 etc.) ----
        _leg_parts = ["pelvic_pitch", "pelvic_roll", "pelvic_yaw", "knee_pitch", "ankle_pitch", "ankle_roll"]
        _env_to_hl_name: dict[str, str] = {}
        # 这里_env_to_hl_name是一个字典，键是Casbot_25DoF的名称，值是hl_motion的名称
        for i, part in enumerate(_leg_parts):
            _env_to_hl_name[f"left_leg_{part}_joint"] = f"leg_l{i+1}_joint"
            _env_to_hl_name[f"right_leg_{part}_joint"] = f"leg_r{i+1}_joint"

        # Precompute hl_motion names for update() name matching in /joint_states
        # self.joint_names就是Casbot_25Dof的顺序以及名称，也就是mujoco的顺序
        # 而这里self._joint_state_names也是Casbot_25Dof的顺序，只是名称变成了left_l1_joint这样和message的情况
        self._joint_state_names: list[str] = [
            _env_to_hl_name.get(name, name) for name in self.joint_names
        ]

        # Auto-compute joint2motor_idx from robot_joint_names if not explicitly given
        # 这里robot_joint_names的顺序就是hl_motion的消息顺序，是由casbot_cfg中定义的顺序，这里就是将mujoco的顺序换成消息顺序
        if self._dof_idx is None and self.hl_cfg.robot_joint_names is not None:
            self._dof_idx = [
                self.hl_cfg.robot_joint_names.index(_env_to_hl_name.get(name, name))
                for name in self.joint_names
            ]

        # ---- ROS2 init ----
        if not rclpy.ok():
            rclpy.init()
        # 创建节点
        self.node = Node("robojudo_casbot")

        # ---- state storage ----
        self._latest_joint_state = None
        self._latest_imu = None

        # ---- subscribers ----
        # 给节点创建订阅者，一旦收到信息执行回调函数self._js_callback
        self._joint_state_sub = self.node.create_subscription(
            JointState,
            self.hl_cfg.joint_state_topic,
            self._js_callback,
            10,
        )
        self._imu_sub = self.node.create_subscription(
            Imu,
            self.hl_cfg.imu_topic,
            self._imu_callback,
            10,
        )

        # ---- publishers ----
        # 这里cmd_names的顺序其实就是casbot_cfg的robot_joint_names的顺序
        cmd_names = self.hl_cfg.robot_joint_names if self.hl_cfg.robot_joint_names else self.joint_names
        self._cmd_msg = JointState()
        self._cmd_msg.name = cmd_names
        self._cmd_msg.velocity = [0.0] * self.num_dofs
        self._cmd_msg.effort = [0.0] * self.num_dofs
        self._cmd_msg.position = [float(p) for p in self._default_pos_robot_order()]

        self._joint_cmd_pub = self.node.create_publisher(
            JointState,
            self.hl_cfg.joint_cmd_topic,
            10,
        )

        # ---- spin thread ----
        # 添加线程执行器
        self._executor = MultiThreadedExecutor()
        # 给线程执行器添加节点
        self._executor.add_node(self.node)
        # 为节点添加线程，这个线程就是那两个订阅者，持续监听收取信息
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True, name="ros2_spin"
        )
        # 开启线程
        self._spin_thread.start()

        # ---- born place alignment ----
        if self.born_place_align:
            self.torso_align = TransformAlignment()

        self.self_check()

    # ---- ROS2 callbacks ----

    def _js_callback(self, msg):
        self._latest_joint_state = msg

    def _imu_callback(self, msg):
        self._latest_imu = msg

    @staticmethod
    def _transform_anchor_imu_to_base(
        waist_yaw: float,
        waist_yaw_vel: float,
        anchor_quat: np.ndarray,
        anchor_ang_vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform waist_yaw_link IMU state into the floating-base frame.

        Quaternions use the project convention [x, y, z, w]. The IMU axes
        are assumed to coincide with the waist_yaw_link axes.
        """
        base_to_anchor = sRot.from_euler("z", waist_yaw)
        base_quat = (sRot.from_quat(anchor_quat) * base_to_anchor.inv()).as_quat()

        # The IMU angular velocity is expressed in the anchor frame. Remove
        # the waist joint contribution, then express the remainder in base.
        base_ang_vel = base_to_anchor.apply(
            anchor_ang_vel - np.array([0.0, 0.0, waist_yaw_vel], dtype=np.float32)
        )
        return base_quat, base_ang_vel

    # ---- Environment interface ----

    def set_gains(self, stiffness, damping):
        self.kps = np.asarray(stiffness)
        self.kds = np.asarray(damping)

    def reset(self):
        if self.born_place_align:
            self.born_place_align = False
            self.update()
            self.born_place_align = True
            self.set_born_place()
            self.update()

    def set_born_place(self, quat: np.ndarray | None = None, pos: np.ndarray | None = None):
        quat_ = self.base_quat if quat is None else quat
        pos_ = self.base_pos if pos is None else pos
        super().set_born_place(quat_, pos_)
        logger.info(f"[CasbotRealEnv] born place set to pos: {pos_}, quat: {quat_}")

    def self_check(self):
        timeout = 5.0
        start = time.time()
        while self._latest_joint_state is None and (time.time() - start) < timeout:
            time.sleep(0.05)
        if self._latest_joint_state is None:
            logger.warning("[CasbotRealEnv] Timeout waiting for joint_state, continuing anyway")
        else:
            logger.info("[CasbotRealEnv] Successfully connected to robot")

    def update(self):
        js = self._latest_joint_state
        if js is not None:
            # 这里获得消息js里的数据，是字典类型为{名称:值}
            name_to_pos = dict(zip(js.name, js.position))
            name_to_vel = dict(zip(js.name, js.velocity))
            # 这里self._joint_state_names已经是mujoco的顺序并且是消息类型的名称，所以这里直接通过名字映射将消息的顺序映射为mujoco即Casbot_25Dof的顺序
            for i, hl_name in enumerate(self._joint_state_names):
                if hl_name in name_to_pos:
                    self._dof_pos[i] = float(name_to_pos[hl_name])
                    self._dof_vel[i] = float(name_to_vel[hl_name])

        # ---- IMU ----
        imu = self._latest_imu
        if imu is not None:
            # The Casbot IMU is mounted on waist_yaw_link, which is also the
            # BeyondMimic torso/anchor, so use its data directly for torso.
            torso_quat = np.array(
                [imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w],
                dtype=np.float32,
            )
            torso_ang_vel = np.array(
                [imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z],
                dtype=np.float32,
            )

            waist_yaw = float(self._dof_pos[self._waist_yaw_idx])
            waist_yaw_vel = float(self._dof_vel[self._waist_yaw_idx])
            base_quat, base_ang_vel = self._transform_anchor_imu_to_base(
                waist_yaw=waist_yaw,
                waist_yaw_vel=waist_yaw_vel,
                anchor_quat=torso_quat,
                anchor_ang_vel=torso_ang_vel,
            )

            if self.born_place_align:
                base_quat = self.base_align.align_quat(base_quat)
                torso_quat = self.base_align.align_quat(torso_quat)

            self._torso_quat = torso_quat
            self._torso_ang_vel = torso_ang_vel
            self._base_quat = base_quat
            self._base_ang_vel = base_ang_vel
            self._base_rpy = sRot.from_quat(base_quat).as_euler("xyz")

        # ---- odometry ----
        if self._base_pos is None:
            self._base_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if self._base_lin_vel is None:
            self._base_lin_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # ---- FK ----
        if self.update_with_fk:
            fk_info = self.fk()
            self._torso_pos = fk_info[self._torso_name]["pos"]
            # Before the first IMU message, use FK as a temporary fallback.
            if imu is None:
                self._torso_quat = fk_info[self._torso_name]["quat"]
                self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]

    def step(self, pd_target, hand_pose=None):
        assert len(pd_target) == self.num_dofs

        if self._dof_idx is not None:
            for env_i, robot_i in enumerate(self._dof_idx):
                self._cmd_msg.position[robot_i] = float(pd_target[env_i])
        else:
            self._cmd_msg.position = [float(p) for p in pd_target]

        if self.enabled:
            self._joint_cmd_pub.publish(self._cmd_msg)

    def _default_pos_robot_order(self):
        """Return default_pos reordered to robot (robot_joint_names) order."""
        if self._dof_idx is not None:
            pos = np.zeros(self.num_dofs, dtype=np.float64)
            for env_i, robot_i in enumerate(self._dof_idx):
                pos[robot_i] = self.default_pos[env_i]
            return pos
        return self.default_pos

    def shutdown(self):
        self.enabled = False

        self._cmd_msg.position = [float(p) for p in self._default_pos_robot_order()]
        for _ in range(5):
            self._joint_cmd_pub.publish(self._cmd_msg)
            time.sleep(0.02)

        self._executor.shutdown()
        self.node.destroy_node()
        logger.info("[CasbotRealEnv] Shutdown complete")
