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

        if self._dof_idx is not None and len(self._dof_idx) != self.num_dofs:
            raise ValueError("joint2motor_idx length must match num_dofs")

        # ---- ROS2 init ----
        if not rclpy.ok():
            rclpy.init()

        self.node = Node("robojudo_casbot")

        # ---- state storage ----
        self._latest_joint_state = None
        self._latest_imu = None

        # ---- subscribers ----
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
        cmd_names = self.hl_cfg.robot_joint_names if self.hl_cfg.robot_joint_names else self.joint_names
        self._cmd_msg = JointState()
        self._cmd_msg.name = cmd_names
        self._cmd_msg.position = [float(p) for p in self.default_pos]
        self._cmd_msg.velocity = [0.0] * self.num_dofs
        self._cmd_msg.effort = [0.0] * self.num_dofs

        self._joint_cmd_pub = self.node.create_publisher(
            JointState,
            self.hl_cfg.joint_cmd_topic,
            10,
        )

        # ---- background publish timer (50Hz safety net) ----
        self._cmd_timer = self.node.create_timer(
            self.hl_cfg.control_dt,
            self._timer_callback,
        )

        # ---- spin thread ----
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True, name="ros2_spin"
        )
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

    def _timer_callback(self):
        """Background 50Hz safety send."""
        if not self.enabled:
            return
        self._joint_cmd_pub.publish(self._cmd_msg)

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
        # ---- joint positions / velocities ----
        js = self._latest_joint_state
        if js is not None:
            msg_names = (
                self.hl_cfg.robot_joint_names
                if self.hl_cfg.robot_joint_names
                else js.name
            )
            name_to_idx = {n: i for i, n in enumerate(js.name)}

            dof_pos = np.zeros(self.num_dofs, dtype=np.float32)
            dof_vel = np.zeros(self.num_dofs, dtype=np.float32)

            for i, name in enumerate(msg_names):
                if i >= self.num_dofs:
                    break
                if name in name_to_idx:
                    idx = name_to_idx[name]
                    if idx < len(js.position):
                        dof_pos[i] = js.position[idx]
                    if idx < len(js.velocity):
                        dof_vel[i] = js.velocity[idx]

            self._dof_pos = dof_pos
            self._dof_vel = dof_vel

        # ---- IMU ----
        imu = self._latest_imu
        if imu is not None:
            quat = np.array(
                [imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w],
                dtype=np.float32,
            )
            ang_vel = np.array(
                [imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z],
                dtype=np.float32,
            )

            if self.born_place_align:
                quat = self.base_align.align_quat(quat)

            self._base_quat = quat
            self._base_ang_vel = ang_vel
            self._base_rpy = sRot.from_quat(quat).as_euler("xyz")

        # ---- odometry ----
        if self._base_pos is None:
            self._base_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if self._base_lin_vel is None:
            self._base_lin_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # ---- FK ----
        if self.update_with_fk:
            fk_info = self.fk()
            self._torso_pos = fk_info[self._torso_name]["pos"]
            self._torso_quat = fk_info[self._torso_name]["quat"]
            self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]

    def step(self, pd_target, hand_pose=None):
        assert len(pd_target) == self.num_dofs

        self._cmd_msg.position = [float(p) for p in pd_target]

        if self.enabled:
            self._joint_cmd_pub.publish(self._cmd_msg)

    def shutdown(self):
        self.enabled = False

        # send default position a few times
        self._cmd_msg.position = [float(p) for p in self.default_pos]
        for _ in range(5):
            self._joint_cmd_pub.publish(self._cmd_msg)
            time.sleep(0.02)

        self._executor.shutdown()
        self.node.destroy_node()
        logger.info("[CasbotRealEnv] Shutdown complete")
