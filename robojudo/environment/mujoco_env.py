import logging
import time

import mujoco
import mujoco_viewer
import numpy as np

from robojudo.environment import Environment, env_registry
from robojudo.environment.env_cfgs import MujocoEnvCfg
from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.utils.util_func import quat_rotate_inverse_np, quatToEuler

logger = logging.getLogger(__name__)


@env_registry.register
class MujocoEnv(Environment):
    cfg_env: MujocoEnvCfg
    # 在g1_beyondmimic中其cfg_env是G1MujocoEnvCfg，这个类又是继承MujocoEnvCfg(在env_cfgs.py脚本中)
    def __init__(self, cfg_env: MujocoEnvCfg, device="cpu"):
        super().__init__(cfg_env=cfg_env, device=device)
        # 获得仿真持续时间，为60s
        self.sim_duration = cfg_env.sim_duration
        # 0.001s
        self.sim_dt = cfg_env.sim_dt
        # 20
        self.sim_decimation = cfg_env.sim_decimation
        # 为0.02s
        self.control_dt = self.sim_dt * self.sim_decimation
        # 获得xml构建机器人
        self.model = mujoco.MjModel.from_xml_path(cfg_env.xml)  # pyright: ignore[reportAttributeAccessIssue]
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)  # pyright: ignore[reportAttributeAccessIssue]
        # mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        # 刚导入xml的时候，self.data是只有机器人qpos全0的值，这里通过前向运动学以及机器人的各个连杆和其父连杆的关系去计算每个点的全部全局位置。
        # 这一步step相当于去更新/初始化传感器数据和self.data的其他数据
        mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

        self.viewer = mujoco_viewer.MujocoViewer(
            self.model,
            self.data,
            width=1200,
            height=900,
            hide_menus=True,
            diable_key_callbacks=True,
        )
        self.viewer.cam.distance = 3.0
        self.viewer.cam.elevation = -10.0
        self.viewer.cam.azimuth = 180.0
        # self.viewer._paused = True

        if cfg_env.visualize_extras:
            self.visualizer = MujocoVisualizer(self.viewer)
        else:
            self.visualizer = None

        self.last_time = time.time()
        # 这一步也是，先给每个全局变量先赋值一下机器人默认全0状态的数据
        self.update()  # get initial state

    def reborn(self, init_qpos=None):
        if init_qpos is not None:
            self.data.qpos[0:7] = init_qpos
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = 0.0
        else:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # pyright: ignore[reportAttributeAccessIssue]
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

    def reset(self):
        if self.born_place_align:  # TODO: merge
            self.born_place_align = False  # disable during reset
            self.update()
            self.born_place_align = True  # enable after reset
            self.set_born_place()
            self.update()

    def set_gains(self, stiffness, damping):
        assert len(stiffness) == self.num_dofs and len(damping) == self.num_dofs
        self.stiffness = np.asarray(stiffness)
        self.damping = np.asarray(damping)

    def self_check(self):
        pass

    def set_born_place(self, quat: np.ndarray | None = None, pos: np.ndarray | None = None):
        quat_ = self.base_quat if quat is None else quat
        pos_ = self.base_pos if pos is None else pos
        super().set_born_place(quat_, pos_)

    def update(self, simple=False):  # TODO: clean sensors in xml
        """simple: only update dof pos & vel"""
        dof_pos = self.data.qpos.astype(np.float32)[-self.num_dofs :]
        dof_vel = self.data.qvel.astype(np.float32)[-self.num_dofs :]
        # 在update函数中实际更新的是self._dof_pos而不是self.dof_pos，但是其实在env_cfgs.py中self.dof_pos是一个@property返回的就是self._dof_pos的副本，所以更新这个也可以
        self._dof_pos = dof_pos.copy()
        self._dof_vel = dof_vel.copy()

        if simple:
            return
        # 将读过来的四元数转化为xyzw的形式
        quat = self.data.qpos.astype(np.float32)[3:7][[1, 2, 3, 0]]
        ang_vel = self.data.qvel.astype(np.float32)[3:6]
        base_pos = self.data.qpos.astype(np.float32)[:3]
        lin_vel = self.data.qvel.astype(np.float32)[0:3]

        if self.born_place_align:
            quat, base_pos = self.base_align.align_transform(quat, base_pos)

        lin_vel = quat_rotate_inverse_np(quat, lin_vel)
        rpy = quatToEuler(quat)

        self._base_rpy = rpy.copy()
        self._base_quat = quat.copy()
        self._base_ang_vel = ang_vel.copy()

        self._base_pos = base_pos.copy()
        self._base_lin_vel = lin_vel.copy()
        # 执行前向运动学，根位置是base_pos和base_quat
        if self.update_with_fk:
            fk_info = self.fk()
            self._fk_info = fk_info.copy()
            self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]
            # 这个四元数很重要，最后得到的并且进入的四元数是_torso_quat而不是base_quat，也是xyzw的形式
            self._torso_quat = fk_info[self._torso_name]["quat"]
            self._torso_pos = fk_info[self._torso_name]["pos"]
    # 环境接受力矩执行步进，循环执行20次PD
    def step(self, pd_target, hand_pose=None):
        assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"

        if hand_pose is not None:
            logger.info("Hand pose-->", hand_pose)

        self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
        if self.viewer.is_alive:
            self.viewer.render()

        for _ in range(self.sim_decimation):
            torque = (pd_target - self.dof_pos) * self.stiffness - self.dof_vel * self.damping
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)

            self.data.ctrl = torque

            mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
            # 每一次PD后都要update去更新self.dof_pos即机器人的关节到达了什么位置
            self.update(simple=True)
        self.update(simple=False)

    def shutdown(self):
        self.viewer.close()


if __name__ == "__main__":
    from robojudo.config.g1.env.g1_mujuco_env_cfg import G1MujocoEnvCfg

    mujoco_env = MujocoEnv(cfg_env=G1MujocoEnvCfg())
    mujoco_env.viewer._paused = False

    while True:
        # mujoco_env.update()
        mujoco_env.step(np.zeros(mujoco_env.num_dofs))
        time.sleep(0.02)
