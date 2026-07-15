import logging

from box import Box

import robojudo.controller
from robojudo.controller import Controller, ControllerHook, CtrlCfg

logger = logging.getLogger(__name__)


class CtrlManager:
    """
    A manager that handles multiple controllers and their interactions.
    """

    def __init__(
        self,
        cfg_ctrls: list[CtrlCfg] | None = None,
        env=None,
        device="cpu",
    ):
        self.cfg_ctrls = cfg_ctrls
        self.env = env
        self.device = device

        controllers = {}
        for cfg_ctrl in self.cfg_ctrls or []:
            ctrl_type = cfg_ctrl.ctrl_type
            if ctrl_type in controllers.keys():
                logger.warning(f"Controller type {ctrl_type} already exists, skipping.")
                continue
            # KeyboardCtrlCfg的类名就是KeyboardCtrl
            ctrl_class: type[Controller] = getattr(robojudo.controller, ctrl_type)
            # 创建类
            controller: Controller = ctrl_class(cfg_ctrl=cfg_ctrl, env=self.env, device=self.device)
            controllers[ctrl_type] = {
                "inst": controller,
                "cfg": cfg_ctrl,
                # "triggers": []
            }

        self.controllers = Box(controllers)

    def reset(self):
        """
        Reset all controllers.
        """
        for controller in self.controllers.values():
            controller.inst.reset()

    def post_step_callback(self, ctrl_data: Box):
        """
        Call post step callback for all controllers.
        """
        # self.process_triggers()
        # 获取对应的状态比如[MOTION_RESET]
        commands = ctrl_data.get("COMMANDS", [])
        for controller in self.controllers.values():
            controller.inst.post_step_callback(commands)

    def get_ctrl_data(self, env_data):
        ctrl_data_all = {}
        ctrl_commands_all = set()
        # 这里会遍历所有controller包括键盘或者手柄
        for ctrl_type, controller in self.controllers.items():
            # 通过两个get_data函数去读取硬件状态
            if isinstance(controller.inst, ControllerHook):
                ctrl_data = controller.inst.get_data_with_hook(prior_ctrl_data=ctrl_data_all, env_data=env_data)
            else:
                ctrl_data = controller.inst.get_data()
            # 如果硬件的事件匹配到配置的triggers映射，则生成命令字符串ctrl_commands例如[MOTION_RESET]
            ctrl_data_triggered, ctrl_commands = controller.inst.process_triggers(ctrl_data)

            ctrl_data_all[ctrl_type] = ctrl_data_triggered

            ctrl_commands_all.update(ctrl_commands)

        ctrl_data_all["COMMANDS"] = list(ctrl_commands_all)
        return Box(ctrl_data_all)


if __name__ == "__main__":
    # Example usage
    from robojudo.config.g1.env.g1_dummy_env_cfg import G1DummyEnvCfg
    from robojudo.controller.ctrl_cfgs import JoystickCtrlCfg, KeyboardCtrlCfg
    from robojudo.environment.dummy_env import DummyEnv

    env = DummyEnv(cfg_env=G1DummyEnvCfg(forward_kinematic=None))
    cfg_ctrls = [KeyboardCtrlCfg(), JoystickCtrlCfg()]

    ctrl_manager = CtrlManager(cfg_ctrls=cfg_ctrls, env=env)  # pyright: ignore[reportArgumentType]
    ctrl_manager.reset()
    ctrl_data = ctrl_manager.get_ctrl_data(None)
    print(ctrl_data)
    # ctrl_manager.post_step_callback()
    print("Controller manager initialized and ready.")
