import os

from robojudo.controller.ctrl_cfgs import CtrlCfg


class CasbotParkourDiffusionCtrlCfg(CtrlCfg):
    """Online terrain-conditioned Flow Matching reference for Casbot Parkour."""

    ctrl_type: str = "ParkourDiffusionCtrl"
    model_path: str = os.environ.get(
        "CASBOT_FLOW_MATCHING_ONNX",
        "/home/casbot/Desktop/whole_body_tracking/logs/flow_matching/casbot_flow_matching/20260818_185937/export/pretrained.onnx",
    )
    # This small denoiser is faster and more reliable on CPU when ORT's thread
    # pool is bounded; CUDA launch/transfers dominate its 50 tiny calls.
    execution_provider: str = "cpu"
    onnx_intra_op_threads: int = 4
    # Local-root command [forward m/s, left m/s, yaw rad/s]. Arrow keys control
    # vx/wz, 1/2 control vy, and Space stops. A joystick may write this vector.
    default_velocity_command: tuple[float, float, float] = (0.5, 0.0, 0.0)
    velocity_command_limits: tuple[float, float, float] = (2.0, 1.0, 2.0)
    linear_velocity_step: float = 0.1
    angular_velocity_step: float = 0.1

    # Legacy option retained for configuration compatibility.  Sparse models
    # now interpolate actual H3 -> ten keyframes and track all 50 dense frames.
    seam_blend: float = 1.0
    seed: int = 42
