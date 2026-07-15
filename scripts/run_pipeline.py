# Fix OMP perfmance issue on ARM platform (Jetson)
# 我们以拿这个python script/run_pipeline -c g1_beyondmimic作为例子来讲解整篇代码
import os
import platform

if platform.machine().startswith("aarch64"):
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import logging
import time

import robojudo.pipeline
# python中要import包里的任何子模块必须先执行包的__init.py所以这句话首先会去执行config/__init__.py
from robojudo.config.config_manager import ConfigManager
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.pipeline.rl_pipeline import RlPipeline

logger = logging.getLogger("robojudo")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="g1",
        help="Name of the config class to use",
    )
    args = parser.parse_args()
    return args

def main():
    # 读取输入参数，我们的g1_beyondmimic就是配置类的名字
    args = parse_args()
    logger.info(f"Using config: {args.config}")
    # 这里初始化的时候就已经通过类名来获得类并且创建了对象
    config_manager = ConfigManager(config_name=args.config)
    # 这里已经得到了g1_beyondmimic的对象
    cfg: RlPipelineCfg = config_manager.get_cfg()
    # pipeline就是整个从环境获取数据交给网络，网络返回数据给环境的整个流程，这里从配置文件获得这个类的名称，在g1_beyondmimic是rlpipeline
    pipeline_type = cfg.pipeline_type
    # 通过pipeline类名获得类
    pipeline_class: type[RlPipeline] = getattr(robojudo.pipeline, pipeline_type)
    logger.info(f"Using pipeline: {pipeline_type} -> {pipeline_class}")
    # 通过类创建对象，并传入配置对象cfg
    pipeline = pipeline_class(cfg=cfg)
    # 如果环境不是仿真而是实机则会走prepare
    if not cfg.env.is_sim:
        pipeline.prepare()

    while True:
        # 记录步进前一次的时间
        time_start = time.time()
        # 步进一次，这里包括整个一次onnx的推理
        pipeline.step()
        # 记录结束后的时间
        time_end = time.time()
        time_diff = time_end - time_start

        # keep the pipeline running at the desired frequency
        if not cfg.run_fullspeed:
            # 这里pipeline.dt是0.02s
            time_diff = pipeline.dt - time_diff
            if time_diff > 0:
                # 去等待到0.02过去
                time.sleep(time_diff)
            else:
                if not cfg.env.is_sim:
                    logger.error(f"Warning: frame drop -> {time_diff}")
                    if time_diff < -0.2:
                        logger.critical("Exiting due to excessive frame drop")
                        pipeline.env.shutdown()
                        time.sleep(10)
                        break


if __name__ == "__main__":
    main()
