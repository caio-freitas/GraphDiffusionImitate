"""
Usage:
Training:
python demo_env.py
"""
import logging
from omegaconf import DictConfig
from imitation.env.pybullet.se2_envs.robot_se2_pickplace import SE2BotPickPlace

log = logging.getLogger(__name__)


def demo():
    log.info("Running demo...")

    robot = SE2BotPickPlace()
    s = robot.reset()

    grasps = robot.grasp_objects[0].all_grasp_abs

    robot.setControlMode("position")
    while(True):
        s = robot.step()




if __name__ == "__main__":
    demo()