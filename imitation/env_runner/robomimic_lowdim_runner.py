import imageio
import logging
import time
from typing import Dict

import gymnasium as gym
import collections
from imitation.agent.base_agent import BaseAgent
from imitation.env_runner.base_runner import BaseRunner
import wandb

log = logging.getLogger(__name__)

class RobomimicEnvRunner(BaseRunner):
    def __init__(self,
                env,
                output_dir,
                action_horizon,
                obs_horizon,
                render=True,
                fps=30,
                output_video=False,
                use_full_pred_after=1) -> None:
        super().__init__(output_dir)
        self.env = env
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.render = render
        self.fps = fps
        self.output_video = output_video
        self.use_full_pred_after = use_full_pred_after 
        self.output_dir = output_dir
        self.curr_video = None
        if self.output_video: # don't create video writer if not needed
            self.start_video()


        # keep a queue of last obs_horizon steps of observations
        self.reset()


    def start_video(self):
        self.curr_video = f"{self.output_dir}/output_{time.time()}.mp4"
        self.video_writer = imageio.get_writer(self.curr_video, fps=30)


    def end_video(self):
        if self.video_writer is not None:
            self.video_writer.close()
        # Log video to WandB
        if wandb.run is not None:
            log.info(f"Logging video to wandb")
            wandb.log({"video": wandb.Video(self.curr_video, fps=self.fps)})
        else:
            log.warning("wandb.run is None, not logging video to wandb")

    def reset(self) -> None:
        self.obs = self.env.reset()
        self.obs_deque = collections.deque(
            [self.obs] * self.obs_horizon, maxlen=self.obs_horizon)

    def run(self, agent: BaseAgent, n_steps: int) -> Dict:
        log.info(f"Running agent {agent.__class__.__name__} for {n_steps} steps")
        if self.output_video:
            self.start_video()
        done = False
        info = {}
        rewards = []
        action_horizon = self.action_horizon
        i = 0
        while i < n_steps:
            actions = agent.get_action(self.obs_deque)
            # Make sure the action is always [[...]]
            if len(actions.shape) == 1:
                log.warning("Action shape is 1D, adding batch dimension")
                actions = actions.reshape(1, -1)
            elif len(actions.shape) == 3:
                log.warning("Action shape is 3D, squeezing batch dimension")
                actions = actions.squeeze(0)
            if i >= self.use_full_pred_after * n_steps:
                log.info(f"Reaching end of episode, action_horizion = pred_horizon = {len(actions)}")
                action_horizon = len(actions)
            for j in range(1, action_horizon + 1): # skip first action
                action = actions[j] 
                if done:
                    self.env.close()
                    if self.output_video:
                        self.end_video()
                    return rewards, info
                obs, reward, done, info = self.env.step(action)
                self.obs_deque.append(obs)
                
                if self.render:
                    self.env.render()
                    # time.sleep(1/self.fps) # TODO properly fix the rendering speed or not

                if self.output_video:
                    # We need to directly grab full observations so we can get image data
                    full_obs = self.env.env._get_observations()

                    # Grab image data (assume relevant camera name is the first in the env camera array)
                    img = full_obs[self.env.env.camera_names[0] + "_image"]

                    # Write to video writer
                    self.video_writer.append_data(img[::-1])

                i += 1
            
        self.env.close()
        if self.output_video:
            self.end_video()

        return rewards, info