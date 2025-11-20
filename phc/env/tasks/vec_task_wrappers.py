# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from gym import spaces
import numpy as np
import torch
from phc.env.tasks.vec_task import VecTaskCPU, VecTaskGPU, VecTaskPython

class VecTaskCPUWrapper(VecTaskCPU):
    def __init__(self, task, rl_device, sync_frame_time=False, clip_observations=5.0):
        super().__init__(task, rl_device, sync_frame_time, clip_observations)
        return

class VecTaskGPUWrapper(VecTaskGPU):
    def __init__(self, task, rl_device, clip_observations=5.0):
        super().__init__(task, rl_device, clip_observations)
        return


class VecTaskPythonWrapper(VecTaskPython):
    def __init__(self, task, rl_device, clip_observations=5.0):
        super().__init__(task, rl_device, clip_observations)

        self._amp_obs_space = spaces.Box(np.ones(task.get_num_amp_obs()) * -np.Inf, np.ones(task.get_num_amp_obs()) * np.Inf)
        
        self._enc_amp_obs_space = spaces.Box(np.ones(task.get_num_enc_amp_obs()) * -np.Inf, np.ones(task.get_num_enc_amp_obs()) * np.Inf)
        return

    def reset(self, env_ids=None):
        out = self.task.reset(env_ids)
        obs_tensor = torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        # If task returns (obs, expert), propagate expert as a second return
        if isinstance(out, (tuple, list)) and len(out) == 2:
            _, expert = out
            return obs_tensor, expert
        return obs_tensor

    def step(self, actions):
        """Override step to support expert actions for distillation"""
        out = self.task.step(actions)

        # Read from task buffers (Isaac Gym style)
        obs = torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        rewards = self.task.rew_buf.to(self.rl_device)
        dones = self.task.reset_buf.to(self.rl_device)
        infos = self.task.extras

        # Check if task returned expert actions (for distillation)
        if isinstance(out, (tuple, list)) and len(out) == 5:
            _, _, _, _, expert = out
            return obs, rewards, dones, infos, expert
        # Check if task has _get_expert_actions method (distillation mode)
        elif hasattr(self.task, '_get_expert_actions'):
            try:
                expert_mus, expert_actions = self.task._get_expert_actions(None, obs)
                expert = {"mus": expert_mus, "actions": expert_actions}
                return obs, rewards, dones, infos, expert
            except Exception:
                # Fallback to standard mode if expert computation fails
                return obs, rewards, dones, infos
        else:
            # Standard mode
            return obs, rewards, dones, infos

    @property
    def amp_observation_space(self):
        return self._amp_obs_space
    
    @property
    def enc_amp_observation_space(self):
        return self._enc_amp_obs_space

    def fetch_amp_obs_demo(self, num_samples):
        return self.task.fetch_amp_obs_demo(num_samples)
    
    @property
    def enc_amp_observation_space(self):
        return self._enc_amp_obs_space
    
    ################ Calm ################
    def fetch_amp_obs_demo_pair(self, num_samples):
        return self.task.fetch_amp_obs_demo_pair(num_samples)

    def fetch_amp_obs_demo_enc_pair(self, num_samples):
        return self.task.fetch_amp_obs_demo_enc_pair(num_samples)

    def fetch_amp_obs_demo_per_id(self, num_samples, motion_ids):
        return self.task.fetch_amp_obs_demo_per_id(num_samples, motion_ids)
