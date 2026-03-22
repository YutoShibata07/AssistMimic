import torch
import os

from rl_games.algos_torch import players
from rl_games.algos_torch import torch_ext
from phc.utils.running_mean_std import RunningMeanStd
from rl_games.common.player import BasePlayer
from learning.distributed_wrapper import DistributedWrapper

import numpy as np
import gc
from gym import spaces


class CommonPlayer(players.PpoPlayerContinuous):

    def __init__(self, config):
        # Handle distributed setup
        self.distributed_wrapper = DistributedWrapper()
        
        # Check if we're in distributed mode and adjust device accordingly
        if self.distributed_wrapper.is_distributed:
            config = config.copy()
            if 'device' in config:
                config['device'] = f'cuda:{self.distributed_wrapper.local_rank}'
        
        BasePlayer.__init__(self, config)
        self.network = config['network']

        self._setup_action_space()
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']

        net_config = self._build_net_config()
        self._build_net(net_config)
        self.first = True
        return

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn
        for t in range(n_games):
            if games_played >= n_games:
                break

            obs_dict = self.env_reset()

            batch_size = 1
            batch_size = self.get_batch_size(obs_dict['obs'], batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            print_game_res = False

            done_indices = []

            with torch.no_grad():
                for n in range(self.max_steps):

                    obs_dict = self.env_reset(done_indices)

                    if has_masks:
                        masks = self.env.get_action_mask()
                        action = self.get_masked_action(obs_dict, masks, is_determenistic)
                    else:
                        action = self.get_action(obs_dict, is_determenistic)

                    # print(obs_dict[0].cpu().numpy())
                    # print("needing a very very fine comb here. ")
                    # import joblib; joblib.dump(obs_dict[0].cpu().numpy(), "a.pkl")
                    # np.abs(joblib.load("a.pkl") - obs_dict[0].cpu().numpy()).sum()

                    # import joblib; joblib.dump(obs_dict['obs'].detach().cpu().numpy(), "a.pkl")
                    # import joblib; np.abs(joblib.load("a.pkl")[0] - obs_dict['obs'][0].detach().cpu().numpy()).sum()
                    # joblib.dump(action, "a.pkl")
                    # joblib.load("a.pkl")[0] - action[0]

                    obs_dict, r, done, info = self.env_step(self.env, action)

                    cr += r
                    steps += 1

                    self._post_step(info)

                    if render:
                        self.env.render(mode='human')
                        time.sleep(self.render_sleep)

                    all_done_indices = done.nonzero(as_tuple=False)
                    done_indices = all_done_indices[::self.num_agents]
                    done_count = len(done_indices)
                    games_played += done_count

                    if done_count > 0:
                        if self.is_rnn:
                            for s in self.states:
                                s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0

                        cur_rewards = cr[done_indices].sum().item()
                        cur_steps = steps[done_indices].sum().item()

                        cr = cr * (1.0 - done.float())
                        steps = steps * (1.0 - done.float())
                        sum_rewards += cur_rewards
                        sum_steps += cur_steps

                        game_res = 0.0
                        if isinstance(info, dict):
                            if 'battle_won' in info:
                                print_game_res = True
                                game_res = info.get('battle_won', 0.5)
                            if 'scores' in info:
                                print_game_res = True
                                game_res = info.get('scores', 0.5)
                        if self.print_stats:
                            if print_game_res:
                                print('reward:', cur_rewards / done_count, 'steps:', cur_steps / done_count, 'w:', game_res)
                            else:
                                print('reward:', cur_rewards / done_count, 'steps:', cur_steps / done_count)

                        sum_game_res += game_res
                        # if batch_size//self.num_agents == 1 or games_played >= n_games:
                        if games_played >= n_games:
                            break

                    done_indices = done_indices[:, 0]

        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life)

        return

    def obs_to_torch(self, obs):
        obs = super().obs_to_torch(obs)
        obs_dict = {'obs': obs}
        return obs_dict

    def get_action(self, obs_dict, is_determenistic=False):
        # Save observations to npy for comparison with holosoma
        import os
        _save_obs_dir = os.environ.get("ASSISTMIMIC_SAVE_OBS_DIR", "")
        if _save_obs_dir:
            if not hasattr(self, '_obs_save_step'):
                self._obs_save_step = 0
            if self._obs_save_step < 100:  # Save first 100 steps
                import numpy as np
                os.makedirs(_save_obs_dir, exist_ok=True)
                obs = obs_dict['obs']
                # For multi-agent: extract recipient observations (agent 1 = odd indices)
                if hasattr(self, 'num_agents') and self.num_agents == 2:
                    recipient_obs_raw = obs[1::2, :].cpu().numpy()  # Recipient at odd indices
                else:
                    recipient_obs_raw = obs.cpu().numpy()

                # Get normalized observation
                if hasattr(self, 'running_mean_std') and self.running_mean_std is not None:
                    obs_norm = self._preproc_obs(obs)
                    if hasattr(self, 'num_agents') and self.num_agents == 2:
                        recipient_obs_norm = obs_norm[1::2, :].cpu().numpy()
                    else:
                        recipient_obs_norm = obs_norm.cpu().numpy()
                else:
                    recipient_obs_norm = recipient_obs_raw

                np.save(f"{_save_obs_dir}/assistmimic_recipient_obs_raw_step{self._obs_save_step:04d}.npy", recipient_obs_raw[0])
                np.save(f"{_save_obs_dir}/assistmimic_recipient_obs_norm_step{self._obs_save_step:04d}.npy", recipient_obs_norm[0])

                if self._obs_save_step == 0:
                    print(f"[OBS-SAVE] Saving observations to {_save_obs_dir}")
                    if hasattr(self, 'running_mean_std') and self.running_mean_std is not None:
                        np.save(f"{_save_obs_dir}/assistmimic_rms_mean.npy",
                                self.running_mean_std.running_mean.cpu().numpy())
                        np.save(f"{_save_obs_dir}/assistmimic_rms_var.npy",
                                self.running_mean_std.running_var.cpu().numpy())
                        print(f"[OBS-SAVE] Saved RMS mean shape: {self.running_mean_std.running_mean.shape}")
                self._obs_save_step += 1

        output = super().get_action(obs_dict['obs'], is_determenistic)
        return output

    def env_step(self, env, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()

        result = env.step(actions)
        if len(result) == 5:
            obs, rewards, dones, infos, expert = result
        else:
            obs, rewards, dones, infos = result

        if hasattr(obs, 'dtype') and obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            return obs, rewards.to(self.device), dones.to(self.device), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return self.obs_to_torch(obs), torch.from_numpy(rewards), torch.from_numpy(dones), infos

    def _build_net(self, config):
        if self.normalize_input:
            if "env" in self.__dict__:
                obs_shape = torch_ext.shape_whc_to_cwh(self.env.task.get_running_mean_size())
            else:
                obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
            self.running_mean_std = RunningMeanStd(obs_shape).to(self.device)
            self.running_mean_std.eval()
        config['mean_std'] = self.running_mean_std
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

        return

    def env_reset(self, env_ids=None):
        obs = self.env.reset(env_ids)
        return self.obs_to_torch(obs)

    def _post_step(self, info):
        return

    def _build_net_config(self):
        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        config = {'actions_num': self.actions_num, 'input_shape': obs_shape, 'num_seqs': self.num_agents}
        return config

    def _setup_action_space(self):
        self.actions_num = self.action_space.shape[0]
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        return
    
class CommonPlayerDiscrete(players.PpoPlayerDiscrete):

    def __init__(self, config):
        BasePlayer.__init__(self, config)
        self.network = config['network']

        self._setup_action_space()
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']

        net_config = self._build_net_config()
        self._build_net(net_config)
        self.first = True
        return

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn
        for t in range(n_games):
            if games_played >= n_games:
                break

            obs_dict = self.env_reset()

            batch_size = 1
            batch_size = self.get_batch_size(obs_dict['obs'], batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            print_game_res = False

            done_indices = []

            with torch.no_grad():
                for n in range(self.max_steps):

                    obs_dict = self.env_reset(done_indices)

                    if has_masks:
                        masks = self.env.get_action_mask()
                        action = self.get_masked_action(obs_dict, masks, is_determenistic)
                    else:
                        action = self.get_action(obs_dict, is_determenistic)

                    # print(obs_dict[0].cpu().numpy())
                    # print("needing a very very fine comb here. ")
                    # import joblib; joblib.dump(obs_dict[0].cpu().numpy(), "a.pkl")
                    # np.abs(joblib.load("a.pkl") - obs_dict[0].cpu().numpy()).sum()

                    # import joblib; joblib.dump(obs_dict['obs'].detach().cpu().numpy(), "a.pkl")
                    # import joblib; np.abs(joblib.load("a.pkl")[0] - obs_dict['obs'][0].detach().cpu().numpy()).sum()
                    # joblib.dump(action, "a.pkl")
                    # joblib.load("a.pkl")[0] - action[0]
                    obs_dict, r, done, info = self.env_step(self.env, action)

                    cr += r
                    steps += 1

                    self._post_step(info)

                    if render:
                        self.env.render(mode='human')
                        time.sleep(self.render_sleep)

                    all_done_indices = done.nonzero(as_tuple=False)
                    done_indices = all_done_indices[::self.num_agents]
                    done_count = len(done_indices)
                    games_played += done_count

                    if done_count > 0:
                        if self.is_rnn:
                            for s in self.states:
                                s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0

                        cur_rewards = cr[done_indices].sum().item()
                        cur_steps = steps[done_indices].sum().item()

                        cr = cr * (1.0 - done.float())
                        steps = steps * (1.0 - done.float())
                        sum_rewards += cur_rewards
                        sum_steps += cur_steps

                        game_res = 0.0
                        if isinstance(info, dict):
                            if 'battle_won' in info:
                                print_game_res = True
                                game_res = info.get('battle_won', 0.5)
                            if 'scores' in info:
                                print_game_res = True
                                game_res = info.get('scores', 0.5)
                        if self.print_stats:
                            if print_game_res:
                                print('reward:', cur_rewards / done_count, 'steps:', cur_steps / done_count, 'w:', game_res)
                            else:
                                print('reward:', cur_rewards / done_count, 'steps:', cur_steps / done_count)

                        sum_game_res += game_res
                        # if batch_size//self.num_agents == 1 or games_played >= n_games:
                        if games_played >= n_games:
                            break

                    done_indices = done_indices[:, 0]

        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life)

        return

    def obs_to_torch(self, obs):
        obs = super().obs_to_torch(obs)
        obs_dict = {'obs': obs}
        return obs_dict

    def get_action(self, obs_dict, is_determenistic=False):
        # Save observations to npy for comparison with holosoma
        import os
        _save_obs_dir = os.environ.get("ASSISTMIMIC_SAVE_OBS_DIR", "")
        if _save_obs_dir:
            if not hasattr(self, '_obs_save_step'):
                self._obs_save_step = 0
            if self._obs_save_step < 100:  # Save first 100 steps
                import numpy as np
                os.makedirs(_save_obs_dir, exist_ok=True)
                obs = obs_dict['obs']
                # For multi-agent: extract recipient observations (agent 1 = odd indices)
                if hasattr(self, 'num_agents') and self.num_agents == 2:
                    recipient_obs_raw = obs[1::2, :].cpu().numpy()  # Recipient at odd indices
                else:
                    recipient_obs_raw = obs.cpu().numpy()

                # Get normalized observation
                if hasattr(self, 'running_mean_std') and self.running_mean_std is not None:
                    obs_norm = self._preproc_obs(obs)
                    if hasattr(self, 'num_agents') and self.num_agents == 2:
                        recipient_obs_norm = obs_norm[1::2, :].cpu().numpy()
                    else:
                        recipient_obs_norm = obs_norm.cpu().numpy()
                else:
                    recipient_obs_norm = recipient_obs_raw

                np.save(f"{_save_obs_dir}/assistmimic_recipient_obs_raw_step{self._obs_save_step:04d}.npy", recipient_obs_raw[0])
                np.save(f"{_save_obs_dir}/assistmimic_recipient_obs_norm_step{self._obs_save_step:04d}.npy", recipient_obs_norm[0])

                if self._obs_save_step == 0:
                    print(f"[OBS-SAVE] Saving observations to {_save_obs_dir}")
                    if hasattr(self, 'running_mean_std') and self.running_mean_std is not None:
                        np.save(f"{_save_obs_dir}/assistmimic_rms_mean.npy",
                                self.running_mean_std.running_mean.cpu().numpy())
                        np.save(f"{_save_obs_dir}/assistmimic_rms_var.npy",
                                self.running_mean_std.running_var.cpu().numpy())
                        print(f"[OBS-SAVE] Saved RMS mean shape: {self.running_mean_std.running_mean.shape}")
                self._obs_save_step += 1

        output = super().get_action(obs_dict['obs'], is_determenistic)
        return output

    def env_step(self, env, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()

        result = env.step(actions)
        if len(result) == 5:
            obs, rewards, dones, infos, expert = result
        else:
            obs, rewards, dones, infos = result

        if hasattr(obs, 'dtype') and obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            return obs, rewards.to(self.device), dones.to(self.device), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return self.obs_to_torch(obs), torch.from_numpy(rewards), torch.from_numpy(dones), infos

    def _build_net(self, config):
        if self.normalize_input:
            obs_shape = torch_ext.shape_whc_to_cwh(self.env.task.get_running_mean_size())
            self.running_mean_std = RunningMeanStd(obs_shape).to(self.device)
            self.running_mean_std.eval()
        config['mean_std'] = self.running_mean_std
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

        return

    def env_reset(self, env_ids=None):
        obs = self.env.reset(env_ids)
        return self.obs_to_torch(obs)

    def _post_step(self, info):
        return

    def _build_net_config(self):
        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        config = {'actions_num': self.actions_num, 'input_shape': obs_shape, 'num_seqs': self.num_agents}
        return config

    def _setup_action_space(self):
        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape
        
        if type(action_space) is spaces.Discrete:
            self.actions_num = action_space.n
            self.is_multi_discrete = False
        if type(action_space) is spaces.Tuple:
            self.actions_num = [action.n for action in action_space]
            self.is_multi_discrete = True
        return
    
    def _change_char_color(self, env_ids):
        base_col = np.array([0.4, 0.4, 0.4])
        range_col = np.array([0.0706, 0.149, 0.2863])
        range_sum = np.linalg.norm(range_col)

        rand_col = np.random.uniform(0.0, 1.0, size=3)
        rand_col = range_sum * rand_col / np.linalg.norm(rand_col)
        rand_col += base_col
        self.vec_env.env.task.set_char_color(rand_col, env_ids)
        return