import os
import sys
import numpy as np
from typing import Optional

from sge.mazemap import Mazemap
from sge.graph import SubtaskGraphCache, SubtaskGraphGenerator
from sge.utils import get_id_from_ind_multihot
from sge.utils import WHITE, BLACK, DARK, LIGHT, GREEN, DARK_RED


class MazeEnv(object):  # single batch

    def __init__(self,
                 game_name: str,
                 graph_param: Optional[str],
                 *,
                 auto_reset_graph: bool = True,
                 game_len: int = 70,
                 gamma: float = 0.99):
        """Create a new MazeEnv.

        Args:
            game_name: `playground` or `mining`.
            graph_param: Denotes predefined preset (train_1, D1_train_1), etc.
                If None, random generation will be used.
            """
        if game_name == 'playground':
            from sge.playground import Playground
            game_config = Playground()
            graph_folder = os.path.join('.', 'data', 'subtask_graph_play')
            filename = 'play_{param}'.format(param=graph_param)

        elif game_name == 'mining':
            from sge.mining import Mining
            game_config = Mining()
            graph_folder = os.path.join('.', 'data', 'subtask_graph_mining')
            filename = 'mining_{param}'.format(param=graph_param)

        self.config = game_config
        self.subtask_list = self.config.subtask_list
        self._auto_reset_graph = auto_reset_graph

        # graph & map
        if graph_param:
            self.graph = SubtaskGraphCache(
                graph_folder, filename, self.config.nb_subtask_type)  # just load all graph
        else:
            self.graph = SubtaskGraphGenerator(game_config, env_name=game_name)

        self.map = Mazemap(game_name, game_config)
        self.gamma = gamma

        # init
        self.game_length = int(np.random.uniform(0.8, 1.2) * game_len)
        self.step_reward = 0.0

    @property
    def max_task(self):
        return self.graph.max_task

    def step(self, action):
        if self.graph.graph_index is None:
            raise RuntimeError('Error: Environment has never been reset()')
        sub_id = -1
        if self.game_over or self.time_over:
            raise ValueError(
                'Environment has already been terminated. need to be reset!')
        oid = self.map.act(action)
        if (action, oid) in self.config.subtask_param_to_id:  # if (action, item) is one of the subtasks
            sid = self.config.subtask_param_to_id[(action, oid)]

            if sid in self.subtask_id_list:  # if sub_id is in the subtask graph
                sub_id = sid
            else:
                #print('Warning! Executed a non-existing subtask')
                pass
        #
        self.reward = self._act_subtask(sub_id)
        self.ret += self.reward*self.gamma
        self.step_count += 1
        self.time_over = self.step_count >= self.game_length
        self.game_over = (self.eligibility*self.mask).sum().item() == 0

        return self._get_state(), self.reward, (self.game_over or self.time_over), self._get_info()

    def reset_graph(self, **kwargs):
        self.graph.reset_graph(**kwargs)

    def reset(self, graph_index=None):  # after every episode
        #if self.seed is not None:
        #    np.random.seed(self.seed)

        # 1. reset graph
        if self._auto_reset_graph:
            self.graph.reset_graph(graph_index=graph_index)

        if self.graph.graph_index is None:
            raise RuntimeError('Error: Environment has never called reset_graph()')

        # 2. reset subtask status
        self.executed_sub_ind = -1
        self.game_over = False
        self.time_over = False
        self.mask = np.ones(self.nb_subtask, dtype=np.uint8)
        self.mask_id = np.zeros(self.max_task, dtype=np.uint8)

        for ind, sub_id in self.graph.ind_to_id.items():
            self.mask_id[sub_id] = 1
        self.completion = np.zeros(self.nb_subtask, dtype=np.int8)
        self.comp_id = np.zeros(self.max_task, dtype=np.uint8)
        self._compute_elig()
        self.step_count, self.ret, self.reward = 0, 0, 0

        # 3. reset map
        self.map.reset(self.subtask_id_list)

        return self._get_state(), self._get_info()

    @property
    def nb_subtask(self):
        return len(self.graph.subtask_id_list)
    @property
    def rew_mag(self):
        return self.graph.rew_mag
    @property
    def subtask_id_list(self):
        return self.graph.subtask_id_list

    def state_spec(self):
        return [
            {'dtype': self.map.get_obs().dtype, 'name': 'observation', 'shape': self.map.get_obs().shape},
            {'dtype': self.mask_id.dtype, 'name': 'mask', 'shape': self.mask_id.shape},
            {'dtype': self.comp_id.dtype, 'name': 'completion', 'shape': self.comp_id.shape},
            {'dtype': self.elig_id.dtype, 'name': 'eligibility', 'shape': self.elig_id.shape},
            {'dtype': int, 'name': 'step', 'shape': ()}
        ]

    def get_actions(self):
        return self.config.legal_actions

    # internal
    def _get_state(self):
        step = self.game_length - self.step_count
        return {
            'observation': self.map.get_obs(),
            'mask': self.mask_id.astype(np.float),
            'completion': self.comp_id.astype(np.float),
            'eligibility': self.elig_id.astype(np.float),
            'step': step
        }

    def _get_info(self):
        return {
            'graph': self.graph
        }

    def _act_subtask(self, sub_id):
        self.executed_sub_ind = -1
        reward = self.step_reward
        if sub_id < 0:
            return reward
        sub_ind = self.graph.id_to_ind[sub_id]
        if self.eligibility[sub_ind] == 1 and self.mask[sub_ind] == 1:
            self.completion[sub_ind] = 1
            self.comp_id[sub_id] = 1
            reward += self.rew_mag[sub_ind]
            self.executed_sub_ind = sub_ind
        self.mask[sub_ind] = 0
        self.mask_id[sub_id] = 0

        self._compute_elig()

        return reward

    def _compute_elig(self):
        self.eligibility = self.graph.get_elig(self.completion)
        self.elig_id = get_id_from_ind_multihot(
            self.eligibility, self.graph.ind_to_id, self.max_task)
