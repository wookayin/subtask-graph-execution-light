import random
import pytest
import sys

from sge.mazeenv import MazeEnv


class TestMazeEnv:

    @pytest.mark.parametrize("game_name, graph_param", [
        ("mining", "train_1"),
        ("playground", "D1_train_1"),
    ])
    def test_maze_env_loaded(self, game_name: str, graph_param: str):
        print("[%s]" % game_name)
        env = MazeEnv(game_name=game_name,
                      graph_param=graph_param,  # scenario (Difficulty)
                      game_len=70,  # episode length
                      )
        s, info = env.reset()
        assert 'observation' in s
        assert 'mask' in s
        assert 'completion' in s
        assert 'eligibility' in s
        assert 'step' in s
        assert 'graph' in info  # SubtaskGraph

        self._examine_graph(env)
        self._run_random_episode(env)

    @pytest.mark.parametrize("game_name, graph_param", [
        ("mining", None),
        ("playground", None),
    ])
    def test_maze_env_generated(self, game_name: str, graph_param):
        print("[%s]" % game_name)
        env = MazeEnv(game_name=game_name,
                      graph_param=graph_param,  # scenario (Difficulty)
                      game_len=70,  # episode length
                      )
        s, info = env.reset()
        assert 'observation' in s
        assert 'mask' in s
        assert 'completion' in s
        assert 'eligibility' in s
        assert 'step' in s
        assert 'graph' in info  # SubtaskGraph

        self._examine_graph(env)
        self._run_random_episode(env)

    def _examine_graph(self, env: MazeEnv):
        """Print brief information about the subtask graph in the env."""
        print(env.graph)

    def _run_random_episode(self, env: MazeEnv, verbose=False):
        # Action spaces are enum.
        action_space = env.get_actions()
        print(action_space)

        # interaction with a random agent
        print("Running a random episode...")
        step, done = 0, False
        t = 0
        R = 0
        while not done:
            action = random.choice(list(action_space))
            s, r, done, info = env.step(action)
            if verbose:
                print('Step={:02d}, Action={}, Reward={:.2f}, Done={}'.format(
                    t, action, r, done))
            t += 1
            R += r
            if t > 200:
                raise RuntimeError("Episode did not finish...")
        print(f"Episode terminated in {t} steps, Return = {R}")

    def test_maze_env_parameterized(self):
        env = MazeEnv(game_name='playground', graph_param=None,
                      auto_reset_graph=False)
        print(env.graph)
        with pytest.raises(RuntimeError):
            env.reset()

        # override pool_size and subtasks.
        env.reset_graph(subtask_pool_size=14, subtasks=12, num_layers=4,
                        subtask_design=[])
        print(env.graph)
        assert env.graph.max_task == 14
        assert env.max_task == env.graph.max_task
        assert env.graph.ntasks == 12
        assert env.graph.nlayer == 4
        # TODO: validate per-layer distractors and subtasks.
        env.reset()
        self._run_random_episode(env)

        # subtasks: As a subset.
        env.reset_graph(subtask_pool_size=14, subtasks=[0, 2, 4, 6, 8, 10],
                        num_layers=2,
                        subtask_design=[])
        print(env.graph)
        assert env.graph.max_task == 14
        assert env.max_task == env.graph.max_task
        assert env.graph.ntasks == 6
        assert env.graph.nlayer == 2
        # TODO: validate per-layer distractors and subtasks.
        env.reset()
        self._run_random_episode(env)

        # subtasks: design
        design = [
            {'id': 0, 'layer': 0, 'distractor': 0},
            {'id': 1, 'layer': 1, 'distractor': 1},
            {'id': 2, 'layer': 2, 'distractor': 0},
        ]
        env.reset_graph(subtask_pool_size=14, subtasks=14, num_layers=3,
                        subtask_design=design)
        env.reset()


if __name__ == '__main__':
    sys.exit(pytest.main(["-s", "-v"] + sys.argv))
