import logging

logger = logging.getLogger()
logger.disabled = True

import unittest

from agent import DQNAgent


class RLAgentTest(unittest.TestCase):
    def test_agent(self) -> None:
        for pretrain_semantic in [False, True]:
            for test_seed in [42]:
                # parameters
                all_params = {
                    "env_str": "room_env:RoomEnv-v1",
                    "max_epsilon": 1.0,
                    "min_epsilon": 0.1,
                    "epsilon_decay_until": 128 * 1,
                    "gamma": 0.65,
                    "capacity": {"episodic": 4, "semantic": 5, "short": 1},
                    "nn_params": {
                        "hidden_size": 4,
                        "num_layers": 2,
                        "n_actions": 3,
                        "embedding_dim": 4,
                        "include_positional_encoding": False,
                    },
                    "num_iterations": 128 * 2,
                    "replay_buffer_size": 2 * 4,
                    "warm_start": 2 * 4,
                    "batch_size": 2,
                    "target_update_interval": 10,
                    "pretrain_semantic": pretrain_semantic,
                    "run_test": True,
                    "num_samples_for_results": 3,
                    "train_seed": test_seed + 5,
                    "plotting_interval": 10,
                    "device": "cpu",
                    "test_seed": test_seed,
                }

                agent = DQNAgent(**all_params)
                agent.train()
                agent.remove_results_from_disk()
