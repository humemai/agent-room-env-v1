import logging

logger = logging.getLogger()
logger.disabled = True

import unittest

from tqdm.auto import tqdm

from agent import HandcraftedAgent


class HandcraftedAgentTest(unittest.TestCase):
    def test_all_agents(self) -> None:
        for policy in tqdm(["random", "episodic_only", "semantic_only"]):
            for test_seed in [42]:
                all_params = {
                    "env_str": "room_env:RoomEnv-v1",
                    "env_config": {
                        "des_size": "l",
                        "question_prob": 1.0,
                        "allow_random_human": True,
                        "allow_random_question": True,
                        "check_resources": True,
                        "seed": test_seed,
                    },
                    "policy": policy,
                    "num_samples_for_results": 3,
                }
                if policy == "random":
                    all_params["capacity"] = {
                        "episodic": 16,
                        "semantic": 16,
                        "short": 1,
                    }
                elif policy == "episodic_only":
                    all_params["capacity"] = {"episodic": 32, "semantic": 0, "short": 1}
                else:
                    all_params["capacity"] = {"episodic": 0, "semantic": 32, "short": 1}

                agent = HandcraftedAgent(**all_params)
                agent.test()
                agent.remove_results_from_disk()
