"""PPO Agent for the RoomEnv1 environment."""

import os
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from humemai.policy import answer_question, encode_observation, manage_memory
from humemai.utils import write_yaml
from tqdm.auto import tqdm

from ..handcrafted import HandcraftedAgent
from .nn import LSTM
from .utils import (plot_results, save_final_results,
                    save_states_actions_probs_values, save_validation,
                    select_action, update_model)


class PPOAgent(HandcraftedAgent):
    """PPO Agent interacting with environment.

    Based on https://github.com/MrSyee/pg-is-all-you-need
    """

    def __init__(
        self,
        env_str: str = "room_env:RoomEnv-v1",
        env_config: dict = {
            "des_size": "l",
            "question_prob": 1.0,
            "allow_random_human": False,
            "allow_random_question": False,
            "check_resources": True,
        },
        num_episodes: int = 10,
        num_rollouts: int = 2,
        epoch_per_rollout: int = 64,
        batch_size: int = 128,
        gamma: float = 0.9,
        tau: float = 0.8,
        epsilon: float = 0.2,
        entropy_weight: float = 0.005,
        capacity: dict = {
            "episodic": 16,
            "semantic": 16,
            "short": 1,
        },
        pretrain_semantic: str | bool = False,
        nn_params: dict = {
            "hidden_size": 64,
            "num_layers": 2,
            "embedding_dim": 64,
            "include_human": "sum",
            "human_embedding_on_object_location": False,
            "fuse_information": "sum",
            "include_positional_encoding": True,
            "max_timesteps": 128,
            "max_strength": 128,
        },
        run_test: bool = True,
        num_samples_for_results: int = 10,
        plotting_interval: int = 10,
        train_seed: int = 42,
        test_seed: int = 42,
        device: str = "cpu",
        default_root_dir: str = "./training-results/",
    ):
        """Initialization.

        Args:
            env_str: This has to be "room_env:RoomEnv-v1"
            env_config: The configuration of the environment.
            num_episodes: The number of iterations to train the agent.
            replay_buffer_size: The size of the replay buffer.
            batch_size: The batch size for training This is the amount of samples
                sampled from the replay buffer.
            max_epsilon: The maximum epsilon.
            min_epsilon: The minimum epsilon.
            gamma: The discount factor.
            capacity: The capacity of each human-like memory systems.
            pretrain_semantic: Whether or not to pretrain the semantic memory
                system.
            nn_params: The parameters for the function approximator.
            run_test: Whether or not to run test.
            num_samples_for_results: The number of samples to validate / test the agent.
            plotting_interval: The interval to plot the results.
            train_seed: The random seed for train.
            test_seed: The random seed for test.
            device: The device to run the agent on. This is either "cpu" or "cuda".
            default_root_dir: default root directory to store the results.

        """
        all_params = deepcopy(locals())
        del all_params["self"]
        del all_params["__class__"]
        self.all_params = deepcopy(all_params)
        self.train_seed = train_seed
        self.test_seed = test_seed
        self.env_config = env_config
        super().__init__(
            env_str=env_str,
            env_config={**self.env_config, "seed": self.train_seed},
            policy="rl",
            num_samples_for_results=num_samples_for_results,
            capacity=capacity,
            pretrain_semantic=pretrain_semantic,
            default_root_dir=default_root_dir,
        )
        write_yaml(self.all_params, os.path.join(self.default_root_dir, "train.yaml"))

        self.val_filenames = []
        self.num_episodes = num_episodes
        self.plotting_interval = plotting_interval
        self.run_test = run_test
        self.device = torch.device(device)
        print(f"Running on {self.device}")

        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.tau = tau
        self.epoch_per_rollout = epoch_per_rollout
        self.num_rollouts = num_rollouts
        self.entropy_weight = entropy_weight

        self.num_steps_in_episode = self.env.unwrapped.des.until
        self.total_maximum_episode_rewards = (
            self.env.unwrapped.total_maximum_episode_rewards
        )

        assert (self.num_rollouts % self.num_episodes) == 0 or (
            self.num_episodes % self.num_rollouts
        ) == 0

        self.num_steps_per_rollout = int(
            self.num_episodes / self.num_rollouts * self.num_steps_in_episode
        )

        self.action2str = {
            0: "episodic",
            1: "semantic",
            2: "forget",
        }
        self.action_space = gym.spaces.Discrete(len(self.action2str))

        self.nn_params = nn_params
        self.nn_params["capacity"] = self.capacity
        self.nn_params["device"] = self.device
        self.nn_params["entities"] = (
            self.env.des.humans + self.env.des.objects + self.env.des.object_locations
        )
        # there is only one relation in v1, so just ignore it.
        self.nn_params["relations"] = []

        self.nn_params["memory_of_interest"] = ["episodic", "semantic", "short"]
        self.nn_params["n_actions"] = len(self.action2str)

        self.actor = LSTM(**self.nn_params, is_actor=True, is_critic=False)
        self.critic = LSTM(**self.nn_params, is_actor=False, is_critic=True)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters())
        self.critic_optimizer = optim.Adam(self.critic.parameters())

        # global stats to save
        self.actor_losses, self.critic_losses = [], []  # training loss
        self.states_all = {"train": [], "val": [], "test": []}
        self.scores_all = {"train": [], "val": [], "test": None}
        self.actions_all = {"train": [], "val": [], "test": []}
        self.actor_probs_all = {"train": [], "val": [], "test": []}
        self.critic_values_all = {"train": [], "val": [], "test": []}

    def create_empty_rollout_buffer(self) -> tuple[list, list, list, list, list, list]:
        """Create empty buffer for training.

        Make sure to call this before and after each rollout.

        Returns:
            states_buffer: The states. actions_buffer: The actions. rewards_buffer: The
            rewards. values_buffer: The values. masks_buffer: The masks.
            log_probs_buffer: The log probabilities.
        """
        # memory for training
        states_buffer: list[dict] = []  # this has to be a list of dictionaries
        actions_buffer: list[torch.Tensor] = []
        rewards_buffer: list[torch.Tensor] = []
        values_buffer: list[torch.Tensor] = []
        masks_buffer: list[torch.Tensor] = []
        log_probs_buffer: list[torch.Tensor] = []

        return (
            states_buffer,
            actions_buffer,
            rewards_buffer,
            values_buffer,
            masks_buffer,
            log_probs_buffer,
        )

    def init_memory_env_reset_encode_observation(self) -> list:
        """Init memory systems, reset environment, and encode observation.

        Returns:
            question: [head, relation, ?, current_time]
        """
        self.init_memory_systems()
        (observation, question), info = self.env.reset()
        encode_observation(self.memory_systems, observation)

        return question

    def step(
        self,
        question: list,
        is_train_val_test: str,
        states_buffer: list | None = None,
        actions_buffer: list | None = None,
        values_buffer: list | None = None,
        log_probs_buffer: list | None = None,
        append_states_actions_probs_values: bool = False,
        append_states: bool = False,
    ) -> tuple[int, bool, list[str]]:
        """Interact with the actual gymnasium environment by taking a step.

        Args:
            question: The question from the environment.
            is_train_val_test: Whether the agent is in train, validation, or test mode.



        Returns:
            reward: reward from the gynmasium environment
            done: whether the environment terminated or not
            question: the question from the environment

        """
        state = self.memory_systems.return_as_a_dict_list()
        action, actor_probs, critic_value = select_action(
            actor=self.actor,
            critic=self.critic,
            state=state,
            is_test=(is_train_val_test in ["val", "test"]),
            states=states_buffer,
            actions=actions_buffer,
            values=values_buffer,
            log_probs=log_probs_buffer,
        )

        if append_states_actions_probs_values:
            if append_states:
                # state is a list, which is a mutable object. So, we need to deepcopy
                # it.
                self.states_all[is_train_val_test].append(deepcopy(state))
            else:
                self.states_all[is_train_val_test].append(None)
            self.actions_all[is_train_val_test].append(action)
            self.actor_probs_all[is_train_val_test].append(actor_probs)
            self.critic_values_all[is_train_val_test].append(critic_value)

        manage_memory(
            self.memory_systems,
            self.action2str[action],
            split_possessive=True,
        )

        answer = str(
            answer_question(self.memory_systems, "episodic_semantic", question)
        )

        (
            (observation, question),
            reward,
            done,
            truncated,
            info,
        ) = self.env.step(answer)

        encode_observation(self.memory_systems, observation)
        done = done or truncated

        return reward, done, question

    def train(self):
        """Train the agent."""

        self.num_validation = 0
        new_episode_starts = True
        score = 0
        episode_idx = 0

        for _ in tqdm(range(self.num_rollouts)):
            (
                states_buffer,
                actions_buffer,
                rewards_buffer,
                values_buffer,
                masks_buffer,
                log_probs_buffer,
            ) = self.create_empty_rollout_buffer()

            for _ in range(self.num_steps_per_rollout):

                if new_episode_starts:
                    question = self.init_memory_env_reset_encode_observation()

                reward, done, question = self.step(
                    question=question,
                    is_train_val_test="train",
                    states_buffer=states_buffer,
                    actions_buffer=actions_buffer,
                    values_buffer=values_buffer,
                    log_probs_buffer=log_probs_buffer,
                    append_states_actions_probs_values=True,
                    append_states=False,
                )
                score += reward

                reward = np.reshape(reward, (1, -1)).astype(np.float64)
                done = np.reshape(done, (1, -1))
                rewards_buffer.append(torch.FloatTensor(reward).to(self.device))
                masks_buffer.append(torch.FloatTensor(1 - done).to(self.device))

                # if episode ends
                if done:
                    episode_idx += 1
                    self.scores_all["train"].append(score)
                    with torch.no_grad():
                        self.validate()

                    score = 0
                    new_episode_starts = True
                else:
                    new_episode_starts = False

            next_state = self.memory_systems.return_as_a_dict_list()
            actor_loss, critic_loss = update_model(
                next_state,
                states_buffer,
                actions_buffer,
                rewards_buffer,
                values_buffer,
                masks_buffer,
                log_probs_buffer,
                self.gamma,
                self.tau,
                self.epoch_per_rollout,
                self.batch_size,
                self.epsilon,
                self.entropy_weight,
                self.actor,
                self.critic,
                self.actor_optimizer,
                self.critic_optimizer,
            )

            self.actor_losses.append(actor_loss)
            self.critic_losses.append(critic_loss)

            # plotting & show training results
            self.plot_results("all", True)

        with torch.no_grad():
            self.test()

        self.env.close()
        save_states_actions_probs_values(
            self.states_all["train"],
            self.actions_all["train"],
            self.actor_probs_all["train"],
            self.critic_values_all["train"],
            self.default_root_dir,
            "train",
        )

    def validate(self) -> None:
        """Validate the agent."""
        self.actor.eval()
        self.critic.eval()

        scores = []
        for idx in range(self.num_samples_for_results):
            question = self.init_memory_env_reset_encode_observation()

            done = False
            score = 0
            while not done:
                if idx == self.num_samples_for_results - 1:
                    append_results = True
                else:
                    append_results = False

                reward, done, question = self.step(
                    question=question,
                    is_train_val_test="val",
                    states_buffer=None,
                    actions_buffer=None,
                    values_buffer=None,
                    log_probs_buffer=None,
                    append_states_actions_probs_values=append_results,
                    append_states=True,
                )
                score += reward

            scores.append(score)

        save_validation(
            scores=scores,
            scores_all_val=self.scores_all["val"],
            default_root_dir=self.default_root_dir,
            num_validation=self.num_validation,
            val_filenames=self.val_filenames,
            actor=self.actor,
            critic=self.critic,
        )

        start = self.num_validation * self.num_steps_in_episode
        end = (self.num_validation + 1) * self.num_steps_in_episode

        save_states_actions_probs_values(
            self.states_all["val"][start:end],
            self.actions_all["val"][start:end],
            self.actor_probs_all["val"][start:end],
            self.critic_values_all["val"][start:end],
            self.default_root_dir,
            "val",
            self.num_validation,
        )

        self.env.close()
        self.num_validation += 1
        self.actor.train()
        self.critic.train()

    def test(self, checkpoint: str = None) -> None:
        """Test the agent.

        Args:
            checkpoint: The checkpoint to load the model from. If None, the model from
                the best validation is used.

        """
        self.env = gym.make(self.env_str, **{**self.env_config, "seed": self.test_seed})
        self.actor.eval()
        self.critic.eval()

        assert len(self.val_filenames) == 1
        self.actor.load_state_dict(
            torch.load(os.path.join(self.val_filenames[0], "actor.pt"))
        )
        self.critic.load_state_dict(
            torch.load(os.path.join(self.val_filenames[0], "critic.pt"))
        )
        if checkpoint is not None:
            self.actor.load_state_dict(os.path.join(torch.load(checkpoint), "actor.pt"))
            self.critic.load_state_dict(
                os.path.join(torch.load(checkpoint), "critic.pt")
            )

        scores = []
        for idx in range(self.num_samples_for_results):
            question = self.init_memory_env_reset_encode_observation()

            done = False
            score = 0
            while not done:
                if idx == self.num_samples_for_results - 1:
                    append_results = True
                else:
                    append_results = False

                reward, done, question = self.step(
                    question=question,
                    is_train_val_test="test",
                    states_buffer=None,
                    actions_buffer=None,
                    values_buffer=None,
                    log_probs_buffer=None,
                    append_states_actions_probs_values=append_results,
                    append_states=True,
                )
                score += reward

            scores.append(score)

        self.scores_all["test"] = scores

        save_states_actions_probs_values(
            self.states_all["test"],
            self.actions_all["test"],
            self.actor_probs_all["test"],
            self.critic_values_all["test"],
            self.default_root_dir,
            "test",
        )

        save_final_results(
            self.scores_all,
            self.actor_losses,
            self.critic_losses,
            self.default_root_dir,
            self,
        )

        self.plot_results("all", True)
        self.env.close()
        self.actor.train()
        self.critic.train()

    def plot_results(self, to_plot: str = "all", save_fig: bool = False) -> None:
        """Plot things for ppo training.

        Args:
            to_plot: what to plot:
                all: everything
                actor_loss: actor loss
                critic_loss: critic loss
                scores: train, val, and test scores
                actor_probs_train: actor probabilities for training
                actor_probs_val: actor probabilities for validation
                actor_probs_test: actor probabilities for test
                critic_values_train: critic values for training
                critic_values_val: critic values for validation
                critic_values_test: critic values for test

        """
        plot_results(
            self.scores_all,
            self.actor_losses,
            self.critic_losses,
            self.actor_probs_all,
            self.critic_values_all,
            self.num_validation,
            self.action_space.n.item(),
            self.num_episodes,
            self.total_maximum_episode_rewards,
            self.default_root_dir,
            to_plot,
            save_fig,
        )
