"""Utility functions for DQN."""

import logging
import os
import random
from collections import deque
from typing import Deque, Literal

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from IPython.display import clear_output
from tqdm.auto import tqdm

from .utils import (
    argmax,
    is_running_notebook,
    list_duplicates_of,
    write_pickle,
    write_yaml,
)
from .segment_tree import SumSegmentTree, MinSegmentTree

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class ReplayBuffer:
    """A simple numpy replay buffer.

    numpy replay buffer is faster than deque or list.
    copied from https://github.com/Curt-Park/rainbow-is-all-you-need

    """

    def __init__(
        self,
        observation_type: Literal["dict", "tensor"],
        size: int,
        obs_dim: tuple = None,
        batch_size: int = 32,
    ):
        """Initialize replay buffer.

        Args:
            observation_type: "dict" or "tensor"
            size: size of the buffer
            batch_size: batch size to sample

        """
        if batch_size > size:
            raise ValueError("batch_size must be smaller than size")
        if observation_type == "dict":
            self.obs_buf = np.array([{}] * size)
            self.next_obs_buf = np.array([{}] * size)
        else:
            raise ValueError("At the moment, observation_type must be 'dict'")
            # self.obs_buf = np.zeros([size, *obs_dim], dtype=np.float32)
            # self.next_obs_buf = np.zeros([size, *obs_dim], dtype=np.float32)

        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        (
            self.ptr,
            self.size,
        ) = (
            0,
            0,
        )

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size


class ReplayBufferNStep:
    """A simple numpy N step replay buffer.
    copied from https://github.com/Curt-Park/rainbow-is-all-you-need

    """

    def __init__(
        self,
        observation_type: str,
        size: int,
        obs_dim: tuple = None,
        batch_size: int = 32,
        n_step: int = 1,
        gamma: float = 0.99,
    ):
        """Initialize replay buffer.

        Args:
            observation_type: "dict" or "tensor"
            size: size of the buffer
            batch_size: batch size to sample

        """
        if observation_type == "dict":
            self.obs_buf = np.array([{}] * size)
            self.next_obs_buf = np.array([{}] * size)
        else:
            self.obs_buf = np.zeros([size, *obs_dim], dtype=np.float32)
            self.next_obs_buf = np.zeros([size, *obs_dim], dtype=np.float32)

        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        (
            self.ptr,
            self.size,
        ) = (
            0,
            0,
        )

        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()

        # make a n-step transition
        rew, next_obs, done = self._get_n_step_info(self.n_step_buffer, self.gamma)
        obs, act = self.n_step_buffer[0][:2]

        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        return self.n_step_buffer[0]

    def sample_batch(self) -> dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)

        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
            # for N-step Learning
            indices=idxs,
        )

    def sample_batch_from_idxs(self, idxs: np.ndarray) -> dict[str, np.ndarray]:
        # for N-step Learning
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

    def _get_n_step_info(
        self, n_step_buffer: Deque, gamma: float
    ) -> tuple[np.int64, np.ndarray, bool]:
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        rew, next_obs, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBufferNStep):
    """Prioritized Replay buffer.

    copied from https://github.com/Curt-Park/rainbow-is-all-you-need


    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight

    """

    def __init__(
        self,
        observation_type: str,
        size: int,
        obs_dim: tuple = None,
        batch_size: int = 32,
        alpha: float = 0.6,
        n_step: int = 1,
        gamma: float = 0.99,
    ):
        """Initialization."""
        assert alpha >= 0

        super(PrioritizedReplayBuffer, self).__init__(
            observation_type, size, obs_dim, batch_size, n_step, gamma
        )
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(
        self,
        obs: np.ndarray,
        act: int,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        """Store experience and priority."""
        transition = super().store(obs, act, rew, next_obs, done)

        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority**self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority**self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size

        return transition

    def sample_batch(self, beta: float = 0.4) -> dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()

        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,
        )

    def update_priorities(self, indices: list[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority**self.alpha
            self.min_tree[idx] = priority**self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self) -> list[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight


def plot_results(
    scores: dict,
    training_loss: list,
    epsilons: list,
    q_values: dict,
    iteration_idx: int,
    number_of_actions: int,
    num_iterations: int,
    total_maximum_episode_rewards: int,
    default_root_dir: str,
    to_plot: str = "all",
    save_fig: bool = False,
) -> None:
    """Plot things for DQN training.

    Args:
        to_plot: what to plot:
            training_td_loss
            epsilons
            scores
            q_values_train
            q_values_val
            q_values_test

    """
    is_notebook = is_running_notebook()

    if is_notebook:
        clear_output(True)

    if to_plot == "all":
        plt.figure(figsize=(20, 13))

        plt.subplot(233)
        if scores["train"]:
            plt.title(
                f"iteration {iteration_idx} out of {num_iterations}. "
                f"training score: {scores['train'][-1]} out of {total_maximum_episode_rewards}"
            )
            plt.plot(scores["train"], label="Training score")
            plt.xlabel("episode")

        if scores["val"]:
            val_means = [
                round(np.mean(scores).item()) for scores in scores["val"]
            ]
            plt.title(
                f"validation score: {val_means[-1]} out of {total_maximum_episode_rewards}"
            )
            plt.plot(val_means, label="Validation score")
            plt.xlabel("episode")

        if scores["test"]:
            plt.title(
                f"test score: {np.mean(scores['test'])} out of {total_maximum_episode_rewards}"
            )
            plt.plot(
                [round(np.mean(scores["test"]).item(), 2)] * len(scores["train"]),
                label="Test score",
            )
            plt.xlabel("episode")
        plt.legend(loc="upper left")

        plt.subplot(231)
        plt.title("training td loss")
        plt.plot(training_loss)
        plt.xlabel("update counts")

        plt.subplot(232)
        plt.title("epsilons")
        plt.plot(epsilons)
        plt.xlabel("update counts")

        plt.subplot(234)
        plt.title("Q-values, train")
        for action_number in range(number_of_actions):
            plt.plot(
                [q_values_[action_number] for q_values_ in q_values["train"]],
                label=f"action {action_number}",
            )
        plt.legend(loc="upper left")
        plt.xlabel("number of actions")

        plt.subplot(235)
        plt.title("Q-values, val")
        for action_number in range(number_of_actions):
            plt.plot(
                [q_values_[action_number] for q_values_ in q_values["val"]],
                label=f"action {action_number}",
            )
        plt.legend(loc="upper left")
        plt.xlabel("number of actions")

        plt.subplot(236)
        plt.title("Q-values, test")
        for action_number in range(number_of_actions):
            plt.plot(
                [q_values_[action_number] for q_values_ in q_values["test"]],
                label=f"action {action_number}",
            )
        plt.legend(loc="upper left")
        plt.xlabel("number of actions")

        plt.subplots_adjust(hspace=0.5)
        if save_fig:
            plt.savefig(os.path.join(default_root_dir, "plot.pdf"))

        if is_notebook:
            plt.show()
        else:
            console(**locals())
            plt.close("all")

    elif to_plot == "training_td_loss":
        plt.figure()
        plt.title("training td loss")
        plt.plot(training_loss)
        plt.xlabel("update counts")

    elif to_plot == "epsilons":
        plt.figure()
        plt.title("epsilons")
        plt.plot(epsilons)
        plt.xlabel("update counts")

    elif to_plot == "scores":
        plt.figure()

        if scores["train"]:
            plt.title(
                f"iteration {iteration_idx} out of {num_iterations}. "
                f"training score: {scores['train'][-1]} out of {total_maximum_episode_rewards}"
            )
            plt.plot(scores["train"], label="Training score")
            plt.xlabel("episode")

        if scores["val"]:
            val_means = [
                round(np.mean(scores).item()) for scores in scores["val"]
            ]
            plt.title(
                f"validation score: {val_means[-1]} out of {total_maximum_episode_rewards}"
            )
            plt.plot(val_means, label="Validation score")
            plt.xlabel("episode")

        if scores["test"]:
            plt.title(
                f"test score: {np.mean(scores['test'])} out of {total_maximum_episode_rewards}"
            )
            plt.plot(
                [round(np.mean(scores["test"]).item(), 2)] * len(scores["train"]),
                label="Test score",
            )
            plt.xlabel("episode")
        plt.legend(loc="upper left")

    elif to_plot == "q_values_train":
        plt.figure()
        plt.title("Q-values, train")
        for action_number in range(number_of_actions):
            plt.plot(
                [q_values_[action_number] for q_values_ in q_values["train"]],
                label=f"action {action_number}",
            )
        plt.legend(loc="upper left")
        plt.xlabel("number of actions")

    elif to_plot == "q_values_val":
        plt.figure()
        plt.title("Q-values, val")
        for action_number in range(number_of_actions):
            plt.plot(
                [q_values_[action_number] for q_values_ in q_values["val"]],
                label=f"action {action_number}",
            )
        plt.legend(loc="upper left")
        plt.xlabel("number of actions")

    elif to_plot == "q_values_test":
        plt.figure()
        plt.title("Q-values, test")
        for action_number in range(number_of_actions):
            plt.plot(
                [q_values_[action_number] for q_values_ in q_values["test"]],
                label=f"action {action_number}",
            )
        plt.legend(loc="upper left")
        plt.xlabel("number of actions")
    else:
        raise ValueError(f"to_plot={to_plot} is not valid.")


def console(
    scores: dict,
    training_loss: list,
    iteration_idx: int,
    num_iterations: int,
    total_maximum_episode_rewards: int,
    **kwargs,
) -> None:
    """Print the dqn training to the console."""
    if scores["train"]:
        tqdm.write(
            f"iteration {iteration_idx} out of {num_iterations}.\n"
            f"training score: "
            f"{scores['train'][-1]} out of {total_maximum_episode_rewards}"
        )

    if scores["val"]:
        val_means = [round(np.mean(scores).item()) for scores in scores["val"]]
        tqdm.write(
            f"validation score: {val_means[-1]} "
            f"out of {total_maximum_episode_rewards}"
        )

    if scores["test"]:
        tqdm.write(
            f"test score: {np.mean(scores['test'])} out of {total_maximum_episode_rewards}"
        )

    tqdm.write(f"training loss: {training_loss[-1]}\n")
    print()


def save_final_results(
    scores: dict,
    training_loss: list,
    default_root_dir: str,
    q_values: dict,
    self: object,
) -> None:
    """Save dqn train / val / test results."""
    results = {
        "train_score": scores["train"],
        "validation_score": [
            {
                "mean": round(np.mean(scores).item(), 2),
                "std": round(np.std(scores).item(), 2),
            }
            for scores in scores["val"]
        ],
        "test_score": {
            "mean": round(np.mean(scores["test"]).item(), 2),
            "std": round(np.std(scores["test"]).item(), 2),
        },
        "training_loss": training_loss,
    }
    write_yaml(results, os.path.join(default_root_dir, "results.yaml"))
    write_yaml(q_values, os.path.join(default_root_dir, "q_values.yaml"))
    write_pickle(self, os.path.join(default_root_dir, "agent.pkl"))


def compute_loss(
    samples: dict[str, np.ndarray],
    device: str,
    dqn: torch.nn.Module,
    dqn_target: torch.nn.Module,
    ddqn: str,
    gamma: float,
) -> torch.Tensor:
    """Return td loss.

    Args:
        samples: A dictionary of samples from the replay buffer.
            obs: np.ndarray,
            act: np.ndarray,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
        device:
        dqn:
        dqn_target:
        ddqn:
        gamma:

    Returns:
        loss: torch.Tensor

    """
    state = samples["obs"]
    next_state = samples["next_obs"]
    action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
    reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
    done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

    # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
    #       = r                       otherwise
    curr_q_value = dqn(state).gather(1, action)
    if ddqn:
        next_q_value = (
            dqn_target(next_state)
            .gather(1, dqn(next_state).argmax(dim=1, keepdim=True))
            .detach()
        )
    else:
        next_q_value = dqn_target(next_state).max(dim=1, keepdim=True)[0].detach()
    mask = 1 - done
    target = (reward + gamma * next_q_value * mask).to(device)

    # calculate dqn loss
    loss = F.smooth_l1_loss(curr_q_value, target)

    return loss


def select_action(
    state: dict,
    greedy: bool,
    dqn: torch.nn.Module,
    epsilon: float,
    action_space: gym.spaces.Discrete,
) -> tuple[int, list]:
    """Select an action from the input state.

    Args:
        state: The current state of the memory systems. This is NOT what the gym env
        gives you. This is made by the agent.
        greedy: always pick greedy action if True
        save_q_value: whether to save the q values or not.

    Returns:
        selected_action: an action to take.
        q_values: a list of q values for each action.

    """
    # epsilon greedy policy
    q_values = dqn(np.array([state])).detach().cpu().tolist()[0]

    if epsilon < np.random.random() or greedy:
        selected_action = argmax(q_values)
    else:
        selected_action = action_space.sample().item()

    return selected_action, q_values


def update_model(
    replay_buffer: ReplayBuffer,
    optimizer: torch.optim.Adam,
    device: str,
    dqn: torch.nn.Module,
    dqn_target: torch.nn.Module,
    ddqn: str,
    gamma: float,
) -> torch.Tensor:
    """Update the model by gradient descent."""
    samples = replay_buffer.sample_batch()

    loss = compute_loss(samples, device, dqn, dqn_target, ddqn, gamma)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def save_validation(
    scores_temp: list,
    scores: dict,
    default_root_dir: str,
    num_validation: int,
    val_filenames: list,
    dqn: torch.nn.Module,
    if_duplicate_take_first: bool = False,
) -> None:
    """Keep the best validation model.

    Args:
        scores_temp: a list of validation scores for the current validation episode.
        scores: a dictionary of scores for train, validation, and test.
        default_root_dir: the root directory where the results are saved.
        num_validation: the current validation episode.
        val_filenames: a list of filenames for the validation models.
        dqn: the dqn model.
        if_duplicate_take_first: if True, take the first duplicate model. This will take
            the higher training loss model. If False, take the last duplicate model.
            This will take the lower training loss model.
    """
    mean_score = round(np.mean(scores_temp).item())
    filename = os.path.join(
        default_root_dir, f"episode={num_validation}_val-score={mean_score}.pt"
    )
    torch.save(dqn.state_dict(), filename)

    val_filenames.append(filename)
    scores["val"].append(scores_temp)

    scores_to_compare = []
    for filename in val_filenames:
        score = int(filename.split("val-score=")[-1].split(".pt")[0].split("/")[-1])
        scores_to_compare.append(score)

    indexes = list_duplicates_of(scores_to_compare, max(scores_to_compare))
    if if_duplicate_take_first:
        file_to_keep = val_filenames[indexes[0]]
    else:
        file_to_keep = val_filenames[indexes[-1]]

    for filename in val_filenames:
        if filename != file_to_keep:
            os.remove(filename)
            val_filenames.remove(filename)


def save_states_q_values_actions(
    states: list,
    q_values: list,
    actions: list,
    default_root_dir: str,
    val_or_test: str,
    num_validation: int | None = None,
) -> None:
    """Save states, q_values, and actions.

    Args:
        states: a list of states.
        q_values: a list of q_values.
        actions: a list of actions.

    """
    if val_or_test.lower() == "val":
        filename = os.path.join(
            default_root_dir,
            f"states_q_values_actions_val_episode={num_validation}.yaml",
        )
    else:
        filename = os.path.join(default_root_dir, "states_q_values_actions_test.yaml")

    assert len(states) == len(q_values) == len(actions)
    to_save = [
        {"state": s, "q_values": q, "action": a}
        for s, q, a in zip(states, q_values, actions)
    ]
    write_yaml(to_save, filename)


def target_hard_update(dqn: torch.nn.Module, dqn_target: torch.nn.Module) -> None:
    """Hard update: target <- local."""
    dqn_target.load_state_dict(dqn.state_dict())
