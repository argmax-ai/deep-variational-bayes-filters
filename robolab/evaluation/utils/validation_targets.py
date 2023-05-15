# Copyright (C) 2019-2023 Volkswagen Aktiengesellschaft,
# Berliner Ring 2, 38440 Wolfsburg, Germany
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from enum import Enum
from enum import auto


class DataNodes(Enum):
    inputs = auto()
    targets = auto()
    controls = auto()
    rewards = auto()
    metas = auto()
    mask = auto()


class SequenceModelNodes(Enum):
    filter_observation_sample = auto()
    filter_reward_sample = auto()
    filter_state_sample = auto()
    predict_observation_sample = auto()
    predict_reward_sample = auto()
    predict_state_sample = auto()
    generate_observation_sample = auto()
    generate_reward_sample = auto()
    generate_state_sample = auto()


class AgentNodes(Enum):
    filter_observation_sample = auto()
    filter_action_sample = auto()
    filter_control_sample = auto()
    filter_rewards = auto()
    filter_internal_rewards = auto()
    filter_critic = auto()
    filter_target_critic = auto()
    dream_observations_sample = auto()
    dream_actions_sample = auto()
    dream_controls_sample = auto()
    dream_rewards = auto()
    dream_internal_rewards = auto()
    dream_from_prior_observations_sample = auto()
    dream_from_prior_actions_sample = auto()
    dream_from_prior_controls_sample = auto()
    dream_from_prior_rewards = auto()
    env_observations_sample = auto()
    env_meta_observations = auto()
    env_actions_sample = auto()
    env_controls_sample = auto()
    env_rewards = auto()
    env_internal_rewards = auto()
    env_critic = auto()
    env_target_critic = auto()
