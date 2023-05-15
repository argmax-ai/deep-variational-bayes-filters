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

from dataclasses import dataclass
from typing import Dict
from typing import List
import torch
from robolab.models.latent_state import LatentState


@dataclass
class EnvReturn:
    observation: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor

    def _to(self, attr, device):
        target = None
        if getattr(self, attr) is not None:
            target = getattr(self, attr).to(device)

        return target

    def _detach(self, attr):
        target = None
        if getattr(self, attr) is not None:
            target = getattr(self, attr).detach()

        return target

    def to(self, device):
        return EnvReturn(
            observation=self._to("observation", device),
            reward=self._to("reward", device),
            done=self._to("done", device),
        )

    def detach(self):
        return EnvReturn(
            observation=self._detach("observation"),
            reward=self._detach("reward"),
            done=self._detach("done"),
        )


@dataclass
class GymReturn(EnvReturn):
    observation: torch.Tensor
    meta: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    info: List[Dict] = None

    def to(self, device):
        return GymReturn(
            observation=self._to("observation", device),
            meta=self._to("meta", device),
            reward=self._to("reward", device),
            done=self._to("done", device),
            info=self.info,
        )

    def detach(self):
        return GymReturn(
            observation=self._detach("observation"),
            meta=self._detach("meta"),
            reward=self._detach("reward"),
            done=self._detach("done"),
            info=self.info,
        )


@dataclass
class FaramaGymReturn(EnvReturn):
    observation: torch.Tensor
    meta: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    info: List[Dict] = None

    def to(self, device):
        return FaramaGymReturn(
            observation=self._to("observation", device),
            meta=self._to("meta", device),
            reward=self._to("reward", device),
            done=self._to("done", device),
            terminated=self._to("terminated", device),
            truncated=self._to("truncated", device),
            info=self.info,
        )

    def detach(self):
        return FaramaGymReturn(
            observation=self._detach("observation"),
            meta=self._detach("meta"),
            reward=self._detach("reward"),
            done=self._detach("done"),
            terminated=self._detach("terminated"),
            truncated=self._detach("truncated"),
            info=self.info,
        )


@dataclass
class DreamEnvReturn(EnvReturn):
    observation: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor

    state: LatentState

    def to(self, device):
        return DreamEnvReturn(
            observation=self._to("observation", device),
            reward=self._to("reward", device),
            done=self._to("done", device),
            state=self._to("state", device),
        )

    def detach(self):
        return DreamEnvReturn(
            observation=self._detach("observation"),
            reward=self._detach("reward"),
            done=self._detach("done"),
            state=self._detach("state"),
        )
