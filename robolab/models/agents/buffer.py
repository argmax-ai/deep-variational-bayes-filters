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

from typing import Optional
from typing import List
import torch


class Buffer(torch.nn.Module):
    _buffer: List[torch.Tensor]
    _is_full_backward_hook: Optional[bool]

    def __init__(self, buffer_size: int, element_size: int):
        super().__init__()
        self.buffer_size = buffer_size
        self.element_size = element_size
        self._buffer = []

    def reset(self, batch_size: int, prefix_element: torch.Tensor) -> torch.Tensor:
        """Reset the agent's control or observation buffer.

        Fills the buffer with zero tensors, most recent value can be
        overwritten by prefix_element instead.

        Parameters
        ----------
        batch_size
        prefix_element
        """
        self._buffer.clear()
        for t in range(self.buffer_size):
            zero_element = torch.zeros(
                (batch_size, self.element_size), device=prefix_element.device
            )
            self._buffer.append(zero_element)

        if prefix_element is not None:
            self._buffer[-1] = prefix_element

        return torch.stack(self._buffer)

    def update(self, value: torch.Tensor) -> Optional[bool]:
        self._buffer.pop(0)
        self._buffer.append(value)
        return True

    def get_buffer(self) -> torch.Tensor:
        return torch.stack(self._buffer)

    def get_latest(self) -> torch.Tensor:
        return self._buffer[-1]

    def get_oldest(self) -> torch.Tensor:
        return self._buffer[0]
