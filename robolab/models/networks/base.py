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

from math import sqrt
import torch
import torch.nn as nn

VAR_OFFSET = 1e-5
STDDEV_OFFSET = sqrt(VAR_OFFSET)


class Dense(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        activation=None,
        hidden_layers=1,
        hidden_units=128,
        hidden_activation=torch.relu,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if isinstance(activation, str):
            self.activation = getattr(torch, activation.lower())
        else:
            self.activation = activation

        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        if isinstance(hidden_activation, str):
            self.hidden_activation = getattr(torch, hidden_activation)
        else:
            self.hidden_activation = hidden_activation

        self._hidden_layers_fn = []
        hidden_in_features = self.in_features
        for _ in range(self.hidden_layers):
            layer = nn.Linear(hidden_in_features, hidden_units)
            hidden_in_features = hidden_units
            self._hidden_layers_fn.append(layer)

        self._hidden_layers_fn = nn.ModuleList(self._hidden_layers_fn)

        if self.hidden_layers > 0:
            in_shape = self.hidden_units
        else:
            in_shape = self.in_features

        self._output_layer = nn.Linear(in_shape, self.out_features)

    def forward(self, x):
        y = x
        for i in range(self.hidden_layers):
            y = self._hidden_layers_fn[i](y)
            if self.hidden_activation:
                y = self.hidden_activation(y)

        y = self._output_layer(y)
        if self.activation:
            y = self.activation(y)

        return y
