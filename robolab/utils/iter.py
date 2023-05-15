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


class IteratorFactory:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def get_iterator(self):
        class_type = self._ITERATOR_CLASS
        return iter(class_type(*self.args, **self.kwargs))


class ParameterIterator:
    def __init__(self, n_samples=None):
        self.n_samples = n_samples


class TilingIterator(ParameterIterator):
    """Wrapper to tile output of another parameter interator.
    First tile is sampled from the child iterator and then repeated without resampling.
    """

    def __init__(self, child, repeats=None):
        self.repeats = repeats

        if isinstance(child, IteratorFactory):
            self.child = child.get_iterator()
        else:
            self.child = child

    def __iter__(self):
        self.buffer = []
        for value in self.child:
            self.buffer.append(value)
            yield value

        if self.repeats is None:
            while True:
                for value in self.buffer:
                    yield value
        else:
            for _ in range(self.repeats - 1):
                for value in self.buffer:
                    yield value


class Tiling(IteratorFactory):
    _ITERATOR_CLASS = TilingIterator


class RepeatingIterator(ParameterIterator):
    """Wrapper to repeat output of another parameter interator."""

    def __init__(self, child, repeats, n_samples=None):
        super().__init__(n_samples=n_samples)
        self.repeats = repeats

        if isinstance(child, IteratorFactory):
            self.child = child.get_iterator()
        else:
            self.child = child

    def __iter__(self):
        if self.n_samples is None:
            while True:
                for value in self.child:
                    for _ in range(self.repeats):
                        yield value
        else:
            for value in self.child:
                for _ in range(self.repeats):
                    yield value


class Repeating(IteratorFactory):
    _ITERATOR_CLASS = RepeatingIterator
