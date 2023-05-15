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

import setuptools

with open("README.md") as fh:
    long_description = fh.read()

gym_reqs = [
    "gymnasium-robotics~=1.2.0",
    "dm_control",
]

setuptools.setup(
    name="robolab",
    version="2.3.0",
    author="Volkswagen Machine Learning Research Lab",
    author_email="philip.becker-ehmck@argmax.ai",
    description="Framework for model-free and model-based reinforcement learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vw-argmax-ai/robolab",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
    ],
    install_requires=[
        "torch~=1.13.0",
        "h5py>=3.6.0,<4.0",
        "matplotlib>=3.4.0,<4.0",
        "seaborn>=0.12.1",
        "pytorch-lightning~=1.9.0",
        "numpy>=1.21.0,<2.0",
        "omegaconf>=2.3.0,<3.0",
        "hydra-core>=1.3.1,<2.0",
        "flatten-dict>=0.4.2",
        "ffmpeg-python==0.2.0",
        "gymnasium~=0.27.0",
        "minigrid>=2.1.0",
        "joblib~=1.2.0",
        "pyvirtualdisplay~=3.0",
        "intervaltree",
        "control~=0.9",
        "torchdiffeq==0.2.3",
        "MarkupSafe==2.0.1",
        "moviepy~=1.0",
        "rliable~=1.0.8",
        "tbparse==0.0.7",
    ],
    extras_require={
        "gym": gym_reqs,
        "all": gym_reqs,
    },
    python_requires=">=3.7,<3.11",
)
