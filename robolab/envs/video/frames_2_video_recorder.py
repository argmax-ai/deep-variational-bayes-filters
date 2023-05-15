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

import os
import ffmpeg
import numpy as np
from .video_recorder import VideoRecorder


class Frames2VideoRecorder(VideoRecorder):
    def __init__(self, directory):
        super().__init__(directory)

        self.buffer = []

    def start(self):
        pass

    def step(self, images):
        self._append(images)

    def stop_and_save(self, fn, in_framerate=60, out_framerate=60, vcodec="libx264"):
        if len(self.buffer) > 0:
            # TODO: fix pipe error
            # self._store_as_video(fn, np.array(self.buffer), in_framerate, out_framerate, vcodec)
            self.buffer = []

    def kill(self):
        pass

    def _append(self, image: np.ndarray):
        self.buffer.append(image)

    def _store_as_video(self, fn, images, in_framerate=60, out_framerate=60, vcodec="libx264"):
        path = os.path.join(self.directory, f"{fn}.mp4")

        n, height, width, channels = images.shape
        process = (
            ffmpeg.input(
                "pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{width}x{height}", r=in_framerate
            )
            .output(path, pix_fmt="yuv420p", vcodec=vcodec, r=out_framerate)
            .overwrite_output()
            .run_async(pipe_stdin=True, quiet=True)
        )

        for frame in images:
            process.stdin.write(frame.astype(np.uint8).tobytes())
        process.stdin.close()
        process.wait()
