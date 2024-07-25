import hashlib
import os.path as osp
import time

import cv2
import imageio
import numpy as np


def convert_frames(array):
    """
    Converts frames from (n, c, h, w) to (n, h, w, c) format, or verifies if already in (n, h, w, c) format.
    Handles cases with different channel sizes (e.g., 3 for RGB, 4 for RGBA).

    Parameters:
    - array: NumPy array with shape either (n, c, h, w) or (n, h, w, c)

    Returns:
    - NumPy array with shape (n, h, w, c)
    """
    argmin = np.argmin(array.shape[1:])

    if argmin == 0:
        # Assumes the format is (n, c, h, w) and converts to (n, h, w, c)
        return array.transpose(0, 2, 3, 1)
    return array


def record_video(save_dir, index, obs, record_format="mp4"):
    # make sure channels are last dim
    obs = convert_frames(obs)

    if record_format == "mp4":
        record_video_mp4(save_dir, index, obs)
    elif record_format == "gif":
        record_video_gif(save_dir, index, obs)
    elif record_format == "npy":
        np.save(osp.join(save_dir, f"render-{index}.npy"), obs)
    elif record_format == "npz":
        np.savez_compressed(osp.join(save_dir, f"render-{index}.npz"), obs)
    else:
        raise ValueError(f"Unknown format to record video: {record_format}")


def unstack_frames(obs):
    # Unstack the frames if stacked, while leaving colors unaltered
    frames = [obs[i] for i in range(obs.shape[0])]
    # frames = np.split(obs, 1, axis=-1)
    # frames = np.concatenate(np.array(frames), axis=0)
    # frames = [np.squeeze(a, axis=0) for a in np.split(frames, frames.shape[0], axis=0)]
    return frames


def record_video_mp4(save_dir, index, obs):
    """Record a video from samples collected at evalutation time."""

    # Create OpenCV video writer
    vname = "render-{}".format(index)
    frameSize = (obs.shape[-2], obs.shape[-3])
    writer = cv2.VideoWriter(
        filename="{}.mp4".format(osp.join(save_dir, vname)),
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=25,
        frameSize=frameSize,
        isColor=True,
    )

    frames = unstack_frames(obs)
    for frame in frames:
        # Add frame to video
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    cv2.destroyAllWindows()
    # Delete the object
    del frames


def record_video_gif(save_dir, index, obs):
    frames = unstack_frames(obs)
    with imageio.get_writer(
        osp.join(save_dir, f"render-{index}.gif"), mode="I"
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


class OpenCVImageViewer(object):
    """Viewer used to render simulations."""

    def __init__(self, q_to_exit=True):
        self._q_to_exit = q_to_exit
        # Create unique identifier
        hash_ = hashlib.sha1()
        hash_.update(str(time.time()).encode("utf-8"))
        # Create window
        self._window_name = str(hash_.hexdigest()[:20])
        cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)
        self._isopen = True

    def __del__(self):
        cv2.destroyWindow(self._window_name)
        self._isopen = False

    def imshow(self, img):
        # Convert image to BGR format
        cv2.imshow(self._window_name, img[:, :, [2, 1, 0]])
        # Listen for escape key, then exit if pressed
        if cv2.waitKey(1) == ord("q") and self._q_to_exit:
            exit()

    @property
    def isopen(self):
        return self._isopen

    def close(self):
        pass
