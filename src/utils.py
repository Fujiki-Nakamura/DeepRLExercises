import os

import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
import torch


def wrap_atari_dqn(env):
    from common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=False)


def get_state(args, observation, state=None, is_initial=False):
    if is_initial:
        state = np.stack(
            [observation for _ in range(args.state_length)], axis=0)
    else:
        state = state.cpu().squeeze(0).numpy()
        state = np.append(
            state[1:, :, :], observation[np.newaxis, :, :], axis=0)

    return torch.from_numpy(state).unsqueeze(0)


def preprocess(observation, last_observation, args):
    # observation = np.maximum(observation, last_observation)
    observation = resize(rgb2gray(observation), (args.state_w, args.state_h))
    return observation.astype(np.float32).reshape(args.state_w, args.state_h)


def save_checkpoint(state, episode, score, logdir):
    filename = f'checkpoint-{episode+1:09}-{score}.pt'
    path = os.path.join(logdir, filename)
    torch.save(state, path)


def get_logger(log_file):
    from logging import getLogger, FileHandler, StreamHandler
    from logging import Formatter, DEBUG, ERROR, INFO  # noqa
    fh = FileHandler(log_file)
    fh.setLevel(INFO)
    sh = StreamHandler()
    sh.setLevel(INFO)

    for handler in [fh, sh]:
        formatter = Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
    logger = getLogger('log')
    logger.setLevel(DEBUG)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger
