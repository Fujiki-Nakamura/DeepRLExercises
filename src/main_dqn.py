import argparse
import datetime as dt
from itertools import count
import os
# import math
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import gym

from utils import get_logger, get_state, preprocess, save_checkpoint
from nets import DQN, ReplayMemory, Transition


steps_done = 0


def select_action(args, state, policy_net):
    global steps_done

    sample = random.random()
    # eps_threshold = args.eps_end + (args.eps_start - args.eps_end) * math.exp(-1. * steps_done / args.eps_decay)  # noqa
    eps_threshold = args.eps_end + (args.eps_start - args.eps_end) / args.exploration_steps  # noqa

    action = torch.tensor([[policy_net.last_action]], device=args.device, dtype=torch.long)
    r = random.random()
    if steps_done % args.action_interval == 0:
        if r > eps_threshold:
            with torch.no_grad():
                action = policy_net(state.to(args.device)).max(dim=1)[1].view(1, 1)  # noqa
        else:
            action = torch.tensor(
                [[random.randrange(args.n_actions)]],
                device=args.device, dtype=torch.long)
        policy_net.last_action = action.item()

    return action


def train(args, policy_net, target_net, memory, optimizer):
    if len(memory) < args.batch_size:
        return

    transitions = memory.sample(args.batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=args.device, dtype=torch.bool)
    non_final_next_states = torch.cat(
        [s for s in batch.next_state if s is not None]).to(args.device)

    state_batch = torch.cat(batch.state).to(args.device)
    action_batch = torch.cat(batch.action).to(args.device)
    reward_batch = torch.cat(batch.reward).to(args.device)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(args.batch_size, device=args.device)
    next_state_values[non_final_mask] = target_net(
        non_final_next_states).max(1)[0].detach()
    expected_state_action_values = reward_batch + args.gamma * next_state_values

    loss = F.smooth_l1_loss(
        state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def main(args):
    global steps_done

    # logs
    expid = dt.datetime.now().strftime('%Y%m%d%H%M%S')
    args.logdir = os.path.join(args.logdir, expid)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    os.chmod(args.logdir, 0o777)

    logger = get_logger(os.path.join(args.logdir, 'main.log'))
    logger.info('Logging at {}'.format(args.logdir))
    logger.info(args)
    writer = SummaryWriter(args.logdir)

    # misc
    random.seed(args.random_state)
    os.environ['PYTHONHASHSEED'] = str(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)

    # env
    env = gym.make(args.env_name).unwrapped
    # model
    args.n_actions = env.action_space.n
    policy_net = DQN(args.state_h, args.state_w, args.n_actions).to(args.device)
    target_net = DQN(args.state_h, args.state_w, args.n_actions).to(args.device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    # optim
    optimizer = optim.RMSprop(
        policy_net.parameters(), lr=args.lr, momentum=args.momentum)
    memory = ReplayMemory(args.replay_memory_size)

    total_rewards = []
    durations = []
    for episode_i in range(args.n_episodes):
        start_time = time.time()
        g = 0

        last_obs = env.reset()
        obs = env.reset()
        processed_obs = preprocess(obs, last_obs, args)
        state = get_state(args, processed_obs, state=None, is_initial=True)
        for t in count():
            if not torch.cuda.is_available():
                env.render()
            steps_done += 1
            last_obs = obs
            action = select_action(args, state, policy_net)
            # _action = _action_space[action.item()]
            _action = action.item()
            obs, reward, done, _ = env.step(_action)
            processed_obs = preprocess(obs, last_obs, args=args)
            g += reward
            reward = torch.tensor([reward], device=args.device)

            if done:
                next_state = None
            else:
                next_state = get_state(args, processed_obs, state)

            memory.push(state, action, next_state, reward)
            state = next_state

            if steps_done % args.train_interval == 0:
                train(args, policy_net, target_net, memory, optimizer)
            if steps_done % args.update_target_net_every_x_timesteps == 0:  # noqa
                target_net.load_state_dict(policy_net.state_dict())
                logger.info(
                    '[{}] Env {} Episode {} Timesteps {} - Updated TargetNet'.format(
                        expid, args.env_name, episode_i + 1, steps_done)
                )

            if done:
                # episode_durations.append(t + 1)
                # plot_durations()
                break

        total_rewards.append(g)
        durations.append(t + 1)
        writer.add_scalar('{}/TotalReward/Train'.format(args.env_name), g, episode_i + 1)

        if (episode_i + 1) % args.save_every_x_episodes == 0:
            state_dict = {
                'epoch': episode_i,
                'policy_net': policy_net.state_dict(),
                'target_net': target_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'return': g,
            }
            save_checkpoint(state_dict, episode_i, g, args.logdir)

        logger.info(
            '[{}] Env {} Episode {} Timesteps {} Time {:.2f}s Duration {} TotalReward {}'.format(
                expid, args.env_name,
                episode_i + 1, steps_done, time.time() - start_time, t + 1, g))

    logger.info('Finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--env_name', type=str, default='Breakout-v0')
    parser.add_argument('--n_actions', type=int, default=0)
    parser.add_argument('--state_w', type=int, default=84)
    parser.add_argument('--state_h', type=int, default=84)
    parser.add_argument('--state_length', type=int, default=4)
    # model
    parser.add_argument('--replay_memory_size', type=int, default=400000)
    # optim
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--momentum', type=float, default=0.95)
    # training
    parser.add_argument('--action_interval', type=int, default=4)
    parser.add_argument('--train_interval', type=int, default=4)
    parser.add_argument('--n_episodes', type=int, default=12000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--eps_start', type=float, default=1.0)
    parser.add_argument('--eps_end', type=float, default=0.1)
    parser.add_argument('--eps_decay', type=int, default=200)
    parser.add_argument('--exploration_steps', type=int, default=1000000)
    parser.add_argument('--update_target_net_every_x_episodes', type=int, default=10)  # noqa
    parser.add_argument('--update_target_net_every_x_timesteps', type=int, default=10000)  # noqa
    parser.add_argument('--save_every_x_episodes', type=int, default=100)
    # misc
    parser.add_argument('--logdir', type=str, default='../logs')
    parser.add_argument('--random_state', type=int, default=42)

    args, _ = parser.parse_known_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main(args)
