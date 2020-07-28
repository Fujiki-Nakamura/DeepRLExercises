# DeepRLExercises

## Usage
- For example, to train an agent for Pong with GPU 0,
```
$ cd src/
$ CUDA_VISIBLE_DEVICES=0 python main_dqn.py --env_name PongNoFrameskip-v4 --replay_memory_size 1000000 loss_name mse_loss --optim_name Adam --lr 1e-4 --batch_size 32

```

## NOTE
- `src/` contains codes for Deep Q Network solutions to Breakout, Atari.

## Results
- Evaluations are done by greedy policies with learned agents.

### Breakout
- Deep Q Network
- trained for 28,400 episodes (although should be reported in number of frames the agent learned).
- main modifications to the setting reported in the paper 2015:
  - Adam(lr=3e-5)

![openaigym video 0 3350 video000000](https://user-images.githubusercontent.com/8359397/88448370-317e9c80-ce78-11ea-9081-5c914dd5841b.gif)
![スクリーンショット 2020-07-25 13 14 59](https://user-images.githubusercontent.com/8359397/88448468-49a2eb80-ce79-11ea-83ee-bbaf182d5912.png)

### Pong
- Deep Q Network same as the Breakout
- Bellow are figures of
  - left & middle: DQN agent-learned behaviors. Finally, the agent seemed to learn to exploit the oponent behavior and "crack" the Pong game (middle).
  - right: learning curves (horizontal axis: episodes, vertical axis: total rewards in an episode)
    - green line: SmoothL1Loss, Adam(3e-5). totally sames as the Breakout agent.
    - gray line: MSELoss, RMSprop(lr=1e-4, momentum=0.)

![pong openaigym video 1 5121 video000000](https://user-images.githubusercontent.com/8359397/88469830-5d5d5900-cf30-11ea-9b02-94858104c0e7.gif)
![pong 2500-episodes openaigym video 0 6461 video000000](https://user-images.githubusercontent.com/8359397/88469833-60f0e000-cf30-11ea-80c1-a6196767e62b.gif)
![スクリーンショット 2020-07-26 11 11 08](https://user-images.githubusercontent.com/8359397/88469872-bb8a3c00-cf30-11ea-88b3-ff3f74602447.png)

## Resources
- https://github.com/berkeleydeeprlcourse
