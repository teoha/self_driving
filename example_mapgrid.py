import argparse

import cv2
import numpy as np
from gym_duckietown.envs import DuckietownEnv

# declare the arguments
parser = argparse.ArgumentParser()

# Do not change this
parser.add_argument('--max_steps', type=int, default=1500, help='max_steps')

# You should set them to different map name and seed accordingly
parser.add_argument('--map-name', '-m', default="map2_0", type=str)
parser.add_argument('--seed', '-s', default=2, type=int)
parser.add_argument('--start-tile', '-st', default="1,1", type=str, help="two numbers separated by a comma")
parser.add_argument('--goal-tile', '-gt', default="3,3", type=str, help="two numbers separated by a comma")
args = parser.parse_args()

env = DuckietownEnv(
    domain_rand=False,
    max_steps=1500,
    map_name=args.map_name,
    seed=args.seed,
    user_tile_start=args.start_tile,
    goal_tile=args.goal_tile,
    randomize_maps_on_reset=False,
    full_transparency=True
)

obs = env.reset()
env.render()

map_img, goal, start_pos = env.get_task_info()
print("start tile:", start_pos, " goal tile:", goal)
# print(map_img)
from . import MapGrid

grid = MapGrid(map_img)


# Show the map image
# White pixels are drivable and black pixels are not.
# Blue pixels indicate lan center
# Each tile has size 100 x 100 pixels
# Tile (0, 0) locates at left top corner.
cv2.imshow("map", map_img)
cv2.waitKey(200)
from .model import Squeezenet
import torch

from .utils import MemoryMapDataset
from . import Policy

model = Squeezenet()
model.load_state_dict(torch.load('./iil_baseline/model.pt', map_location=torch.device('cpu')))

policy_optimizer = torch.optim.Adam(model.parameters())

input_shape = (120,160)
dataset = MemoryMapDataset(25000, (3, *input_shape), (2,), "")
learner = Policy(
    path={(7,7):(6,7), (6,7):(5,7),(5,7):(4,7), (4,7):(3,7),(3,7):(2,7),(2,7):(1,7),(1,7):(1,6),
            (1,6): (1,5), (1,5): (1,4), (1,4): (1,3), (1,3):(1,2), (1,2): (1,1)},
    map_grid=grid,
    goal_tile=goal,
    model=model,
    optimizer=policy_optimizer,
    storage_location="",
    dataset = dataset
)



obs, reward, done, info = env.step((0,0))
curr_pos = info['curr_pos']

for i in range(2000):
    action = learner.predict(obs, curr_pos)
    print(f'action={action}')
    obs, reward, done, info = env.step(action)
    if done: break
    curr_pos = info['curr_pos']

    print('Steps = %s, Timestep Reward=%.3f, curr_tile:%s'
          % (env.step_count, reward, curr_pos))
    env.render()