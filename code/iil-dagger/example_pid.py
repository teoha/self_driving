import argparse

import cv2
import numpy as np
from gym_duckietown.envs import DuckietownEnv

# declare the arguments
parser = argparse.ArgumentParser()

# Do not change this
parser.add_argument('--max_steps', type=int, default=1500, help='max_steps')

# You should set them to different map name and seed accordingly
parser.add_argument('--map-name', '-m', default="map4_0", type=str)
parser.add_argument('--seed', '-s', default=2, type=int)
parser.add_argument('--start-tile', '-st', default="1,13", type=str, help="two numbers separated by a comma")
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

# Show the map image
# White pixels are drivable and black pixels are not.
# Blue pixels indicate lan center
# Each tile has size 100 x 100 pixels
# Tile (0, 0) locates at left top corner.
cv2.imshow("map", map_img)
cv2.waitKey(200)

# please remove this line for your own policy
# actions = np.loadtxt('./map4_0_seed2_start_1,13_goal_3,3.txt', delimiter=',')

from .teacher import PIDPolicy
from .teacher import JunctionPolicy

# teacher = PIDPolicy(env, 0.5)
teacher = JunctionPolicy(None, env, 0.5)

for i in range(1000):
    action = teacher.predict(obs)
    obs, reward, done, info = env.step(action)
    curr_pos = info['curr_pos']

    print('Steps = %s, Timestep Reward=%.3f, curr_tile:%s'
          % (env.step_count, reward, curr_pos))
    env.render()