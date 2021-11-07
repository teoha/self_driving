import argparse

from PIL import Image
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
parser.add_argument('--model', '-d', default="model", type=str)
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
from . import BFS

grid = MapGrid(map_img)
planner = BFS(map_img,goal,start_pos,grid.get_grid())
path = planner.search()
# print(path)

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
model.load_state_dict(torch.load('./iil_baseline/' + args.model + '.pt', map_location=torch.device('cpu')))

policy_optimizer = torch.optim.Adam(model.parameters())

# path = {}
# path1_0 = {}
# for i in range(5):
#     path[(i, 1)] = (i+1,1)
#     path[(i,0)] = (i, 1)
#     path[(i,2)] = (i, 1)
# for i in range(70):
#     path[(i, 1)] = (i+1,1)
#     path[(i,0)] = (i, 1)
#     path[(i,2)] = (i, 1)

# path2 = {(7,7):(6,7), (6,7):(5,7),(5,7):(4,7), (4,7):(3,7),(3,7):(2,7),(2,7):(1,7),(1,7):(1,6),
#             (1,6): (1,5), (1,5): (1,4), (1,4): (1,3), (1,3):(1,2), (1,2): (1,1)}
# path2.clear()

# for i in range(8):
#     path2[(0, i)] = (1,i)
#     path2[(2,i)] = (1,i)
#     path2[(i,8)] = (i, 7)
#     path2[(i,6)] = (i, 7)
#     path2[(i+1,7)] = (i,7)
#     path2[(1,i+1)] = (1,i)
# path2[(7,1)] = (6,1)

input_shape = (40,80)
dataset = MemoryMapDataset(25000, (3, *input_shape), (2,), "")
learner = Policy(
    path=path,
    map_grid=grid,
    goal_tile=goal,
    model=model,
    optimizer=policy_optimizer,
    storage_location="",
    dataset = dataset
)



obs, reward, done, info = env.step((0,0))
curr_pos = info['curr_pos']

def preprocess_observation(observation):
    # print(type(observation), observation.shape)
    img = Image.fromarray(observation)
    # img.show()
    img = img.resize((80, 60))
    # img.show()
    img = img.crop((0, 20, 80, 60))
    # # img.show()
    thresh = 80
    fn = lambda x : 255 if x > thresh else 0
    img = img.convert('L')
    # img.show()
    img = img.point(fn, mode='1')
    # img.show()
    img = img.convert('RGB')
    # img.show()
    obs = np.array(img)
    
    # print(type(obs), obs.shape)
    return obs

actions = []

for i in range(10000):
    obs = preprocess_observation(obs)
    action = learner.predict(obs, curr_pos)
    try:
        action = action.numpy()
    except AttributeError:
        pass
    print(f'action={action}')
    actions.append(action)
    obs, reward, done, info = env.step(action)
    curr_pos = info['curr_pos']

    if curr_pos == goal: break

    print('Steps = %s, Timestep Reward=%.3f, curr_tile:%s'
          % (env.step_count, reward, curr_pos))
    env.render()



# dump the controls using numpy
np.savetxt(f'./{args.map_name}_seed{args.seed}_start_{start_pos[0]},{start_pos[1]}_goal_{goal[0]},{goal[1]}.txt',
           actions, delimiter=',')