import argparse

from PIL import Image
import cv2
import numpy as np
import torch
from gym_duckietown.envs import DuckietownEnv

from .planner import MapGrid, BFS
from .model import Squeezenet

from .utils import MemoryMapDataset
from . import Policy


def get_config():
    # declare the arguments
    parser = argparse.ArgumentParser()

    # Do not change this
    parser.add_argument('--max_steps', type=int, default=1500, help='max_steps')

    # You should set them to different map name and seed accordingly
    parser.add_argument('--map-name', '-m', default="map2_0", type=str)
    parser.add_argument('--seed', '-s', default=2, type=int)
    parser.add_argument('--start-tile', '-st', default="1,1", type=str, help="two numbers separated by a comma")
    parser.add_argument('--goal-tile', '-gt', default="3,3", type=str, help="two numbers separated by a comma")
    parser.add_argument('--model', '-d', default="model_map1_0bw", type=str)
    args = parser.parse_args()
    return args

def launch_env(args):
    env = DuckietownEnv(
        domain_rand=False,
        max_steps=1500,
        map_name=args.map_name,
        seed=args.seed,
        user_tile_start=args.start_tile,
        goal_tile=args.goal_tile,
        randomize_maps_on_reset=False
    )
    return env

def get_path(map_img, goal, start_pos):
    grid = MapGrid(map_img)
    planner = BFS(goal, start_pos, grid.get_grid())
    path = planner.search()
    return path

def get_policy(goal, args, map_img, start_pos):
    model = Squeezenet()
    model.load_state_dict(torch.load('./pretrained/' + args.model + '.pt', map_location=torch.device('cpu')))

    policy_optimizer = torch.optim.Adam(model.parameters())

    input_shape = (40,80)
    dataset = MemoryMapDataset(25000, (3, *input_shape), (2,), "")
    policy = Policy(
        goal_tile=goal,
        model=model,
        optimizer=policy_optimizer,
        storage_location="",
        dataset = dataset,
        grid=MapGrid(map_img),
        start_pos=start_pos
    )

    return policy

# def preprocess_observation(observation):
#     img = Image.fromarray(observation)
#     img = img.resize((80, 60))
#     img = img.crop((0, 20, 80, 60))
#     thresh = 80
#     fn = lambda x : 255 if x > thresh else 0
#     img = img.convert('L')
#     img = img.point(fn, mode='1')
#     img = img.convert('RGB')
#     obs = np.array(img)
    
#     return obs

def main():
    args = get_config()
    env = launch_env(args)
    env.render()

    map_img, goal, start_pos = env.get_task_info()
    print("start tile:", start_pos, " goal tile:", goal)

    # Show the map image
    # White pixels are drivable and black pixels are not.
    # Blue pixels indicate lan center
    # Each tile has size 100 x 100 pixels
    # Tile (0, 0) locates at left top corner.
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    map_img_clone = cv2.resize(map_img.copy(), (960, 540))  
    cv2.imshow("map", map_img_clone)
    cv2.waitKey(200)

    # path = get_path(map_img, goal, start_pos)
    total_reward=0
    steps=0

    policy = get_policy(goal, args, map_img, start_pos)

    obs, reward, done, info = env.step((0,0))
    curr_pos = info['curr_pos']

    actions = []

    for i in range(10000):
        steps+=1
        action = policy.predict(obs, curr_pos)
        try:
            action = action.numpy()
        except AttributeError:
            pass
        # print(f'action={action}')
        actions.append(action)
        obs, reward, done, info = env.step(action)
        total_reward+=reward
        curr_pos = info['curr_pos']

        if curr_pos == goal: break

        print('Steps = %s, Timestep Reward=%.3f, curr_tile:%s'
            % (env.step_count, reward, curr_pos))
        env.render()

    # dump the controls using numpy
    np.savetxt(f'../control_files/{args.map_name}_seed{args.seed}_start_{start_pos[0]},{start_pos[1]}_goal_{goal[0]},{goal[1]}.txt',
            actions, delimiter=',')
    print("REWARD: {}".format(total_reward/steps))
if __name__ == "__main__":
    main()