import argparse
import cv2
import numpy as np
from gym_duckietown.envs import DuckietownEnv

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


    actions = np.loadtxt(f'../control_files/{args.map_name}_seed{args.seed}_start_{args.start_tile}_goal_{args.goal_tile}.txt', delimiter=',')

    obs, reward, done, info = env.step((0,0))
    curr_pos = info['curr_pos']

    total = 0

    for a in actions:
        obs, reward, done, info = env.step(a)
        curr_pos = info['curr_pos']
        # if reward == -1000:
        #     input()
        total += reward
        print('Steps = %s, Timestep Reward=%.3f, curr_tile:%s'
            % (env.step_count, reward, curr_pos))
        print(f'Total reward: {total}, Average reward: {total/env.step_count}')
        env.render()

if __name__ == "__main__":
    main()