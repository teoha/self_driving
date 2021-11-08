from .learner import NeuralNetworkPolicy
import math
import numpy as np

REF_VELOCITY = 0.5
ADJ_STEPS = 40
ANGLE_THRESHOLD = 0.1
ANGLE_DECAY = math.pi / 100

class Policy(NeuralNetworkPolicy):
    def __init__(self, path, map_grid, goal_tile, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = path
        self.map_grid = map_grid
        self.goal_tile = goal_tile

        # Current, previous tiles, and previous action
        self.prev_tile = None
        self.prev_tile_step = None
        self.cur_tile = None
        self.prev_act = None

        # Time steps taken in turning
        self.turn_step = 0

        # Direction faced - 0,1,2,3: right,up,left,down wrt map_img
        self.face = None 

        # For rotating 180 degrees when facing wrong direction
        self.adj_step = 0
        self.adjust_done = True

    def predict(self, obs, cur_pos=None):
        if cur_pos is None:
            return 0, 0

        self.cur_tile = cur_pos
        if self.cur_tile == self.goal_tile:
            return 0, 0

        # Just started - use NN
        if self.prev_tile is None:
            # initialise prev_tiles
            self.prev_tile = self.cur_tile
            self.prev_tile_step = self.cur_tile
            self.prev_act = super().predict(obs)
        # Adjusting angle - rotate
        elif not self.adjust_done:
            self.prev_act = self.adjust_face()
        # Going straight - use NN
        elif not self.is_turning():
            self.prev_act = super().predict(obs)
        # Turning - predetermined action
        else:
            self.prev_act = self.get_turn_act()
        
        # Entered new tile
        if self.prev_tile_step != self.cur_tile:
            print(self.prev_tile_step, self.cur_tile)
            # Update prev_tiles
            self.prev_tile = self.prev_tile_step
            self.prev_tile_step = self.cur_tile

            # Update direction faced
            self.update_face()

            # Reset any turning counters
            self.turn_step = 0

            # Facing wrong direction - Rotate 180 degrees
            if self.to_adjust():
                self.adjust_done = False

        return self.prev_act

    def is_turning(self):
        '''
        returns 1 if turning left
        returns -1 if turning right
        returns 0 if going straight
        '''
        pi, pj = self.prev_tile
        ci, cj = self.cur_tile
        ni, nj = self.path[self.cur_tile]
        dpi = ci - pi
        dpj = cj - pj        
        dni = ni - ci
        dnj = nj - cj
        if dpi == dni and dpj == dnj or dpi == 0 and dpj == 0:
            return 0
        if dpi * dnj == -1 or dpj * dni == 1:
            return 1
        return -1

    def get_turn_act(self):
        # New turn action
        if self.turn_step == 0:
            vel = REF_VELOCITY
            ang = math.pi / 2 if self.is_turning() == 1 else -math.pi / 2
        # Continued turn action
        else:
            vel, ang = self.prev_act

        self.turn_step += 1

        # Decay angle turned over time
        if ang < 0:
            ang += ANGLE_DECAY
            ang = min(ang, 0)
        else:
            ang -= ANGLE_DECAY
            ang = max(ang, 0)

        return vel, ang


    def right(self):
        return REF_VELOCITY, -math.pi / 2

    def left(self):
        return REF_VELOCITY, math.pi / 2

    def get_dir_next_tile(self, t1, t2):
        i, j = t1
        ni, nj = t2
        # Right
        if ni == i + 1:
            d = 0
        # Up
        elif nj == j - 1:
            d = 1
        # Left
        elif ni == i - 1:
            d = 2
        # Down
        elif nj == j + 1:
            d = 3
        else:
            return None
        return d

    def update_face(self):
        '''
        Update direction faced
        0, 1, 2, 3: Right, Up, Left, Down
        '''
        self.face = self.get_dir_next_tile(self.prev_tile, self.cur_tile)
        return self.face
    
    def get_dir_path(self):
        '''
        Returns direction to get to next tile according to path
        0, 1, 2, 3: Right, Up, Left, Down
        '''
        return self.get_dir_next_tile(self.cur_tile, self.path[self.cur_tile])

    def adjust_face(self):
        '''
        Returns action to rotate, keeps track of # steps taken rotating
        0, math.pi / 2
        '''
        dir_path = self.get_dir_path()
        if self.face is None or dir_path is None:
            return None, None
        if self.face == dir_path:
            return 0, 0
        self.adj_step += 1
        # self.turn_delta = (dir_path - self.face) % 4

        print(self.adj_step, ADJ_STEPS)
        self.adjust_done = self.adj_step >= ADJ_STEPS
        
        if self.adjust_done:
            self.adj_step = 0

        return 0, math.pi / 2

    def to_adjust(self):
        '''
        Returns boolean whether turning around is needed
        '''
        if self.face is None:
            return False
        dir_path = self.get_dir_path()
        self.turn_delta = (dir_path - self.face) % 4
        return self.turn_delta == 2
        