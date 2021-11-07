from .learner import NeuralNetworkPolicy
import math
import numpy as np

REF_VELOCITY = 0.5
TURN_STEPS = 40

class Policy(NeuralNetworkPolicy):
    def __init__(self, path, map_grid, goal_tile, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = path
        self.map_grid = map_grid
        self.goal_tile = goal_tile
        self.prev_tile = None
        self.prev_tile_step = None
        self.cur_tile = None
        self.prev_act = None
        self.time_step = 0
        self.face = None #0,1,2,3: right,up,left,down wrt mapimg
        self.turn_step = 0
        self.adjust_done = True

    def predict(self, obs, cur_pos=None):
        if cur_pos is None:
            return 0, 0

        self.cur_tile = cur_pos
        if self.cur_tile == self.goal_tile:
            return 0, 0

        # TODO: Starting movement
        # if self.prev_tile is None and self.time_step < 5:
        #     self.time_step += 1
        #     return 0, -math.pi / 2
        if not self.adjust_done:
            # print("adjust face")
            self.prev_act = self.adjust_face()
        elif self.prev_tile is None or not self.is_turning():
            # print("NN")
            if self.prev_tile is None:
                self.prev_tile = self.cur_tile
                self.prev_tile_step = self.cur_tile
            # Use NN instructions
            self.prev_act = super().predict(obs)
        # Continue action if in same tile
        elif self.prev_tile == self.cur_tile and self.is_turning():
            # print("prev", self.is_turning())
            return self.prev_act
        elif self.path is None:
            self.prev_act = self.left()
        else:
            # print("get dire")
            self.prev_act = self.get_dir()
        # print(self.cur_tile, self.prev_tile)
        if self.prev_tile_step != self.cur_tile:
            print(self.prev_tile_step, self.cur_tile)
            self.update_face()
            self.prev_tile = self.prev_tile_step
            self.prev_tile_step = self.cur_tile
            # print(self.face, self.get_dir_path())
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

    def get_dir(self):
        i = self.is_turning()
        if i == 0:
            return self.prev_act
        if i == 1:
            return self.left()
        return self.right()


    def right(self):
        return REF_VELOCITY, -math.pi / 4

    def left(self):
        return REF_VELOCITY, math.pi / 4

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
            d = None
        return d

    def update_face(self):
        self.face = self.get_dir_next_tile(self.prev_tile_step, self.cur_tile)
        return self.face
    
    def get_dir_path(self):
        return self.get_dir_next_tile(self.cur_tile, self.path[self.cur_tile])

    def adjust_face(self):
        '''
        to adjust direction faced if wrong direction
        '''
        dir_path = self.get_dir_path()
        if self.face is None or dir_path is None:
            return None, None
        if self.face == dir_path:
            return 0, 0
        self.turn_step += 1
        self.turn_delta = (dir_path - self.face) % 4

        print(self.turn_step, TURN_STEPS)
        self.adjust_done = self.turn_step >= (TURN_STEPS)
        if self.adjust_done:
            self.turn_step = 0

        if self.turn_delta == 1:
            return self.left()
        elif self.turn_delta == 3:
            return self.right()
        return 0, math.pi / 2

    def to_adjust(self):
        if self.face is None:
            return False
        dir_path = self.get_dir_path()
        self.turn_delta = (dir_path - self.face) % 4
        return self.turn_delta == 2
        
        
    

    

    

    