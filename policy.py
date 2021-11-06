from .learner import NeuralNetworkPolicy
import math

REF_VELOCITY = 0.5

class Policy(NeuralNetworkPolicy):
    def __init__(self, path, map_grid, goal_tile, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = path
        self.map_grid = map_grid
        self.goal_tile = goal_tile
        self.prev_tile = None
        self.cur_tile = None
        self.prev_act = None

    def predict(self, obs, cur_pos=None):
        if cur_pos is None:
            return 0, 0
        # Environment gives inverted indices
        j, i = cur_pos
        self.cur_tile = (i, j)
        if self.cur_tile == self.goal_tile:
            return 0, 0
        # self.prev_act = super().predict(obs)
        # return self.prev_act
        # TODO: Starting movement
        if self.prev_tile is None:
            pass
        if self.prev_tile is None or not self.is_turning():
            if self.prev_tile is None:
                self.prev_tile = self.cur_tile
            # Use NN instructions
            self.prev_act = super().predict(obs)
            # return self.prev_act
        # Continue action if in same tile
        elif self.prev_tile == self.cur_tile:
            return self.prev_act
        elif self.path is None:
            self.prev_act = self.left()
            # return self.prev_act
        else:
            self.prev_act = self.get_dir()
        # print(self.cur_tile, self.prev_tile)
        if self.prev_tile != self.cur_tile and self.path[self.prev_tile] != self.cur_tile:
            self.prev_tile = self.path[self.prev_tile]
        return self.prev_act

    def is_turning(self):
        '''
        returns 1 if turning right
        returns -1 if turning left
        returns 0 if going straight
        '''
        pi, pj = self.prev_tile
        ci, cj = self.cur_tile
        ni, nj = self.path[self.cur_tile]
        dpi = ci - pi
        dpj = cj - pj        
        dni = ni - ci
        dnj = nj - cj
        if dpi == dni and dpj == dnj:
            return 0
        if dpi * dnj == -1 or dpj * dni == 1:
            return 1
        return -1

    def get_dir(self):
        i = self.is_turning()
        if i == 0:
            return self.prev_act
        if i == 1:
            return self.right()
        return self.left()


    def right(self):
        return REF_VELOCITY, -math.pi / 4

    def left(self):
        return REF_VELOCITY, math.pi / 4

    

    

    

    