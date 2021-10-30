from .pid_policy import PIDPolicy
import math

class JunctionPolicy(PIDPolicy):
    def __init__(self, path=None, *args):
        super().__init__(*args)
        self.prev_tile = None
        self.cur_tile = None
        self.prev_act = None
        self.path = path
    
    def is_junction(self, i, j):
        mp = self.env.map_data['tiles']
        tile = mp[i][j]
        return 'way' in tile
    
    def predict(self, obs):
        j, _, i = self.env.cur_pos
        i, j = int(i), int(j)
        self.cur_tile = (i, j)
        if not self.is_junction(i, j) or self.prev_tile is None:
            self.prev_tile = self.cur_tile
            return super().predict(obs)
        if self.prev_tile == self.cur_tile:
            return self.prev_act
        if self.path is None:
            return self.right()
        return self.get_dir()
    
    def get_dir(self):
        pi, pj = self.prev_tile
        ci, cj = self.cur_tile
        ni, nj = self.path[self.cur_tile]
        dpi = ci - pi
        dpj = cj - pj        
        dni = ni - ci
        dnj = nj - cj
        if dpi == dni and dpj == dnj:
            return 0, 0
        if dpi * dnj == -1 or dpj * dni == 1:
            return self.right()
        return self.left()


    def right(self):
        return self.ref_velocity, -math.pi / 4

    def left(self):
        return self.ref_velocity, math.pi / 8
        