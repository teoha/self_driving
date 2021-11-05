import numpy as np

class MapGrid:
    '''
    self.grid is a boolean array whether a tile is drivable
    self.letter_grid is a letter array classifying the tile as
    J: Junction, T: Turn, D: Drivable, N: Not Drivable
    e.g. for map2_0, top left is (0,0), (height, width)
    NNNNNNNNN
    NTDJDJDTN
    NDNDNDNDN
    NJDJDJDJN
    NDNDNDNDN
    NJDJDJDJN
    NDNDNDNDN
    NTDJDJDTN
    NNNNNNNNN
    '''
    def __init__(self, map_img, tile_size=100):
        self.map_img = map_img
        self.tile_size = tile_size
        self.height, self.width, _ = map_img.shape
        self.get_grid()
        self.get_letter_grid()

    def get_grid(self):
        self.tile_width = int(self.width / self.tile_size)
        self.tile_height = int(self.height / self.tile_size)
        self.grid = [[False for _ in range(self.tile_width)] for _ in range(self.tile_height)]
        for i in range(self.tile_height):
            for j in range(self.tile_width):
                pixels = self.map_img[self.tile_size * i: self.tile_size * (i + 1), self.tile_size * j: self.tile_size * (j + 1)]
                self.grid[i][j] = np.mean(pixels, axis=(0,1))[0] > 0
        return self.grid

    def is_drivable(self, i, j):
        if i < 0 or i >= self.tile_height or j < 0 or j >= self.tile_width:
            return False
        return self.grid[i][j]
    
    def get_drivable_around(self, i, j):
        n = 0
        for k in (-1, 0, 1):
            for m in (-1, 0, 1):
                if k != 0 and m != 0: continue
                if k == 0 and m == 0: continue
                if not self.is_drivable(i + k, j + m): continue
                n += 1
        return n
    
    def is_junction(self, i, j):
        return self.is_drivable(i, j) and self.get_drivable_around(i, j) > 2
    
    def is_turn(self, i, j):
        return (self.is_drivable(i, j) 
                and not self.is_junction(i, j) 
                and not self.is_straight(i, j))

    def is_straight(self, i, j):
        hor = self.is_drivable(i, j - 1) and self.is_drivable(i, j + 1)
        ver = self.is_drivable(i - 1, j) and self.is_drivable(i + 1, j)
        return hor or ver

    def get_letter_grid(self):
        self.letter_grid = [['N' for _ in range(self.tile_width)] for _ in range(self.tile_height)]
        for i in range(self.tile_height):
            for j in range(self.tile_width):
                # s += str(self.get_drivable_around(i, j))
                if self.is_junction(i, j):
                    self.letter_grid[i][j] = 'J'
                elif self.is_turn(i, j):
                    self.letter_grid[i][j] = 'T'
                elif self.is_drivable(i, j):
                    self.letter_grid[i][j] = 'D'
            print(''.join(self.letter_grid[i]))
        return self.letter_grid
        

