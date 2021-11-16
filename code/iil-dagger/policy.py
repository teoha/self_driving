from .learner import NeuralNetworkPolicy
from .lane_detection import *
import math

REF_VELOCITY = 0.5
ADJ_STEPS = 40
ANGLE_THRESHOLD = 0.2
ANGLE_DECAY = math.pi / 100
PERIOD=0.02

class Policy(NeuralNetworkPolicy):
    """
    Policy used to output control files

    ...
    
    Attributes
    ----------
    path : dict
        path that defines the next_tile for each drivable tile on the grid
    goal_tile : int, int
        tuple of the goal tile

    Methods
    -------
    predict(obs):
        returns action based on observation obs

    
    """
    def __init__(self, path, goal_tile, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = path
        self.goal_tile = goal_tile
        self.reached_goal = False

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

        # Localized global orientation
        self.orientation=None
        self.x=None
        self.y=None


    def predict(self, obs, cur_pos=None):
        if cur_pos is None:
            return 0, 0

        self.cur_tile = cur_pos
        if self.cur_tile == self.goal_tile:
            self.reached_goal = True
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

            # Update previous tiles
            self.prev_tile = self.prev_tile_step
            self.prev_tile_step = self.cur_tile

            # Update direction faced
            self.update_face()

            # Reset any turning counters
            self.turn_step = 0

            # Facing wrong direction - Rotate 180 degrees
            if self.to_adjust():
                self.adjust_done = False

            #Localization w.r.t center of right lane only if going straight
            pose=get_pose(obs)
            # print("pose:{},{}".format(*pose))
            # print("{},{},{}".format(self.prev_tile, self.prev_tile_step,self.cur_tile))
            # print("turning:{}".format(self.is_turning()))
            # input()
            if pose is not None and self.is_turning()==0:
                orientation, displacement=pose
                rough_orientation=self.get_dir_next_tile(self.prev_tile,self.cur_tile) #get rough orientation (EWNS)

                # Localize position
                rough_x = self.cur_tile[0] if self.cur_tile[0]==self.prev_tile[0] else max(self.cur_tile[0],self.prev_tile[0])
                rough_y = self.cur_tile[1] if self.cur_tile[1]==self.prev_tile[1] else max(self.cur_tile[1],self.prev_tile[1])

                if rough_orientation==0:
                    self.x=rough_x
                    self.y=rough_y+0.75-displacement
                elif rough_orientation==1:
                    self.x=rough_x+0.75-displacement
                    self.y=rough_y
                elif rough_orientation==2:
                    self.x=rough_x
                    self.y=rough_y+0.25+displacement
                elif rough_orientation==3:
                    self.x=rough_x+0.25+displacement
                    self.y=rough_y
                else:
                    print("invalid orientation")

                # Localize orientation
                self.orientation=rough_orientation+orientation

                print("================")
                print("x:{}, y:{}, theta:{}".format(self.x, self.y, self.orientation))
                print("================")
                input()
                

            # print("{},{} =================================================================".format(self.prev_tile_step, self.cur_tile))
            # input()


        else:
            # Localize based on actions since last localization
            # This localization does not start until the first tile is reached
            if None not in (self.x, self.y,self.orientation):
                self.x, self.y, self.orientation = self.get_new_pose(self.x,self.y,self.orientation,*self.prev_act, PERIOD)
                print("================")
                print("x:{}, y:{}, theta:{}".format(self.x, self.y, self.orientation))
                print("================")
                #input()

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
        # print("dpi:{}, dpj:{}, dni:{}, dnj:{}".format(dpi,dpj,dni,dnj))
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

        need_correction = self.need_correction() or 0
    
        # Decay angle turned over time
        if need_correction == 1 or need_correction == 0 and ang < 0:
            ang += ANGLE_DECAY
            ang = min(ang, 0)
        elif need_correction == -1 or need_correction == 0 and ang >= 0:
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

    def get_new_pose(self, x,y,theta,v,omega,t):
        new_x=x-(v/omega)*np.sin(theta)+(v/omega)*np.sin(theta+omega*t)
        new_y=y+(v/omega)*np.cos(theta)-(v/omega)*np.cos(theta+omega*t)
        new_theta=theta+omega*t

        return new_x, new_y, new_theta

    def get_next_tile_exact(self):
        '''
        get exact coordinates for center of lane of next tile
        '''
        if self.cur_tile is None:
            return None
        next_tile = self.path[self.cur_tile]

        x, y = next_tile
        
        if next_tile == self.goal_tile:
            return (x + 0.5, y + 0.5)
        
        d_path = self.get_dir_path()

        if d_path == 0:
            return x, y + 0.75
        elif d_path == 1:
            return x + 0.75, y + 1
        elif d_path == 2:
            return x + 1, y + 0.25
        else:
            return x + 0.25, y


    def get_ideal_angle(self):
        if None in (self.x, self.y, self.orientation):
            return None
        nx, ny = self.get_next_tile_exact()
        cx, cy = self.x, self.y
        dx = nx - cx
        dy = ny - cy
        ang = math.atan(dy / dx)
        # Adjust from 1st/4th quad to 3rd/2nd quad
        if dx < 0:
            ang += math.pi
        return ang


    def need_correction(self):
        '''
        Returns if correction needed comparing ideal angle and cur_angle
        0, 1, -1: not needed, clockwise turning, anti clockwise turning
        '''
        ideal_angle = self.get_ideal_angle()
        cur_angle = self.orientation
        if ideal_angle is None or cur_angle is None:
            return None
        d_angle = (ideal_angle - cur_angle) % (2 * math.pi)
        if abs(d_angle) < ANGLE_THRESHOLD:
            return 0
        # clockwise
        if d_angle < math.pi:
            return 1
        # anti-clockwise
        return -1
        
        