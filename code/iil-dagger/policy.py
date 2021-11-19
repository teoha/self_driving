from .learner import NeuralNetworkPolicy
from .lane_detection import *
from .planner import MapGrid, BFS, get_next_pose
import math

REF_VELOCITY = 0.5
ADJ_STEPS = 45
ANGLE_THRESHOLD = 0.05
ANGLE_DECAY = math.pi / 100
PERIOD=0.05

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
    def __init__(self, goal_tile, grid, start_pos,*args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.path = path
        self.goal_tile = goal_tile
        self.start_pos = start_pos
        self.reached_goal = False
        self.grid=grid
        self.pose=None

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
        self.adjust_done = False

        # Localized global orientation
        self.orientation=None
        self.x=None
        self.y=None

        # variables for updating position
        self.gain = 1.0
        self.trim = 0.0
        self.radius = 0.0318
        self.k = 27.0
        self.limit = 1.0

        # Current path
        self.path=None

        # First relative pose before adjustment
        self.initial_pose=None

        # Current Action
        self.current_action=None

        # Last orientation
        self.last_orientation=None

        # Angle to adjust
        self.adj_angle=None
        self.adj_action=None
        self.adj_steps_req=None

    def predict(self, obs, cur_pos=None):
        if cur_pos is None:
            return 0, 0

        self.cur_tile = cur_pos
        if self.cur_tile == self.goal_tile:
            self.reached_goal = True
            return 0, 0
        prev_prev_act=self.prev_act

        # GET ACTION
        # Just started - adjust first
        if self.prev_tile is None:
            # initialise prev_tiles
            self.prev_tile = self.cur_tile
            self.prev_tile_step = self.cur_tile
            self.adjust_done=False
            return self.predict(obs, cur_pos)

        # LOCALIZATION
        # Get relative pose w.r.t lane
        if not self.adjust_done: #Initial adjustment
            self.pose=get_pose(obs,isTurn=True, horizon=1/2, houghTreshold=50, white_treshold=100,minLineLength=70)
        elif self.cur_tile==self.start_pos and self.grid.is_turn(self.cur_tile[1],self.cur_tile[0]): #Initial turning
            self.pose=get_pose(obs,isTurn=True,horizon=1/2, houghTreshold=200, white_treshold=100, side_tresholds=1/2,minLineLength=40)
        else:
            self.pose=get_pose(obs,True)

        if self.pose is not None:
            orientation, displacement = self.pose
            print("angle:{}, displacement:{}".format(*self.pose))
            if self.cur_tile==self.start_pos and self.grid.is_turn(self.cur_tile[1],self.cur_tile[0]):
                print("RESET")
                self.x, self.y, self.orientation = 0, 0.75-displacement, orientation

        # Relative localization in turn
        # if self.grid.is_turn(self.cur_tile[1],self.cur_tile[0]):
            # pose w.r.t to center of right lane
            # print("================")
            # if self.pose is not None:
            #     print("angle:{}, displacement:{}".format(*self.pose))
            #     # input()

        # Entered new tile
        if self.prev_tile_step != self.cur_tile:

            # Update previous tiles
            self.prev_tile = self.prev_tile_step
            self.prev_tile_step = self.cur_tile

            # Update direction faced
            self.update_face()

            # Reset any turning counters
            self.turn_step = 0
            self.localize_tile_change()

        elif None not in (self.x, self.y,self.orientation) and prev_prev_act is not None:
            # If localization fails for in junction and turns
            # Localize based on actions since last localization
            self.step(np.array(prev_prev_act))
            print("================")
            print("action to localize:{}".format(prev_prev_act))
            # After initial adjustment done while still in the starting tile
            if cur_pos==self.start_pos and self.adjust_done:
                print("Initial tile localization")

            print("x:{}, y:{}, theta:{}".format(self.x, self.y, self.orientation))
            print("================")
            # input()

        print("CURRENT ACTION: {}".format(self.current_action))
        
        # DETERMINE ACTION
        # Robot still in initial tile and initial adjustment is completed
        if self.cur_tile==self.start_pos and self.adjust_done:
            if self.grid.is_turn(self.cur_tile[1],self.cur_tile[0]) and self.pose is not None:
                self.prev_act=0.4, -self.pose[0]*2-self.pose[1]*5
            else:
                self.prev_act = super().predict(obs)
        # Adjusting angle - rotate
        elif not self.adjust_done:
            self.prev_act = self.adjust_face()
        # Going straight - use NN
        elif self.current_action==(1,0) and not self.is_facing_jn():
        # elif self.current_action==(1,0) and not self.grid.is_junction(*cur_pos[::-1]):
            self.prev_act = super().predict(obs)
        else:
            self.prev_act = self.get_turn_act()

        return self.prev_act

    def localize_tile_change(self):
        # Localize rough pose
        rough_orientation=self.get_dir_next_tile(self.prev_tile,self.cur_tile) #get rough orientation (EWNS)            
        rough_x = self.cur_tile[0] if self.cur_tile[0]==self.prev_tile[0] else max(self.cur_tile[0],self.prev_tile[0])
        rough_y = self.cur_tile[1] if self.cur_tile[1]==self.prev_tile[1] else max(self.cur_tile[1],self.prev_tile[1])
        self.last_orientation=rough_orientation


        # Determine current action
        # Check if tile is in plan, if not re-plan and follow new path
        if self.path is None or not (*self.cur_tile,self.face) in self.path:
            self.path=self.get_path(self.goal_tile, (*self.cur_tile,rough_orientation))
        self.current_action=self.path[(*self.cur_tile ,rough_orientation)]

        nx, ny, _ = get_next_pose((*self.cur_tile, self.last_orientation), self.current_action)
        face_trouble = self.grid.is_turn(ny, nx)

        # Localization after reaching FIRST new tile using actions
        if self.prev_tile == self.start_pos:
            orientation, displacement = self.orientation, 0.75-self.y

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
            self.orientation=(rough_orientation*np.pi/2)+orientation

            print("================")
            print("x:{}, y:{}, theta:{}".format(self.x, self.y, self.orientation))
            print("================")
            # input()


        # Localization w.r.t center of right lane only if going straight
        # elif self.pose is not None and self.grid.is_straight(self.cur_tile[1], self.cur_tile[0]):
        # elif self.pose is not None and self.grid.is_straight(self.cur_tile[1], self.cur_tile[0]) and not face_turn:
        elif self.pose is not None and self.grid.is_straight(*self.cur_tile[::-1]) and not self.grid.is_junction(*self.cur_tile[::-1]):
            
            orientation, displacement=self.pose

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
            self.orientation=(rough_orientation*np.pi/2)+orientation
            print("================")
            print("x:{}, y:{}, theta:{}".format(self.x, self.y, self.orientation))
            print("================")
        # Cross boundary
        elif self.x is not None and self.y is not None:
            d = self.get_dir_next_tile(self.prev_tile, self.cur_tile)
            if d == 0:
                self.x = self.cur_tile[0]
            elif d == 1:
                self.y = self.prev_tile[1]
            elif d == 2:
                self.x = self.prev_tile[0]
            else:
                self.y = self.cur_tile[1]
        input()

    
    def is_facing_jn(self):
        """
        Determine whether facing junction or not
        """
        cx, cy = self.cur_tile
        if self.grid.is_junction(cy, cx):
            return True
        
        nx, ny, _ = get_next_pose((*self.cur_tile, self.last_orientation), self.current_action)

        return self.grid.is_junction(ny, nx)

    def get_turn_act(self):
        # New turn action
        if self.turn_step == 0:
            vel = REF_VELOCITY
            ang = -self.current_action[1] * math.pi / 2
        # Continued turn action
        else:
            vel, ang = self.prev_act

        self.turn_step += 1

        need_correction = self.need_correction() * 2

        # Faster if going straight
        if abs(need_correction) < ANGLE_THRESHOLD:
            vel = 0.7
        elif need_correction > 0:
            ang = min(need_correction, math.pi / 2)
        elif need_correction < 0:
            ang = max(need_correction, -math.pi / 2)
    
        # Decay angle turned over time
        if ang < 0:
            ang += ANGLE_DECAY
            ang = min(ang, 0)
        else:
            ang -= ANGLE_DECAY
            ang = max(ang, 0)

        return vel, ang

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

    def adjust_face(self):
        '''
        Returns action to rotate, keeps track of # steps taken rotating
        0, math.pi / 2
        '''

        threshold=1
        action=self.adj_action if self.adj_action is not None else (0, math.pi / 2)

        # Record first instance of detecting lane and the orientation from localizing
        if self.pose is not None and self.adj_angle is None:
            orientation, displacement = self.pose
            self.initial_pose=self.pose
            self.adj_angle=-orientation
            self.adj_steps_req=abs(orientation)/0.07054
            self.adj_step += 1
            self.adj_action=(0, math.pi/2) if self.adj_angle>0 else (0, -math.pi/2)
            # self.adjust_done = orientation>-threshold and orientation<threshold # turn until relatively straight to a lane
            action=self.adj_action
        elif self.adj_angle is not None:
            self.adj_step += 1

        if self.adj_steps_req is not None:
            self.adjust_done=self.adj_step>=self.adj_steps_req

        # After adjustment, localize in hypothetical tile to monitor displacement within tile
        if self.adjust_done:
            self.adj_step = 0
            orientation, displacement = self.initial_pose
            self.x, self.y, self.orientation = 0, 0.75-displacement, orientation+self.adj_angle
            self.adj_angle=None
            return 0,0
        print("init action:{}".format(action))

        return action

    def get_next_tile_exact(self):
        '''
        get exact coordinates for center of lane of next tile
        '''
        if self.cur_tile is None:
            return None
        next_tile = get_next_pose((*self.cur_tile, self.last_orientation), self.current_action)

        x, y, theta = next_tile
        
        if next_tile == self.goal_tile:
            return (x + 0.5, y + 0.5)
        
        d_path = self.get_dir_next_tile(self.cur_tile, (x,y))
        print(x, y, d_path)
        if d_path == 0:
            return x + 1, y + 0.75
        elif d_path == 1:
            return x + 0.75, y
        elif d_path == 2:
            return x, y + 0.25
        else:
            return x + 0.25, y + 1


    def get_ideal_angle(self):
        if None in (self.x, self.y, self.orientation):
            return None
        nx, ny = self.get_next_tile_exact()
        cx, cy = self.x, self.y
        dx = nx - cx
        dy = cy - ny
        ang = math.atan(dy / dx)
        # Adjust from 1st/4th quad to 3rd/2nd quad
        if dx < 0:
            ang += math.pi
        print(f'nx={nx},ny={ny},cx={cx},cy={cy},angle={ang}')
        return ang % (2 * math.pi)


    def need_correction(self):
        '''
        Returns if correction needed comparing ideal angle and cur_angle
        0, 1, -1: not needed, clockwise turning, anti clockwise turning
        '''
        ideal_angle = self.get_ideal_angle()
        cur_angle = self.orientation
        if ideal_angle is None or cur_angle is None:
            return None
        cur_angle %= 2 * math.pi
        d_angle = (ideal_angle - cur_angle) % (2 * math.pi)
        print(f'ideal_angle={ideal_angle},cur_angle={cur_angle}, d_angle={d_angle}')
        if d_angle > math.pi:
            d_angle -= 2 * math.pi
        return d_angle
        
        
    def update_physics(self, action, delta_time=None):
        if delta_time is None:
            delta_time = 1/30
        wheelVels = action * 1.2 * 1
        print(wheelVels)
        prev_pos = np.array([self.x,0,self.y])
        # Update the robot's position
        pos, self.orientation = self._update_pos(pos=np.array([self.x,0,self.y]),
                                                   angle=self.orientation,
                                                   wheel_dist=0.102,
                                                   wheelVels=wheelVels,
                                                   deltaTime=delta_time)

        self.x=pos[0]
        self.y=pos[2]

    def _update_pos(self, pos, angle, wheel_dist, wheelVels, deltaTime):
        """
        Update the position of the robot, simulating differential drive

        returns new_pos, new_angle
        """

        Vl, Vr = wheelVels
        l = wheel_dist

        # If the wheel velocities are the same, then there is no rotation
        if Vl == Vr:
            pos = pos + deltaTime * Vl * self.get_dir_vec(angle)
            return pos, angle

        # Compute the angular rotation velocity about the ICC (center of curvature)
        w = (Vr - Vl) / l

        # Compute the distance to the center of curvature
        r = (l * (Vl + Vr)) / (2 * (Vl - Vr))

        # Compute the rotation angle for this time step
        rotAngle = w * deltaTime

        # Rotate the robot's position around the center of rotation
        r_vec = self.get_right_vec(angle)
        px, py, pz = pos
        cx = px + r * r_vec[0]
        cz = pz + r * r_vec[2]
        npx, npz = self.rotate_point(px, pz, cx, cz, rotAngle)
        pos = np.array([npx, py, npz])

        # Update the robot's direction angle
        angle += rotAngle
        return pos, angle
        
    def step(self, action):
        vel, angle = action

        # Distance between the wheels
        baseline = 0.102

        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # print("{}, {}".format(type((self.gain + self.trim)),type(k_r)))

        # adjusting k by gain and trim
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l


        omega_r = (vel + 0.5 * angle * baseline) / self.radius
        omega_l = (vel - 0.5 * angle * baseline) / self.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)

        vels = np.array([u_l_limited, u_r_limited])

        self.update_physics(vels)

    def get_right_vec(self, angle=None):
        """
        Vector pointing to the right of the agent
        """
        if angle == None:
            angle = self.cur_angle

        x = math.sin(angle)
        z = math.cos(angle)
        return np.array([x, 0, z])

    def rotate_point(self, px, py, cx, cy, theta):
        """
        Rotate a 2D point around a center
        """

        dx = px - cx
        dy = py - cy

        new_dx = dx * math.cos(theta) + dy * math.sin(theta)
        new_dy = dy * math.cos(theta) - dx * math.sin(theta)

        return cx + new_dx, cy + new_dy

    def get_dir_vec(self, angle=None):
        """
        Vector pointing in the direction the agent is looking
        """
        if angle == None:
            angle = self.cur_angle

        x = math.cos(angle)
        z = -math.sin(angle)
        return np.array([x, 0, z])

    def get_path(self,goal, start_pose):
        planner = BFS(goal, start_pose, self.grid.get_grid())
        path = planner.search()
        return path
        