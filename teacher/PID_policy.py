import math
import numpy as np

POSITION_THRESHOLD = 0.04
REF_VELOCITY = 0.7
FOLLOWING_DISTANCE = 0.24
AGENT_SAFETY_GAIN = 1.15

DEFAULT_FRAMERATE = 30
DELTA_TIME = 1.0 / DEFAULT_FRAMERATE

KP = 0.8
KI = 0.2
KD = 0.5

class PIDPolicy:
    """
    A Pure Pusuit controller class to act as an expert to the model
    ...
    Methods
    -------
    forward(images)
        makes a model forward pass on input images
    
    loss(*args)
        takes images and target action to compute the loss function used in optimization

    predict(observation)
        takes an observation image and predicts using env information the action
    """
    def __init__(self, env, ref_velocity=REF_VELOCITY, Kp=KP, Ki=KI, Kd=KD):
        """
        Parameters
        ----------
        ref_velocity : float
            duckiebot maximum velocity (default 0.7)
        """
        self.env = env
        self.ref_velocity = ref_velocity
        self.Kp=Kp
        self.Ki=Ki
        self.Kd=Kd
        
        self.clear()

    def clear(self):
        self.ITerm = 0.0
        self.last_error = 0.0

    def predict(self, obs):
        """
        Parameters
        ----------
        observation : image
            image of current observation from simulator
        Returns
        -------
        action: list
            action having velocity and omega of current observation
        """
        e, i, d = self.get_pid_val()
        print(e, i, d)
        if e is None:
            return 0, 0
            
        ux = self.Kp * e + self.Ki * i + self.Kd * d
        vel = self.ref_velocity
        
        ratio = ux / vel
        ratio = max(-1, ratio)
        ratio = min(1, ratio)
        
        angle = math.asin(ratio)
        
        return vel, angle
        
    def get_pid_val(self):
        if not self.env.full_transparency:
            print("not simulated environment, cannot get error")
            return None, None, None
        e = self.get_error()
        i = self.get_integral()
        d = self.get_dif()
        return e, i, d    
    
    def get_error(self):
        info = self.env.get_agent_info()
        if 'lane_position' not in info['Simulator']:
            return self.last_error
        e = info['Simulator']['lane_position']['dist']
        return e
    
    def get_integral(self):
        cur_err = self.get_error()
        self.ITerm += cur_err * DELTA_TIME
        return self.ITerm
        
    def get_dif(self):
        cur_err = self.get_error()
        d_err = cur_err - self.last_error
        self.last_error = cur_err
        return d_err / DELTA_TIME
    
