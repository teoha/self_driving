import math
import numpy as np
from gym_duckietown.simulator import AGENT_SAFETY_RAD

POSITION_THRESHOLD = 0.04
REF_VELOCITY = 0.7
FOLLOWING_DISTANCE = 0.24
AGENT_SAFETY_GAIN = 1.15


class PID:
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
    def __init__(self, env, ref_velocity=REF_VELOCITY, P=0.2, I=0.0 ,D=0.0):
        """
        Parameters
        ----------
        ref_velocity : float
            duckiebot maximum velocity (default 0.7)
        following_distance : float
            distance used to follow the trajectory in pure pursuit (default 0.24)
        """
        self.env = env
        self.ref_velocity = ref_velocity
        self.Kp=P
        self.Ki=I
        self.Kd=D
        
        self.clear()

    def clear(self):
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0
        self.last_time = 0.0

    def predict(self):
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
        lane_pos=self.env.get_agent_info()
        angle_rad=lane_pos['Simulator']['lane_position']['angle_rad']
        if angle_rad>=0:
            error=np.mod(angle_rad,np.pi)
        else:
            error=-np.mod(angle_rad,np.pi)
        print(error)
        #print(error)
        current_time=lane_pos['Simulator']['timestamp']
        delta_time=current_time-self.last_time
        delta_error = error - self.last_error

        self.PTerm=self.Kp*error
        self.ITerm+=error*delta_time

        self.DTerm=0.0
        if delta_time > 0:
            self.DTerm=delta_error/delta_time

        self.last_time=current_time
        self.last_error=error

        #print('P:{}, I:{}, D:{}'.format(self.PTerm,self.Ki * self.ITerm,self.Kd * self.DTerm))

        steering=self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

        return np.array([self.ref_velocity,steering])
    

