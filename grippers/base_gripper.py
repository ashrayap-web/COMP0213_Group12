"""Base gripper interface"""

import numpy as np
import pybullet as p
from abc import ABC, abstractmethod


class SimGripper(ABC):
    """Abstract base class for gripper simulation"""
    
    def __init__(self, pos=None, target_obj=None):
        self.OBJ = target_obj
        self.start_pos = np.array(pos if pos is not None else [np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(0.6, 1)], dtype=float)
        self.grab_start_pos = None
        self.grab_end_pos = None
    
    @abstractmethod
    def set_orientation(self):
        pass
    
    @abstractmethod
    def open_gripper(self):
        pass
    
    @abstractmethod
    def close_gripper(self):
        pass
    
    @abstractmethod
    def move_gripper(self, x, y, z, force=80):
        pass
    
    @abstractmethod
    def move_towards_obj(self):
        pass
    
    def set_orientation(self):
        obj_pos, _ = p.getBasePositionAndOrientation(self.OBJ.body_id)
        d_vector = np.array(obj_pos) - self.start_pos 
        dx, dy, dz = d_vector
        if np.abs(dx) < 1e-3 and np.abs(dy) < 1e-3:
            roll, yaw, pitch = -np.pi/2, 0, np.pi/2
        else:
            pitch  = np.atan2(-dz, np.sqrt(dx**2 + dy**2)) 
            yaw = np.atan2(dy, dx)
            roll = 0
            
        # Add noise to orientation
        yaw += np.random.normal(0, np.pi/72)
        pitch += np.random.normal(0, np.pi/72)
        return p.getQuaternionFromEuler([roll, pitch, yaw])
    
    def is_success(self):
        start_pos_obj = np.array(self.OBJ.pos_grab_before)
        end_pos_obj = np.array(self.OBJ.pos_grab_after)
        dist_diff_vec = (np.array(self.grab_end_pos) - np.array(self.grab_start_pos)) - (end_pos_obj - start_pos_obj)
        _, angular_vel = p.getBaseVelocity(self.OBJ.body_id)
        if dist_diff_vec[2] < 0.01 and np.linalg.norm(angular_vel) < 0.25:
            print(f"\033[32m{self.OBJ.name}: SUCCESS\033[0m")
            return 1.0
        else:
            print(f"\033[31m{self.OBJ.name}: FAILURE\033[0m")
            return 0.0
