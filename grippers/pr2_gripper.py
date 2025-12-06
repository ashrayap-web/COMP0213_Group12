"""PR2 two-finger gripper implementation"""

import numpy as np
import pybullet as p
from .base_gripper import SimGripper


class PR2Gripper(SimGripper):
    """PR2 two-finger gripper"""
    
    def __init__(self, urdf_file, pos=None, orientation=None, target_obj=None):
        super().__init__(pos, target_obj)
        self.orientation = np.array(orientation if orientation is not None else self.set_orientation(), dtype=float)
        self.body_id = p.loadURDF(urdf_file, basePosition=self.start_pos.tolist(), baseOrientation=self.orientation.tolist())
        self.cid = p.createConstraint(self.body_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0,0,0], self.start_pos.tolist(), [0,0,0,1], self.orientation.tolist())

    def close_gripper(self):
        for joint in [0,2]:
            p.setJointMotorControl2(self.body_id, joint, p.POSITION_CONTROL, targetPosition=0.1, maxVelocity=2, force=10)

    def open_gripper(self):
        for joint in [0,2]:
            p.setJointMotorControl2(self.body_id, joint, p.POSITION_CONTROL, targetPosition=0.95, maxVelocity=2, force=10)

    def move_gripper(self, x, y, z, force=80):
        p.changeConstraint(self.cid, jointChildPivot=[x, y, z], jointChildFrameOrientation = self.orientation, maxForce=force)

    def move_towards_obj(self):
        obj_pos, _ = p.getBasePositionAndOrientation(self.OBJ.body_id)
        obj_pos = np.array(obj_pos); obj_pos[2] += 0.005
        curr_pos, _ = p.getBasePositionAndOrientation(self.body_id)
        d_vec = obj_pos - np.array(curr_pos)
        pos_step = obj_pos - 0.30 * (d_vec / np.linalg.norm(d_vec))
        self.move_gripper(pos_step[0], pos_step[1], pos_step[2], force=1000)
