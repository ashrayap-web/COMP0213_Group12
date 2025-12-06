"""Three-finger hand (SDH) gripper implementation"""

import numpy as np
import pybullet as p
from .base_gripper import SimGripper


class ThreeFingerHand(SimGripper):
    """SDH three-finger gripper"""
    
    GRASP_JOINTS = [1, 4, 7]
    PRESHAPE_JOINTS = [2, 5, 8]
    UPPER_JOINTS = [3, 6, 9]

    def __init__(self, urdf_file, pos=None, orientation=None, target_obj=None):
        super().__init__(pos, target_obj)
        base_orientation_quat = self.set_orientation()
        TWIST_ANGLE = np.pi / 2 
        twist_quat = p.getQuaternionFromEuler([np.pi/2, 0, TWIST_ANGLE])
        twisted_orientation_quat = p.multiplyTransforms(
            positionA=[0, 0, 0],
            orientationA=base_orientation_quat,
            positionB=[0, 0, 0],
            orientationB=twist_quat,
        )[1] 
        self.orientation = np.array(orientation if orientation is not None else twisted_orientation_quat, dtype=float)
        self.body_id = p.loadURDF(urdf_file, basePosition=self.start_pos.tolist(), baseOrientation=self.orientation.tolist(), globalScaling=1.0)
        self.gripper_id = self.body_id 
        self.open = False
        self.num_joints = p.getNumJoints(self.body_id)
        
        # Filter the lists. If joint index >= num_joints, remove it.
        self.GRASP_JOINTS = [j for j in self.GRASP_JOINTS if j < self.num_joints]
        self.PRESHAPE_JOINTS = [j for j in self.PRESHAPE_JOINTS if j < self.num_joints]
        self.UPPER_JOINTS = [j for j in self.UPPER_JOINTS if j < self.num_joints]

        # Movement Constraint
        self.cid = p.createConstraint(
            parentBodyUniqueId=self.body_id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0,0,0],
            childFramePosition= self.start_pos.tolist(),
            parentFrameOrientation = [0,0,0,1],
            childFrameOrientation = self.orientation.tolist()
        )

    def preshape(self):
        """Move fingers into preshape pose."""
        for i in [2, 5, 8]:
            p.setJointMotorControl2(self.gripper_id, i, p.POSITION_CONTROL,
                                    targetPosition=-0.7, maxVelocity=0.5, force=1)
        self.open = False

    def open_gripper(self):
        """Gradually open fingers until fully open."""
        closed, iteration = True, 0
        while closed and not self.open:
            joints = self.get_joint_positions()
            closed = False
            for k in range(self.num_joints):
                if k in self.PRESHAPE_JOINTS and joints[k] >= 0.9:
                    self._apply_joint_command(k, joints[k] - 0.01)
                    closed = True
                elif k in self.UPPER_JOINTS and joints[k] <= 0.9:
                    self._apply_joint_command(k, joints[k] - 0.01)
                    closed = True
                elif k in self.GRASP_JOINTS and joints[k] <= 0.9:
                    self._apply_joint_command(k, joints[k] - 0.01)
                    closed = True
            iteration += 1
            if iteration > 1000:
                break
            p.stepSimulation()
        self.open = True

    def _apply_joint_command(self, joint, target):
        p.setJointMotorControl2(self.gripper_id, joint, p.POSITION_CONTROL,
                                targetPosition=target, maxVelocity=2, force=9999)

    def get_joint_positions(self):
        return [p.getJointState(self.gripper_id, i)[0] for i in range(self.num_joints)]

    def close_gripper(self): 
        """Close gripper to grab object"""
        self._apply_joint_command(joint=7, target=-0.5)
        for j in [1, 4, 7]:
            self._apply_joint_command(joint=j, target=0.3)
        self.open = False
        
    def move_gripper(self, x, y, z, force=80):
        p.changeConstraint(
            self.cid,
            jointChildPivot=[x, y, z],
            jointChildFrameOrientation = self.orientation,
            maxForce=force
        )

    def move_towards_obj(self):
        min_dist = 0.17
        z_offset = 0
        obj_pos, _ = p.getBasePositionAndOrientation(self.OBJ.body_id)
        obj_pos = np.array(obj_pos); obj_pos[2] += z_offset
        curr_pos, _ = p.getBasePositionAndOrientation(self.body_id)
        d_vec = obj_pos - np.array(curr_pos)
        pos_step = obj_pos - min_dist *(d_vec / np.linalg.norm(d_vec)) 
        self.move_gripper(pos_step[0], pos_step[1], pos_step[2], force=1000)
