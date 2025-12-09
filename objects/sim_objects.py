"""Simulation objects for grasp experiments"""

import pybullet as p


class SimObject:
    """Base class for simulation objects"""
    
    def __init__(self, name, urdf_file=None, pos=None, orientation=None, scale=1.0):
        self.name = name
        self.pos = pos
        self.orientation = orientation
        self.body_id = p.loadURDF(urdf_file, basePosition=self.pos, baseOrientation=self.orientation, globalScaling=scale)
        self.pos_grab_before = None
        self.pos_grab_after = None


class CubeObject(SimObject):
    """Cube object for grasping"""
    
    def __init__(self, name, urdf_file="cube_small.urdf", pos=None, orientation=None, scale=1.3):
        super().__init__(name, urdf_file, pos, orientation, scale)


class CylinderObject(SimObject):
    """Cylinder object for grasping"""
    
    def __init__(self, name, urdf_file="urdf_files/cylinder.urdf", pos=None, orientation=None, scale=0.55):
        super().__init__(name, urdf_file, pos, orientation, scale)
        p.changeDynamics(self.body_id, -1, lateralFriction=1.0)
