import math
import numpy as np
from scipy.spatial.transform import Rotation as R

def euler_to_axis(roll,pitch,yaw):
    '''
    将roll,pitch,yaw转换成绕特定轴的旋转角度
    '''
    roll,pitch,yaw = map(lambda x:(x/180)*math.pi, [roll,pitch,yaw])
    yawMatrix = np.matrix([
    [math.cos(yaw), -math.sin(yaw), 0],
    [math.sin(yaw), math.cos(yaw), 0],
    [0, 0, 1]
    ])

    pitchMatrix = np.matrix([
    [math.cos(pitch), 0, math.sin(pitch)],
    [0, 1, 0],
    [-math.sin(pitch), 0, math.cos(pitch)]
    ])

    rollMatrix = np.matrix([
    [1, 0, 0],
    [0, math.cos(roll), -math.sin(roll)],
    [0, math.sin(roll), math.cos(roll)]
    ])

    R = yawMatrix * pitchMatrix * rollMatrix

    # multi = 1 / (2 * math.sin(theta))

    rx = (R[2, 1] - R[1, 2])
    ry = (R[0, 2] - R[2, 0])
    rz = (R[1, 0] - R[0, 1])

    # norm = math.sqrt(rx**2+ry**2+rz**2)
    # rx, ry, rz = rx/norm, ry/norm, rz/norm

    theta = math.acos(((R[0, 0] + R[1, 1] + R[2, 2]) - 1) / 2)
    theta = theta/math.pi*180
    return theta, rx,ry,rz

def euler_to_rotation_matrix(yaw,pitch,roll):
    yaw,pitch,roll = yaw/180*np.pi, pitch/180*np.pi, roll/180*np.pi
    z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                  [np.sin(yaw),  np.cos(yaw), 0],
                  [0,0,1]])
    y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                  [0,1,0],
                  [-np.sin(pitch),0, np.cos(pitch)]])

    x = np.array([[1,0,0],
                  [0, np.cos(roll), -np.sin(roll)],
                  [0, np.sin(roll),  np.cos(roll)]])


def axis_to_rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def euler_to_4x4matrix(roll,pitch,yaw):
    rotation = R.from_euler('zyx',[yaw,pitch,roll],degrees=True)
    m = rotation.as_matrix()
    m_new = np.zeros([4,4])
    m_new[0:3,0:3] = m
    m_new[:,-1] = np.array([0,0,0,1])
    m_new[-1,:] = np.array([0,0,0,1])
    return m_new