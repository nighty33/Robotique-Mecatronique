import numpy as np
from abc import abstractmethod

from numpy import random
from numpy.linalg.linalg import solve
import homogeneous_transform as ht
import math
import test as t

from scipy import optimize

class RobotModel:
    def getNbJoints(self):
        """
        Returns
        -------
        length : int
            The number of joints for the robot
        """
        return len(self.getJointsNames())

    def getMotorsNames(self):
        """
        Returns
        -------
        motor_names : string[]
            The list of names for the motors
        """
        return [name + "_motor" for name in self.getJointsNames()]

    def getSensorsNames(self):
        """
        Returns
        -------
        motor_names : string[]
            The list of names for the motors
        """
        return [name + "_sensor" for name in self.getJointsNames()]

    @abstractmethod
    def getJointsNames(self):
        """
        Returns
        -------
        joint_names : string[]
            The list of names given to robot joints
        """

    @abstractmethod
    def getJointsLimits(self):
        """
        Returns
        -------
        np.array
            The values limits for the robot joints, each row is a different
            joint, column 0 is min, column 1 is max
        """

    @abstractmethod
    def getOperationalDimensionNames(self):
        """
        Returns
        -------
        joint_names : string array
            The list of names of the operational dimensions
        """

    @abstractmethod
    def getOperationalDimensionLimits(self):
        """
        Returns
        -------
        limits : np.array(x,2)
            The values limits for the operational dimensions, each row is a
            different dimension, column 0 is min, column 1 is max
        """

    @abstractmethod
    def getBaseFromToolTransform(self, joints):
        """
        Parameters
        ----------
        joints_position : np.array
            The values of the joints of the robot in joint space

        Returns
        -------
        np.array
            The transformation matrix from base to tool
        """

    @abstractmethod
    def computeMGD(self, q):
        """
        Parameters
        ----------
        q : np.array
            The values of the joints of the robot in joint space

        Returns
        -------
        np.array
            The coordinate of the effectors in the operational space
        """

    @abstractmethod
    def computeJacobian(self, joints):
        """
        Parameters
        ----------
        joints : np.array
            The values of the joints of the robot in joint space

        Returns
        -------
        np.array
            The jacobian of the robot for given joints values
        """

    @abstractmethod
    def analyticalMGI(self, target):
        """
        Parameters
        ----------
        joints : np.arraynd shape(n,)
            The current values of the joints of the robot in joint space
        target : np.arraynd shape(m,)
            The target in operational space

        Returns
        -------
        nb_solutions : int
            The number of solutions for the given target, -1 if there is an
            infinity of solutions
        joint_pos : np.ndarray shape(X,) or None
            One of the joint configuration which allows to reach the provided
            target. If no solution is available, returns None.
        """

    def computeMGI(self, joints, target, method):
        """
        Parameters
        ----------
        joints : np.ndarray shape(n,)
            The current position of joints in angular space
        target : np.ndarray shape(m,)
            The target in operational space
        method : str
            The method used to compute MGI, available choices:
            - analyticalMGI
            - jacobianInverse
            - jacobianTransposed
        seed : None or int
            The seed used for inner random components if needed
        """
        if method == "analyticalMGI":
            nb_sols, sol = self.analyticalMGI(target)
            t.test_method(target, sol, getRobotModel(self.getRobotName()))
            return sol
        elif method == "jacobianInverse":
            ret = self.solveJacInverse(joints, target)
            t.test_method(target, ret, getRobotModel(self.getRobotName()))
            return ret
        elif method == "jacobianTransposed":
            ret = self.solveJacTransposed(joints, target)
            t.test_method(target, ret, getRobotModel(self.getRobotName()))
            return ret
        raise RuntimeError("Unknown method: " + method)

    @abstractmethod
    def getRobotName(self):
        """
        Returns
        -------
        robot_name : name of the robot
        """

    def solveJacInverse(self, joints, target):
        """
        Parameters
        ----------
        joints: np.ndarray shape(n,)
            The initial position for the search in angular space
        target: np.ndarray shape(n,)
            The wished target for the tool in operational space
        """
        # TODO: implement
        iter = 0
        max_step = 0.1
        size = len(target)
        J = None
        inv_J = None
        deg_tol = 0.001
        v_dist = (target - self.computeMGD(joints))
        while np.linalg.norm(v_dist) > deg_tol and iter < 50:
            a = np.zeros(size)
            J = self.computeJacobian(joints)
            if(J.shape[0]==J.shape[1]):
                if np.linalg.det(J) != 0:
                    inv_J = np.linalg.inv(J)
                    v_dist = (target - self.computeMGD(joints))
                    epsilon = inv_J @ v_dist
                    if np.linalg.norm(epsilon) > max_step:
                        epsilon = epsilon/np.linalg.norm(epsilon)*max_step
                    joints = joints + epsilon
                else:
                    for i in range(size):
                        a[i] = np.random.uniform(0.0, 0.01)
                    joints = joints + a
            else:
                raise RuntimeError("Matrix non-invertible: '" + J + "'")
            iter+=1

        return joints

    def distance(self, joints ,target):
        return np.linalg.norm(self.computeMGD(joints) - target , 2)

    def jacobian_function(self, joints, target):
        size = len(joints)
        J = self.computeJacobian(joints)
        v_dist = (target - self.computeMGD(joints))
        return -2 * J.transpose() @ v_dist

    def solveJacTransposed(self, joints, target):

        result = optimize.minimize(self.distance, joints, target, jac=self.jacobian_function)
        return result.x


class RobotRT(RobotModel):
    """
    Model a robot with a 2 degrees of freedom: 1 rotation and 1 translation

    The operational space of the robot is 2 dimensional because it can only move inside a plane
    """

    def __init__(self):
        self.W = 0.05
        self.L0 = 1.0
        self.L1 = 0.2
        self.L2 = 0.25 + self.W/2  # Distance including the offset
        self.max_q1 = 0.25
        self.T_0_1 = ht.translation([0, 0, self.L0+self.W/2])
        self.T_1_2 = ht.translation([self.L1, 0, 0])
        self.T_2_E = ht.translation([0.0, -self.L2, 0]) @ ht.rot_z(np.pi)

    def getRobotName(self):
        return "RobotRT"

    def getJointsNames(self):
        return ["q1", "q2"]

    def getJointsLimits(self):
        return np.array([[-np.pi, np.pi], [0, 0.55]], dtype=np.double)

    def getOperationalDimensionNames(self):
        return ["x", "y"]

    def getOperationalDimensionLimits(self):
        dimension = math.sqrt(self.L2**2 + self.L1**2)

        return np.array([[-dimension, dimension],  [-dimension, dimension]])

    def getBaseFromToolTransform(self, joints):
        T_0_1 = self.T_0_1 @ ht.rot_z(joints[0])
        T_1_2 = self.T_1_2 @ ht.translation(joints[1] * np.array([1, 0, 0]))
        return T_0_1 @ T_1_2 @ self.T_2_E

    def computeMGD(self, q):
        tool_pos = self.getBaseFromToolTransform(q) @ np.array([0, 0, 0, 1])
        return tool_pos[:2]

    def analyticalMGI(self, target):
        x = target[0]
        y = target[1]

        dist = np.linalg.norm(target[:2])
        min = math.sqrt(self.max_q1**2 + self.L1**2)
        max = math.sqrt(self.max_q1**2 + (self.L1+self.max_q1)**2 )
        if dist < min or dist > max:
            return 0 , None

        angle1 = math.atan2(self.max_q1, x)
        dist_cur = math.sqrt( (dist**2) - (x**2))
        angle2= math.atan2(y,dist_cur)
        return 1, [angle1+angle2, dist_cur-self.L1]

    def computeJacobian(self, joints):
        # TODO: implement
        T_0_1 = self.T_0_1 @ ht.rot_z(joints[0])
        T_1_2 = self.T_1_2 @ ht.translation(joints[1] * np.array([1, 0, 0]))

        dev_T_q1 = self.T_0_1 @ ht.d_rot_z(joints[0]) @ T_1_2 @ self.T_2_E
        dev_T_q2 = T_0_1 @ self.T_1_2 @ ht.d_translation(np.array([1, 0, 0])) @ self.T_2_E

        J = np.zeros((2, 2), dtype=np.double)
        J[:, 0] = dev_T_q1[:2, -1]
        J[:, 1] = dev_T_q2[:2, -1]
        return J


class RobotRRR(RobotModel):
    """
    Model a robot with 3 degrees of freedom along different axis
    """

    def __init__(self):
        self.W = 0.05
        self.L0 = 1.0 + self.W/2
        self.L1 = 0.5
        self.L2 = 0.4
        self.L3 = 0.3 + self.W/2
        self.T_0_1 = ht.translation([0, 0, self.L0])
        self.T_1_2 = ht.translation([0, self.L1, 0])
        self.T_2_3 = ht.translation([0.0, self.L2, 0])
        self.T_3_E = ht.translation([0.0, self.L3, 0])

    def getRobotName(self):
        return "RobotRRR"

    def getJointsNames(self):
        return ["q1", "q2", "q3"]

    def getJointsLimits(self):
        return np.array([[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]], dtype=np.double)

    def getOperationalDimensionNames(self):
        return ["x", "y", "z"]

    def getOperationalDimensionLimits(self):
        dim_x = self.L1+self.L2+self.L3
        dim_y = dim_x
        dim_z = self.L2+self.L3

        return np.array([[-dim_x,dim_x],[-dim_y,dim_y],[self.L0-dim_z,self.L0+dim_z]])

    def getBaseFromToolTransform(self, joints):
        T_0_1 = self.T_0_1 @ ht.rot_z(joints[0])
        T_1_2 = self.T_1_2 @ ht.rot_x(joints[1])
        T_2_3 = self.T_2_3 @ ht.rot_x(joints[2])
        return T_0_1 @ T_1_2 @ T_2_3 @ self.T_3_E

    def computeMGD(self, q):
        tool_pos = self.getBaseFromToolTransform(q) @ np.array([0, 0, 0, 1])
        return tool_pos[:3]

    def cosine(self,x,y,L1,L2):
        D = math.sqrt(x**2+y**2)
        alpha = math.acos( (L1**2 + D**2 - L2**2) / (2*L1*D) )
        beta = math.acos( (L1**2 + L2**2 - D**2) / (2*L2*L1) )
        nb_sol = 2

        if(D == L1 + L2):
            nb_sol = 1

        return alpha , beta , nb_sol

    def analyticalMGI(self, target):

        x =target[0]
        y =target[1]
        z =target[2]

        x_lim , y_lim , z_lim = self.getOperationalDimensionLimits()

        if(x < x_lim[0] or x > x_lim[1]
            or y < y_lim[0] or y > y_lim[1]
            or z < z_lim[0] or z > z_lim[1]):
            print("too far")
            return 0 , [0,0,0]

        q1 = np.pi/2 - math.atan2(y,x)

        nb_sol = 0
        sols = []

        for q in [q1, q1+np.pi]:
            x_1_2, y_1_2 , z_1_2 , o_1_2 = (ht.invert_transform(self.T_0_1) @ ht.rot_z(q) @ np.concatenate((target[:3],[1])))

            dist = math.sqrt((y_1_2-self.L1)**2+z_1_2**2)
            if (dist < abs(self.L2 - self.L3)) or dist > (self.L2 + self.L3):
                break
            else:
                a , b , nb = self.cosine(y_1_2 -self.L1,z_1_2 , self.L2 , self.L3)
                phi = math.atan2(z_1_2, y_1_2 - self.L1)
                sols.append([q,phi-a, np.pi - b])
                sols.append([q,phi+a,b-np.pi])
                nb_sol+=nb

        if(len(sols)== 0):
            print("any solutions")
            return 0 , None

        joints = [-sols[0][0],sols[0][1],sols[0][2]]
        
        return nb_sol , np.array(joints , dtype=np.double)

    def computeJacobian(self, joints):
        # TODO: implement
        T_0_1 = self.T_0_1 @ ht.rot_z(joints[0])
        T_1_2 = self.T_1_2 @ ht.rot_x(joints[1])
        T_2_3 = self.T_2_3 @ ht.rot_x(joints[2])

        dev_T_q1 = self.T_0_1 @ ht.d_rot_z(joints[0]
                                           ) @ T_1_2 @ T_2_3 @ self.T_3_E
        dev_T_q2 = T_0_1 @ self.T_1_2 @ ht.d_rot_x(
            joints[1]) @ T_2_3 @ self.T_3_E
        dev_T_q3 = T_0_1 @ T_1_2 @ self.T_2_3 @ ht.d_rot_x(
            joints[2]) @ self.T_3_E

        J = np.zeros((3, 3), dtype=np.double)
        J[:, 0] = dev_T_q1[:3, -1]
        J[:, 1] = dev_T_q2[:3, -1]
        J[:, 2] = dev_T_q3[:3, -1]
        return J


class LegRobot(RobotModel):
    """
    Model of a simple robot leg with 4 degrees of freedom
    """

    def __init__(self):
        # TODO: implement
        self.W = 0.05
        self.L0 = 1.0
        self.L1 = 0.5
        self.L2 = 0.3
        self.L3 = 0.3
        self.L4 = 0.2
        self.T_0_1 = ht.translation([0, 0, self.L0])
        self.T_1_2 = ht.translation([0, self.L1, 0])
        self.T_2_3 = ht.translation([0.0, self.L2, 0])
        self.T_3_4 = ht.translation([0.0, self.L3, 0])
        self.T_4_E = ht.translation([0.0, self.L4, 0])

    def getRobotName(self):
        return "LegRobot"

    def getJointsNames(self):
        return ["q1", "q2", "q3", "q4"]

    def getJointsLimits(self):
        angle_lim = np.array([-np.pi, np.pi])
        L = np.zeros((4, 2))
        for d in range(4):
            L[d, :] = angle_lim
        return L

    def getOperationalDimensionNames(self):
        return ["x", "y", "z", "r32"]

    def getOperationalDimensionLimits(self):
        # TODO: implement
        dim_x = self.L1+self.L2+self.L3+self.L4
        dim_y = dim_x
        dim_z = self.L2+self.L3+self.L4
        return np.array([[-dim_x,dim_x],[-dim_y,dim_y],[self.L0-dim_z,self.L0+dim_z]])

    def getBaseFromToolTransform(self, joints):
        # TODO: implement
        T_0_1 = self.T_0_1 @ ht.rot_z(joints[0])
        T_1_2 = self.T_1_2 @ ht.rot_x(joints[1])
        T_2_3 = self.T_2_3 @ ht.rot_x(joints[2])
        T_3_4 = self.T_3_4 @ ht.rot_x(joints[3])
        return T_0_1 @ T_1_2 @ T_2_3 @ T_3_4 @ self.T_4_E

    def extractMGD(self, T):
        """
        T : np.arraynd shape(4,4)
           An homogeneous transformation matrix
        """
        # TODO: implement
        new = np.zeros(4, dtype=np.double)
        new[0] = T[0,3]
        new[1] = T[1,3]
        new[2] = T[2,3]
        new[3] = T[2,1]
        return new

    def computeMGD(self, joints):
        # TODO: implement
        T = self.getBaseFromToolTransform(joints)
        return np.append(T[:-1, -1], T[1,2])

    def cosine(self,x,y,L1,L2):
        D = math.sqrt(x**2+y**2)
        if (D < abs(L1 - L2)) or D > (L1 + L2):
            return 0,0,0
        alpha = math.acos( (L1**2 + D**2 - L2**2) / (2*L1*D) )
        beta = math.acos( (L1**2 + L2**2 - D**2) / (2*L2*L1) )
        nb_sol = 2

        if(D == L1 + L2):
            nb_sol = 1

        return alpha , beta , nb_sol
    
    def analyticalMGI(self, target):
        sols = []
        nb_sols=0
        theta = np.pi/2 - math.atan2(target[1],target[0])
        for q0 in [theta, theta + np.pi]:
            x_1_2 , y_1_2 , z_1_2 , o_1_2 = ht.rot_z(q0) @ ht.invert_transform(self.T_0_1) @ np.concatenate((target[:3],[1]))        

            r3_2 = -math.asin(target[3])

            y3_1_2 = y_1_2 - math.cos(r3_2) * self.L4 - self.L1
            z3_1_2 = z_1_2 - math.sin(r3_2) * self.L4

            dist = math.sqrt(y3_1_2**2+z3_1_2**2)
            if (dist < abs(self.L2 - self.L3)) or dist > (self.L2 + self.L3):
                break
            else:
                a , b , sol = self.cosine(y3_1_2 , z3_1_2 , self.L2 , self.L3)
                phi = math.atan2(z3_1_2,y3_1_2 )
                alpha = phi + a
                beta = b - np.pi
                alpha2 = phi -a
                beta2 = np.pi - b
                q3 = r3_2 - alpha - beta
                sols.append([-q0,alpha,beta,q3])
                sols.append([-q0,alpha2,beta2,q3])
                nb_sols+= sol
            
        if nb_sols == 0:
            return 0, None
        return nb_sols, sols[0]

    def computeJacobian(self, joints):
        # TODO: implement
        T_0_1 = self.T_0_1 @ ht.rot_z(joints[0])
        T_1_2 = self.T_1_2 @ ht.rot_x(joints[1])
        T_2_3 = self.T_2_3 @ ht.rot_x(joints[2])
        T_3_4 = self.T_3_4 @ ht.rot_x(joints[3])

        dev_T_q1 = self.T_0_1 @ ht.d_rot_z(joints[0]
            ) @ T_1_2 @ T_2_3 @ T_3_4 @ self.T_4_E
        dev_T_q2 = T_0_1 @ self.T_1_2 @ ht.d_rot_x(
            joints[1]) @ T_2_3 @ T_3_4 @ self.T_4_E
        dev_T_q3 = T_0_1 @ T_1_2 @ self.T_2_3 @ ht.d_rot_x(
            joints[2]) @ T_3_4 @ self.T_4_E
        dev_T_q4 = T_0_1 @ T_1_2 @ T_2_3 @ self.T_3_4 @ ht.d_rot_x(
            joints[3]) @ self.T_4_E

        J = np.zeros((3, 4), dtype=np.double)
        J[:, 0] = dev_T_q1[:3, -1]
        J[:, 1] = dev_T_q2[:3, -1]
        J[:, 2] = dev_T_q3[:3, -1]
        J[:, 3] = dev_T_q4[:3, -1]
        return J


def getRobotModel(robot_name):
    robot = None
    if robot_name == "RobotRT":
        robot = RobotRT()
    elif robot_name == "RobotRRR":
        robot = RobotRRR()
    elif robot_name == "LegRobot":
        robot = LegRobot()
    else:
        raise RuntimeError("Unknown robot name: '" + robot_name + "'")
    return robot
