import numpy as np
from abc import abstractmethod
import homogeneous_transform as ht


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
            return sol
        elif method == "jacobianInverse":
            return self.solveJacInverse(joints, target)
        elif method == "jacobianTransposed":
            return self.solveJacTransposed(joints, target)
        raise RuntimeError("Unknown method: " + method)

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
        return joints

    def solveJacTransposed(self, joints, target):
        """
        Parameters
        ----------
        joints: np.ndarray shape(n,)
            The initial position for the search in angular space
        target: np.ndarray shape(n,)
            The wished target for the tool in operational space
        """
        # TODO: implement
        return joints


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

    def getJointsNames(self):
        return ["q1", "q2"]

    def getJointsLimits(self):
        return np.array([[-np.pi, np.pi], [0, 0.55]], dtype=np.double)

    def getOperationalDimensionNames(self):
        return ["x", "y"]

    def getOperationalDimensionLimits(self):
        # TODO: implement
        return np.array([[-1, 1],  [-1, 1]])

    def getBaseFromToolTransform(self, joints):
        T_0_1 = self.T_0_1 @ ht.rot_z(joints[0])
        T_1_2 = self.T_1_2 @ ht.translation(joints[1] * np.array([1, 0, 0]))
        return T_0_1 @ T_1_2 @ self.T_2_E

    def computeMGD(self, q):
        tool_pos = self.getBaseFromToolTransform(q) @ np.array([0, 0, 0, 1])
        return tool_pos[:2]

    def analyticalMGI(self, target):
        # TODO: implement
        return 0, np.array([0, 0], dtype=np.double)

    def computeJacobian(self, joints):
        # TODO: implement
        T_0_1 = self.T_0_1 @ ht.rot_z(joints[0])
        T_1_2 = self.T_1_2 @ ht.translation(joints[1] * np.array([1, 0, 0]))
        
        dev_T_q1 = self.T_0_1 @ ht.d_rot_z(joints[0]) @ T_1_2 @ self.T_2_E
        dev_T_q2 = T_0_1 @ self.T_1_2 @ htd_.translation(joints[1] * np.array([1, 0, 0])) @ self.T_2_E
        
        J = np.zeros((2, 2), dtype=np.double)
        J[:, 0] = dev_T_q1.T
        J[:, 1] = dev_T_q2.T
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

    def getJointsNames(self):
        return ["q1", "q2", "q3"]

    def getJointsLimits(self):
        return np.array([[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]], dtype=np.double)

    def getOperationalDimensionNames(self):
        return ["x", "y", "z"]

    def getOperationalDimensionLimits(self):
        # TODO: implement
        return np.array([[-1, 1], [-1, 1], [-1, 1]])

    def getBaseFromToolTransform(self, joints):
        T_0_1 = self.T_0_1 @ ht.rot_z(joints[0])
        T_1_2 = self.T_1_2 @ ht.rot_x(joints[1])
        T_2_3 = self.T_2_3 @ ht.rot_x(joints[2])
        return T_0_1 @ T_1_2 @ T_2_3 @ self.T_3_E

    def computeMGD(self, q):
        tool_pos = self.getBaseFromToolTransform(q) @ np.array([0, 0, 0, 1])
        return tool_pos[:3]

    def analyticalMGI(self, target):
        # TODO: implement
        return 1, np.array([0, 0, 0], dtype=np.double)

    def computeJacobian(self, joints):
        # TODO: implement
        T_0_1 = self.T_0_1 @ ht.rot_z(joints[0])
        T_1_2 = self.T_1_2 @ ht.rot_x(joints[1])
        T_2_3 = self.T_2_3 @ ht.rot_x(joints[2])
        
        dev_T_q1 = self.T_0_1 @ ht.d_rot_z(joints[0]) @ T_1_2 @ T_2_3 @ self.T_3_E
        dev_T_q2 = T_0_1 @ self.T_1_2 @ ht.d_rot_x(joints[1]) @ T_2_3 @ self.T_3_E
        dev_T_q3 = T_0_1 @ T_1_2 @ self.T_2_3 @ ht.d_rot_x(joints[2]) @ self.T_3_E
        
        J = np.zeros((3, 3), dtype=np.double)
        J[:, 0] = dev_T_q1.T
        J[:, 1] = dev_T_q2.T
        J[:, 2] = dev_T_q3.T
        return J


class LegRobot(RobotModel):
    """
    Model of a simple robot leg with 4 degrees of freedom
    """
    def __init__(self):
        # TODO: implement
        pass

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
        return np.array([[-1, 1], [-1, 1], [-1, 1], [-1, 1]])

    def getBaseFromToolTransform(self, joints):
        # TODO: implement
        raise NotImplementedError()

    def extractMGD(self, T):
        """
        T : np.arraynd shape(4,4)
           An homogeneous transformation matrix
        """
        # TODO: implement
        raise NotImplementedError()

    def computeMGD(self, joints):
        # TODO: implement
        raise NotImplementedError()

    def analyticalMGI(self, target):
        # TODO: implement
        raise NotImplementedError()

    def computeJacobian(self, joints):
        # TODO: implement
        raise NotImplementedError()


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
