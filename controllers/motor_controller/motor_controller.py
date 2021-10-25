"""
motor_controller controller

A simple controller to set the target of a robot
"""

import csv
import os
from controller import Robot
import numpy as np

import robots

ALLOWED_MODES = ['computeMGD', 'analyticalMGI', 'jacobianInverse', 'jacobianTransposed']
DEFAULT_CONTROL_MODE = 'analyticalMGI'  # The mode used for the simulation see ALLOWED_MODES
# DEFAULT_CONTROL_MODE = 'jacobianInverse'  # The mode used for the simulation see ALLOWED_MODES
# DEFAULT_CONTROL_MODE = 'jacobianTransposed'  # The mode used for the simulation see ALLOWED_MODES

#DEFAULT_ROBOT = 'RobotRT'
DEFAULT_ROBOT = 'RobotRRR'
#DEFAULT_ROBOT = 'LegRobot'

# Setting parameters from environment variables
control_mode = os.environ.get('CONTROL_MODE', DEFAULT_CONTROL_MODE)
robot_name = os.environ.get('ROBOT_NAME', DEFAULT_ROBOT)

default_rt_target = [0.2,-0.25]
default_rrr_target = [0.6, 0.3, 1.0]
default_leg_target = [0.0, 0.7, 0.6, -1.0]
robot_targets = {
    'RobotRT': {
        'computeMGD': [1.57, 0.1],
        'analyticalMGI': default_rt_target,
        'jacobianInverse': default_rt_target,
        'jacobianTransposed': default_rt_target
    },
    'RobotRRR': {
        'computeMGD': [1.57, 1.57, 1.57],
        'analyticalMGI': default_rrr_target,
        'jacobianInverse': default_rrr_target,
        'jacobianTransposed': default_rrr_target
    },
    'LegRobot': {
        'computeMGD': [1.57, -1.57, 1.57, -1.57],
        'analyticalMGI': default_leg_target,
        'jacobianInverse': default_leg_target,
        'jacobianTransposed': default_leg_target
    }
}
targets = robot_targets[robot_name][control_mode]

model = robots.getRobotModel(robot_name)
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Initializing motors
motors = []
for name in model.getMotorsNames():
    motors.append(robot.getDevice(name))

# Initializing sensors
sensors = []
for name in model.getSensorsNames():
    sensors.append(robot.getDevice(name))
    sensors[-1].enable(timestep)

with open('robot_data.csv', 'w') as output_file:
    logger = csv.DictWriter(output_file, ["t", "variable", "order", "source", "value"])
    logger.writeheader()

    t = 0.0  # [s]
    while robot.step(timestep) != -1:
        nb_joints = model.getNbJoints()
        q = np.zeros(nb_joints)
        target_op_pos = None
        target_q = targets
        # If using MGI, retrieve op_pos from targets
        if control_mode != "computeMGD":
            target_q = model.computeMGI(q, targets, control_mode)
        # Setting motors and writing joints target + measurements
        for i in range(nb_joints):
            motors[i].setPosition(target_q[i])
            q[i] = sensors[i].getValue()
            joint_name = model.getJointsNames()[i]
            logger.writerow({"t": t, "variable": joint_name, "order": 0, "source": "target", "value": target_q[i]})
            logger.writerow({"t": t, "variable": joint_name, "order": 0, "source": "sensor", "value": q[i]})
        # Measuring and writing operational target
        op_pos = model.computeMGD(q)
        op_dim_names = model.getOperationalDimensionNames()
        for i in range(len(op_dim_names)):
            name = op_dim_names[i]
            logger.writerow({"t": t, "variable": name, "order": 0, "source": "computeMGD", "value": op_pos[i]})
            if control_mode != "computeMGD":
                logger.writerow({"t": t, "variable": name, "order": 0, "source": "target", "value": targets[i]})
        t += timestep / 10**3
