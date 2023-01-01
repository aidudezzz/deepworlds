import math

import numpy as np


def normalize_to_range(value, min, max, new_min, new_max):
    value = float(value)
    min = float(min)
    max = float(max)
    new_min = float(new_min)
    new_max = float(new_max)
    return (new_max - new_min) / (max - min) * (value - max) + new_max


def get_distance_from_target(robot_node, target_node):
    robot_coordinates = robot_node.getField('translation').getSFVec3f()
    target_coordinate = target_node.getField('translation').getSFVec3f()

    dx = robot_coordinates[0] - target_coordinate[0]
    dy = robot_coordinates[1] - target_coordinate[1]
    distance_from_target = math.sqrt(dx * dx + dy * dy)
    return distance_from_target


def get_angle_from_target(robot_node,
                          target_node,
                          is_true_angle=False,
                          is_abs=False):

    robot_angle = robot_node.getField('rotation').getSFRotation()[3]

    robot_coordinates = robot_node.getField('translation').getSFVec3f()
    target_coordinate = target_node.getField('translation').getSFVec3f()

    x_r = (target_coordinate[0] - robot_coordinates[0])
    y_r = (target_coordinate[1] - robot_coordinates[1])

    y_r = -y_r

    # robotWorldAngle = math.atan2(robot_coordinates[1], robot_coordinates[0])

    if robot_angle < 0.0: robot_angle += 2 * np.pi

    x_f = x_r * math.sin(robot_angle) - \
          y_r * math.cos(robot_angle)

    y_f = x_r * math.cos(robot_angle) + \
          y_r * math.sin(robot_angle)

    # print("x_f: {} , y_f: {}".format(x_f, y_f) )
    if is_true_angle:
        x_f = -x_f
    angle_diff = math.atan2(y_f, x_f)

    if is_abs:
        angle_diff = abs(angle_diff)

    return angle_diff
