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
                          is_abs=False):
    """
    Returns the angle between the facing vector of the robot and the target position.
    Explanation can be found here https://math.stackexchange.com/a/14180.
    :param robot_node: The robot Webots node
    :type robot_node: controller.node.Node
    :param target_node: The target Webots node
    :type target_node: controller.node.Node
    :param is_abs: Whether to return the absolute value of the angle.
    :type is_abs: bool
    :return: The angle between the facing vector of the robot and the target position
    :rtype: float, [-π, π]
    """
    # The sign of the z-axis is needed to flip the rotation sign, because Webots seems to randomly
    # switch between positive and negative z-axis as the robot rotates.
    robot_rotation = robot_node.getField('rotation').getSFRotation()
    robot_angle = robot_rotation[3] if robot_rotation[2] > 0 else -robot_rotation[3]

    robot_coordinates = robot_node.getField('translation').getSFVec3f()
    target_coordinate = target_node.getField('translation').getSFVec3f()

    x_r = (target_coordinate[0] - robot_coordinates[0])
    y_r = (target_coordinate[1] - robot_coordinates[1])

    angle_diff = math.remainder(math.atan2(y_r, x_r) - robot_angle, math.tau)
    return abs(angle_diff) if is_abs else angle_diff
