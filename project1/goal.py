import numpy as np
import jac


class Goal(object):
    """
    A trivial goal that is always satisfied.
    """

    def __init__(self):
        pass

    def is_satisfied(self, state):
        """
        Determine if the query state satisfies this goal or not.
        """
        return True


class RelocateGoal(Goal):
    """
    The goal for relocating tasks.
    (i.e., pushing the target object into a circular goal region.)
    """

    def __init__(self, x_g=0.2, y_g=-0.2, r_g=0.1):
        """
        args: x_g: The x-coordinate of the center of the goal region.
              y_g: The y-coordinate of the center of the goal region.
              r_g: The radius of the goal region.
        """
        super(RelocateGoal, self).__init__()
        self.x_g, self.y_g, self.r_g = x_g, y_g, r_g

    def is_satisfied(self, state):
        """
        Check if the state satisfies the RelocateGoal or not.
        args: state: The state to check.
                     Type: dict, {"stateID", int, "stateVec", numpy.ndarray}
        """
        stateVec = state["stateVec"]
        x_tgt, y_tgt = stateVec[7], stateVec[8] # position of the target object
        if np.linalg.norm([x_tgt - self.x_g, y_tgt - self.y_g]) < self.r_g:
            return True
        else:
            return False


class GraspGoal(Goal):
    """
    The goal for grasping tasks.
    (i.e., approaching the end-effector to a pose that can grasp the target object.)
    """

    def __init__(self):
        super(GraspGoal, self).__init__()
        self.jac_solver = jac.JacSolver() # the jacobian solver

    def is_satisfied(self, state):
        """
        Check if the state satisfies the GraspGoal or not.
        args: state: The state to check.
                     Type: dict, {"stateID", int, "stateVec", numpy.ndarray}
        returns: True or False.
        """
        ########## TODO ##########
        # Task 3
        # Test with:        python main.py --task 3
        #python -u main.py --task 3 2>&1 | awk '!/Intel MKL WARNING/ && !/Intel oneAPI Math Kernel Library 2025.0/'

        stateVec = state["stateVec"]

        # Robot joint angles and target cube pose in SE(2).
        q = stateVec[:7]
        x_tgt, y_tgt, theta_tgt = stateVec[7:10]

        # Forward kinematics returns the end-effector pose in the robot base frame.
        pos_ee_base, quat_ee = self.jac_solver.forward_kinematics(q)

        # Convert the end-effector position to the world frame.
        x_ee = pos_ee_base[0] - 0.4
        y_ee = pos_ee_base[1] - 0.2

        # The robot base has zero world rotation, so the end-effector yaw is unchanged.
        yaw_ee = self.jac_solver.bullet_client.getEulerFromQuaternion(quat_ee)[2]

        # Relative 2D position of the cube center w.r.t. the end-effector center.
        rel = np.array([x_tgt - x_ee, y_tgt - y_ee])

        # Unit vector along the end-effector plane in the x-y view.
        t = np.array([np.cos(yaw_ee), np.sin(yaw_ee)])

        # Unit normal to the end-effector plane.
        n = np.array([-np.sin(yaw_ee), np.cos(yaw_ee)])

        # d1: perpendicular distance from the cube center to the gripper plane.
        d1 = np.abs(np.dot(rel, n))

        # d2: distance along the gripper opening direction; cube must lie between fingers.
        d2 = np.abs(np.dot(rel, t))

        def angle_diff(a, b):
            # Smallest absolute wrapped angle difference in [0, pi].
            d = (a - b + np.pi) % (2 * np.pi) - np.pi
            return np.abs(d)

        # The cube can be grasped from either pair of parallel sides.
        gamma = min(
            angle_diff(yaw_ee, theta_tgt),
            angle_diff(yaw_ee, theta_tgt + np.pi / 2.0),
            angle_diff(yaw_ee, theta_tgt + np.pi),
            angle_diff(yaw_ee, theta_tgt + 3.0 * np.pi / 2.0),
        )

        return (d1 < 0.01) and (d2 < 0.02) and (gamma < 0.2)

        ##########################
        