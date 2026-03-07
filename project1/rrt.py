import numpy as np
import time
import samplers
import utils


class Tree(object):
    """
    The tree class for Kinodynamic RRT.
    """

    def __init__(self, pdef):
        """
        args: pdef: The problem definition instance.
                    Type: pdef.ProblemDefinition
        """
        self.pdef = pdef
        self.dim_state = self.pdef.get_state_dimension()
        self.nodes = []
        self.stateVecs = np.empty(shape=(0, self.dim_state))

    def add(self, node):
        """
        Add a new node into the tree.
        """
        self.nodes.append(node)
        self.stateVecs = np.vstack((self.stateVecs, node.state['stateVec']))
        assert len(self.nodes) == self.stateVecs.shape[0]

    def nearest(self, rstateVec):
        """
        Find the node in the tree whose state vector is nearest to "rstateVec".
        """
        dists = self.pdef.distance_func(rstateVec, self.stateVecs)
        nnode_id = np.argmin(dists)
        return self.nodes[nnode_id]

    def size(self):
        """
        Query the size (number of nodes) of the tree. 
        """
        return len(self.nodes)
    

class Node(object):
    """
    The node class for Tree.
    """

    def __init__(self, state):
        """
        args: state: The state associated with this node.
                     Type: dict, {"stateID": int, 
                                  "stateVec": numpy.ndarray}
        """
        self.state = state
        self.control = None # the control asscoiated with this node
        self.parent = None # the parent node of this node

    def get_control(self):
        return self.control

    def get_parent(self):
        return self.parent

    def set_control(self, control):
        self.control = control

    def set_parent(self, pnode):
        self.parent = pnode
    

class KinodynamicRRT(object):
    """
    The Kinodynamic Rapidly-exploring Random Tree (RRT) motion planner.
    """

    def __init__(self, pdef):
        """
        args: pdef: The problem definition instance for the problem to be solved.
                    Type: pdef.ProblemDefinition
        """
        self.pdef = pdef
        self.tree = Tree(pdef)
        self.state_sampler = samplers.StateSampler(self.pdef) # state sampler
        self.control_sampler = samplers.ControlSampler(self.pdef) # control sampler


    def solve(self, time_budget):
        """
            The main algorithm of Kinodynamic RRT.
            args:  time_budget: The planning time budget (in seconds).
            returns: is_solved: True or False.
                          plan: The motion plan found by the planner,
                                represented by a sequence of tree nodes.
                                Type: a list of rrt.Node
        """
        ########## TODO ##########
        # Task 2: Part 2
        # Test with:        python main.py --task 2
        # python -u main.py --task 2 2>&1 | awk '!/Intel MKL WARNING/ && !/Intel oneAPI Math Kernel Library 2025.0/'

        solved = False
        plan = None
        time_start = time.time()

        def get_ee_xy(state):
            # Recover the end-effector position in world x-y from the robot joints.
            q = state["stateVec"][:7]
            pos_ee_base, _ = self.pdef.panda_sim.jac_solver.forward_kinematics(q)
            return np.array([pos_ee_base[0] - 0.4, pos_ee_base[1] - 0.2])

        def get_ee_yaw(state):
            q = state["stateVec"][:7]
            _, quat_ee = self.pdef.panda_sim.jac_solver.forward_kinematics(q)
            return self.pdef.panda_sim.bullet_client.getEulerFromQuaternion(quat_ee)[2]

        def get_obj_xy(state):
            # The yellow target cube is the first object stored in the state vector.
            return np.array(state["stateVec"][7:9])

        def get_obj_yaw(state):
            return state["stateVec"][9]

        def get_goal_xy():
            return np.array([self.pdef.goal.x_g, self.pdef.goal.y_g])

        def angle_diff(a, b):
            d = (a - b + np.pi) % (2 * np.pi) - np.pi
            return np.abs(d)

        # Task 2 uses a circular relocation goal; Task 3 uses a grasp goal.
        is_relocate = hasattr(self.pdef.goal, "x_g") and hasattr(self.pdef.goal, "y_g")
        if is_relocate:
            goal_xy = get_goal_xy()

        def guide_cost(state, mode):
            ee_xy = get_ee_xy(state)
            obj_xy = get_obj_xy(state)

            if is_relocate:
                to_goal = goal_xy - obj_xy
                dist_obj_goal = np.linalg.norm(to_goal)

                if dist_obj_goal < 1e-8:
                    push_dir = np.array([1.0, 0.0])
                else:
                    push_dir = to_goal / dist_obj_goal

                # Approach a point slightly behind the target so the robot can push it forward.
                pre_push_xy = obj_xy - 0.06 * push_dir

                if mode == "approach":
                    return np.linalg.norm(ee_xy - pre_push_xy) + 0.25 * dist_obj_goal
                else:
                    return 2.0 * dist_obj_goal + 0.4 * np.linalg.norm(ee_xy - pre_push_xy)

            else:
                ee_yaw = get_ee_yaw(state)
                obj_yaw = get_obj_yaw(state)

                gamma = min(
                    angle_diff(ee_yaw, obj_yaw),
                    angle_diff(ee_yaw, obj_yaw + np.pi / 2.0),
                    angle_diff(ee_yaw, obj_yaw + np.pi),
                    angle_diff(ee_yaw, obj_yaw + 3.0 * np.pi / 2.0),
                )

                rel = obj_xy - ee_xy
                t = np.array([np.cos(ee_yaw), np.sin(ee_yaw)])
                n = np.array([-np.sin(ee_yaw), np.cos(ee_yaw)])

                d1 = np.abs(np.dot(rel, n))
                d2 = np.abs(np.dot(rel, t))

                return 2.0 * d1 + 1.0 * d2 + 0.5 * gamma

        def nearest_guided(mode):
            # Pick the existing tree node that is best for the current subtask.
            best_node = None
            best_cost = np.inf
            for node in self.tree.nodes:
                c = guide_cost(node.state, mode)
                if c < best_cost:
                    best_cost = c
                    best_node = node
            return best_node

        def sample_control_guided(node, mode, k):
            # Sample several controls and keep the outcome with the lowest guide cost.
            # If any candidate already reaches the goal, accept it immediately.
            best_ctrl = None
            best_state = None
            best_cost = np.inf

            for _ in range(k):
                ctrl = np.random.uniform(
                    self.control_sampler.low,
                    self.control_sampler.high,
                    self.control_sampler.dim
                )
                pstate, valid = self.pdef.propagate(node.state, ctrl)
                if (not valid) or (not self.pdef.is_state_valid(pstate)):
                    continue

                if self.pdef.goal.is_satisfied(pstate):
                    self.pdef.panda_sim.restore_state(node.state)
                    return ctrl, pstate

                c = guide_cost(pstate, mode)
                if c < best_cost:
                    best_cost = c
                    best_ctrl = ctrl
                    best_state = pstate

            # Restore the simulator so planning visuals do not stay on a rejected sample.
            self.pdef.panda_sim.restore_state(node.state)
            return best_ctrl, best_state

        start_state = self.pdef.get_start_state()
        root = Node(start_state)

        self.tree = Tree(self.pdef)
        self.tree.add(root)

        while time.time() - time_start < time_budget:
            # Keep a small amount of uniform exploration to avoid overcommitting too early.
            if np.random.rand() < 0.15:
                state_vec = self.state_sampler.sample()
                nearest = self.tree.nearest(state_vec)
                ctrl, outcome = self.control_sampler.sample_to(nearest, state_vec, k=10)
            else:
                # Task 2 uses approach/push; task 3 uses grasp guidance.
                if is_relocate:
                    mode = "approach" if np.random.rand() < 0.7 else "push"
                else:
                    mode = "grasp"

                nearest = nearest_guided(mode)
                ctrl, outcome = sample_control_guided(nearest, mode, k=30)

            if ctrl is None or outcome is None:
                continue

            new_node = Node(outcome)
            new_node.set_control(ctrl)
            new_node.set_parent(nearest)
            self.tree.add(new_node)

            if self.pdef.goal.is_satisfied(new_node.state):
                # Recover the plan by following parent pointers back to the root.
                plan = []
                node = new_node
                while node is not None:
                    plan.append(node)
                    node = node.get_parent()
                plan.reverse()
                return True, plan

        return solved, plan



'''
            solved = False
            plan = None
            time_start = time.time()


            def get_ee_xy(state):
                # Recover the end-effector position in world x-y from the robot joints.
                q = state["stateVec"][:7]
                pos_ee_base, _ = self.pdef.panda_sim.jac_solver.forward_kinematics(q)
                return np.array([pos_ee_base[0] - 0.4, pos_ee_base[1] - 0.2])

            def get_obj_xy(state):
                # The yellow target cube is the first object stored in the state vector.
                return np.array(state["stateVec"][7:9])

            def get_goal_xy():
                return np.array([self.pdef.goal.x_g, self.pdef.goal.y_g])


            # Task 2 uses a circular relocation goal; Task 3 uses a grasp goal.
            goal_xy = get_goal_xy()
            is_relocate = hasattr(self.pdef.goal, "x_g") and hasattr(self.pdef.goal, "y_g")
            if is_relocate:
                goal_xy = np.array([self.pdef.goal.x_g, self.pdef.goal.y_g])


            def guide_cost(state, mode):
                ee_xy = get_ee_xy(state)
                obj_xy = get_obj_xy(state)

                to_goal = goal_xy - obj_xy
                dist_obj_goal = np.linalg.norm(to_goal)

                if dist_obj_goal < 1e-8:
                    push_dir = np.array([1.0, 0.0])
                else:
                    push_dir = to_goal / dist_obj_goal

                # Approach a point slightly behind the target so the robot can push it forward.
                pre_push_xy = obj_xy - 0.06 * push_dir

                if mode == "approach":
                    return np.linalg.norm(ee_xy - pre_push_xy) + 0.25 * dist_obj_goal
                else:
                    return 2.0 * dist_obj_goal + 0.4 * np.linalg.norm(ee_xy - pre_push_xy)

            def nearest_guided(mode):
                # Pick the existing tree node that is best for the current subtask.
                best_node = None
                best_cost = np.inf
                for node in self.tree.nodes:
                    c = guide_cost(node.state, mode)
                    if c < best_cost:
                        best_cost = c
                        best_node = node
                return best_node

            def sample_control_guided(node, mode, k):
                # Sample several controls and keep the outcome with the lowest guide cost.
                # If any candidate already reaches the goal, accept it immediately.
                best_ctrl = None
                best_state = None
                best_cost = np.inf

                for _ in range(k):
                    ctrl = np.random.uniform(
                        self.control_sampler.low,
                        self.control_sampler.high,
                        self.control_sampler.dim
                    )
                    pstate, valid = self.pdef.propagate(node.state, ctrl)
                    if (not valid) or (not self.pdef.is_state_valid(pstate)):
                        continue

                    if self.pdef.goal.is_satisfied(pstate):
                        self.pdef.panda_sim.restore_state(node.state)
                        return ctrl, pstate

                    c = guide_cost(pstate, mode)
                    if c < best_cost:
                        best_cost = c
                        best_ctrl = ctrl
                        best_state = pstate

                # Restore the simulator so planning visuals do not stay on a rejected sample.
                self.pdef.panda_sim.restore_state(node.state)
                return best_ctrl, best_state

            start_state = self.pdef.get_start_state()
            root = Node(start_state)

            self.tree = Tree(self.pdef)
            self.tree.add(root)

            while time.time() - time_start < time_budget:
                # Keep a small amount of uniform exploration to avoid overcommitting too early.
                if np.random.rand() < 0.15:
                    state_vec = self.state_sampler.sample()
                    nearest = self.tree.nearest(state_vec)
                    ctrl, outcome = self.control_sampler.sample_to(nearest, state_vec, k=10)
                else:
                    # Most expansions are guided either to approach the target or push it to the goal.
                    mode = "approach" if np.random.rand() < 0.7 else "push"
                    nearest = nearest_guided(mode)
                    ctrl, outcome = sample_control_guided(nearest, mode, k=30)

                if ctrl is None or outcome is None:
                    continue

                new_node = Node(outcome)
                new_node.set_control(ctrl)
                new_node.set_parent(nearest)
                self.tree.add(new_node)

                if self.pdef.goal.is_satisfied(new_node.state):
                    # Recover the plan by following parent pointers back to the root.
                    plan = []
                    node = new_node
                    while node is not None:
                        plan.append(node)
                        node = node.get_parent()
                    plan.reverse()
                    return True, plan

            ##########################
            return solved, plan
'''