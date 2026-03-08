import numpy as np
import copy
import sim
import goal
import rrt
import utils


########## TODO ##########
def _control_list(plan):
    controls = []
    for node in plan:
        ctrl = node.get_control()
        if ctrl is not None:
            controls.append(np.array(ctrl, dtype=float).copy())
    return controls


def _ee_xy(pdef, state):
    q = state["stateVec"][:sim.pandaNumDofs]
    pos_ee_base, _ = pdef.panda_sim.jac_solver.forward_kinematics(q)
    return np.array([pos_ee_base[0] - 0.4, pos_ee_base[1] - 0.2])


def _non_target_xy(state):
    state_vec = state["stateVec"]
    num_objects = (state_vec.shape[0] - sim.pandaNumDofs) // 3
    pts = []
    for obj_id in range(1, num_objects):
        base = sim.pandaNumDofs + 3 * obj_id
        pts.append(state_vec[base:base + 2])
    if len(pts) == 0:
        return np.empty((0, 2))
    return np.vstack(pts)


def rollout_plan(pdef, controls):
    """
    Roll out a control sequence from the start state and rebuild the corresponding plan.
    returns: solved: True or False
                  plan: rebuilt list of rrt.Node, or None on invalid rollout
    """
    start_state = pdef.get_start_state()
    root = rrt.Node(start_state)
    plan = [root]
    parent = root
    state = start_state

    for ctrl in controls:
        ctrl = np.array(ctrl, dtype=float).copy()
        pstate, valid = pdef.propagate(state, ctrl)
        if (not valid) or (not pdef.is_state_valid(pstate)):
            pdef.panda_sim.restore_state(start_state)
            return False, None

        node = rrt.Node(pstate)
        node.set_parent(parent)
        node.set_control(ctrl)
        plan.append(node)
        parent = node
        state = pstate

    solved = pdef.goal.is_satisfied(state)
    pdef.panda_sim.restore_state(start_state)
    if not solved:
        return False, None
    return True, plan


def trajectory_cost(pdef, plan, verbose=False):
    """
    Score a motion plan by execution time, end-effector path length, and clutter motion.
    Lower is better.
    """
    if plan is None or len(plan) == 0:
        return np.inf

    duration_cost = 0.0
    ee_path_cost = 0.0
    clutter_motion_cost = 0.0

    for i in range(1, len(plan)):
        prev_state = plan[i - 1].state
        curr_state = plan[i].state
        ctrl = plan[i].get_control()

        if ctrl is not None:
            duration_cost += float(ctrl[3])

        ee_prev = _ee_xy(pdef, prev_state)
        ee_curr = _ee_xy(pdef, curr_state)
        ee_path_cost += np.linalg.norm(ee_curr - ee_prev)

        clutter_prev = _non_target_xy(prev_state)
        clutter_curr = _non_target_xy(curr_state)
        if clutter_prev.shape[0] > 0:
            clutter_motion_cost += np.sum(np.linalg.norm(clutter_curr - clutter_prev, axis=1))

    total_cost = duration_cost + 2.0 * ee_path_cost + 5.0 * clutter_motion_cost

    if verbose:
        print("Trajectory cost breakdown:")
        print("  total control duration =", duration_cost)
        print("  end-effector path length =", ee_path_cost)
        print("  non-target object motion =", clutter_motion_cost)
        print("  total weighted cost =", total_cost)

    return total_cost


def optimize_plan(pdef, plan, num_iterations=250, seed=0, verbose=False):
    """
    Stochastically optimize a plan by perturbing, shortening, or removing controls.
    returns: optimized_plan, stats
    """
    rng = np.random.default_rng(seed)
    best_controls = _control_list(plan)
    best_plan = copy.deepcopy(plan)
    best_cost = trajectory_cost(pdef, best_plan)

    if verbose:
        print("Starting trajectory optimization...")
        print("Initial number of controls:", len(best_controls))
        trajectory_cost(pdef, best_plan, verbose=True)

    low = np.array([pdef.bounds_ctrl.low[0], pdef.bounds_ctrl.low[1], pdef.bounds_ctrl.low[2], 0.05])
    high = np.array([pdef.bounds_ctrl.high[0], pdef.bounds_ctrl.high[1], pdef.bounds_ctrl.high[2], 0.60])

    for it in range(num_iterations):
        if len(best_controls) == 0:
            break

        candidate_controls = [ctrl.copy() for ctrl in best_controls]
        move = rng.choice(["drop", "shrink", "perturb"], p=[0.25, 0.30, 0.45])

        if move == "drop" and len(candidate_controls) > 1:
            drop_idx = int(rng.integers(len(candidate_controls)))
            candidate_controls.pop(drop_idx)
            if verbose:
                print("Iteration", it, ": trying to remove control", drop_idx)
        else:
            idx = int(rng.integers(len(candidate_controls)))

            if move == "shrink":
                scale = rng.uniform(0.6, 0.95)
                candidate_controls[idx][3] *= scale
                if verbose:
                    print("Iteration", it, ": shrinking duration of control", idx, "by factor", scale)
            else:
                noise = np.array([
                    rng.normal(0.0, 0.03),
                    rng.normal(0.0, 0.03),
                    rng.normal(0.0, 0.12),
                    rng.normal(0.0, 0.06),
                ])
                candidate_controls[idx] += noise
                if verbose:
                    print("Iteration", it, ": perturbing control", idx, "with noise", noise)

            candidate_controls[idx] = np.clip(candidate_controls[idx], low, high)

        solved, candidate_plan = rollout_plan(pdef, candidate_controls)
        if not solved:
            if verbose:
                print("  candidate rejected: rollout invalid or goal not satisfied")
            continue

        candidate_cost = trajectory_cost(pdef, candidate_plan)
        if verbose:
            print("  candidate cost =", candidate_cost, "best so far =", best_cost)

        if candidate_cost < best_cost:
            if verbose:
                print("  accepted improvement")
            best_cost = candidate_cost
            best_controls = candidate_controls
            best_plan = candidate_plan

    if verbose:
        print("Finished optimization.")
        print("Final number of controls:", len(best_controls))
        trajectory_cost(pdef, best_plan, verbose=True)

    stats = {
        "initial_num_controls": max(len(plan) - 1, 0),
        "optimized_num_controls": max(len(best_plan) - 1, 0),
        "initial_cost": trajectory_cost(pdef, plan),
        "optimized_cost": best_cost,
    }
    return best_plan, stats




def plan_and_optimize(pdef, planner, time_budget=120.0, num_iterations=250, seed=0, verbose=False):
    """
    Solve the planning problem and optimize the returned plan if successful.
    returns: solved, plan, optimized_plan, stats
    """
    solved, plan = planner.solve(time_budget)
    if not solved:
        return False, None, None, None

    optimized_plan, stats = optimize_plan(
        pdef,
        plan,
        num_iterations=num_iterations,
        seed=seed,
        verbose=verbose,
    )
    return True, plan, optimized_plan, stats


##########################
