import time
import argparse
import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc
import numpy as np
import sim
from pdef import Bounds, ProblemDefinition
from goal import RelocateGoal, GraspGoal
import rrt
import utils
import opt


def setup_pdef(panda_sim):
  pdef = ProblemDefinition(panda_sim)
  dim_state = pdef.get_state_dimension()
  dim_ctrl = pdef.get_control_dimension()

  # define bounds for state and control space
  bounds_state = Bounds(dim_state)
  for j in range(sim.pandaNumDofs):
    bounds_state.set_bounds(j, sim.pandaJointRange[j, 0], sim.pandaJointRange[j, 1])
  for j in range(sim.pandaNumDofs, dim_state):
    if ((j - sim.pandaNumDofs) % 3 == 2):
      bounds_state.set_bounds(j, -np.pi, np.pi)
    else:
      bounds_state.set_bounds(j, -0.3, 0.3)
  pdef.set_state_bounds(bounds_state)

  bounds_ctrl = Bounds(dim_ctrl)
  bounds_ctrl.set_bounds(0, -0.2, 0.2)
  bounds_ctrl.set_bounds(1, -0.2, 0.2)
  bounds_ctrl.set_bounds(2, -1.0, 1.0)
  bounds_ctrl.set_bounds(3, 0.4, 0.6)
  pdef.set_control_bounds(bounds_ctrl)
  return pdef


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--task", type=int, choices=[1, 2, 3, 4])
  args = parser.parse_args()

  # set up the simulation
  pgui = utils.setup_bullet_client(p.GUI)
  panda_sim = sim.PandaSim(pgui)

  # Task 1: Move the Robot with Jacobian-based Projection
  if args.task == 1:
    pdef = setup_pdef(panda_sim)

    ctrls = [[0.02, 0, 0.2, 10],
             [0, 0.02, 0.2, 10],
             [-0.02, 0, -0.2, 10],
             [0, -0.02, -0.2, 10]]
    errs = []
    for _ in range(10):
      for ctrl in ctrls:
        wpts_ref = utils.extract_reference_waypoints(panda_sim, ctrl)
        wpts, _ = panda_sim.execute(ctrl)
        err_pos = np.mean(np.linalg.norm(wpts[:, 0:2] - wpts_ref[:, 0:2], axis=1))
        err_orn = np.mean(np.abs(wpts[:, 2] - wpts_ref[:, 2]))
        print("The average Cartesian error for executing the last control:")
        print("Position: %f meters\t Orientation: %f rads" % (err_pos, err_orn))
        errs.append([err_pos, err_orn])
    errs = np.array(errs)
    print("\nThe average Cartesian error for the entire exeution:")
    print("Position: %f meters\t Orientation: %f rads" % (errs[:, 0].mean(), errs[:, 1].mean()))


  else:
    # configure the simulation and the problem
    utils.setup_env(panda_sim)
    pdef = setup_pdef(panda_sim)

    # Task 2: Kinodynamic RRT Planning for Relocating
    if args.task == 2:
      goal = RelocateGoal()
      pdef.set_goal(goal)

      planner = rrt.KinodynamicRRT(pdef)
      time_st = time.time()
      solved, plan = planner.solve(120.0)
      print("Running time of rrt.KinodynamicRRT.solve(): %f secs" % (time.time() - time_st))

      if solved:
        print("The Plan has been Found:")
        panda_sim.restore_state(pdef.get_start_state())
        for _ in range(2):
          panda_sim.step()
        panda_sim.restore_state(pdef.get_start_state())
        utils.execute_plan(panda_sim, plan)
        while True:
          pass

    # Task 3: Kinodynamic RRT Planning for Grasping
    elif args.task == 3:
      goal = GraspGoal()
      pdef.set_goal(goal)

      planner = rrt.KinodynamicRRT(pdef)
      time_st = time.time()
      solved, plan = planner.solve(120.0)
      print("Running time of rrt.KinodynamicRRT.solve(): %f secs" % (time.time() - time_st))

      if solved:
        print("The Plan has been Found:")
        panda_sim.restore_state(pdef.get_start_state())
        for _ in range(2):
          panda_sim.step()
        panda_sim.restore_state(pdef.get_start_state())
        utils.execute_plan(panda_sim, plan)
        panda_sim.grasp()
        while True:
          pass





    # Task 4: Trajectory Optimization
    elif args.task == 4:
      ########## TODO ##########
      # Task 4
      # Run to Evaluate: python main.py --task 4

      goal = RelocateGoal()
      pdef.set_goal(goal)

      # Best run from experiments for screenshots
      seed = 3
      num_iterations = 300
      task_name = "Relocation"

      print("\n" + "=" * 70)
      print(f"Starting Task 4 visualization run: seed={seed}, iterations={num_iterations}")

      # Camera angle for screenshots
      pgui.resetDebugVisualizerCamera(
        cameraDistance=0.9,
        cameraYaw=120,
        cameraPitch=-45,
        cameraTargetPosition=[-0.1, -0.1, 0.0]
      )

      # Rebuild planner
      planner = rrt.KinodynamicRRT(pdef)

      # Reset to start state
      panda_sim.restore_state(pdef.get_start_state())
      for _ in range(2):
        panda_sim.step()

      time_st = time.time()
      solved, plan, opt_plan, stats = opt.plan_and_optimize(
        pdef,
        planner,
        time_budget=120.0,
        num_iterations=num_iterations,
        seed=seed,
        verbose=False,
      )
      runtime = time.time() - time_st

      print("Running time of planning + optimization: %f secs" % runtime)
      print("Solved:", solved)
      print("Task:", task_name)
      print("Seed:", seed)
      print("Optimization iterations:", num_iterations)
      print("Runtime (s):", runtime)

      if solved:
        initial_controls = stats["initial_num_controls"]
        optimized_controls = stats["optimized_num_controls"]
        initial_cost = stats["initial_cost"]
        optimized_cost = stats["optimized_cost"]

        if initial_cost != 0:
          improvement = 100.0 * (initial_cost - optimized_cost) / initial_cost
        else:
          improvement = 0.0

        print("Initial controls:", initial_controls)
        print("Optimized controls:", optimized_controls)
        print("Initial cost:", initial_cost)
        print("Optimized cost:", optimized_cost)
        print("Cost improvement (%):", improvement)

        # ------------------------------------------------------------
        # SHOW INITIAL TRAJECTORY FIRST
        # ------------------------------------------------------------
        print("\nShowing INITIAL trajectory...")
        panda_sim.restore_state(pdef.get_start_state())
        for _ in range(2):
          panda_sim.step()
        panda_sim.restore_state(pdef.get_start_state())
        utils.execute_plan(panda_sim, plan)

        print("Initial trajectory finished. Take screenshot now if needed.")
        time.sleep(20.0)

        # ------------------------------------------------------------
        # RESET AND SHOW OPTIMIZED TRAJECTORY
        # ------------------------------------------------------------
        print("\nShowing OPTIMIZED trajectory...")
        panda_sim.restore_state(pdef.get_start_state())
        for _ in range(2):
          panda_sim.step()
        panda_sim.restore_state(pdef.get_start_state())
        utils.execute_plan(panda_sim, opt_plan)

        print("Optimized trajectory finished. Leave window open for screenshot.")
        while True:
          pass

      else:
        print("No solution found.")
      ##########################



'''    
    #CODE USED TO PRODUCE OUTPUTS IN FILES
    # Task 4: Trajectory Optimization
    elif args.task == 4:
      ########## TODO ##########
      # Task 4
      # Run to Evaluate: python main.py --task 4

      #Loop for Producing Experiment Results:
      # Choose task
      goal = RelocateGoal()
      # goal = GraspGoal()
      pdef.set_goal(goal)

      seeds = [0, 1, 2, 3, 4, 5]
      iteration_list = [0, 100, 300]

      task_name = "Relocation" if isinstance(goal, RelocateGoal) else "Grasping"
      csv_path = f"task4_{task_name.lower()}_results.csv"

      # Create CSV with header if it does not already exist
      file_exists = os.path.exists(csv_path)
      with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
          writer.writerow([
            "task",
            "seed",
            "iterations",
            "solved",
            "runtime_s",
            "initial_controls",
            "optimized_controls",
            "initial_cost",
            "optimized_cost",
            "cost_improvement_percent"
          ])

        for seed in seeds:
          for num_iterations in iteration_list:
            print("\n" + "=" * 70)
            print(f"Starting experiment: seed={seed}, iterations={num_iterations}")

            # Rebuild planner each run
            planner = rrt.KinodynamicRRT(pdef)

            # Reset to the same start state before each run
            panda_sim.restore_state(pdef.get_start_state())
            for _ in range(2):
              panda_sim.step()

            time_st = time.time()
            solved, plan, opt_plan, stats = opt.plan_and_optimize(
              pdef,
              planner,
              time_budget=120.0,
              num_iterations=num_iterations,
              seed=seed,
              verbose=False,
            )
            runtime = time.time() - time_st

            print("Running time of planning + optimization: %f secs" % runtime)
            print("Solved:", solved)
            print("Task:", task_name)
            print("Seed:", seed)
            print("Optimization iterations:", num_iterations)
            print("Runtime (s):", runtime)

            if solved:
              initial_controls = stats["initial_num_controls"]
              optimized_controls = stats["optimized_num_controls"]
              initial_cost = stats["initial_cost"]
              optimized_cost = stats["optimized_cost"]

              if initial_cost != 0:
                improvement = 100.0 * (initial_cost - optimized_cost) / initial_cost
              else:
                improvement = 0.0

              print("Initial controls:", initial_controls)
              print("Optimized controls:", optimized_controls)
              print("Initial cost:", initial_cost)
              print("Optimized cost:", optimized_cost)
              print("Cost improvement (%):", improvement)

              writer.writerow([
                task_name,
                seed,
                num_iterations,
                True,
                runtime,
                initial_controls,
                optimized_controls,
                initial_cost,
                optimized_cost,
                improvement
              ])
            else:
              print("Planner failed for this experiment.")

              writer.writerow([
                task_name,
                seed,
                num_iterations,
                False,
                runtime,
                "",
                "",
                "",
                "",
                ""
              ])

      print("\nSaved results to:", csv_path)

      ####################

'''