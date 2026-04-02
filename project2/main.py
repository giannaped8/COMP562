import os
import numpy as np
import argparse
import trimesh
import alg
import utils


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, choices=["bunny", "cow", "duck"], default="bunny")
    parser.add_argument("--task", type=int, choices=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    # load the mesh file and visualize
    mesh_path = "./meshes/%s.stl" % args.mesh
    mesh = trimesh.load(mesh_path)
    print("The mesh file was loaded by the path: %s" % os.path.abspath(mesh_path))
    print("Information of the mesh:")
    print("  Number of Faces: %d" % len(mesh.faces))
    print("  Number of Vertices: %d" % len(mesh.vertices))
    print("  The Center of Mass:", mesh.center_mass)
    print("\n")
    #_ = utils.plot_mesh(mesh)

    # Task 1: Primitive Wrenches Calculation 
    if args.task == 1:
        #grasp = np.random.choice(np.arange(len(mesh.faces)), size=3, replace=False)
        grasp = [0, 249, 484]
        print("The grasp:", grasp)
        con_pts = utils.get_centroid_of_triangles(mesh, grasp)
        print("The contact points of the given grasp:")
        print(con_pts)
        utils.plot_grasp(mesh, grasp)
        W = alg.primitive_wrenches(mesh, grasp)
        utils.check_wrenches(mesh, grasp, W)

    # Task 2: Grasp Quality Evaluation 
    if args.task == 2:
        grasp = [0, 249, 484]
        print("The grasp:", grasp)
        con_pts = utils.get_centroid_of_triangles(mesh, grasp)
        print("The contact points of the given grasp:")
        print(con_pts)
        utils.plot_grasp(mesh, grasp)
        Q = alg.eval_Q(mesh, grasp)
        print("The quality of the given grasp: %f \n" % Q)
    
    # Task 3: Sample a Stable Grasp
    elif args.task == 3:
        grasp, Q = alg.sample_stable_grasp(mesh, thresh=0.01)
        print("The grasp:", grasp)
        con_pts = utils.get_centroid_of_triangles(mesh, grasp)
        print("The contact points of the stable grasp:")
        print(con_pts)
        print("The quality of the given grasp: %f \n" % Q)
        utils.plot_grasp(mesh, grasp)

    # Task 4: Optimize the Given Grasp
    elif args.task == 4:
        grasp = [80, 165, 444]
        traj = alg.optimize_grasp(mesh, grasp)
        print("The quality of the given initial grasp: %f" % alg.eval_Q(mesh, traj[0]))
        print("The quality of the optimized grasp: %f" % alg.eval_Q(mesh, traj[-1]))
        utils.plot_traj(mesh, traj)


    # Task 5: Sample and Optimize a Grasp under Reachability Constraint
    # TESTING:       python main.py --task 5
    elif args.task == 5:
        ##########################
        #traj = alg.optimize_reachable_grasp(mesh, r=1.0)
        #print("The quality of the given initial grasp: %f" % alg.eval_Q(mesh, traj[0]))
        #print("The quality of the optimized grasp: %f" % alg.eval_Q(mesh, traj[-1]))
        #utils.plot_traj(mesh, traj)
        ##########################

        # Test the reachability-constrained optimizer for both required r values.
        r_vals = [0.5, 1]
        for r_val in r_vals:
            print(f"Reachability Measure: r = {r_val}")

            # Track the best optimized trajectory among several random trials.
            Q_max = -np.inf
            best_traj = []

            for i in range(5):
                traj = alg.optimize_reachable_grasp(mesh, r=r_val)

                # Evaluate the final grasp in this trajectory.
                Q_opt = alg.eval_Q(mesh, traj[-1])
                grasp_opt = [int(x) for x in traj[-1]]

                # Keep the trajectory with the highest final quality.
                if Q_opt > Q_max:
                    Q_max = Q_opt
                    best_traj = traj

                # Print the sampled initial grasp and the optimized result for this trial.
                print(f"Trial {i + 1}")
                print("  Initial grasp:", traj[0])
                print("  Initial Q:", alg.eval_Q(mesh, traj[0]))
                print("  Optimized grasp:", grasp_opt)
                print("  Optimized Q:", Q_opt)
                print("  Trajectory length:", len(traj))
                print()

            # Report and visualize the best trajectory found for this reachability value.
            print("The maximum optimized grasp quality : %f" % Q_max)
            utils.plot_traj(mesh, best_traj)
        ##########################


