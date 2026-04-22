import csv
import time
import numpy as np
import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc

import sim
import alg

# RUN: python task3_experiments.py

"""
Run a set of Task 3 Particle Filtering experiments for the report.

This script tests different combinations of:
    1. The number of particles used by the online Particle Filter.
    2. The sigma value used in the Gaussian likelihood function.
    3. The delta value used when perturbing particles after resampling.

Task 3 is different from Task 2 because the observations are generated online.
Instead of using the fixed observations from obvs.npy, each run commands the robot
in simulation until the stick's spherical end touches an obstacle.

Because online observation generation is stochastic, each hyperparameter
combination is run multiple times. The script saves each individual trial to
task3_results.csv and also saves an averaged summary to task3_summary.csv.
"""


# Candidate particle counts to test.
#
# A larger particle count usually gives the filter a better chance of
# representing the true robot base pose, but it also makes each iteration slower.
num_particles_list = [1000, 2500, 5000]


# Candidate sigma values to test.
#
# Sigma controls how sharply the filter penalizes particles whose predicted
# contact location is not close to an obstacle surface.
#
# Smaller sigma:
#     Gives high weight only to particles that explain the observation very well.
#
# Larger sigma:
#     Is more forgiving of noisy online observations, but may make the weights
#     less selective.
sigma_list = [0.05, 0.08, 0.1]


# Candidate delta values to test.
#
# Delta is the standard deviation of the Gaussian noise added to particles after
# resampling. It helps keep the particle set from collapsing too quickly.
#
# Smaller delta:
#     Keeps particles close to the current estimate.
#
# Larger delta:
#     Adds more exploration, but can make the estimate noisier.
delta_list = [0.02, 0.03, 0.05]


# Number of repeated trials for each hyperparameter combination.
#
# Task 3 can vary from run to run because it uses random online exploration, so
# repeated trials make the comparison more reliable.
num_trials = 2


# Ground-truth robot base pose used in main.py.
#
# The online Particle Filter is trying to estimate this pose.
loc_gt = np.array([-0.3, -0.3, 0.9])


# Output files.
#
# task3_results.csv stores every individual trial.
# task3_summary.csv stores averages for each hyperparameter combination.
results_file = "task3_results.csv"
summary_file = "task3_summary.csv"


def pose_error(est, target):
    """
    Compute one scalar error value between the estimated and true robot base pose.

    The theta error is wrapped into [-pi, pi] so that angles near -pi and pi are
    treated as close to each other instead of far apart.
    """
    theta_error = (est[2] - target[2] + np.pi) % (2 * np.pi) - np.pi
    return np.linalg.norm([est[0] - target[0], est[1] - target[1], theta_error])


def make_sim():
    """
    Create a fresh PyBullet simulation for one Task 3 experiment trial.

    DIRECT mode is used instead of GUI mode so the experiment batch runs faster
    and does not open a new visualization window for every trial.
    """
    bullet_client = bc.BulletClient(connection_mode=p.DIRECT)
    bullet_client.setAdditionalSearchPath(pd.getDataPath())
    bullet_client.setTimeStep(sim.SimTimeStep)
    bullet_client.resetSimulation()
    bullet_client.setGravity(0, 0, 0)
    panda_sim = sim.PandaSim(bullet_client, loc=loc_gt)
    return bullet_client, panda_sim


# Column names for the per-trial CSV output.
fieldnames = [
    "num_particles", "sigma", "delta", "trial",
    "est_x", "est_y", "est_theta",
    "error", "runtime_sec", "status"
]


# This list stores one dictionary per experiment run.
# Each dictionary contains the hyperparameters, the estimated pose, the error,
# runtime, and whether the run succeeded.
results = []


# Run every hyperparameter combination and save each trial immediately.
#
# Saving after every trial is useful for Task 3 because online experiments can be
# slow. If a later trial fails, earlier successful results are still preserved.
with open(results_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    # Loop over every particle count.
    for num_particles in num_particles_list:

        # Loop over every sigma value.
        for sigma in sigma_list:

            # Loop over every delta value.
            for delta in delta_list:

                # Repeat each setting to measure run-to-run variation.
                for trial in range(num_trials):
                    print(
                        f"Running trial {trial + 1}: "
                        f"num_particles={num_particles}, sigma={sigma}, delta={delta}"
                    )

                    bullet_client = None
                    start = time.time()

                    try:
                        # Create a new simulation for this trial so previous
                        # robot states do not affect the next experiment.
                        bullet_client, panda_sim = make_sim()

                        # Run Task 3 online Particle Filtering.
                        #
                        # plot=False disables particle visualization so the
                        # batch experiment runs faster.
                        est = alg.particle_filter_online(
                            panda_sim,
                            num_particles,
                            sigma=sigma,
                            delta=delta,
                            plot=False
                        )

                        # Measure runtime and compare the estimate to ground truth.
                        runtime = time.time() - start
                        error = pose_error(est, loc_gt)

                        # Store a successful trial.
                        row = {
                            "num_particles": num_particles,
                            "sigma": sigma,
                            "delta": delta,
                            "trial": trial + 1,
                            "est_x": est[0],
                            "est_y": est[1],
                            "est_theta": est[2],
                            "error": error,
                            "runtime_sec": runtime,
                            "status": "ok"
                        }

                        print(f"  est={est}, error={error:.4f}, runtime={runtime:.1f}s")

                    except Exception as exc:
                        # If one trial fails, record the failure and continue
                        # testing the remaining hyperparameter combinations.
                        runtime = time.time() - start
                        row = {
                            "num_particles": num_particles,
                            "sigma": sigma,
                            "delta": delta,
                            "trial": trial + 1,
                            "est_x": "",
                            "est_y": "",
                            "est_theta": "",
                            "error": "",
                            "runtime_sec": runtime,
                            "status": f"{type(exc).__name__}: {exc}"
                        }

                        print(f"  failed after {runtime:.1f}s: {row['status']}")

                    finally:
                        # Disconnect from PyBullet to avoid accumulating
                        # simulations in memory.
                        if bullet_client is not None:
                            bullet_client.disconnect()

                    # Save the row in memory and write it to disk immediately.
                    results.append(row)
                    writer.writerow(row)
                    f.flush()


# Build a summary table by averaging successful trials for each hyperparameter
# combination.
summary = []

for num_particles in num_particles_list:
    for sigma in sigma_list:
        for delta in delta_list:
            rows = [
                row for row in results
                if row["num_particles"] == num_particles
                and row["sigma"] == sigma
                and row["delta"] == delta
                and row["status"] == "ok"
            ]

            if len(rows) == 0:
                continue

            # Convert the successful trial errors and runtimes into arrays so
            # their mean, standard deviation, and minimum can be computed.
            errors = np.array([row["error"] for row in rows], dtype=float)
            runtimes = np.array([row["runtime_sec"] for row in rows], dtype=float)

            summary.append({
                "num_particles": num_particles,
                "sigma": sigma,
                "delta": delta,
                "num_successful_trials": len(rows),
                "mean_error": np.mean(errors),
                "std_error": np.std(errors),
                "min_error": np.min(errors),
                "mean_runtime_sec": np.mean(runtimes)
            })


# Sort the summary so the best average result appears first.
summary.sort(key=lambda row: row["mean_error"])


# Write the summary CSV file.
with open(summary_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "num_particles", "sigma", "delta",
        "num_successful_trials",
        "mean_error", "std_error", "min_error",
        "mean_runtime_sec"
    ])
    writer.writeheader()
    writer.writerows(summary)

print(f"Saved trial results to {results_file}")
print(f"Saved summary results to {summary_file}")


# Print the best hyperparameter combination according to mean error.
if len(summary) > 0:
    best = summary[0]
    print(
        "Best average combo: "
        f"num_particles={best['num_particles']}, "
        f"sigma={best['sigma']}, "
        f"delta={best['delta']}, "
        f"mean_error={best['mean_error']:.4f}"
    )
