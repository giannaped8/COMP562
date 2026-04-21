import csv
import numpy as np
import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc

import sim
import alg
import utils

#RUN THIS FILE:     python task2_experiments.py

"""
Run the full set of Task 2 Particle Filtering experiments required for the report.

This script tests different combinations of:
    1. The number of particles used by the Particle Filter.
    2. The sigma value used in the Gaussian likelihood function.
    3. The delta value used when perturbing particles after resampling.

For each combination, the Particle Filter is run multiple times because the algorithm
uses random initialization, random resampling, and random Gaussian perturbations.

The script saves the estimated robot base pose for each run to a CSV file so that
the results can be compared and summarized in the report.
"""


# The particle counts requested in the project report.
# A larger number of particles should usually give a more accurate and robust estimate,
# but it also increases computation time.
num_particles_list = [1000, 2500]


# The sigma values requested in the project report.
# Sigma is the standard deviation of the Gaussian used to convert contact distance
# into a particle likelihood.
#
# Smaller sigma:
#     Penalizes particles strongly if their predicted contact distance is not close to zero.
#
# Larger sigma:
#     Gives more similar weights to particles, even if their predicted contact distances differ.
sigma_list = [0.01, 0.05, 0.2]


# The delta values requested in the project report.
# Delta controls the standard deviation of the Gaussian noise added to each particle
# after resampling.
#
# Smaller delta:
#     Keeps resampled particles close to their previous values.
#
# Larger delta:
#     Spreads particles out more, which can help exploration but may reduce stability.
delta_list = [0.001, 0.01, 0.1]


# Number of repeated trials for each hyperparameter combination.
# Multiple trials are useful because Particle Filtering is stochastic.
num_trials = 3


# Ground-truth robot base pose used in main.py.
# The Particle Filter is trying to estimate this pose.
#
# Format:
#     [x position, y position, theta orientation]
loc_gt = np.array([-0.3, -0.3, 0.9])


# Load the pre-collected observations provided by the project.
# Each observation is a robot joint configuration where the probe is touching an obstacle.
#
# Shape:
#     (# of observations, 7)
# where each row contains 7 robot joint angles.
obvs = utils.load_npy("obvs.npy")


# This list stores one dictionary per experiment run.
# Each dictionary contains the hyperparameters, the trial number, the estimated pose,
# and the estimation error.
results = []


# Loop over every required number of particles.
for num_particles in num_particles_list:

    # Loop over every required sigma value.
    for sigma in sigma_list:

        # Loop over every required delta value.
        for delta in delta_list:

            # Repeat the same hyperparameter setting multiple times to observe randomness.
            for trial in range(num_trials):
                print(
                    f"Running trial {trial + 1}: "
                    f"num_particles={num_particles}, sigma={sigma}, delta={delta}"
                )

                # Create a PyBullet client in DIRECT mode.
                #
                # DIRECT mode runs the physics simulation without opening the GUI.
                # This makes batch experiments much faster than using p.GUI.
                bullet_client = bc.BulletClient(connection_mode=p.DIRECT)

                # Configure PyBullet similarly to Task 2 in main.py.
                #
                # setAdditionalSearchPath:
                #     Allows PyBullet to find built-in assets such as plane.urdf.
                #
                # setTimeStep:
                #     Sets the simulation time step.
                #
                # resetSimulation:
                #     Clears any previous simulation state.
                #
                # setGravity:
                #     Gravity is set to zero because this project focuses on
                #     contact-based pose estimation rather than dynamics.
                bullet_client.setAdditionalSearchPath(pd.getDataPath())
                bullet_client.setTimeStep(sim.SimTimeStep)
                bullet_client.resetSimulation()
                bullet_client.setGravity(0, 0, 0)

                # Create a Panda robot simulation at the known ground-truth base pose.
                #
                # Although the Particle Filter does not directly receive loc_gt as its estimate,
                # the simulator uses this pose to generate the robot/environment configuration
                # for visualization and comparison.
                panda_sim = sim.PandaSim(bullet_client, loc=loc_gt)

                # Run the implemented Particle Filtering algorithm.
                #
                # args:
                #     panda_sim:
                #         The PyBullet simulation instance.
                #
                #     obvs:
                #         The 100 provided contact observations.
                #
                #     num_particles:
                #         The number of particles used in this experiment.
                #
                #     sigma:
                #         The Gaussian standard deviation used to calculate particle weights.
                #
                #     delta:
                #         The Gaussian perturbation scale added after resampling.
                #
                #     plot:
                #         False disables plotting so the batch experiments run faster.
                #
                # returns:
                #     est:
                #         The estimated robot base pose as [x, y, theta].
                est = alg.particle_filter(
                    panda_sim,
                    obvs,
                    num_particles,
                    sigma=sigma,
                    delta=delta,
                    plot=False
                )

                # Compute the Euclidean distance between the estimated pose and ground truth.
                #
                # This gives one simple scalar error value for comparing experiments.
                # It combines x error, y error, and theta error into one number.
                theta_error = (est[2] - loc_gt[2] + np.pi) % (2 * np.pi) - np.pi
                error = np.linalg.norm([est[0] - loc_gt[0], est[1] - loc_gt[1], theta_error])

                # Save the result from this trial.
                #
                # These values will be written to task2_results.csv after all experiments finish.
                results.append({
                    "num_particles": num_particles,
                    "sigma": sigma,
                    "delta": delta,
                    "trial": trial + 1,
                    "est_x": est[0],
                    "est_y": est[1],
                    "est_theta": est[2],
                    "error": error
                })

                # Disconnect from PyBullet before starting the next trial.
                #
                # This prevents simulations from accumulating in memory.
                bullet_client.disconnect()


# Save all experiment results to a CSV file.
#
# The CSV file can be opened in Excel, Numbers, Google Sheets, or read back into Python.
# Each row represents one Particle Filter run.
with open("task2_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "num_particles",
            "sigma",
            "delta",
            "trial",
            "est_x",
            "est_y",
            "est_theta",
            "error"
        ]
    )

    # Write the column names as the first row.
    writer.writeheader()

    # Write one row for each saved experiment result.
    writer.writerows(results)


print("Saved results to task2_results.csv")
