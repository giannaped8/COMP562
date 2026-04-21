import numpy as np
import sim
import utils
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


OBS_CENTER = np.array([[0.9, 0], [0.25, 0.5], [-0.3, 0.5], [-1, 0.1], [0.3, -0.8]])
OBS_RADIUS = np.array([0.5, 0.3, 0.2, 0.5, 0.4])
SPH_RADIUS = 0.02

FK_Solver = utils.FKSolver() # forward kinematics solver


########## Task 1: Particle Weights Calculation ##########
# TESTING:  python main.py --task 1
def dist_to_closest_obs(x, y):
    """
    Find the distance between the stick's sphere end centered at (x, y) 
    to its closest cylinder obstacle.
    args:       x: The x-coordinate of the spherical end.
                y: The y-coordinate of the spherical end. 
    returns: dist: The distance to the closest obstacle.
    """
    ########## TODO ##########
    dist = 0.0
    obs_dists = np.sqrt((OBS_CENTER[:, 0] - x) ** 2 + (OBS_CENTER[:, 1] - y) ** 2)
    surface_dists = obs_dists - (OBS_RADIUS + SPH_RADIUS)
    dist = surface_dists[np.argmin(np.abs(surface_dists))]
    ##########################
    return dist


def cal_weights(particles, obv, sigma=0.05):
    """
    Calculate the weights for particles based on the given observation.
    args:  particles: The particles represented by their states
                      Type: numpy.ndarray of shape (# of particles, 3)
                 obv: The given observation (the robot's joint angles).
                      Type: numpy.ndarray of shape (7,)
               sigma: The standard deviation of the Gaussian distribution
                      for calculating likelihood (default: 0.05).
    returns: weights: The weights of all particles.
                      Type: numpy.ndarray of shape (# of particles,)
    """
    ########## TODO ##########
    #sigma = 0.5    #[0.05, 0.1, 0.5]
    #print("Sigma = ", sigma)

    # Use forward kinematics to find the spherical end position in the robot base frame.
    tip_x_local, tip_y_local = FK_Solver.forward_kinematics_2d(obv)

    dists = []

    # For each particle, transform the spherical end into the world frame.
    for px, py, theta in particles:
        tip_x_world = px + np.cos(theta) * tip_x_local - np.sin(theta) * tip_y_local
        tip_y_world = py + np.sin(theta) * tip_x_local + np.cos(theta) * tip_y_local

        # Compute the signed distance from the predicted spherical end position
        # to the closest obstacle.
        dists.append(dist_to_closest_obs(tip_x_world, tip_y_world))

    dists = np.array(dists)

    # Compute Gaussian likelihoods in log form for numerical stability.
    # The normalizing constant is omitted because weights are normalized afterward.
    log_likelihoods = -0.5 * (dists / sigma) ** 2

    # Subtract the largest log likelihood before exponentiating.
    # This prevents all likelihoods from underflowing to zero when sigma is small.
    log_likelihoods -= np.max(log_likelihoods)

    likelihoods = np.exp(log_likelihoods)
    total_likelihood = np.sum(likelihoods)

    # If numerical issues still occur, fall back to uniform weights.
    # This prevents np.random.choice from receiving NaN probabilities.
    if total_likelihood == 0 or not np.isfinite(total_likelihood):
        weights = np.ones(len(particles)) / len(particles)
    else:
        weights = likelihoods / total_likelihood

    ##########################
    return weights



def most_likely_particle(particles, obv):
    """
    Find the most likely particle.
    args:  particles: The particles represented by their states
                      Type: numpy.ndarray of shape (# of particles, 3)
                 obv: The given observation (the robot's joint angles). 
                      Type: numpy.ndarray of shape (7,)
    returns:     idx: The index of the most likely particle
                      Type: int
    """
    ########## TODO ##########
    idx = 0
    weights = cal_weights(particles, obv)
    idx = np.argmax(weights)
    ##########################
    return idx

















########## Task 2: Particle Filter ##########
#   TESTING:  python main.py --task 2
def particle_filter(panda_sim, obvs, num_particles, sigma=0.05, delta=0.01, plot=True):
    """
    The Particle Filtering algorithm. 
    args:    panda_sim: The instance of the simulation. 
                        Type: sim.PandaSim (provided)
                  obvs: The given observations (the robot's joint angles)
                        Type: numpy.ndarray of shape (# of observations, 7)
         num_particles: The number of particles. 
                        Type: int
                 sigma: The standard deviation of the Gaussian distribution
                        for calculating likelihood (default: 0.05).
                 delta: The scale of the Gaussian for perturbing particles.
                        (default: 0.01)
                  plot: Whether to enable particle plot (True or False).
    returns:       est: The estimate of the pose of the robot base. 
                        Type: numpy.ndarray of shape (3,)
    """
    # initialize the particles and weights 
    particles = np.random.uniform(low=[-1, -1, -np.pi], high=[1, 1, np.pi], size=(num_particles, 3))
    weights = np.ones(shape=(num_particles,)) / num_particles

    # configure the visualization
    if plot:
        ax = utils.config_plot_ax()
        utils.plot_pf(ax, particles, panda_sim.loc)
        plt.pause(0.01)

    for obv in obvs:
        panda_sim.set_joint_values(obv)

        ########## TODO ##########
        # Update particle weights using the current contact observation.
        weights = cal_weights(particles, obv, sigma=sigma)

        # Resample particles according to their weights.
        # Particles with larger weights are more likely to be copied.
        resampled_idxs = np.random.choice(
            num_particles,
            size=num_particles,
            replace=True,
            p=weights
        )
        particles = particles[resampled_idxs]

        # Add Gaussian noise after resampling to prevent particle collapse.
        particles += np.random.normal(
            loc=0.0,
            scale=delta,
            size=particles.shape
        )

        # Keep x and y inside the original valid sampling range.
        particles[:, 0] = np.clip(particles[:, 0], -1, 1)
        particles[:, 1] = np.clip(particles[:, 1], -1, 1)

        # Wrap theta back into [-pi, pi].
        particles[:, 2] = (particles[:, 2] + np.pi) % (2 * np.pi) - np.pi

        ##########################

        # plot the particles in the visualization
        if plot:
            utils.plot_pf(ax, particles, panda_sim.loc)
            plt.pause(0.01)
    est = particles.mean(0)
    return est















########## Task 3: Generate Observations Online ##########
#   TESTING:  python main.py --task 3
def get_one_obv(panda_sim):
    """
    Control the robot in simulation to obtain an observation.
    args: panda_sim: The instance of the simulation. 
                     Type: sim.PandaSim (provided)
    returns:    obv: One observation found by this function.
                     Type: numpy.ndarray of shape (7,)
    """
    ########## TODO ##########
    # We collect an online observation by randomly moving the robot's end-effector
    # until the spherical end of the stick touches one of the cylinder obstacles.
    #
    # The observation is the robot's 7 joint angles at the moment of touch.

    obv = None
    max_steps = 8000
    segment_steps = 75
    speed = 0.35

    v = np.zeros(6)

    for step in range(max_steps):

        # Choose a new random Cartesian direction every few simulation steps.
        #
        # The first three entries of v are translational velocity.
        # The last three entries are rotational velocity, which we keep at zero.
        if step % segment_steps == 0:
            direction = np.random.normal(size=3)

            # Most useful motion for contacting vertical cylinders is in x-y.
            # Keep z motion smaller so the probe does not wander too aggressively up/down.
            direction[2] *= 0.25

            direction_norm = np.linalg.norm(direction)
            if direction_norm < 1e-8:
                continue

            direction = direction / direction_norm

            v = np.zeros(6)
            v[0:3] = speed * direction

        # Save the current joint configuration before moving.
        #
        # If the robot causes an invalid collision, we can roll back to this
        # previous safe configuration.
        prev_jpos, _, _ = panda_sim.get_joint_states()
        prev_jpos = np.array(prev_jpos[:7])

        # Move the robot one control step using the selected Cartesian velocity.
        panda_sim.execute(v)

        # If the spherical end of the stick touches an obstacle, this is a valid
        # contact observation. Return the current 7 robot joint angles.
        if panda_sim.is_touch():
            jpos, _, _ = panda_sim.get_joint_states()
            obv = np.array(jpos[:7])
            return obv

        # If there is a collision but it is not a valid stick-sphere touch,
        # undo the movement and pick a new direction.
        if panda_sim.is_collision():
            panda_sim.set_joint_values(prev_jpos)

            # Force a new random direction on the next loop iteration.
            segment_steps = 1

        else:
            # Restore the normal segment length after a safe move.
            segment_steps = 75

    # If no touch is found, return the current joint configuration as a fallback.
    # In most runs this should not be reached, but it prevents returning None.
    jpos, _, _ = panda_sim.get_joint_states()
    obv = np.array(jpos[:7])

    ##########################
    return obv












def particle_filter_online(panda_sim, num_particles, sigma=0.05, delta=0.01, plot=True):
    """
    The online Particle Filtering algorithm. 
    args:     panda_sim: The instance of the simulation. 
                         Type: sim.PandaSim (provided)
          num_particles: The number of particles. 
                         Type: int
                  sigma: The standard deviation of the Gaussian distribution
                         for calculating likelihood (default: 0.05).
                  delta: The scale of the Gaussian for perturbing particles.
                         (default: 0.01)
                   plot: Whether to enable particle plot (True or False).
    returns:        est: The estimate of the pose of the robot base. 
                         Type: numpy.ndarray of shape (3,)
    """
    # # initialize the particles and weights 
    particles = np.random.uniform(low=[-1, -1, -np.pi], high=[1, 1, np.pi], size=(num_particles, 3))
    weights = np.ones(shape=(num_particles,)) / num_particles

    # configure the visualization
    if plot:
        ax = utils.config_plot_ax()
        utils.plot_pf(ax, particles, panda_sim.loc)
        plt.pause(0.01)

    ########## TODO ##########
    # Number of online contact observations to collect.
    #
    # Each iteration collects one new touch observation, updates particle weights,
    # resamples particles, and perturbs them.

    num_iters = 100  # The number of iterations. Feel free to change the value
    for _ in range(num_iters):

        # Move the robot until the spherical probe touches an obstacle.
        # The returned observation is the 7D joint configuration at contact.
        obv = get_one_obv(panda_sim)

        # Update particle weights using the newly collected observation.
        weights = cal_weights(particles, obv, sigma=sigma)

        # Resample particles according to their normalized weights.
        # Higher-weight particles are more likely to be selected multiple times.
        resampled_idxs = np.random.choice(
            num_particles,
            size=num_particles,
            replace=True,
            p=weights
        )
        particles = particles[resampled_idxs]

        # Add Gaussian perturbation after resampling.
        #
        # This prevents particle collapse, where all particles become identical
        # after repeated resampling.
        particles += np.random.normal(
            loc=0.0,
            scale=delta,
            size=particles.shape
        )

        # Keep x and y inside the original state-space bounds.
        particles[:, 0] = np.clip(particles[:, 0], -1, 1)
        particles[:, 1] = np.clip(particles[:, 1], -1, 1)

        # Wrap theta back into [-pi, pi].
        particles[:, 2] = (particles[:, 2] + np.pi) % (2 * np.pi) - np.pi

        # Plot the particles in the visualization.
        if plot:
            utils.plot_pf(ax, particles, panda_sim.loc)
            plt.pause(0.01)

    ##########################
    est = particles.mean(0)
    return est



