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

    # ---------------------------------------------------------------------------
    # Pre-defined arm configurations for diverse workspace coverage.
    # Each row is a set of 7 joint angles. Cycling through these puts the sphere
    # at different heights and horizontal positions in the robot's LOCAL frame,
    # which constrains all three pose dimensions (x, y, theta) in the filter.
    # ---------------------------------------------------------------------------
    _PROBE_CONFIGS = np.array([
        [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],  # default: mid-height
        [0.6, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],  # base rotated right
        [-0.6, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],  # base rotated left
        [0.0, -0.5, 0.0, -2.0, 0.0, 1.8, 0.785],  # arm extended / higher
        [0.0, -1.0, 0.0, -2.6, 0.0, 1.3, 0.785],  # arm pulled in / lower
        [1.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],  # far right rotation
        [-1.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],  # far left rotation
    ])

    def safe_is_touch():
        try:
            return panda_sim.is_touch()
        except TypeError:
            return False

    def safe_is_collision():
        try:
            return panda_sim.is_collision()
        except TypeError:
            return False

    obv = None

    max_attempts = 70  # = 10 full cycles through _PROBE_CONFIGS
    steps_per_attempt = 350  # max simulation steps per probe direction
    min_steps = 8  # ignore contacts for first N steps to flush stale cache

    # Remember the original configuration so we can restore it after every attempt.
    # set_joint_values / resetJointState zeros velocity too — more reliable than
    # save/restoreState in GUI mode, which can leave motor velocity targets active.
    jpos_init, _, _ = panda_sim.get_joint_states()
    start_joints = np.array(jpos_init[:7])

    for attempt in range(max_attempts):
        # ── 1. Pick an arm configuration ────────────────────────────────────
        # Cycle through the pre-defined poses so we cover the full workspace.
        probe_joints = _PROBE_CONFIGS[attempt % len(_PROBE_CONFIGS)]
        panda_sim.set_joint_values(probe_joints)
        # A few physics steps flush the contact cache after the teleport.
        for _ in range(5):
            panda_sim.step()

        # ── 2. Pick a random horizontal probe direction ──────────────────────
        angle = np.random.uniform(0, 2 * np.pi)
        speed = 0.25
        v = np.array([np.cos(angle) * speed,
                      np.sin(angle) * speed,
                      0.0, 0.0, 0.0, 0.0])

        # ── 3. Move until touch, arm collision, or step budget exhausted ─────
        found = False
        steps = 0
        for _ in range(steps_per_attempt):
            # Only record after min_steps to avoid residual contacts from reset
            if safe_is_touch() and steps >= min_steps:              # DEBUGGING
            #if panda_sim.is_touch() and steps >= min_steps:
                jpos, _, _ = panda_sim.get_joint_states()
                obv = np.array(jpos[:7])
                found = True
                break
            # Arm (non-tip) collision → abort this direction
            if safe_is_collision() and not safe_is_touch():          # DEBUGGING
            #if panda_sim.is_collision() and not panda_sim.is_touch():
                break
            panda_sim.execute(v)
            steps += 1

        # ── 4. Reset to starting configuration ──────────────────────────────
        panda_sim.set_joint_values(start_joints)
        for _ in range(5):
            panda_sim.step()

        if found:
            return obv

    # Fallback: no touch found — return current joints to prevent None.
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
    # initialize the particles and weights
    particles = np.random.uniform(low=[-1, -1, -np.pi], high=[1, 1, np.pi], size=(num_particles, 3))
    weights = np.ones(shape=(num_particles,)) / num_particles

    # configure the visualization
    if plot:
        ax = utils.config_plot_ax()
        utils.plot_pf(ax, particles, panda_sim.loc)
        plt.pause(0.01)

    num_iters = 100
    for _ in range(num_iters):

        ########## TODO ##########
        obv = get_one_obv(panda_sim)

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


