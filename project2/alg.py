import numpy as np
import itertools as it
import scipy.spatial
import utils
import time


########## Task 1: Primitive Wrenches ##########

def primitive_wrenches(mesh, grasp, mu=0.2, n_edges=8):
    """
    Find the primitive wrenches for each contact of a grasp.
    args:   mesh: The object mesh model.
                  Type: trimesh.base.Trimesh
           grasp: The indices of the mesh triangles being contacted.
                  Type: list of int
              mu: The friction coefficient of the mesh surface.
                  (default: 0.2)
         n_edges: The number of edges of the friction polyhedral cone.
                  Type: int (default: 8)
    returns:   W: The primitive wrenches.
                  Type: numpy.ndarray of shape (len(grasp) * n_edges, 6)
    """
    ########## TODO ##########
    # TESTING:      python main.py --task 1
    W = np.zeros(shape=(len(grasp) * n_edges, 6))

    # Use triangle centroids as contact points and measure torque about the center of mass.
    contacts = utils.get_centroid_of_triangles(mesh, grasp)
    cm = mesh.center_mass

    for j, tr_id in enumerate(grasp):
        # Outward unit normal of the contacted face.
        n = mesh.face_normals[tr_id]
        n = n / np.linalg.norm(n)

        # Build an orthonormal basis for the tangent plane at the contact.
        ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t1 = np.cross(n, ref)
        t1 = t1 / np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        t2 = t2 / np.linalg.norm(t2)

        # Lever arm from the object center of mass to the contact point.
        r = contacts[j] - cm

        for k in range(n_edges):
            # Sample one edge of the polyhedral friction cone in the tangent plane.
            theta = 2 * np.pi * k / n_edges
            tangent = np.cos(theta) * t1 + np.sin(theta) * t2

            # Primitive force has unit normal component and tangential magnitude mu.
            f = n + mu * tangent

            # Wrench torque is induced by the force acting at the contact point.
            tau = np.cross(r, f)

            row = j * n_edges + k
            W[row, :3] = f
            W[row, 3:] = tau

    ##########################
    return W







########## Task 2: Grasp Quality Evaluation ##########

def eval_Q(mesh, grasp, mu=0.2, n_edges=8, lmbd=1.0):
    """
    Evaluate the L1 quality of a grasp.
    args:   mesh: The object mesh model.
                  Type: trimesh.base.Trimesh
           grasp: The indices of the mesh triangles being contacted.
                  Type: list of int
              mu: The friction coefficient of the mesh surface.
                  (default: 0.2)
         n_edges: The number of edges of the friction polyhedral cone.
                  Type: int (default: 8)
            lmbd: The scale of torque magnitude.
                  (default: 1.0)
    returns:   Q: The L1 quality score of the given grasp.
    """
    ########## TODO ##########
    #   TESTING:
    #           python main.py --task 2 --mesh bunny
    #           python main.py --task 2 --mesh cow
    #           python main.py --task 2 --mesh duck

    #   EXPERIMENTS: (L1 = grasp quality)
    #   lmbd = [0.2, 0.5, 1]
    #   print("The value of lamba is: ", lmbd)

    #for counting values in Task 4
    if not hasattr(eval_Q, "counter"):
        eval_Q.counter = 0
    eval_Q.counter += 1

    Q = -np.inf

    # First compute all primitive wrenches for the grasp.
    W = primitive_wrenches(mesh, grasp, mu=mu, n_edges=n_edges)

    # Scale the torque part of each wrench by lambda before building the hull.
    W_scaled = W.copy()
    W_scaled[:, 3:] *= lmbd

    # The convex hull of the primitive wrenches defines the feasible wrench space.
    hull = scipy.spatial.ConvexHull(W_scaled)

    distances = []
    for eq in hull.equations:
        # Each hyperplane is stored as a · x + b = 0.
        a = eq[:-1]
        b = eq[-1]

        # Signed distance from the origin to this hyperplane is -b / ||a||.
        distances.append(-b / np.linalg.norm(a))

    # L1 grasp quality is the minimum signed distance over all hull facets.
    Q = np.min(distances)

    ##########################
    return Q










########## Task 3: Stable Grasp Sampling ##########

def sample_stable_grasp(mesh, thresh=0.0):
    """
    Sample a stable grasp such that its L1 quality is larger than a threshold.
    args:     mesh: The object mesh model.
                    Type: trimesh.base.Trimesh
            thresh: The threshold for stable grasp.
                    (default: 0.0)
    returns: grasp: The stable grasp represented by the indices of triangles.
                    Type: list of int
                 Q: The L1 quality score of the sampled grasp, 
                    expected to be larger than thresh.
    """
    ########## TODO ##########
    # TESTING:          python main.py --task 3
    # thresh=0.01, in main.py & lmbd=1.0 in eval_Q() default
    grasp = None
    Q = -np.inf

    while True:
        # Randomly sample a 3-contact grasp using distinct triangle indices.
        grasp = np.random.choice(len(mesh.faces), size=3, replace=False).tolist()

        # Evaluate the sampled grasp using the L1 quality from Task 2.
        Q = eval_Q(mesh, grasp)

        # Stop once the sampled grasp is stable enough.
        if Q > thresh:
            return grasp, Q

    ##########################
    return grasp, Q








########## Task 4: Grasp Optimization ##########

def find_neighbors(mesh, tr_id, eta=1):
    """
    Find the eta-order neighbor faces (triangles) of tr_id on the mesh model.
    args:       mesh: The object mesh model.
                      Type: trimesh.base.Trimesh
               tr_id: The index of the query face (triangle).
                      Type: int
                 eta: The maximum order of the neighbor faces:
                      Type: int
    returns: nbr_ids: The list of the indices of the neighbor faces.
                      Type: list of int
    """
    ########## TODO ##########
    # TESTING:          python main.py --task 4
    # grasp = [80, 165, 444] in main.py & lmbd=1.0 in eval_Q() default

    # Build a graph where neighboring faces share a vertex.
    nbr_ids = []
    adjacency = {}
    for a, b in mesh.face_neighborhood:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    # Start the search from the query triangle.
    visited = {tr_id}
    frontier = {tr_id}

    # Expand outward eta times to collect all neighbors within eta hops.
    for _ in range(eta):
        next_frontier = set()
        for f in frontier:
            next_frontier |= adjacency.get(f, set())
        next_frontier -= visited
        visited |= next_frontier
        frontier = next_frontier

    # Remove the query triangle itself and return only its neighbors.
    visited.remove(tr_id)
    nbr_ids = list(visited)

    ##########################
    return nbr_ids


def local_optimal(mesh, grasp):
    """
    Find the optimal neighbor grasp of the given grasp.
    args:     mesh: The object mesh model.
                    Type: trimesh.base.Trimesh
             grasp: The indices of the mesh triangles being contacted.
                    Type: list of int
    returns: G_opt: The optimal neighbor grasp with the highest quality.
                    Type: list of int
             Q_max: The L1 quality score of G_opt.
    """
    ########## TODO ##########
    G_opt = None
    Q_max = -np.inf

    candidate_lists = []
    for tr_id in grasp:
        # For each contact, allow either staying on the same face or moving to a neighbor.
        nbrs = find_neighbors(mesh, tr_id, eta=1)
        candidate_lists.append([tr_id] + nbrs)

    # Start with the current grasp as the best known candidate.
    G_opt = list(grasp)
    Q_max = eval_Q(mesh, G_opt)

    # Check every combination of neighboring faces as a candidate neighbor grasp.
    for cand in it.product(*candidate_lists):
        cand = list(cand)

        # Skip grasps that reuse the same triangle for multiple contacts.
        if len(set(cand)) < len(cand):
            continue

        Q = eval_Q(mesh, cand)
        if Q > Q_max:
            G_opt = cand
            Q_max = Q

    ##########################
    return G_opt, Q_max

def optimize_grasp(mesh, grasp):
    """
    Optimize the given grasp and return the trajectory.
    args:    mesh: The object mesh model.
                   Type: trimesh.base.Trimesh
            grasp: The indices of the mesh triangles being contacted.
                   Type: list of int
    returns: traj: The trajectory of the grasp optimization.
                   Type: list of grasp (each grasp is a list of int)



    # Initialize the trajectory with the starting grasp.
    traj = [list(grasp)]
    current = list(grasp)
    current_Q = eval_Q(mesh, current)

    # Repeatedly move to the best neighboring grasp until no improvement is possible.
    while True:
        G_opt, Q_opt = local_optimal(mesh, current)
        if Q_opt > current_Q:
            current = G_opt
            current_Q = Q_opt
            traj.append(current)
        else:
            break

    """
    traj = []
    ########## TODO ##########
    # Reset the quality-evaluation counter for this optimization run.
    eval_Q.counter = 0

    traj = [list(grasp)]
    current = list(grasp)
    current_Q = eval_Q(mesh, current)

    while True:
        G_opt, Q_opt = local_optimal(mesh, current)
        if Q_opt > current_Q:
            current = list(G_opt)
            current_Q = Q_opt
            traj.append(list(current))
        else:
            break

    # Save the number of eval_Q calls used only for optimization.
    eval_count = eval_Q.counter

    # Estimate the probability of getting a grasp at least this good by random sampling.
    n_trials = 5000
    hits = 0
    for _ in range(n_trials):
        rand_grasp = np.random.choice(len(mesh.faces), size=3, replace=False).tolist()
        rand_Q = eval_Q(mesh, rand_grasp)
        if rand_Q >= current_Q:
            hits += 1

    prob = hits / n_trials

    print("Number of grasp quality evaluations:", eval_count)
    print(f"Estimated probability of randomly obtaining optimized quality: {prob:.6f} ({hits}/{n_trials})")

    ##########################
    return traj
















########## Task 5: Grasp Optimization with Reachability ##########

def optimize_reachable_grasp(mesh, r=0.5):
    """
    Sample a reachable grasp and optimize it.
    args:    mesh: The object mesh model.
                   Type: trimesh.base.Trimesh
                r: The reachability measure. (default: 0.5)
    returns: traj: The trajectory of the grasp optimization.
                   Type: list of grasp (each grasp is a list of int) 
    """
    traj = []
    ########## TODO ##########
    #TESTING:       python main.py --task 5

    def is_reachable(grasp):
        points = utils.get_centroid_of_triangles(mesh, grasp)
        psi = np.mean(points, axis=0)
        avg_dist = np.mean(np.linalg.norm(points - psi, axis=1))
        return avg_dist < r

    while True:
        grasp = np.random.choice(len(mesh.faces), size=3, replace=False).tolist()
        if is_reachable(grasp):
            break

    current = list(grasp)
    current_Q = eval_Q(mesh, current)
    traj = [list(current)]

    while True:
        candidate_lists = []
        for tr_id in current:
            nbrs = find_neighbors(mesh, tr_id, eta=1)
            candidate_lists.append([tr_id] + nbrs)

        best_grasp = list(current)
        best_Q = current_Q

        for cand in it.product(*candidate_lists):
            cand = list(cand)
            if len(set(cand)) < len(cand):
                continue
            if not is_reachable(cand):
                continue

            Q = eval_Q(mesh, cand)
            if Q > best_Q:
                best_grasp = cand
                best_Q = Q

        if best_Q > current_Q:
            current = best_grasp
            current_Q = best_Q
            traj.append(list(current))
        else:
            break

    ##########################
    return traj
