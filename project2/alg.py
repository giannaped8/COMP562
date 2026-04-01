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
    W = np.zeros(shape=(len(grasp) * n_edges, 6))

    contacts = utils.get_centroid_of_triangles(mesh, grasp)
    cm = mesh.center_mass

    for j, tr_id in enumerate(grasp):
        n = mesh.face_normals[tr_id]
        n = n / np.linalg.norm(n)

        ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t1 = np.cross(n, ref)
        t1 = t1 / np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        t2 = t2 / np.linalg.norm(t2)

        r = contacts[j] - cm

        for k in range(n_edges):
            theta = 2 * np.pi * k / n_edges
            tangent = np.cos(theta) * t1 + np.sin(theta) * t2
            f = n + mu * tangent
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


    TESTING
    python main.py --task 2 --mesh bunny
    python main.py --task 2 --mesh cow
    python main.py --task 2 --mesh duck
    """
    ########## TODO ##########
    Q = -np.inf
    W = primitive_wrenches(mesh, grasp, mu=mu, n_edges=n_edges)
    W_scaled = W.copy()
    W_scaled[:, 3:] *= lmbd

    hull = scipy.spatial.ConvexHull(W_scaled)

    distances = []
    for eq in hull.equations:
        a = eq[:-1]
        b = eq[-1]
        distances.append(-b / np.linalg.norm(a))

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
    grasp = None
    Q = -np.inf

    while True:
        grasp = np.random.choice(len(mesh.faces), size=3, replace=False).tolist()
        Q = eval_Q(mesh, grasp)
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
    #nbr_ids = []
    adjacency = {}
    for a, b in mesh.face_neighborhood:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    visited = {tr_id}
    frontier = {tr_id}

    for _ in range(eta):
        next_frontier = set()
        for f in frontier:
            next_frontier |= adjacency.get(f, set())
        next_frontier -= visited
        visited |= next_frontier
        frontier = next_frontier

    visited.remove(tr_id)
    return list(visited)
    ##########################
    #return nbr_ids

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
    #G_opt = None
    #Q_max = -np.inf

    candidate_lists = []
    for tr_id in grasp:
        nbrs = find_neighbors(mesh, tr_id, eta=1)
        candidate_lists.append([tr_id] + nbrs)

    G_opt = list(grasp)
    Q_max = eval_Q(mesh, G_opt)

    for cand in it.product(*candidate_lists):
        cand = list(cand)
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
    """
    traj = []
    ########## TODO ##########
    traj = [grasp]
    current = grasp
    current_Q = eval_Q(mesh, current)

    while True:
        G_opt, Q_opt = local_optimal(mesh, current)
        if Q_opt > current_Q:
            current = G_opt
            current_Q = Q_opt
            traj.append(current)
        else:
            break

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
   

    ##########################
    return traj
