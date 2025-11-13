import numpy as np

def compute_distance_matrices(coords):
    """
    coords : dict {id: (x,y)}
    Retourne (M, E, index) où
    M = matrice Manhattan
    E = matrice Euclidienne
    index = liste ordonnée des ids (pour correspondance)
    """
    ids = sorted(coords.keys())
    n = len(ids)

    M = np.zeros((n, n))
    E = np.zeros((n, n))

    for a in range(n):
        xa, ya = coords[ids[a]]
        for b in range(n):
            xb, yb = coords[ids[b]]

            dx = xb - xa
            dy = yb - ya

            # Manhattan
            M[a, b] = abs(dx) + abs(dy)

            # Euclidienne
            E[a, b] = np.sqrt(dx*dx + dy*dy)

    return M, E, ids
