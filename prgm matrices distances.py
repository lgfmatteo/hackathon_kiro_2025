import math

EARTH_RADIUS = 6_371_000      # en mètres
DEG_TO_RAD   = math.pi / 180  # conversion degrés → radians

def latlon_to_cartesian(phi_i, lambda_i, phi_j, lambda_j, phi0):
    """
    phi_i, lambda_i, phi_j, lambda_j, phi0 en DEGRÉS.
    phi0 : latitude de référence (par ex. le dépôt).
    """
    dphi  = phi_j - phi_i
    dlamb = lambda_j - lambda_i

    dy = EARTH_RADIUS * DEG_TO_RAD * dphi
    dx = EARTH_RADIUS * math.cos(DEG_TO_RAD * phi0) * DEG_TO_RAD * dlamb

    return dx, dy

def manhattan_distance(phi_i, lambda_i, phi_j, lambda_j, phi0):
    dx, dy = latlon_to_cartesian(phi_i, lambda_i, phi_j, lambda_j, phi0)
    return abs(dx) + abs(dy)

def euclidean_distance(phi_i, lambda_i, phi_j, lambda_j, phi0):
    dx, dy = latlon_to_cartesian(phi_i, lambda_i, phi_j, lambda_j, phi0)
    return math.hypot(dx, dy)

def distance_matrices(points, phi0):
    """
    points = liste de (phi, lambda) en degrés.
    renvoie (M_manhattan, M_euclidienne)
    """
    n = len(points)
    M_manh = [[0.0] * n for _ in range(n)]
    M_eucl = [[0.0] * n for _ in range(n)]

    for i in range(n):
        phi_i, lam_i = points[i]
        for j in range(n):
            if i == j:
                continue
            phi_j, lam_j = points[j]
            M_manh[i][j] = manhattan_distance(phi_i, lam_i, phi_j, lam_j, phi0)
            M_eucl[i][j] = euclidean_distance(phi_i, lam_i, phi_j, lam_j, phi0)

    return M_manh, M_eucl

# Exemple d'utilisation
if __name__ == "__main__":
    points = [
        (48.8566, 2.3522),   # Paris
        (45.7640, 4.8357),   # Lyon
        (43.6047, 1.4442)    # Toulouse
    ]
    phi0 = points[0][0]  # on prend la latitude de Paris comme référence
    M_manh, M_eucl = distance_matrices(points, phi0)

    print("Manhattan :")
    for row in M_manh:
        print(row)

    print("\nEuclidienne :")
    for row in M_eucl:
        print(row)
