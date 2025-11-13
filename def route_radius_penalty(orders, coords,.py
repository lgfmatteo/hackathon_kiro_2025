def route_radius_penalty(orders, coords, cradius_f):
    """
    orders : liste des IDs d’ordres de la route [i1, i2, ..., in]
    coords : dict {i: (x_i, y_i)} pour tous les ordres
    cradius_f : coefficient cradius de la famille fr
    """
    # 1. calcul du diamètre (max distance Euclidienne)
    diam2 = 0.0  # on peut travailler au carré pour éviter des sqrt inutiles
    m = len(orders)
    for a in range(m):
        i = orders[a]
        xi, yi = coords[i]
        for b in range(a + 1, m):
            j = orders[b]
            xj, yj = coords[j]
            dx = xj - xi
            dy = yj - yi
            dist2 = dx*dx + dy*dy
            if dist2 > diam2:
                diam2 = dist2

    # si la route n'a qu'un seul point (m <= 1), le diamètre=0
    radius2 = diam2 / 4.0
    return cradius_f * radius2

