import math
def build_distance_matrices(df, phi0=None):
    if phi0 is None:
        phi0 = df.iloc[0]["latitude"]

    lats = df["latitude"].tolist()
    lons = df["longitude"].tolist()
    n = len(df)

    M = [[manhattan_distance(lats[i], lons[i], lats[j], lons[j], phi0)
          for j in range(n)] for i in range(n)]
    E = [[euclidean_distance(lats[i], lons[i], lats[j], lons[j], phi0)
          for j in range(n)] for i in range(n)]

    return M, E

