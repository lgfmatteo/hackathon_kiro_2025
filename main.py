import pandas as pd
import numpy as np

# Choix de l'instance
s = '10'

# Parsing des données
vehicles_df = pd.read_csv('instances/vehicles.csv')

instance_df = pd.read_csv('instances/instance_' + s + '.csv')
depot = instance_df[instance_df['id'] == 0].iloc[0]
orders = instance_df[instance_df['id'] != 0].to_dict('records')

# Fonctions utiles
EARTH_RADIUS = 6.371e6
TWO_PI_OVER_360 = 2 * np.pi / 360

def latlon_to_cartesian(phi_i, lambd_i, phi_j, lambd_j, phi_0):
    delta_y = EARTH_RADIUS * TWO_PI_OVER_360 * (phi_j - phi_i)
    delta_x = EARTH_RADIUS * np.cos(TWO_PI_OVER_360 * phi_0) * TWO_PI_OVER_360 * (lambd_j - lambd_i)
    return delta_x, delta_y

def manhattan_distance(phi_i, lambd_i, phi_j, lambd_j, phi0=depot['longitude']):
    delta_x, delta_y = latlon_to_cartesian(phi_i, lambd_i, phi_j, lambd_j, phi0)
    return abs(delta_x) + abs(delta_y)

def euclidean_distance(phi_i, lambd_i, phi_j, lambd_j, phi0):
    delta_x, delta_y = latlon_to_cartesian(phi_i, lambd_i, phi_j, lambd_j, phi0)
    return np.sqrt(delta_x ** 2 + delta_y ** 2)

# Algo Glouton simple
def glouton_simple(orders, vehicles_df):
    routes = []
    
    for order in orders:
        suitable_vehicles = vehicles_df[vehicles_df['max_capacity'] >= order['order_weight']]
        cheapest_vehicle = suitable_vehicles.sort_values('rental_cost').iloc[0]
        
        route = {
            'family': int(cheapest_vehicle['family']),
            'order_1': int(order['id'])
        }
        routes.append(route)
    
    return pd.DataFrame(routes)

# Nouveau glouton
def glouton(orders, vehicles_df):
    routes = []
    aVoir = orders.copy()
    
    while aVoir:
        route_orders = []
        current_weight = 0
        vehicle = vehicles_df.iloc[0]
        
        actual = aVoir.pop(0)
        route_orders.append(actual)
        current_weight += actual['order_weight']
        
        while True:
            feasible = [o for o in aVoir if current_weight + o['order_weight'] <= vehicle['max_capacity']]
            
            if not feasible:
                break
            
            nearest = feasible[0]
            current_distance = manhattan_distance(nearest['longitude'], nearest['latitude'], 
                                                 actual['longitude'], actual['latitude'])
            
            for other in feasible:
                dist = manhattan_distance(other['longitude'], other['latitude'], 
                                        actual['longitude'], actual['latitude'])
                if dist < current_distance:
                    nearest = other
                    current_distance = dist
            
            route_orders.append(nearest)
            current_weight += nearest['order_weight']
            aVoir.remove(nearest)
            actual = nearest
        
        route_dict = {'family': int(vehicle['family'])}
        for i, order in enumerate(route_orders, 1):
            route_dict[f'order_{i}'] = int(order['id'])
        routes.append(route_dict)
    
    df = pd.DataFrame(routes)
    for col in df.columns:
        if col.startswith('order_'):
            df[col] = df[col].astype('Int64')
    return df

df = glouton(orders, vehicles_df)
df.to_csv('route_instance_' + s + '.csv')

