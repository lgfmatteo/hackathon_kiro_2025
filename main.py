import pandas as pd
import numpy as np

# Choix de l'instance
s = '10'

# Parsing des données
vehicles_df = pd.read_csv('instances/vehicles.csv')

instance_df = pd.read_csv('instances/instance_' + s + '.csv')
depot = instance_df[instance_df['id'] == 0].iloc[0]
orders = instance_df[instance_df['id'] != 0].sort_values('window_start').to_dict('records')

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

solution = glouton_simple(orders, vehicles_df)
solution.to_csv('routes_instance_' + s + '.csv', index=False)