import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Set
import time
from dataclasses import dataclass
from copy import deepcopy
import random

# Constantes
EARTH_RADIUS = 6.371e6
TWO_PI_OVER_360 = 2 * np.pi / 360
T_PERIOD = 86400  # 1 day in seconds
OMEGA = 2 * np.pi / T_PERIOD

@dataclass
class Order:
    id: int
    latitude: float
    longitude: float
    weight: float
    window_start: float
    window_end: float
    delivery_duration: float
    x: float = 0.0
    y: float = 0.0

@dataclass
class Vehicle:
    family: int
    max_capacity: float
    rental_cost: float
    fuel_cost: float
    radius_cost: float
    speed: float
    parking_time: float
    fourier_cos: List[float]
    fourier_sin: List[float]

@dataclass
class Route:
    family: int
    orders: List[int]
    load: float = 0.0
    cost: float = 0.0
    
class VRPTWSolver:
    def __init__(self, vehicles_df: pd.DataFrame, instance_df: pd.DataFrame):
        # Parse depot
        depot_row = instance_df[instance_df['id'] == 0].iloc[0]
        self.depot_lat = depot_row['latitude']
        self.depot_lon = depot_row['longitude']
        
        # Parse orders
        orders_df = instance_df[instance_df['id'] != 0].copy()
        self.orders = {}
        for _, row in orders_df.iterrows():
            order = Order(
                id=int(row['id']),
                latitude=row['latitude'],
                longitude=row['longitude'],
                weight=row['order_weight'],
                window_start=row['window_start'],
                window_end=row['window_end'],
                delivery_duration=row['delivery_duration']
            )
            # Convert to Cartesian
            order.x, order.y = self._latlon_to_cartesian(
                self.depot_lat, self.depot_lon,
                order.latitude, order.longitude
            )
            self.orders[order.id] = order
        
        # Parse vehicles
        self.vehicles = []
        for _, row in vehicles_df.iterrows():
            vehicle = Vehicle(
                family=int(row['family']),
                max_capacity=row['max_capacity'],
                rental_cost=row['rental_cost'],
                fuel_cost=row['fuel_cost'],
                radius_cost=row['radius_cost'],
                speed=row['speed'],
                parking_time=row['parking_time'],
                fourier_cos=[row[f'fourier_cos_{i}'] for i in range(4)],
                fourier_sin=[row[f'fourier_sin_{i}'] for i in range(4)]
            )
            self.vehicles.append(vehicle)
        
        # Precompute distance matrices
        self._precompute_distances()
        
    def _latlon_to_cartesian(self, phi_i, lambd_i, phi_j, lambd_j):
        delta_y = EARTH_RADIUS * TWO_PI_OVER_360 * (phi_j - phi_i)
        delta_x = EARTH_RADIUS * np.cos(TWO_PI_OVER_360 * self.depot_lat) * TWO_PI_OVER_360 * (lambd_j - lambd_i)
        return delta_x, delta_y
    
    def _precompute_distances(self):
        n = len(self.orders)
        order_ids = list(self.orders.keys())
        
        # Manhattan distances (for travel time)
        self.manhattan_dist = {}
        # Euclidean distances (for radius cost)
        self.euclidean_dist = {}
        
        # Depot to orders
        for oid in order_ids:
            order = self.orders[oid]
            self.manhattan_dist[(0, oid)] = abs(order.x) + abs(order.y)
            self.manhattan_dist[(oid, 0)] = self.manhattan_dist[(0, oid)]
            self.euclidean_dist[(0, oid)] = np.sqrt(order.x**2 + order.y**2)
            self.euclidean_dist[(oid, 0)] = self.euclidean_dist[(0, oid)]
        
        # Order to order
        for i, oid1 in enumerate(order_ids):
            o1 = self.orders[oid1]
            for oid2 in order_ids[i+1:]:
                o2 = self.orders[oid2]
                dx = o2.x - o1.x
                dy = o2.y - o1.y
                self.manhattan_dist[(oid1, oid2)] = abs(dx) + abs(dy)
                self.manhattan_dist[(oid2, oid1)] = self.manhattan_dist[(oid1, oid2)]
                self.euclidean_dist[(oid1, oid2)] = np.sqrt(dx**2 + dy**2)
                self.euclidean_dist[(oid2, oid1)] = self.euclidean_dist[(oid1, oid2)]
    
    def _get_time_factor(self, vehicle: Vehicle, t: float) -> float:
        """Compute time-dependent factor gamma_f(t)"""
        t_mod = t % T_PERIOD
        factor = 0.0
        for n in range(4):
            factor += vehicle.fourier_cos[n] * np.cos(n * OMEGA * t_mod)
            factor += vehicle.fourier_sin[n] * np.sin(n * OMEGA * t_mod)
        return factor
    
    def _get_travel_time(self, vehicle: Vehicle, from_id: int, to_id: int, departure_time: float) -> float:
        """Compute travel time from one location to another"""
        dist = self.manhattan_dist[(from_id, to_id)]
        base_time = dist / vehicle.speed + vehicle.parking_time
        time_factor = self._get_time_factor(vehicle, departure_time)
        return base_time * time_factor
    
    def _is_route_feasible(self, route: Route, vehicle: Vehicle) -> Tuple[bool, List[float], List[float]]:
        """Check if route is feasible and return arrival/departure times"""
        if route.load > vehicle.max_capacity:
            return False, [], []
        
        current_time = 0.0  # Start at midnight
        arrivals = []
        departures = [current_time]
        
        prev_id = 0
        for order_id in route.orders:
            order = self.orders[order_id]
            
            # Travel to order
            travel_time = self._get_travel_time(vehicle, prev_id, order_id, current_time)
            arrival = current_time + travel_time
            
            # Wait if arriving before window
            if arrival < order.window_start:
                arrival = order.window_start
            
            # Check if arrival is within window
            if arrival > order.window_end + 1e-5:  # Tolerance
                return False, [], []
            
            arrivals.append(arrival)
            departure = arrival + order.delivery_duration
            departures.append(departure)
            
            current_time = departure
            prev_id = order_id
        
        return True, arrivals, departures
    
    def _compute_route_cost(self, route: Route, vehicle: Vehicle) -> float:
        """Compute total cost of a route"""
        # Rental cost
        cost = vehicle.rental_cost
        
        # Fuel cost
        fuel_dist = self.manhattan_dist[(0, route.orders[0])]
        for i in range(len(route.orders) - 1):
            fuel_dist += self.manhattan_dist[(route.orders[i], route.orders[i+1])]
        fuel_dist += self.manhattan_dist[(route.orders[-1], 0)]
        cost += vehicle.fuel_cost * fuel_dist
        
        # Radius cost
        if len(route.orders) > 1:
            max_dist_sq = 0.0
            for i in range(len(route.orders)):
                for j in range(i+1, len(route.orders)):
                    dist = self.euclidean_dist[(route.orders[i], route.orders[j])]
                    max_dist_sq = max(max_dist_sq, dist**2)
            cost += vehicle.radius_cost * 0.5 * max_dist_sq
        
        return cost
    
    def _find_best_vehicle_for_route(self, route: Route) -> Tuple[int, float]:
        """Find the best vehicle type for a given route"""
        best_family = 0
        best_cost = float('inf')
        
        for vehicle in self.vehicles:
            if route.load <= vehicle.max_capacity:
                feasible, _, _ = self._is_route_feasible(route, vehicle)
                if feasible:
                    cost = self._compute_route_cost(route, vehicle)
                    if cost < best_cost:
                        best_cost = cost
                        best_family = vehicle.family
        
        return best_family, best_cost
    
    def _cluster_orders_by_time_and_location(self, max_cluster_size: int = 30) -> List[List[int]]:
        """Cluster orders by time windows and geographical proximity"""
        order_list = list(self.orders.values())
        
        # Sort by time window start
        order_list.sort(key=lambda o: o.window_start)
        
        clusters = []
        current_cluster = []
        
        for order in order_list:
            if not current_cluster:
                current_cluster.append(order.id)
            else:
                # Check time compatibility and cluster size
                first_order = self.orders[current_cluster[0]]
                time_compatible = (order.window_start < first_order.window_end)
                
                if time_compatible and len(current_cluster) < max_cluster_size:
                    current_cluster.append(order.id)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [order.id]
        
        if current_cluster:
            clusters.append(current_cluster)
        
        return clusters
    
    def _savings_algorithm(self, order_ids: List[int], vehicle: Vehicle) -> List[Route]:
        """Clarke-Wright savings algorithm for a subset of orders"""
        # Initialize: one route per order
        routes = []
        for oid in order_ids:
            route = Route(family=vehicle.family, orders=[oid], load=self.orders[oid].weight)
            routes.append(route)
        
        # Compute savings
        savings = []
        for i, r1 in enumerate(routes):
            for j, r2 in enumerate(routes[i+1:], i+1):
                if r1.load + r2.load <= vehicle.max_capacity:
                    # Savings = dist(0,i) + dist(j,0) - dist(i,j)
                    i_id = r1.orders[-1]
                    j_id = r2.orders[0]
                    saving = (self.manhattan_dist[(0, i_id)] + 
                             self.manhattan_dist[(j_id, 0)] - 
                             self.manhattan_dist[(i_id, j_id)])
                    savings.append((saving, i, j))
        
        # Sort by decreasing savings
        savings.sort(reverse=True)
        
        # Merge routes
        merged = set()
        for saving_val, i, j in savings:
            if i in merged or j in merged:
                continue
            
            # Try to merge route i and route j
            new_orders = routes[i].orders + routes[j].orders
            new_load = routes[i].load + routes[j].load
            
            if new_load <= vehicle.max_capacity:
                test_route = Route(family=vehicle.family, orders=new_orders, load=new_load)
                feasible, _, _ = self._is_route_feasible(test_route, vehicle)
                
                if feasible:
                    routes[i] = test_route
                    merged.add(j)
        
        # Remove merged routes
        return [r for idx, r in enumerate(routes) if idx not in merged]
    
    def _insertion_heuristic(self, unassigned: Set[int], routes: List[Route]) -> List[Route]:
        """Insert unassigned orders into existing routes or create new ones"""
        while unassigned:
            best_insertion = None
            best_cost_increase = float('inf')
            
            for order_id in list(unassigned):
                order = self.orders[order_id]
                
                # Try inserting into existing routes
                for route_idx, route in enumerate(routes):
                    vehicle = self.vehicles[route.family - 1]
                    
                    if route.load + order.weight > vehicle.max_capacity:
                        continue
                    
                    # Try all insertion positions
                    for pos in range(len(route.orders) + 1):
                        new_orders = route.orders[:pos] + [order_id] + route.orders[pos:]
                        new_route = Route(
                            family=route.family,
                            orders=new_orders,
                            load=route.load + order.weight
                        )
                        
                        feasible, _, _ = self._is_route_feasible(new_route, vehicle)
                        if feasible:
                            old_cost = self._compute_route_cost(route, vehicle)
                            new_cost = self._compute_route_cost(new_route, vehicle)
                            cost_increase = new_cost - old_cost
                            
                            if cost_increase < best_cost_increase:
                                best_cost_increase = cost_increase
                                best_insertion = ('insert', route_idx, pos, order_id)
                
                # Try creating a new route with cheapest vehicle
                for vehicle in self.vehicles:
                    if order.weight <= vehicle.max_capacity:
                        new_route = Route(family=vehicle.family, orders=[order_id], load=order.weight)
                        feasible, _, _ = self._is_route_feasible(new_route, vehicle)
                        if feasible:
                            cost = self._compute_route_cost(new_route, vehicle)
                            if cost < best_cost_increase:
                                best_cost_increase = cost
                                best_insertion = ('new', vehicle.family, order_id)
                        break
            
            if best_insertion is None:
                print(f"Warning: Cannot assign order(s): {unassigned}")
                break
            
            # Apply best insertion
            if best_insertion[0] == 'insert':
                _, route_idx, pos, order_id = best_insertion
                route = routes[route_idx]
                route.orders.insert(pos, order_id)
                route.load += self.orders[order_id].weight
            else:
                _, family, order_id = best_insertion
                routes.append(Route(
                    family=family,
                    orders=[order_id],
                    load=self.orders[order_id].weight
                ))
            
            unassigned.remove(order_id)
        
        return routes
    
    def _local_search_2opt(self, routes: List[Route], time_limit: float) -> List[Route]:
        """Improve routes using 2-opt and inter-route exchanges"""
        start_time = time.time()
        improved = True
        iteration = 0
        
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            iteration += 1
            
            # Intra-route 2-opt
            for route in routes:
                if len(route.orders) < 4:
                    continue
                
                vehicle = self.vehicles[route.family - 1]
                current_cost = self._compute_route_cost(route, vehicle)
                
                for i in range(len(route.orders) - 2):
                    for j in range(i + 2, len(route.orders)):
                        # Reverse segment [i+1, j]
                        new_orders = route.orders[:i+1] + route.orders[i+1:j+1][::-1] + route.orders[j+1:]
                        new_route = Route(family=route.family, orders=new_orders, load=route.load)
                        
                        feasible, _, _ = self._is_route_feasible(new_route, vehicle)
                        if feasible:
                            new_cost = self._compute_route_cost(new_route, vehicle)
                            if new_cost < current_cost - 0.01:
                                route.orders = new_orders
                                current_cost = new_cost
                                improved = True
                                break
                    if improved:
                        break
            
            # Inter-route relocate (move one order from one route to another)
            if time.time() - start_time < time_limit * 0.8:
                for i, route1 in enumerate(routes):
                    for j, route2 in enumerate(routes):
                        if i == j or len(route1.orders) <= 1:
                            continue
                        
                        vehicle1 = self.vehicles[route1.family - 1]
                        vehicle2 = self.vehicles[route2.family - 1]
                        
                        for pos1 in range(len(route1.orders)):
                            order_id = route1.orders[pos1]
                            order = self.orders[order_id]
                            
                            if route2.load + order.weight > vehicle2.max_capacity:
                                continue
                            
                            # Try inserting into route2
                            for pos2 in range(len(route2.orders) + 1):
                                new_orders1 = route1.orders[:pos1] + route1.orders[pos1+1:]
                                new_orders2 = route2.orders[:pos2] + [order_id] + route2.orders[pos2:]
                                
                                new_route1 = Route(family=route1.family, orders=new_orders1, 
                                                 load=route1.load - order.weight)
                                new_route2 = Route(family=route2.family, orders=new_orders2,
                                                 load=route2.load + order.weight)
                                
                                feasible1, _, _ = self._is_route_feasible(new_route1, vehicle1)
                                feasible2, _, _ = self._is_route_feasible(new_route2, vehicle2)
                                
                                if feasible1 and feasible2:
                                    old_cost = (self._compute_route_cost(route1, vehicle1) +
                                              self._compute_route_cost(route2, vehicle2))
                                    new_cost = (self._compute_route_cost(new_route1, vehicle1) +
                                              self._compute_route_cost(new_route2, vehicle2))
                                    
                                    if new_cost < old_cost - 0.01:
                                        route1.orders = new_orders1
                                        route1.load = new_route1.load
                                        route2.orders = new_orders2
                                        route2.load = new_route2.load
                                        improved = True
                                        break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        return routes
    
    def _optimize_vehicle_selection(self, routes: List[Route]) -> List[Route]:
        """For each route, try to use a cheaper vehicle if possible"""
        for route in routes:
            best_family, best_cost = self._find_best_vehicle_for_route(route)
            route.family = best_family
            route.cost = best_cost
        return routes
    
    def solve(self, time_limit: float = 540) -> List[Route]:
        """Main solving function"""
        start_time = time.time()
        n_orders = len(self.orders)
        print(f"Solving with {n_orders} orders and {len(self.vehicles)} vehicle types")
        
        # Phase 1: Initial clustering
        print("Phase 1: Clustering orders...")
        clusters = self._cluster_orders_by_time_and_location(max_cluster_size=25)
        print(f"Created {len(clusters)} clusters")
        
        # Phase 2: Build initial routes using savings algorithm
        print("Phase 2: Building initial routes...")
        all_routes = []
        
        for cluster_idx, cluster in enumerate(clusters):
            # Use medium vehicle for initial construction
            vehicle = self.vehicles[min(1, len(self.vehicles)-1)]
            cluster_routes = self._savings_algorithm(cluster, vehicle)
            all_routes.extend(cluster_routes)
            
            if (time.time() - start_time) > time_limit * 0.3:
                break
        
        print(f"Initial routes: {len(all_routes)}")
        
        # Phase 3: Handle unassigned orders
        assigned_orders = set()
        for route in all_routes:
            assigned_orders.update(route.orders)
        unassigned = set(self.orders.keys()) - assigned_orders
        
        if unassigned:
            print(f"Phase 3: Inserting {len(unassigned)} unassigned orders...")
            all_routes = self._insertion_heuristic(unassigned, all_routes)
        
        # Phase 4: Local search improvement
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 30:
            print(f"Phase 4: Local search (time remaining: {remaining_time:.1f}s)...")
            all_routes = self._local_search_2opt(all_routes, remaining_time * 0.8)
        
        # Phase 5: Optimize vehicle selection
        print("Phase 5: Optimizing vehicle selection...")
        all_routes = self._optimize_vehicle_selection(all_routes)
        
        # Remove empty routes
        all_routes = [r for r in all_routes if r.orders]
        
        # Compute total cost
        total_cost = sum(r.cost for r in all_routes)
        print(f"\nFinal solution: {len(all_routes)} routes, total cost: {total_cost:.2f}€")
        print(f"Total time: {time.time() - start_time:.2f}s")
        
        return all_routes
    
    def save_solution(self, routes: List[Route], filename: str):
        """Save solution to CSV file"""
        max_orders = max(len(r.orders) for r in routes) if routes else 0
        
        # Build dataframe
        data = []
        for route in routes:
            row = {'family': int(route.family)}
            for i, order_id in enumerate(route.orders, 1):
                row[f'order_{i}'] = int(order_id)
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Ensure all order columns exist
        for i in range(1, max_orders + 1):
            if f'order_{i}' not in df.columns:
                df[f'order_{i}'] = None
        
        # Reorder columns
        cols = ['family'] + [f'order_{i}' for i in range(1, max_orders + 1)]
        df = df[cols]
        
        # Convert to Int64 (pandas nullable integer type to handle NaN)
        df['family'] = df['family'].astype('Int64')
        for i in range(1, max_orders + 1):
            df[f'order_{i}'] = df[f'order_{i}'].astype('Int64')
        
        # Save
        df.to_csv(filename, index=False)
        print(f"Solution saved to {filename}")


def main():
    import sys
    
    # Parse instance number
    if len(sys.argv) > 1:
        instance = sys.argv[1]
    else:
        instance = '07'
    
    print(f"Loading instance {instance}...")
    
    # Load data
    vehicles_df = pd.read_csv('instances/vehicles.csv')
    instance_df = pd.read_csv(f'instances/instance_{instance}.csv')
    
    # Solve
    solver = VRPTWSolver(vehicles_df, instance_df)
    routes = solver.solve(time_limit=540)  # 9 minutes max
    
    # Save
    solver.save_solution(routes, f'routes_{instance}.csv')
    
    print("\nDone!")


if __name__ == '__main__':
    main()