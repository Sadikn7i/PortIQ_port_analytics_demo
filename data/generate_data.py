import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

print("=" * 60)
print("GENERATING GHIH PORT ANALYTICS DEMO DATA")
print("=" * 60)

# ============================================================================
# 1. TRAFFIC FORECASTING DATA
# ============================================================================
print("\nGenerating traffic forecasting data...")

start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 12, 31)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

base_volume = 400
trend = np.linspace(0, 100, len(date_range))
seasonal = 50 * np.sin(2 * np.pi * np.arange(len(date_range)) / 365)
weekly = 30 * np.sin(2 * np.pi * np.arange(len(date_range)) / 7)
noise = np.random.normal(0, 30, len(date_range))

containers = base_volume + trend + seasonal + weekly + noise
containers = np.maximum(containers, 50).astype(int)

weekday = pd.Series(date_range).dt.dayofweek
weekend_mask = weekday >= 5
containers[weekend_mask] = (containers[weekend_mask] * 0.7).astype(int)

traffic_df = pd.DataFrame({
    'date': date_range,
    'containers_arrived': containers,
    'day_of_week': pd.Series(date_range).dt.day_name(),
    'month': pd.Series(date_range).dt.month,
    'is_weekend': weekend_mask,
    'year': pd.Series(date_range).dt.year
})

print(f"SUCCESS: Created {len(traffic_df)} days of traffic data")

# ============================================================================
# 2. DWELL TIME PREDICTION DATA
# ============================================================================
print("\nGenerating dwell time prediction data...")

n_containers = 5000

origins = ['Ethiopia', 'China', 'UAE', 'Saudi Arabia', 'India', 'Turkey', 'Kenya', 'Somalia']
origin_weights = [0.35, 0.20, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03]

cargo_types = ['Electronics', 'Textiles', 'Food', 'Machinery', 'Chemicals', 'Vehicles', 'General']
cargo_weights = [0.20, 0.25, 0.15, 0.10, 0.08, 0.07, 0.15]

customs_status = ['Fast', 'Normal', 'Inspection', 'Delayed']
customs_weights = [0.40, 0.45, 0.10, 0.05]

container_ids = [f'CNT{str(i).zfill(6)}' for i in range(1, n_containers + 1)]
container_origins = np.random.choice(origins, n_containers, p=origin_weights)
container_cargo = np.random.choice(cargo_types, n_containers, p=cargo_weights)
container_customs = np.random.choice(customs_status, n_containers, p=customs_weights)

dwell_times = []
for origin, customs, cargo in zip(container_origins, container_customs, container_cargo):
    if origin == 'Ethiopia':
        base = 50
    elif origin in ['UAE', 'Saudi Arabia']:
        base = 95
    elif origin in ['China', 'India']:
        base = 85
    else:
        base = 75
    
    if customs == 'Fast':
        multiplier = 0.7
    elif customs == 'Normal':
        multiplier = 1.0
    elif customs == 'Inspection':
        multiplier = 1.5
    else:
        multiplier = 2.0
    
    if cargo == 'Food':
        multiplier *= 0.6
    elif cargo == 'Machinery':
        multiplier *= 1.2
    
    dwell = base * multiplier + np.random.normal(0, 15)
    dwell = max(12, min(300, dwell))
    dwell_times.append(round(dwell, 1))

weights = np.random.uniform(5, 40, n_containers).round(1)
truck_available = np.random.choice(['Yes', 'No', 'Delayed'], n_containers, p=[0.70, 0.20, 0.10])
arrival_dates = pd.date_range('2024-01-01', '2024-12-31', periods=n_containers)

dwell_df = pd.DataFrame({
    'container_id': container_ids,
    'origin_country': container_origins,
    'cargo_type': container_cargo,
    'customs_status': container_customs,
    'weight_tons': weights,
    'truck_available': truck_available,
    'arrival_date': arrival_dates,
    'dwell_time_hours': dwell_times
})

print(f"SUCCESS: Created {len(dwell_df)} container records")

# ============================================================================
# 3. ROUTE OPTIMIZATION DATA
# ============================================================================
print("\nGenerating route optimization data...")

locations = ['Doraleh_Port', 'DIFTZ', 'Damerjog_Port', 'City_Center', 'Ethiopia_Border', 'Airport']

distance_matrix = {
    'Doraleh_Port': {'DIFTZ': 15, 'Damerjog_Port': 25, 'City_Center': 10, 'Ethiopia_Border': 100, 'Airport': 12},
    'DIFTZ': {'Doraleh_Port': 15, 'Damerjog_Port': 35, 'City_Center': 20, 'Ethiopia_Border': 95, 'Airport': 22},
    'Damerjog_Port': {'Doraleh_Port': 25, 'DIFTZ': 35, 'City_Center': 22, 'Ethiopia_Border': 110, 'Airport': 28},
    'City_Center': {'Doraleh_Port': 10, 'DIFTZ': 20, 'Damerjog_Port': 22, 'Ethiopia_Border': 105, 'Airport': 8},
    'Ethiopia_Border': {'Doraleh_Port': 100, 'DIFTZ': 95, 'Damerjog_Port': 110, 'City_Center': 105, 'Airport': 102},
    'Airport': {'Doraleh_Port': 12, 'DIFTZ': 22, 'Damerjog_Port': 28, 'City_Center': 8, 'Ethiopia_Border': 102}
}

n_routes = 1000

truck_ids = [f'TRK{str(i).zfill(4)}' for i in range(1, n_routes + 1)]
route_origins = []
route_destinations = []
distances = []
traffic_levels = []
fuel_costs = []
time_taken = []

for _ in range(n_routes):
    origin = random.choice(locations)
    destination = random.choice([loc for loc in locations if loc != origin])
    
    distance = distance_matrix[origin].get(destination, 0)
    
    traffic = random.choice(['Low', 'Medium', 'High'])
    
    if traffic == 'Low':
        speed = 60
    elif traffic == 'Medium':
        speed = 45
    else:
        speed = 30
    
    time_min = (distance / speed * 60) + np.random.normal(0, 5)
    time_min = max(10, time_min)
    
    fuel = distance * 0.80
    if traffic == 'High':
        fuel *= 1.3
    elif traffic == 'Medium':
        fuel *= 1.1
    
    route_origins.append(origin)
    route_destinations.append(destination)
    distances.append(distance)
    traffic_levels.append(traffic)
    time_taken.append(round(time_min, 1))
    fuel_costs.append(round(fuel, 2))

cargo_weights = np.random.uniform(5, 30, n_routes).round(1)

route_df = pd.DataFrame({
    'truck_id': truck_ids,
    'origin': route_origins,
    'destination': route_destinations,
    'distance_km': distances,
    'cargo_weight_tons': cargo_weights,
    'traffic_level': traffic_levels,
    'time_taken_minutes': time_taken,
    'fuel_cost_usd': fuel_costs
})

print(f"SUCCESS: Created {len(route_df)} route records")

# ============================================================================
# 4. ANOMALY DETECTION DATA
# ============================================================================
print("\nAdding anomalies to container data...")

anomaly_df = dwell_df.copy()
n_anomalies = int(len(anomaly_df) * 0.05)
anomaly_indices = np.random.choice(anomaly_df.index, n_anomalies, replace=False)

anomaly_df['is_anomaly'] = False
anomaly_df.loc[anomaly_indices, 'is_anomaly'] = True

for idx in anomaly_indices:
    anomaly_type = random.choice(['fast', 'slow', 'cost'])
    
    if anomaly_type == 'fast':
        anomaly_df.loc[idx, 'dwell_time_hours'] = np.random.uniform(8, 15)
    elif anomaly_type == 'slow':
        anomaly_df.loc[idx, 'dwell_time_hours'] = np.random.uniform(200, 300)

print(f"SUCCESS: Added {n_anomalies} anomalies to data")

# ============================================================================
# SAVE ALL DATA
# ============================================================================
print("\nSaving data files...")

traffic_df.to_csv('traffic_data.csv', index=False)
print("SAVED: traffic_data.csv")

dwell_df.to_csv('dwell_time_data.csv', index=False)
print("SAVED: dwell_time_data.csv")

route_df.to_csv('route_data.csv', index=False)
print("SAVED: route_data.csv")

anomaly_df.to_csv('anomaly_data.csv', index=False)
print("SAVED: anomaly_data.csv")

print("\n" + "=" * 60)
print("DATA GENERATION SUMMARY")
print("=" * 60)
print(f"Traffic forecasting data: {len(traffic_df)} days")
print(f"Dwell time prediction data: {len(dwell_df)} containers")
print(f"Route optimization data: {len(route_df)} routes")
print(f"Anomaly detection data: {len(anomaly_df)} containers ({n_anomalies} anomalies)")
print("\nAll data generated successfully!")
print("=" * 60)