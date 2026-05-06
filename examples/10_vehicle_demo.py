"""Electric Vehicle Environment Demo

Demonstrates the VehicleEnv usage with various scenarios and integration with GridEnv.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from powerzoo.envs.grid import DistGridEnv
from powerzoo.envs.resource import BatteryEnv, VehicleEnv

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'x10_vehicle_demo')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def print_section(title):
    """Print section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# Scenario 1: Basic Single-Trip Vehicle
# ============================================================
print_section("Scenario 1: Basic Single-Trip Vehicle (Default)")

# Create a vehicle with default single trip schedule
vehicle = VehicleEnv(
    E_max_kWh=60.0,
    soc_init=0.5,
    p_charge_max_kW=7.0,
    delta_t_minutes=60
)

print("\nVehicle Configuration:")
print(f"  Battery: {vehicle.capacity_mwh * 1000:.0f} kWh")
print(f"  Charging: {vehicle.p_charge_max_mw * 1000:.1f} kW (Level 2)")
print(f"  Initial SOC: {vehicle.soc_init:.1%}")

print("\nCommute Schedule:")
for i, trip in enumerate(vehicle.commute_schedule, 1):
    print(f"  Trip {i}: Depart {trip['departure']:.1f}h -> "
          f"Arrive {trip['arrival']:.1f}h, "
          f"Energy {trip['energy'] * 1000:.1f} kWh")

vehicle.reset()
print(f"\nInitial State: SOC={vehicle.soc:.3f}, Home={vehicle.is_home}")

# Simulate a few hours
print("\nSimulation (first 10 hours):")
for hour in range(24):
    action = -0.007 if vehicle.is_home else 0.0  # Charge when home
    vehicle.step(action)
    info = vehicle.status()
    home_icon = "home" if info['is_home'] else "away"
    print(f"  {hour:2d}h: {home_icon} SOC={info['soc']:.3f}, "
          f"P={info['current_p_mw'] * 1000:5.1f}kW")

# ============================================================
# Scenario 2: Multiple Daily Trips
# ============================================================
print_section("Scenario 2: Complex Daily Pattern with Multiple Trips")

# Create a vehicle with multiple trips (work + lunch + errands)
vehicle_multi = VehicleEnv(
    E_max_kWh=60.0,
    soc_init=0.8,
    p_charge_max_kW=7.0,
    commute_schedule=[
        {'departure': 8.0, 'arrival': 9.0, 'energy_kWh': 10.0},  # Morning commute to work
        {'departure': 12.0, 'arrival': 13.0, 'energy_kWh': 5.0},  # Lunch break trip
        {'departure': 17.0, 'arrival': 18.0, 'energy_kWh': 8.0},  # Evening errands
        {'departure': 20.0, 'arrival': 21.0, 'energy_kWh': 10.0},  # Return home
    ],
    delta_t_minutes=60.0,  # 1-hour time steps
)

print("\nVehicle Configuration:")
print(f"  Number of trips: {len(vehicle_multi.commute_schedule)}")
print(f"  Time step: {vehicle_multi.delta_t_minutes:.0f} minutes")

print("\nCommute Schedule:")
for i, trip in enumerate(vehicle_multi.commute_schedule, 1):
    print(f"  Trip {i}: {trip['departure']:5.1f}h -> {trip['arrival']:5.1f}h, "
          f"{trip['energy'] * 1000:5.1f} kWh")

total_energy = sum(t['energy'] for t in vehicle_multi.commute_schedule) * 1000
print(f"\n  Total daily commute energy: {total_energy:.1f} kWh")

# Run 24-hour simulation
vehicle_multi.reset()
print("\n24-Hour Simulation:")
print("  Time | Home | SOC   | Power  | Status")
print("  -----|------|-------|--------|-------")

for hour in range(24):
    action = -0.007 if vehicle_multi.is_home else 0.0
    vehicle_multi.step(action)
    info = vehicle_multi.status()

    home_str = " Yes " if info['is_home'] else " No  "
    ready = "OK" if info['departure_ready'] else "LOW"
    home_icon = "home" if info['is_home'] else "away"
    print(f"  {hour:2d}h {home_icon}|{home_str}| {info['soc']:.3f} | "
          f"{info['current_p_mw'] * 1000:5.1f}kW | {ready}")

# ============================================================
# Scenario 3: Integration with Distribution Grid
# ============================================================
print_section("Scenario 3: Multiple EVs in Distribution Grid")


def plot_graph(grid, state):
    # Visualization: Plot grid with 3 EVs after 24 hours
    # print("\nVisualizing grid with 3 EVs...")
    plotter = grid.case.plotter
    plotter.pos = None  # Reset layout cache

    fig, ax = plotter.plot_power_flow_with_resources(
        nodes_df=state['nodes'],
        lines_df=state['lines'],
        grid_env=grid,
        figsize=(18, 10),
        layout='feeder',
        v_min=0.90,
        v_max=1.10,
        flow_label_format='pq',
        title='IEEE 33-Bus with 3 Electric Vehicles (After 24h)',
        resource_offset=(0.2, 1.4),
        resource_size=0.9
    )

    plot_path = f'{OUTPUT_DIR}/3evs_at_{grid.time_step - 1}h.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    # print(f"  Saved: {plot_path}")
    plt.close()


# Create distribution grid
grid = DistGridEnv()

# Add multiple EVs with different patterns
ev1 = VehicleEnv(
    parent=grid,
    bus_id=10,
    E_max_kWh=60.0,
    soc_init=0.5,
    commute_schedule=[
        {'departure': 8.0, 'arrival': 9.0, 'energy_kWh': 8.0},
        {'departure': 18.0, 'arrival': 19.0, 'energy_kWh': 8.0},
    ],
    delta_t_minutes=60.0,
    p_charge_max_kW=7,
)

ev2 = VehicleEnv(
    parent=grid,
    bus_id=15,
    E_max_kWh=40.0,
    soc_init=0.7,
    commute_schedule=[
        {'departure': 7.0, 'arrival': 8.0, 'energy_kWh': 5.0},
        {'departure': 17.0, 'arrival': 18.0, 'energy_kWh': 5.0},
    ],
    delta_t_minutes=60.0,
)

ev3 = VehicleEnv(
    parent=grid,
    bus_id=20,
    E_max_kWh=75.0,
    soc_init=0.6,
    commute_schedule=[
        {'departure': 9.0, 'arrival': 10.0, 'energy_kWh': 12.0},
        {'departure': 19.0, 'arrival': 20.0, 'energy_kWh': 12.0},
    ],
    delta_t_minutes=60.0,
)

print(f"\nGrid Setup:")
print(f"  Number of EVs: 3")
print(f"  EV1 ({ev1.resource_id}): Bus {ev1.bus_id}, {ev1.capacity_mwh * 1000:.0f} kWh")
print(f"  EV2 ({ev2.resource_id}): Bus {ev2.bus_id}, {ev2.capacity_mwh * 1000:.0f} kWh")
print(f"  EV3 ({ev3.resource_id}): Bus {ev3.bus_id}, {ev3.capacity_mwh * 1000:.0f} kWh")

# Reset grid
state, info = grid.reset(day_id=10)
print(f"\nGrid Initial State:")
print(f"  Day ID: {grid.day_id}")
print(f"  Voltage range: [{state['safety_info']['v_min_actual']:.4f}, "
      f"{state['safety_info']['v_max_actual']:.4f}] p.u.")
print(f"  Network loss: {state['p_loss_MW']:.4f} MW")

# Simulate coordinated charging - 24 hours
print("\nCoordinated Charging - 24 Hour Simulation:")
print("  Time | EV1 SOC | EV2 SOC | EV3 SOC | Total P | Status")
print("  -----|---------|---------|---------|---------|--------")

plot_periods = [1, 9, 19, 20]
for hour in range(24):
    # Charge all EVs at home
    action = {
        ev1.resource_id: -ev1.p_charge_max_mw / 1000 if ev1.is_home else 0.0,
        ev2.resource_id: -0.005 if ev2.is_home else 0.0,
        ev3.resource_id: -0.010 if ev3.is_home else 0.0,
    }

    state, reward, done, truncated, info = grid.step(action)

    total_p = sum([ev1.current_p_mw, ev2.current_p_mw, ev3.current_p_mw]) * 1000  # to kW
    status = (
        f"{'H' if ev1.is_home else 'A'}"
        f"{'H' if ev2.is_home else 'A'}"
        f"{'H' if ev3.is_home else 'A'}"
    )
    print(f"  {hour:2d}h  | {ev1.soc:.3f}  | {ev2.soc:.3f}  | {ev3.soc:.3f}  | "
          f"{total_p:6.1f}kW | {status}")
    if hour in plot_periods:
        plot_graph(grid, state)

# ============================================================
# Scenario 4: Smart Charging Strategy
# ============================================================
print_section("Scenario 4: Smart Charging with Price Signal")

# Create vehicle for smart charging
vehicle_smart = VehicleEnv(
    E_max_kWh=60.0,
    soc_init=0.4,
    soc_departure_min=0.85,
    p_charge_max_kW=7.0,
    commute_schedule=[
        {'departure': 8.0, 'arrival': 18.0, 'energy_kWh': 15.0},
    ],
    delta_t_minutes=60.0,  # 1-hour steps
)

# Simulated electricity price ($/kWh)
price_profile = np.array([
    0.05, 0.05, 0.05, 0.05, 0.05, 0.05,  # 0-5h: Low (night)
    0.08, 0.10, 0.15, 0.12, 0.10, 0.09,  # 6-11h: Rising (morning)
    0.10, 0.12, 0.15, 0.18, 0.20, 0.18,  # 12-17h: High (afternoon)
    0.15, 0.12, 0.10, 0.08, 0.07, 0.06,  # 18-23h: Declining (evening)
])


def smart_charging_strategy(vehicle, price, time_to_departure):
    """Smart charging based on price and SOC"""
    if not vehicle.is_home:
        return 0.0

    # Priority 1: Must charge if SOC below departure requirement and time is running out
    if vehicle.soc < vehicle.soc_departure_min and time_to_departure < 3:
        return -0.007  # Maximum charge

    # Priority 2: Charge when price is low
    if price < 0.08 and vehicle.soc < 0.9:
        return -0.007

    # Priority 3: V2G when price is very high and SOC is sufficient
    if price > 0.15 and vehicle.soc > vehicle.soc_departure_min + 0.1:
        return 0.005  # Discharge to grid

    return 0.0


vehicle_smart.reset()
print("\nSmart Charging Strategy:")
print("  - Charge at night (low price)")
print("  - V2G during peak hours (high price)")
print("  - Ensure departure readiness")

print("\n24-Hour Smart Charging:")
print("  Time | Price | SOC   | Action    | Ready")
print("  -----|-------|-------|-----------|------")

total_cost = 0.0
total_revenue = 0.0

for hour in range(24):
    price = price_profile[hour]
    time_to_departure = (8 - hour) % 24

    action = smart_charging_strategy(vehicle_smart, price, time_to_departure)
    vehicle_smart.step(action)
    info = vehicle_smart.status()

    # Calculate cost/revenue
    energy_kwh = abs(info['current_p_mw']) * 1000 * 1.0  # 1 hour
    if info['current_p_mw'] < 0:  # Charging
        total_cost += energy_kwh * price
        action_str = f"Charge {abs(info['current_p_mw']) * 1000:.1f}kW"
    elif info['current_p_mw'] > 0:  # V2G
        total_revenue += energy_kwh * price
        action_str = f"V2G {info['current_p_mw'] * 1000:.1f}kW"
    else:
        action_str = "Idle"

    ready = "OK" if info['departure_ready'] else "LOW"
    print(f"  {hour:2d}h  | ${price:.2f} | {info['soc']:.3f} | "
          f"{action_str:11s} | {ready}")

print(f"\nEconomic Summary:")
print(f"  Total charging cost: ${total_cost:.2f}")
print(f"  Total V2G revenue: ${total_revenue:.2f}")
print(f"  Net cost: ${total_cost - total_revenue:.2f}")
print(f"  Final SOC: {vehicle_smart.soc:.1%}")
print(f"  Departure ready: {'Yes' if vehicle_smart.check_departure_ready() else 'No'}")

print("\n" + "=" * 70)
print("Demo Complete!")
print("=" * 70)
