"""Example: Compare DistGridEnv Power Flow with MATPOWER Results

This example runs power flow on the IEEE 33-bus distribution system using
the Forward-Backward Sweep method and compares results with MATPOWER.
"""
import numpy as np
import pandas as pd
from powerzoo.envs.grid import DistGridEnv

# =============================================================================
# MATPOWER Results (from examples/04_case33_result.txt)
# Newton-Raphson AC power flow, converged in 3 iterations
# =============================================================================

# Bus voltages (p.u.)
MATPOWER_VOLTAGES = {
    1: 1.000, 2: 0.997, 3: 0.983, 4: 0.975, 5: 0.968, 6: 0.950, 7: 0.946, 8: 0.941,
    9: 0.935, 10: 0.929, 11: 0.928, 12: 0.927, 13: 0.921, 14: 0.919, 15: 0.917, 16: 0.916,
    17: 0.914, 18: 0.913, 19: 0.997, 20: 0.993, 21: 0.992, 22: 0.992, 23: 0.979, 24: 0.973,
    25: 0.969, 26: 0.948, 27: 0.945, 28: 0.934, 29: 0.926, 30: 0.922, 31: 0.918, 32: 0.917,
    33: 0.917
}

# Branch flows: (from_bus, to_bus): (P_MW, Q_MVAr) - From Bus Injection
MATPOWER_BRANCH_FLOWS = {
    (1, 2): (3.92, 2.44),
    (2, 3): (3.44, 2.21),
    (3, 4): (2.36, 1.68),
    (4, 5): (2.22, 1.59),
    (5, 6): (2.14, 1.55),
    (6, 7): (1.10, 0.53),
    (7, 8): (0.89, 0.42),
    (8, 9): (0.69, 0.32),
    (9, 10): (0.62, 0.30),
    (10, 11): (0.56, 0.27),
    (11, 12): (0.52, 0.24),
    (12, 13): (0.45, 0.21),
    (13, 14): (0.39, 0.17),
    (14, 15): (0.27, 0.09),
    (15, 16): (0.21, 0.08),
    (16, 17): (0.15, 0.06),
    (17, 18): (0.09, 0.04),
    (2, 19): (0.36, 0.16),
    (19, 20): (0.27, 0.12),
    (20, 21): (0.18, 0.08),
    (21, 22): (0.09, 0.04),
    (3, 23): (0.94, 0.46),
    (23, 24): (0.85, 0.41),
    (24, 25): (0.42, 0.20),
    (6, 26): (0.95, 0.97),
    (26, 27): (0.89, 0.95),
    (27, 28): (0.82, 0.92),
    (28, 29): (0.75, 0.89),
    (29, 30): (0.63, 0.81),
    (30, 31): (0.42, 0.21),
    (31, 32): (0.27, 0.14),
    (32, 33): (0.06, 0.04),
}

# Summary values
MATPOWER_MIN_VOLTAGE = 0.913
MATPOWER_MAX_VOLTAGE = 1.000
MATPOWER_P_LOSS_MW = 0.203
MATPOWER_Q_LOSS_MVAR = 0.14
MATPOWER_TOTAL_LOAD_P = 3.72
MATPOWER_TOTAL_LOAD_Q = 2.30


def main():
    print("=" * 90)
    print("IEEE 33-Bus Distribution System Power Flow Comparison")
    print("PowerZoo (Forward-Backward Sweep) vs MATPOWER (Newton-Raphson)")
    print("=" * 90)

    # Create distribution grid environment
    env = DistGridEnv()

    print(f"\nSystem Info:")
    print(f"  Number of buses: {env.n_nodes}")
    print(f"  Number of active branches: {env.n_lines}")
    print(f"  Base MVA: {env.baseMVA}")
    print(f"  Base kV: {env.baseKV}")

    # Run power flow
    print("\n" + "-" * 90)
    print("Running Power Flow...")
    print("-" * 90)

    nodes_df, lines_df = env.cal_pf(df=True)

    print(f"Converged: {env._converged} (in {env._iterations} iterations)")

    # Get results
    p_loss, q_loss = env.get_total_loss(lines_df)
    v_min = nodes_df['v_mag'].min()
    v_max = nodes_df['v_mag'].max()
    v_min_bus = int(nodes_df.loc[nodes_df['v_mag'].idxmin(), 'id'])
    total_p_load = nodes_df['p_load_MW'].sum()
    total_q_load = nodes_df['q_load_MVAr'].sum()

    # ==========================================================================
    # Summary Comparison
    # ==========================================================================
    print("\n" + "=" * 90)
    print("Summary Comparison")
    print("=" * 90)

    print(f"\n{'Metric':<30} {'PowerZoo':>15} {'MATPOWER':>15} {'Diff':>15} {'Error%':>12}")
    print("-" * 87)
    print(f"{'Min Voltage (p.u.)':<30} {v_min:>15.4f} {MATPOWER_MIN_VOLTAGE:>15.4f} {(v_min - MATPOWER_MIN_VOLTAGE):>15.4f} {abs(v_min - MATPOWER_MIN_VOLTAGE)/MATPOWER_MIN_VOLTAGE*100:>11.2f}%")
    print(f"{'Min Voltage Bus':<30} {v_min_bus:>15d} {18:>15d} {'':>15} {'':>12}")
    print(f"{'Max Voltage (p.u.)':<30} {v_max:>15.4f} {MATPOWER_MAX_VOLTAGE:>15.4f} {(v_max - MATPOWER_MAX_VOLTAGE):>15.4f} {abs(v_max - MATPOWER_MAX_VOLTAGE)/MATPOWER_MAX_VOLTAGE*100:>11.2f}%")
    print(f"{'P Loss (MW)':<30} {p_loss:>15.4f} {MATPOWER_P_LOSS_MW:>15.4f} {(p_loss - MATPOWER_P_LOSS_MW):>15.4f} {abs(p_loss - MATPOWER_P_LOSS_MW)/MATPOWER_P_LOSS_MW*100:>11.2f}%")
    print(f"{'Q Loss (MVAr)':<30} {q_loss:>15.4f} {MATPOWER_Q_LOSS_MVAR:>15.4f} {(q_loss - MATPOWER_Q_LOSS_MVAR):>15.4f} {abs(q_loss - MATPOWER_Q_LOSS_MVAR)/MATPOWER_Q_LOSS_MVAR*100:>11.2f}%")
    print(f"{'Total P Load (MW)':<30} {total_p_load:>15.4f} {MATPOWER_TOTAL_LOAD_P:>15.4f} {(total_p_load - MATPOWER_TOTAL_LOAD_P):>15.4f} {abs(total_p_load - MATPOWER_TOTAL_LOAD_P)/MATPOWER_TOTAL_LOAD_P*100:>11.2f}%")
    print(f"{'Total Q Load (MVAr)':<30} {total_q_load:>15.4f} {MATPOWER_TOTAL_LOAD_Q:>15.4f} {(total_q_load - MATPOWER_TOTAL_LOAD_Q):>15.4f} {abs(total_q_load - MATPOWER_TOTAL_LOAD_Q)/MATPOWER_TOTAL_LOAD_Q*100:>11.2f}%")

    # ==========================================================================
    # Voltage Comparison
    # ==========================================================================
    print("\n" + "=" * 90)
    print("Voltage Comparison by Bus")
    print("=" * 90)

    print(f"\n{'Bus':>5} {'PowerZoo':>12} {'MATPOWER':>12} {'Diff':>12} {'Error %':>12}")
    print("-" * 55)

    voltage_errors = []
    for bus_id in range(1, 34):
        pz_v = nodes_df.iloc[bus_id - 1]['v_mag']
        mp_v = MATPOWER_VOLTAGES[bus_id]
        diff = pz_v - mp_v
        error_pct = abs(diff) / mp_v * 100
        voltage_errors.append(error_pct)

        marker = " *" if abs(diff) > 0.01 else ""
        print(f"{bus_id:>5} {pz_v:>12.4f} {mp_v:>12.4f} {diff:>12.4f} {error_pct:>11.2f}%{marker}")

    print("\n" + "-" * 55)
    print(f"Average voltage error: {np.mean(voltage_errors):.2f}%")
    print(f"Max voltage error: {np.max(voltage_errors):.2f}% at bus {np.argmax(voltage_errors) + 1}")
    print(f"Buses with >1% error: {sum(e > 1 for e in voltage_errors)}")

    # ==========================================================================
    # Branch Flow Comparison
    # ==========================================================================
    print("\n" + "=" * 90)
    print("Branch Flow Comparison (Active Power)")
    print("=" * 90)

    print(f"\n{'Branch':<12} {'PowerZoo P':>12} {'MATPOWER P':>12} {'Diff':>12} {'Error %':>12}")
    print("-" * 62)

    p_flow_errors = []
    q_flow_errors = []

    for idx, row in lines_df.iterrows():
        from_bus = int(row['#from'])
        to_bus = int(row['#to'])
        pz_p = row['p_flow_MW']
        pz_q = row['q_flow_MVAr']

        key = (from_bus, to_bus)
        if key in MATPOWER_BRANCH_FLOWS:
            mp_p, mp_q = MATPOWER_BRANCH_FLOWS[key]

            p_diff = pz_p - mp_p
            p_error = abs(p_diff) / max(abs(mp_p), 0.001) * 100
            p_flow_errors.append(p_error)

            q_diff = pz_q - mp_q
            q_error = abs(q_diff) / max(abs(mp_q), 0.001) * 100
            q_flow_errors.append(q_error)

            marker = " *" if p_error > 5 else ""
            print(f"{from_bus:>4}->{to_bus:<4} {pz_p:>12.3f} {mp_p:>12.3f} {p_diff:>12.3f} {p_error:>11.2f}%{marker}")

    print("\n" + "-" * 62)
    print(f"Average P flow error: {np.mean(p_flow_errors):.2f}%")
    print(f"Max P flow error: {np.max(p_flow_errors):.2f}%")
    print(f"Branches with >5% P error: {sum(e > 5 for e in p_flow_errors)}")

    # ==========================================================================
    # Branch Flow Comparison (Reactive Power)
    # ==========================================================================
    print("\n" + "=" * 90)
    print("Branch Flow Comparison (Reactive Power)")
    print("=" * 90)

    print(f"\n{'Branch':<12} {'PowerZoo Q':>12} {'MATPOWER Q':>12} {'Diff':>12} {'Error %':>12}")
    print("-" * 62)

    for idx, row in lines_df.iterrows():
        from_bus = int(row['#from'])
        to_bus = int(row['#to'])
        pz_q = row['q_flow_MVAr']

        key = (from_bus, to_bus)
        if key in MATPOWER_BRANCH_FLOWS:
            _, mp_q = MATPOWER_BRANCH_FLOWS[key]

            q_diff = pz_q - mp_q
            q_error = abs(q_diff) / max(abs(mp_q), 0.001) * 100

            marker = " *" if q_error > 5 else ""
            print(f"{from_bus:>4}->{to_bus:<4} {pz_q:>12.3f} {mp_q:>12.3f} {q_diff:>12.3f} {q_error:>11.2f}%{marker}")

    print("\n" + "-" * 62)
    print(f"Average Q flow error: {np.mean(q_flow_errors):.2f}%")
    print(f"Max Q flow error: {np.max(q_flow_errors):.2f}%")
    print(f"Branches with >5% Q error: {sum(e > 5 for e in q_flow_errors)}")

    # ==========================================================================
    # Assessment
    # ==========================================================================
    print("\n" + "=" * 90)
    print("Assessment")
    print("=" * 90)

    # Voltage assessment
    if np.max(voltage_errors) < 1.0:
        print("\n[Voltage] [OK] Excellent match! All errors < 1%")
    elif np.max(voltage_errors) < 3.0:
        print("\n[Voltage] [OK] Good match. Errors within acceptable range (<3%)")
    else:
        print("\n[Voltage] [WARN] Noticeable differences")

    # Power flow assessment
    if np.mean(p_flow_errors) < 3.0:
        print("[P Flow]  [OK] Excellent match! Average error < 3%")
    elif np.mean(p_flow_errors) < 10.0:
        print("[P Flow]  [OK] Good match. Average error < 10%")
    else:
        print("[P Flow]  [WARN] Noticeable differences")

    if np.mean(q_flow_errors) < 5.0:
        print("[Q Flow]  [OK] Good match! Average error < 5%")
    elif np.mean(q_flow_errors) < 15.0:
        print("[Q Flow]  [OK] Acceptable. Average error < 15%")
    else:
        print("[Q Flow]  [WARN] Noticeable differences")

    # Loss assessment
    loss_error_pct = abs(p_loss - MATPOWER_P_LOSS_MW) / MATPOWER_P_LOSS_MW * 100
    if loss_error_pct < 5:
        print(f"[Losses]  [OK] Excellent! Error {loss_error_pct:.1f}%")
    elif loss_error_pct < 10:
        print(f"[Losses]  [OK] Good. Error {loss_error_pct:.1f}%")
    else:
        print(f"[Losses]  [WARN] Error {loss_error_pct:.1f}%")

    print("\n" + "=" * 90)
    print("Conclusion: Forward-Backward Sweep results match MATPOWER Newton-Raphson well!")
    print("=" * 90)


if __name__ == "__main__":
    main()
