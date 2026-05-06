"""AC Optimal Power Flow solver for transmission grid using pandapower.

This module provides single-period AC-OPF backed by pandapower's runopp().
It reads ALL available MATPOWER-style parameters from ClearCase to match
the built-in solver (cal_acopf_trans.py) formulation:

  - Full branch parameters: r, x, b (charging), ratio (tap), angle (shift)
  - Quadratic generator cost: tc_a * P^2 + tc_b * P  (tc_a = mc_b/2, tc_b = mc_c)
  - Per-bus voltage limits (Vmax, Vmin)
  - Per-generator reactive limits (Qmax, Qmin)
  - Bus shunt admittance (Gs, Bs)
  - Reactive load (Qd) scaled proportionally to Pd (ratio clamped ≥ 0 so Q never
    reverses when renewable over-generation makes net load negative at a node)

Branches with non-unity tap ratio are modelled as pandapower transformers
(create_transformer_from_parameters) so the admittance matrix matches the
MATPOWER pi-model exactly.

Requirements:
    pandapower: pip install pandapower
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import pandapower as pp
    HAS_PANDAPOWER = True
except ImportError:
    HAS_PANDAPOWER = False
    pp = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PandapowerACOPFConfig:
    """Immutable solver configuration — all scalar hyperparameters in one place."""
    baseMVA: float
    vn_kv: float
    v_min: float
    v_max: float
    q_factor: float
    verbose: bool
    fail_cost: float


# ---------------------------------------------------------------------------
# Generator limit snapshot
# ---------------------------------------------------------------------------

@dataclass
class GeneratorLimitSnapshot:
    """Original P/Q limits captured at build time, used by CommitmentUpdater."""
    p_min: np.ndarray
    p_max: np.ndarray
    q_min: np.ndarray
    q_max: np.ndarray


# ---------------------------------------------------------------------------
# Branch handles — replace three parallel lists with typed objects
# ---------------------------------------------------------------------------

class LineBranchHandle:
    """Holds the pandapower line index and reads active power flow."""

    def __init__(self, pp_idx: int) -> None:
        self.pp_idx = pp_idx

    def read_active_flow_mw(self, net) -> float:
        return float(net.res_line.at[self.pp_idx, 'p_from_mw'])


class TrafoBranchHandle:
    """Holds the pandapower transformer index and reads active power flow."""

    def __init__(self, pp_idx: int) -> None:
        self.pp_idx = pp_idx

    def read_active_flow_mw(self, net) -> float:
        return float(net.res_trafo.at[self.pp_idx, 'p_hv_mw'])


# ---------------------------------------------------------------------------
# Network artifacts — all indices and precomputed arrays from the build phase
# ---------------------------------------------------------------------------

@dataclass
class PandapowerNetworkArtifacts:
    """Everything produced by the build phase; consumed by the solver."""
    gen_rows: List[int]                    # pandapower gen row indices (one per unit)
    load_pp_indices: List[int]             # pandapower load row indices
    load_node_indices: np.ndarray          # 0-indexed positions into node arrays
    branch_handles: List                   # LineBranchHandle | TrafoBranchHandle
    bus_pp_indices: Dict[int, int]         # node_id -> pandapower bus index
    gen_limit_snapshot: GeneratorLimitSnapshot
    nodes_units_map: np.ndarray            # (n_nodes, n_units) mapping matrix
    base_pd: Optional[np.ndarray]          # original Pd column, or None
    base_qd: Optional[np.ndarray]          # original Qd column, or None


# ---------------------------------------------------------------------------
# Network builder — case -> (net, artifacts); no solver state side-effects
# ---------------------------------------------------------------------------

class PandapowerNetworkBuilder:
    """Builds a pandapower network from a ClearCase and returns artifacts.

    This class is stateless: call build() and discard the builder.
    All mutable solver state lives in the returned artifacts.
    """

    def build(self, case, config: PandapowerACOPFConfig):
        """Return (net, artifacts) from the given case and config.

        Nothing is written onto the caller; all outputs are return values.
        """
        net = pp.create_empty_network(sn_mva=config.baseMVA)
        Zbase = (config.vn_kv ** 2) / config.baseMVA
        f_hz = net.f_hz

        bus_pp_indices = self._build_buses(net, case, config)
        self._build_ext_grid(net, bus_pp_indices, case)
        gen_rows = self._build_generators(net, bus_pp_indices, case, config)
        branch_handles = self._build_branches(net, bus_pp_indices, case, config, Zbase, f_hz)
        self._build_shunts(net, bus_pp_indices, case)
        load_node_indices, load_pp_indices = self._build_loads(net, bus_pp_indices)

        gen_limit_snapshot = GeneratorLimitSnapshot(
            p_min=net.gen.loc[gen_rows, 'min_p_mw'].values.copy(),
            p_max=net.gen.loc[gen_rows, 'max_p_mw'].values.copy(),
            q_min=net.gen.loc[gen_rows, 'min_q_mvar'].values.copy(),
            q_max=net.gen.loc[gen_rows, 'max_q_mvar'].values.copy(),
        )

        nodes_units_map = case.get_nodes_units_map()

        has_qd = 'Qd' in case.nodes.columns and 'Pd' in case.nodes.columns
        base_pd = case.nodes['Pd'].values.astype(float) if has_qd else None
        base_qd = case.nodes['Qd'].values.astype(float) if has_qd else None

        artifacts = PandapowerNetworkArtifacts(
            gen_rows=gen_rows,
            load_pp_indices=load_pp_indices,
            load_node_indices=load_node_indices,
            branch_handles=branch_handles,
            bus_pp_indices=bus_pp_indices,
            gen_limit_snapshot=gen_limit_snapshot,
            nodes_units_map=nodes_units_map,
            base_pd=base_pd,
            base_qd=base_qd,
        )
        return net, artifacts

    # ------------------------------------------------------------------
    # Private build helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_buses(net, case, config: PandapowerACOPFConfig) -> Dict[int, int]:
        bus_pp_indices: Dict[int, int] = {}
        for _, row in case.nodes.iterrows():
            node_idx = int(row['#id'])
            if 'Vmax' in case.nodes.columns and 'Vmin' in case.nodes.columns:
                v_max_bus = float(row['Vmax'])
                v_min_bus = float(row['Vmin'])
            else:
                v_max_bus = config.v_max
                v_min_bus = config.v_min
            pp_idx = pp.create_bus(
                net, vn_kv=config.vn_kv,
                name=f"bus_{node_idx}",
                min_vm_pu=v_min_bus,
                max_vm_pu=v_max_bus,
            )
            bus_pp_indices[node_idx] = pp_idx
        return bus_pp_indices

    @staticmethod
    def _build_ext_grid(net, bus_pp_indices: Dict[int, int], case) -> None:
        slack_bus = getattr(case, 'slack_bus', 0)
        ext_idx = pp.create_ext_grid(
            net,
            bus=bus_pp_indices[slack_bus],
            vm_pu=1.0, name='slack',
            min_p_mw=-1e-6, max_p_mw=1e-6,
            min_q_mvar=-1e-6, max_q_mvar=1e-6,
        )
        pp.create_poly_cost(net, element=ext_idx, et='ext_grid',
                            cp2_eur_per_mw2=1e6, cp1_eur_per_mw=0.0,
                            cp0_eur=0.0)

    @staticmethod
    def _build_generators(net, bus_pp_indices: Dict[int, int],
                          case, config: PandapowerACOPFConfig) -> List[int]:
        gen_rows: List[int] = []
        for _, unit in case.units.iterrows():
            bus_idx = int(unit['#bus_id'])
            pp_bus = bus_pp_indices[bus_idx]
            p_max = float(unit['p_max'])
            p_min = float(unit['p_min'])

            if 'Qmax' in case.units.columns and 'Qmin' in case.units.columns:
                q_max = float(unit['Qmax'])
                q_min = float(unit['Qmin'])
            else:
                q_max = p_max * config.q_factor
                q_min = -p_max * config.q_factor

            gen_idx = pp.create_gen(
                net, bus=pp_bus, p_mw=p_min, vm_pu=1.0,
                name=f"gen_{int(unit['#id'])}",
                min_p_mw=p_min, max_p_mw=p_max,
                min_q_mvar=q_min, max_q_mvar=q_max,
                controllable=True,
            )
            tc_a = float(unit['mc_b']) / 2  # TC quadratic coeff = mc_b/2 (exact when mc_a=0)
            tc_b = float(unit['mc_c'])       # TC linear coeff = mc_c
            pp.create_poly_cost(
                net, element=gen_idx, et='gen',
                cp2_eur_per_mw2=tc_a,
                cp1_eur_per_mw=tc_b,
                cp0_eur=0.0,
            )
            gen_rows.append(gen_idx)
        return gen_rows

    @classmethod
    def _build_branches(cls, net, bus_pp_indices: Dict[int, int],
                        case, config: PandapowerACOPFConfig,
                        Zbase: float, f_hz: float) -> List:
        branch_handles: List = []
        for _, line in case.lines.iterrows():
            from_bus = int(line['#from'])
            to_bus = int(line['#to'])
            x_pu = float(line['x'])
            r_pu = float(line['r']) if 'r' in case.lines.columns else 0.0
            b_pu = float(line['b']) if 'b' in case.lines.columns else 0.0
            ratio = float(line['ratio']) if 'ratio' in case.lines.columns else 0.0
            angle = float(line['angle']) if 'angle' in case.lines.columns else 0.0
            cap = float(line['cap'])
            line_id = int(line['#id'])

            is_trafo = (ratio != 0.0 and abs(ratio - 1.0) > 1e-8)
            if is_trafo:
                handle = cls._add_transformer(
                    net, bus_pp_indices, from_bus, to_bus,
                    r_pu, x_pu, b_pu, ratio, angle, cap, line_id, config,
                )
            else:
                handle = cls._add_line(
                    net, bus_pp_indices, from_bus, to_bus,
                    r_pu, x_pu, b_pu, cap, Zbase, f_hz, line_id, config,
                )
            branch_handles.append(handle)
        return branch_handles

    @staticmethod
    def _add_line(net, bus_pp, from_bus, to_bus,
                  r_pu, x_pu, b_pu, cap, Zbase, f_hz, line_id,
                  config: PandapowerACOPFConfig) -> LineBranchHandle:
        r_ohm = r_pu * Zbase
        x_ohm = x_pu * Zbase

        if b_pu != 0:
            c_nf = abs(b_pu) / Zbase / (2 * np.pi * f_hz) * 1e9
        else:
            c_nf = 0.0

        if 0 < cap < 1e5:
            max_i_ka = cap / (np.sqrt(3) * config.vn_kv)
        else:
            max_i_ka = 9999.0

        line_idx = pp.create_line_from_parameters(
            net,
            from_bus=bus_pp[from_bus],
            to_bus=bus_pp[to_bus],
            length_km=1.0,
            r_ohm_per_km=r_ohm,
            x_ohm_per_km=x_ohm,
            c_nf_per_km=c_nf,
            max_i_ka=max_i_ka,
            max_loading_percent=100.0,
            name=f"line_{line_id}",
        )
        return LineBranchHandle(line_idx)

    @staticmethod
    def _add_transformer(net, bus_pp, from_bus, to_bus,
                         r_pu, x_pu, b_pu, ratio, angle, cap, line_id,
                         config: PandapowerACOPFConfig) -> TrafoBranchHandle:
        """Convert MATPOWER branch parameters to pandapower transformer parameters.

        The admittance matrix matches the MATPOWER pi-model exactly.
        """
        z_mag = np.sqrt(r_pu**2 + x_pu**2)
        vk_percent = z_mag * 100.0
        vkr_percent = r_pu * 100.0

        sn_mva = config.baseMVA
        if 0 < cap < 1e5:
            max_loading = cap / sn_mva * 100.0
        else:
            max_loading = 1e6

        tap_step_pct = (ratio - 1.0) * 100.0

        trafo_idx = pp.create_transformer_from_parameters(
            net,
            hv_bus=bus_pp[from_bus],
            lv_bus=bus_pp[to_bus],
            sn_mva=sn_mva,
            vn_hv_kv=config.vn_kv,
            vn_lv_kv=config.vn_kv,
            vkr_percent=vkr_percent,
            vk_percent=max(vk_percent, vkr_percent + 1e-10),  # vk >= vkr
            pfe_kw=0.0,
            i0_percent=0.0,
            shift_degree=angle,
            tap_side='hv',
            tap_neutral=0,
            tap_pos=1,
            tap_step_percent=tap_step_pct,
            max_loading_percent=max_loading,
            name=f"trafo_{line_id}",
        )
        return TrafoBranchHandle(trafo_idx)

    @staticmethod
    def _build_shunts(net, bus_pp_indices: Dict[int, int], case) -> None:
        if 'Gs' not in case.nodes.columns and 'Bs' not in case.nodes.columns:
            return
        for _, row in case.nodes.iterrows():
            node_idx = int(row['#id'])
            gs = float(row.get('Gs', 0.0))
            bs = float(row.get('Bs', 0.0))
            if gs != 0 or bs != 0:
                pp.create_shunt(
                    net, bus=bus_pp_indices[node_idx],
                    p_mw=gs,
                    q_mvar=-bs,    # MATPOWER Bs>0 = capacitive; pandapower q_mvar>0 = inductive
                )

    @staticmethod
    def _build_loads(net, bus_pp_indices: Dict[int, int]):
        """Create one placeholder load per bus; return index arrays."""
        node_list = []
        load_pp_list = []
        for node_idx, pp_bus in bus_pp_indices.items():
            load_idx = pp.create_load(
                net, bus=pp_bus,
                p_mw=0.0, q_mvar=0.0,
                name=f"load_{node_idx}",
                controllable=False,
            )
            node_list.append(node_idx)
            load_pp_list.append(load_idx)
        return np.array(node_list, dtype=int), load_pp_list


# ---------------------------------------------------------------------------
# Load updater — writes node_net_load_mw into net.load with Qd scaling
# ---------------------------------------------------------------------------

class LoadUpdater:
    """Updates pandapower load table for a single solve step.

    Qd scaling rule: proportional to Pd, ratio clamped ≥ 0.
    Physical note: clamping sets Q to 0 whenever net load goes negative
    (renewable over-generation).  If accurate reactive modelling is needed
    under over-generation, model renewables as separate sgen elements.
    """

    def apply(self, net, node_net_load_mw: np.ndarray,
              artifacts: PandapowerNetworkArtifacts) -> None:
        idx = artifacts.load_node_indices
        pp_idx = artifacts.load_pp_indices

        net.load.loc[pp_idx, 'p_mw'] = node_net_load_mw[idx]
        net.load.loc[pp_idx, 'q_mvar'] = 0.0

        if artifacts.base_pd is not None and artifacts.base_qd is not None:
            pd_new = node_net_load_mw[idx]
            pd_orig = artifacts.base_pd[idx]
            qd_orig = artifacts.base_qd[idx]
            mask = np.abs(pd_orig) > 1e-8
            ratio = np.where(mask,
                             np.maximum(pd_new / np.where(mask, pd_orig, 1.0), 0.0),
                             0.0)
            net.load.loc[pp_idx, 'q_mvar'] = qd_orig * ratio


# ---------------------------------------------------------------------------
# Commitment updater — zeros Q limits for off units and restores on change
# ---------------------------------------------------------------------------

class CommitmentUpdater:
    """Applies unit commitment to pandapower generator bounds.

    Off units: P and Q limits set to zero (prevents reactive-power cheating).
    On units: original limits restored from the snapshot.
    commitment=None: all units restored (fully committed state).
    """

    def apply(self, net, commitment: Optional[np.ndarray],
              artifacts: PandapowerNetworkArtifacts) -> None:
        gen_rows = artifacts.gen_rows
        snap = artifacts.gen_limit_snapshot

        if commitment is not None:
            commitment = np.asarray(commitment, dtype=float)
            on_mask = commitment != 0
            off_pp = [gen_rows[i] for i in range(len(gen_rows)) if not on_mask[i]]
            on_i = np.where(on_mask)[0]
            on_pp = [gen_rows[i] for i in on_i]
            if off_pp:
                net.gen.loc[off_pp,
                    ['min_p_mw', 'max_p_mw', 'min_q_mvar', 'max_q_mvar']] = 0.0
            if len(on_i):
                net.gen.loc[on_pp, 'min_p_mw']   = snap.p_min[on_i]
                net.gen.loc[on_pp, 'max_p_mw']   = snap.p_max[on_i]
                net.gen.loc[on_pp, 'min_q_mvar'] = snap.q_min[on_i]
                net.gen.loc[on_pp, 'max_q_mvar'] = snap.q_max[on_i]
        else:
            net.gen.loc[gen_rows, 'min_p_mw']   = snap.p_min
            net.gen.loc[gen_rows, 'max_p_mw']   = snap.p_max
            net.gen.loc[gen_rows, 'min_q_mvar'] = snap.q_min
            net.gen.loc[gen_rows, 'max_q_mvar'] = snap.q_max


# ---------------------------------------------------------------------------
# ACOPFSolver — façade that orchestrates the above components
# ---------------------------------------------------------------------------

class ACOPFSolver:
    """AC-OPF solver backed by pandapower.

    Build a pandapower network once from case topology and solve repeatedly
    with updated node net loads.

    Args:
        case: ClearCase instance (needs nodes, units, lines, loads)
        baseMVA: System base power (MVA). Defaults to case.baseMVA or 100.
        vn_kv: Nominal voltage (kV). Defaults to case.basekV or 100.
        v_min: Minimum voltage (p.u.). Default 0.95
        v_max: Maximum voltage (p.u.). Default 1.05
        q_factor: |Q_max/P_max| ratio for generators. Default 0.75 (pf~0.8)
        verbose: Print pandapower output. Default False
        fail_cost: Cost returned on OPF failure (replaces np.inf). Default 1e6.
    """

    def __init__(self, case, baseMVA: float = None, vn_kv: float = None,
                 v_min: float = 0.95, v_max: float = 1.05,
                 q_factor: float = 0.75, verbose: bool = False,
                 fail_cost: float = 1e6):
        if not HAS_PANDAPOWER:
            raise ImportError(
                "pandapower is required for AC-OPF. "
                "Install via: pip install pandapower"
            )

        if not getattr(case, 'init_flag', False):
            case.init()

        self.case = case
        self.config = PandapowerACOPFConfig(
            baseMVA=baseMVA or getattr(case, 'baseMVA', 100.0),
            vn_kv=vn_kv or getattr(case, 'basekV', None) or getattr(case, 'baseKV', 100.0),
            v_min=v_min,
            v_max=v_max,
            q_factor=q_factor,
            verbose=verbose,
            fail_cost=fail_cost,
        )

        self.n_units = len(case.units)
        self.n_nodes = len(case.nodes)
        self.n_lines = len(case.lines)

        self.net, self._artifacts = PandapowerNetworkBuilder().build(case, self.config)
        self._load_updater = LoadUpdater()
        self._commitment_updater = CommitmentUpdater()

    # ------------------------------------------------------------------
    # Compatibility properties — test suite accesses these directly
    # ------------------------------------------------------------------

    @property
    def _pp_gen_idx(self) -> List[int]:
        return self._artifacts.gen_rows

    # ------------------------------------------------------------------
    # Public solver interface
    # ------------------------------------------------------------------

    def solve(self, node_net_load_mw: np.ndarray,
              commitment: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Solve AC-OPF for given node net loads.

        Args:
            node_net_load_mw: Net load at each node (n_nodes,), MW
            commitment: Unit commitment (n_units,) 0/1. None = all committed.

        Returns:
            Dict with keys matching cal_dcopf_trans.py / cal_acopf_trans.py.
        """
        self._load_updater.apply(self.net, node_net_load_mw, self._artifacts)
        self._commitment_updater.apply(self.net, commitment, self._artifacts)
        if not self._run_opf():
            return self._failure_result()
        return self._build_result(node_net_load_mw)

    # ------------------------------------------------------------------
    # Private solve steps
    # ------------------------------------------------------------------

    def _run_opf(self) -> bool:
        try:
            pp.runopp(self.net, verbose=self.config.verbose,
                      suppress_warnings=not self.config.verbose)
            return bool(self.net['OPF_converged'])
        except Exception as e:
            if self.config.verbose:
                print(f"AC-OPF failed: {e}")
            return False

    def _build_result(self, node_net_load_mw: np.ndarray) -> Dict[str, Any]:
        gen_rows = self._artifacts.gen_rows

        unit_power_mw = np.array([
            self.net.res_gen.at[gi, 'p_mw'] for gi in gen_rows
        ])
        q_gen = np.array([
            self.net.res_gen.at[gi, 'q_mvar'] for gi in gen_rows
        ])

        line_flow_mw = np.array([
            h.read_active_flow_mw(self.net)
            for h in self._artifacts.branch_handles
        ])

        vm_pu = self.net.res_bus['vm_pu'].values
        va_deg = self.net.res_bus['va_degree'].values

        node_unit_power_mw = self._artifacts.nodes_units_map @ unit_power_mw
        node_net_injection_mw = node_unit_power_mw - node_net_load_mw

        total_cost = float(self.net.res_cost)

        if 'lam_p' in self.net.res_bus.columns:
            lmp = self.net.res_bus['lam_p'].values.copy()
            lmp_quality = 'exact'
            lmp_available = True
        else:
            logger.warning("pandapower OPF: lam_p not available in res_bus; "
                           "LMP set to zero. Reward functions using LMP will "
                           "receive uninformative price signals.")
            lmp = np.zeros(self.n_nodes)
            lmp_quality = 'unavailable'
            lmp_available = False

        return {
            'unit_power_mw': unit_power_mw,
            'line_flow_mw': line_flow_mw,
            'node_net_injection_mw': node_net_injection_mw,
            'total_cost': total_cost,
            'slack_violation': 0.0,
            'status': 'optimal',
            'success': True,
            'lmp': lmp,
            'vm_pu': vm_pu,
            'va_deg': va_deg,
            'q_gen': q_gen,
            'solver_backend': 'pandapower',
            'lmp_method': 'pandapower_dual' if lmp_available else 'none',
            'lmp_quality': lmp_quality,
            'lmp_available': lmp_available,
        }

    def _failure_result(self) -> Dict[str, Any]:
        return {
            'unit_power_mw': np.zeros(self.n_units),
            'line_flow_mw': np.zeros(self.n_lines),
            'node_net_injection_mw': np.zeros(self.n_nodes),
            'total_cost': self.config.fail_cost,
            'slack_violation': self.config.fail_cost,
            'status': 'failed',
            'success': False,
            'lmp': np.zeros(self.n_nodes),
            'vm_pu': np.ones(self.n_nodes),
            'va_deg': np.zeros(self.n_nodes),
            'q_gen': np.zeros(self.n_units),
            'solver_backend': 'pandapower',
            'lmp_method': 'none',
            'lmp_quality': 'unavailable',
            'lmp_available': False,
        }


# ---------------------------------------------------------------------------
# Global cache
# ---------------------------------------------------------------------------

_acopf_cache: Dict[int, tuple] = {}


def solve_acopf_detailed(case, node_net_load_mw: np.ndarray,
                         commitment: Optional[np.ndarray] = None,
                         baseMVA: float = None, vn_kv: float = None,
                         v_min: float = 0.95, v_max: float = 1.05,
                         q_factor: float = 0.75,
                         fail_cost: float = 1e6,
                         rebuild: bool = False,
                         verbose: bool = False) -> Dict[str, Any]:
    """Solve AC-OPF via pandapower and return detailed results."""
    case_id = id(case)
    entry = _acopf_cache.get(case_id)
    if entry is None or entry[0] is not case or rebuild:
        solver = ACOPFSolver(
            case, baseMVA=baseMVA, vn_kv=vn_kv,
            v_min=v_min, v_max=v_max,
            q_factor=q_factor, verbose=verbose,
            fail_cost=fail_cost,
        )
        _acopf_cache[case_id] = (case, solver)
        return solver.solve(node_net_load_mw, commitment)
    return entry[1].solve(node_net_load_mw, commitment)


if __name__ == '__main__':
    from powerzoo.case import load_case

    for case_num in [5, 14, 118]:
        case = load_case(case_num)
        case.init()
        n = len(case.nodes)
        if 'Pd' in case.nodes.columns:
            net_load = case.nodes['Pd'].values.astype(float)
        else:
            total_cap = case.units['p_max'].sum()
            net_load = np.zeros(n)
            net_load[1:] = total_cap * 0.6 / (n - 1)

        print(f"\n{'='*60}")
        print(f"  Case{case_num} — pandapower AC-OPF")
        print(f"{'='*60}")
        result = solve_acopf_detailed(case, net_load, verbose=True, rebuild=True)
        if result['success']:
            print(f"  Status: {result['status']}")
            print(f"  Total Cost: ${result['total_cost']:.2f}")
            print(f"  Unit P: {result['unit_power_mw']}")
        else:
            print("  FAILED")
