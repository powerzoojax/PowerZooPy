import importlib.util
import os
import sys
import warnings
from functools import wraps

data_dir = os.path.dirname(__file__)

_CASE_SUBDIRS = ["transmission", "distribution"]


def hide_traceback(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(e)

    return wrap


def _load_case_from_mfile(mfile_path: str, mock: bool = True):
    """Load a MATPOWER .m file and return a ClearCase-compatible object.

    Converts MATPOWER bus/gen/branch/gencost tables to the internal DataFrame
    schema expected by ClearCase (F7 fix).

    Args:
        mfile_path: Absolute or relative path to a MATPOWER .m file.
        mock: Passed to ClearCase.__init__ (enables CaseMocker). Default True.

    Returns:
        A ClearCase instance populated from the .m file.
    """
    import numpy as np
    from powerzoo.case.CaseBase import ClearCase, DataFrame
    from powerzoo.case.source_mfile.MfileModel import MFile

    with open(mfile_path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    m = MFile()
    m.read(text)

    bus = m.bus.reset_index(drop=True)
    n_buses = len(bus)
    node_data = [[float(bus.loc[i, "bus_i"]), float(i), 0.0] for i in range(n_buses)]
    nodes = DataFrame(["id", "x", "y"], node_data)

    gen = m.gen.reset_index(drop=True)
    gencost = m.gencost.reset_index(drop=True)
    n_gen = len(gen)
    unit_data = []
    for i in range(n_gen):
        bus_id = float(gen.loc[i, "bus"])
        p_max = float(gen.loc[i, "Pmax"])
        p_min = float(gen.loc[i, "Pmin"])
        # MATPOWER gencost type 2: TC(p) = c2·p² + c1·p + c0  (A=c2, B=c1, C=c0)
        # Convert to marginal cost: MC(p) = 2·c2·p + c1 → mc_a=0, mc_b=2·c2, mc_c=c1
        if i < len(gencost):
            tc_c2 = float(gencost.loc[i, "A"]) if "A" in gencost.columns else 0.0
            tc_c1 = float(gencost.loc[i, "B"]) if "B" in gencost.columns else 0.0
            mc_a = 0.0
            mc_b = 2.0 * tc_c2
            mc_c = tc_c1
        else:
            mc_a, mc_b, mc_c = 0.0, 0.0, 0.0
        unit_data.append([float(i + 1), bus_id, mc_a, mc_b, mc_c, p_max, p_min])
    units = DataFrame(["id", "bus_id", "mc_a", "mc_b", "mc_c", "p_max", "p_min"], unit_data)

    branch = m.branch.reset_index(drop=True)
    n_branch = len(branch)
    line_data = []
    for i in range(n_branch):
        fbus = float(branch.loc[i, "fbus"])
        tbus = float(branch.loc[i, "tbus"])
        x = float(branch.loc[i, "x"])
        rate_a = float(branch.loc[i, "rateA"]) if "rateA" in branch.columns else 0.0
        cap = rate_a if rate_a > 0 else 0.0
        line_data.append([float(i + 1), fbus, tbus, x, 0.0, cap])
    lines = DataFrame(["id", "from", "to", "x", "floor", "cap"], line_data)

    load_rows = []
    load_id = 1
    for i in range(n_buses):
        pd_mw = float(bus.loc[i, "Pd"]) if "Pd" in bus.columns else 0.0
        bus_id = float(bus.loc[i, "bus_i"])
        d_max = max(pd_mw, 0.0)
        d_min = 0.0
        load_rows.append([float(load_id), bus_id, 0.0, 0.0, 0.0, d_max, d_min])
        load_id += 1
    loads = DataFrame(["id", "bus_id", "mc_a", "mc_b", "mc_c", "d_max", "d_min"], load_rows)

    case_name = os.path.splitext(os.path.basename(mfile_path))[0]

    class _MFileClearCase(ClearCase):
        def __init__(self, _mock):
            self.nodes = nodes
            self.units = units
            self.lines = lines
            self.loads = loads
            super().__init__(_mock)

    case_obj = _MFileClearCase(mock)
    case_obj.name = case_name
    return case_obj


def _find_case_file(case_name: str, grid_type: str = None) -> str:
    """Locate a Case*.py file inside the case sub-packages.

    Search order:
    1. If *grid_type* is given, look in that sub-directory first.
    2. Search all ``_CASE_SUBDIRS``.
    3. Fall back to the ``case/`` root (for legacy / generated files).

    Returns:
        Absolute path to the Python case file.

    Raises:
        FileNotFoundError: if no matching file is found.
    """
    filename = f"{case_name}.py"

    if grid_type:
        preferred = os.path.join(data_dir, grid_type, filename)
        if os.path.exists(preferred):
            return preferred

    for subdir in _CASE_SUBDIRS:
        candidate = os.path.join(data_dir, subdir, filename)
        if os.path.exists(candidate):
            return candidate

    root_candidate = os.path.join(data_dir, filename)
    if os.path.exists(root_candidate):
        return root_candidate

    raise FileNotFoundError(
        f"{case_name} not found in {_CASE_SUBDIRS} or case root"
    )


@hide_traceback
def load_case(case_id, case_source="", mock=True, grid_type=None):
    """Load a power system case by name or file path.

    Args:
        case_id: Case identifier.  Accepted forms:
            - Integer or string (e.g. ``5``, ``'Case5'``, ``'case5'``,
              ``'case33bw'``): loads the matching built-in Python case file.
              The ``case`` prefix is case-insensitive.
            - String ending in ``'.m'``: treated as a MATPOWER case file path
              and converted on-the-fly.
        case_source: (deprecated) Sub-directory inside ``case/`` to search.
        mock: Passed to the case constructor (enables CaseMocker).
        grid_type: Optional.  ``'transmission'`` or ``'distribution'``.
            Restricts lookup to the corresponding sub-directory and emits a
            warning if the loaded case's ``GRID_TYPE`` does not match.

    Returns:
        ClearCase instance.
    """
    if isinstance(case_id, str) and case_id.endswith(".m"):
        path = case_id if os.path.isabs(case_id) else os.path.join(os.getcwd(), case_id)
        assert os.path.exists(path), f"MATPOWER file NOT EXISTS: {path}"
        return _load_case_from_mfile(path, mock=mock)

    case_str = str(case_id)
    if case_str.lower().startswith('case'):
        case_name = 'Case' + case_str[4:]
    else:
        case_name = f'Case{case_str}'

    if case_source:
        case_path = os.path.join(data_dir, case_source, f"{case_name}.py")
        assert os.path.exists(case_path), f"{case_name} NOT EXISTS in {case_source}"
    else:
        case_path = _find_case_file(case_name, grid_type=grid_type)

    spec = importlib.util.spec_from_file_location(case_name, case_path)
    case_py = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = case_py
    spec.loader.exec_module(case_py)
    case_class = getattr(case_py, case_name)
    case_object = case_class(mock)
    case_object.name = case_name

    if grid_type and getattr(case_object, "GRID_TYPE", ""):
        if case_object.GRID_TYPE != grid_type:
            warnings.warn(
                f"Case '{case_name}' has GRID_TYPE='{case_object.GRID_TYPE}', "
                f"but grid_type='{grid_type}' was requested.",
                UserWarning,
                stacklevel=2,
            )

    return case_object


# ---------------------------------------------------------------------------
# Convenience re-exports so `from powerzoo.case import Case5` works
# ---------------------------------------------------------------------------
from .transmission import *  # noqa: F401,F403
from .distribution import *  # noqa: F401,F403


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------
from ._registry import list_cases  # noqa: F401


# ---------------------------------------------------------------------------
# Matplotlib / numpy cosmetics (non-critical)
# ---------------------------------------------------------------------------
try:
    import numpy as np

    np.set_printoptions(precision=4, suppress=True)

    from matplotlib import pyplot as plt
    import platform

    plt.rcParams["font.sans-serif"] = (
        ["SimSun"] if platform.system() == "Windows" else ["Arial Unicode MS"]
    )
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.unicode_minus"] = False
except ModuleNotFoundError:
    pass
