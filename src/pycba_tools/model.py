"""
CBAModel: a pycba continuous beam with named load cases, load combinations and
named nodes, analysed one load combination at a time.

pycba's BeamAnalysis analyses a single load matrix. A Pynite FEModel3D, on the
other hand, holds load cases, load combinations and the results of every
combination. CBAModel provides that model-level container on top of pycba so
that pycba models can be serialized and have their results extracted exactly
like Pynite models (see pycba_tools.serialize and pycba_tools.reports).

Conventions
-----------
* Inputs (spans, stiffnesses, restraints, load matrix rows, the global-coordinate
  load helpers) use **pycba's** conventions: positive transverse loads act
  downward, positive moment loads are counter-clockwise, span numbers inside load
  matrix rows are 1-based.
* The Pynite-shaped result views (`CBAModel.nodes`, `CBAModel.members`) and every
  `pycba_tools.reports` function use **Pynite's** conventions for a member along
  the global X axis loaded in Y, so that result trees are interchangeable with
  those from pynite-tools:

    ===========  ===========================  =========================
    Pynite       pycba                        Relationship
    ===========  ===========================  =========================
    RxnFY        vertical reaction            same sign (up positive)
    RxnMZ        moment reaction              same sign (CCW positive)
    DY / RZ      nodal displacement/rotation  same sign
    Fy (shear)   V                            same sign
    Mz (moment)  M                            Mz = -M
    dy           D                            same sign
    ===========  ===========================  =========================

  pycba is a planar beam solver, so the out-of-plane, axial and torsional
  results (FX, FZ, MX, MY, DX, DZ, RX, RY, Fz, My, axial, torque, dx, dz)
  are always zero.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
from pycba import BeamAnalysis
from pycba.load import factor_LM
from pycba.section import SectionEI
from pycba.types import MemberType
from pycba.utils import supports_to_R

DEFAULT_CASE = "Case 1"
DEFAULT_COMBO = "Combo 1"

# Number of columns in a pycba load matrix row, by load type
LOAD_ROW_LENGTHS = {1: (3,), 2: (4,), 3: (5,), 4: (4,), 5: (4, 6)}  # 6: >= 3

_TOL = 1e-9


@dataclass
class LoadCombo:
    """
    A named load combination: {load case name: factor}. Mirrors Pynite's
    LoadCombo (name, factors, combo_tags).
    """

    name: str
    factors: dict[str, float] = field(default_factory=dict)
    combo_tags: Optional[list[str] | str] = None

    def AddLoadCase(self, case_name: str, factor: float) -> None:
        """Adds (or replaces) a load case factor, as Pynite's LoadCombo does."""
        self.factors[case_name] = factor

    def DeleteLoadCase(self, case_name: str) -> None:
        """Removes a load case from the combination, as Pynite's LoadCombo does."""
        del self.factors[case_name]


class CBAModel:
    """
    A continuous beam, its load cases and load combinations and, once analysed,
    the pycba results of every load combination.

    Parameters
    ----------
    L : sequence of float, optional
        Span lengths. Required unless 'node_locations' is given.
    EI : float or sequence of float
        Flexural rigidity, one value for all spans or one per span.
    R : sequence of float, optional
        pycba restraint vector, two entries per node (vertical, rotation):
        -1 = fixed, 0 = free, +k = spring stiffness. Give 'R' or 'supports'.
    supports : sequence, optional
        pycba's friendly alternative to 'R': one support name per node
        ("pin", "roller", "fixed", "free") or a raw [vertical, rotation] pair.
    eletype : sequence, optional
        pycba element type per span (1/"FF", 2/"FP", 3/"PF", 4/"PP").
        Default is fixed-fixed (continuous) for every span.
    D : sequence, optional
        pycba prescribed displacement vector (same length as 'R'); None for
        DOFs without a prescribed displacement. Only allowed at fixed DOFs.
    GAv : float or sequence, optional
        Shear rigidity per span (Timoshenko spans). Default is None.
    kf : float or sequence, optional
        Winkler foundation modulus per span. Default is None.
    name : str
        The name of the (physical) member, used as the member key in result
        trees. Default is "M1".
    node_names : sequence of str, optional
        Names of the nodes at the span boundaries (left to right). Default is
        "N0", "N1", ...
    node_locations : sequence of float, optional
        Global X coordinates of the nodes. Defaults to the cumulative span
        lengths starting at 0. When given without 'L', the spans are derived
        from it. Storing the locations keeps node coordinates exact (no
        floating point drift from summing the spans).
    npts : int
        Number of pycba evaluation points per span. Default is 500.
    """

    def __init__(
        self,
        L: Optional[Sequence[float]] = None,
        EI: float | Sequence[float] = 1.0,
        R: Optional[Sequence[float]] = None,
        supports: Optional[Sequence] = None,
        eletype: Optional[Sequence] = None,
        D: Optional[Sequence[Optional[float]]] = None,
        GAv: Optional[float | Sequence[Optional[float]]] = None,
        kf: Optional[float | Sequence[Optional[float]]] = None,
        name: str = "M1",
        node_names: Optional[Sequence[str]] = None,
        node_locations: Optional[Sequence[float]] = None,
        npts: int = 500,
    ):
        if L is None and node_locations is None:
            raise ValueError("Provide the span lengths 'L' or the 'node_locations'.")
        if node_locations is not None:
            node_locations = [float(x) for x in node_locations]
            spans = list(np.diff(node_locations))
            if L is None:
                L = spans
            elif len(L) != len(spans) or not np.allclose(L, spans):
                raise ValueError("'L' does not agree with 'node_locations'.")
        self.L: list[float] = [float(x) for x in L]
        n_spans = len(self.L)
        if n_spans == 0:
            raise ValueError("The model needs at least one span.")
        if any(span <= 0 for span in self.L):
            raise ValueError(f"Span lengths must be positive: {self.L}")
        n_nodes = n_spans + 1

        self.EI: list[float] = _per_span(EI, n_spans, "EI", allow_none=False)
        self.GAv: Optional[list[Optional[float]]] = (
            None if GAv is None else _per_span(GAv, n_spans, "GAv")
        )
        self.kf: Optional[list[Optional[float]]] = (
            None if kf is None else _per_span(kf, n_spans, "kf")
        )

        if (R is None) == (supports is None):
            raise ValueError("Provide exactly one of 'R' or 'supports'.")
        if supports is not None:
            R = supports_to_R(supports, n_nodes)
        self.R: list[float] = [float(r) for r in R]
        if len(self.R) != 2 * n_nodes:
            raise ValueError(f"'R' needs {2 * n_nodes} entries, got {len(self.R)}.")

        if eletype is None:
            eletype = [1] * n_spans
        self.eletype: list[int] = [MemberType.coerce(e) for e in eletype]
        if len(self.eletype) != n_spans:
            raise ValueError(f"'eletype' needs {n_spans} entries, got {len(self.eletype)}.")

        if D is None:
            D = [None] * (2 * n_nodes)
        self.D: list[Optional[float]] = [None if d is None else float(d) for d in D]
        if len(self.D) != 2 * n_nodes:
            raise ValueError(f"'D' needs {2 * n_nodes} entries, got {len(self.D)}.")
        for idx, (d, r) in enumerate(zip(self.D, self.R)):
            if d is not None and r >= 0:
                raise ValueError(
                    f"A prescribed displacement is only supported at a fixed DOF "
                    f"(R = -1); DOF {idx} has R = {r}."
                )

        self.name = name
        if node_names is None:
            node_names = [f"N{idx}" for idx in range(n_nodes)]
        self.node_names: list[str] = list(node_names)
        if len(self.node_names) != n_nodes or len(set(self.node_names)) != n_nodes:
            raise ValueError(f"'node_names' needs {n_nodes} unique names.")
        if node_locations is None:
            node_locations = [0.0] + list(np.cumsum(self.L))
        self.node_locations: list[float] = [float(x) for x in node_locations]

        self.npts = int(npts)
        self.load_cases: dict[str, list[list]] = {}
        self.load_combos: dict[str, LoadCombo] = {}
        self.analyses: dict[str, BeamAnalysis] = {}
        self._stations: dict[str, list[dict[str, np.ndarray]]] = {}
        self._reactions: dict[str, np.ndarray] = {}
        self._nodes: Optional[dict[str, "NodeView"]] = None
        self._members: Optional[dict[str, "MemberView"]] = None

    # ------------------------------------------------------------------
    # Interop with pycba objects
    # ------------------------------------------------------------------

    @classmethod
    def from_beam_analysis(
        cls, beam_analysis: BeamAnalysis, case: str = DEFAULT_CASE, **kwargs
    ) -> "CBAModel":
        """
        Returns a CBAModel with the geometry of the pycba 'beam_analysis' and its
        load matrix as load case 'case'. Extra keyword arguments (e.g. 'name',
        'node_names') are passed to CBAModel.
        """
        beam = beam_analysis.beam
        GAv = list(getattr(beam, "mbr_GAv", []) or []) or None
        kf = list(getattr(beam, "mbr_kf", []) or []) or None
        model = cls(
            L=list(beam.mbr_lengths),
            EI=list(beam.mbr_EIs),
            R=list(beam.restraints),
            eletype=list(beam.mbr_eletype),
            D=list(beam.prescribed_displacements),
            GAv=GAv if GAv and any(g is not None for g in GAv) else None,
            kf=kf if kf and any(k is not None for k in kf) else None,
            npts=kwargs.pop("npts", beam_analysis.npts),
            **kwargs,
        )
        for load in beam.loads or []:
            model.add_load(load, case=case)
        return model

    def to_beam_analysis(
        self, combo_name: Optional[str] = None, case: Optional[str] = None
    ) -> BeamAnalysis:
        """
        Returns a new (unanalysed) pycba BeamAnalysis for the load combination
        'combo_name' or the load case 'case' (give one, not both). With neither,
        the beam is returned without loads.
        """
        if combo_name is not None and case is not None:
            raise ValueError("Give 'combo_name' or 'case', not both.")
        if combo_name is not None:
            LM = self.combined_loads(combo_name)
        elif case is not None:
            LM = [list(row) for row in self.load_cases[case]]
        else:
            LM = []
        return self._beam_analysis(LM)

    # ------------------------------------------------------------------
    # Loads and load combinations
    # ------------------------------------------------------------------

    @property
    def n_spans(self) -> int:
        return len(self.L)

    @property
    def length(self) -> float:
        return self.node_locations[-1] - self.node_locations[0]

    def add_load(self, load: Sequence[float], case: str = DEFAULT_CASE) -> None:
        """
        Adds a raw pycba load matrix row (e.g. [span, 2, P, a]) to load 'case'.
        The span number is 1-based.
        """
        row = _validate_load_row(load, self.n_spans)
        self.load_cases.setdefault(case, []).append(row)
        self._reset_results()

    def add_point_load(self, P: float, x: float, case: str = DEFAULT_CASE) -> None:
        """
        Adds a point load 'P' (positive down) at global location 'x'.
        A load located exactly on an internal node is applied to the end of the
        span to the left of the node.
        """
        span, a = self._locate(x)
        self.add_load([span + 1, 2, float(P), a], case)

    def add_moment_load(self, M: float, x: float, case: str = DEFAULT_CASE) -> None:
        """
        Adds a concentrated moment 'M' (positive counter-clockwise) at global
        location 'x'.
        """
        span, a = self._locate(x)
        self.add_load([span + 1, 4, float(M), a], case)

    def add_distributed_load(
        self,
        w1: float,
        w2: float,
        x1: Optional[float] = None,
        x2: Optional[float] = None,
        case: str = DEFAULT_CASE,
    ) -> None:
        """
        Adds a linearly varying distributed load from 'w1' at global location
        'x1' to 'w2' at 'x2' (positive down). Defaults to the full beam. The load
        is split at the nodes into one pycba load per span it covers.
        """
        X = self.node_locations
        x1 = X[0] if x1 is None else float(x1)
        x2 = X[-1] if x2 is None else float(x2)
        if x2 < x1:
            raise ValueError(f"x2 ({x2}) must not be less than x1 ({x1}).")
        self._check_within(x1)
        self._check_within(x2)
        if x2 - x1 <= _TOL:
            return

        def w_at(x: float) -> float:
            return w1 + (w2 - w1) * (x - x1) / (x2 - x1)

        for span in range(self.n_spans):
            s0, s1 = X[span], X[span + 1]
            start, end = max(x1, s0), min(x2, s1)
            if end - start <= _TOL:
                continue
            wa, wb = w_at(start), w_at(end)
            a, c = start - s0, end - start
            full_span = abs(a) <= _TOL and abs(c - self.L[span]) <= _TOL
            if wa == wb:
                row = [span + 1, 1, wa] if full_span else [span + 1, 3, wa, a, c]
            else:
                row = [span + 1, 5, wa, wb] if full_span else [span + 1, 5, wa, wb, a, c]
            self.add_load(row, case)

    def add_load_combo(
        self,
        name: str,
        factors: dict[str, float],
        combo_tags: Optional[list[str] | str] = None,
    ) -> None:
        """
        Adds a load combination: {load case name: factor}. Same signature as
        Pynite's FEModel3D.add_load_combo.
        """
        self.load_combos[name] = LoadCombo(name, dict(factors), combo_tags)
        self._reset_results()

    def combined_loads(self, combo_name: str) -> list[list]:
        """
        Returns the factored pycba load matrix of load combination 'combo_name'.
        Load cases named in the combination that have no loads contribute
        nothing (as in Pynite).
        """
        LM = []
        for case, factor in self.load_combos[combo_name].factors.items():
            if factor == 0.0 or case not in self.load_cases:
                continue
            LM.extend(factor_LM(self.load_cases[case], float(factor)))
        return LM

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze(
        self, npts: Optional[int] = None, check_stability: bool = True
    ) -> None:
        """
        Analyses every load combination with pycba. As in Pynite, a model
        without load combinations gets a default 'Combo 1' = 1.0 x 'Case 1'.

        The pycba BeamAnalysis of each combination is kept in
        'self.analyses[combo_name]' for use with pycba's own tools (plotting,
        dataframes, etc.).
        """
        if npts is not None:
            self.npts = int(npts)
        if not self.load_combos:
            self.add_load_combo(DEFAULT_COMBO, {DEFAULT_CASE: 1.0})
        self._reset_results()
        for combo_name in self.load_combos:
            beam_analysis = self._beam_analysis(self.combined_loads(combo_name))
            beam_analysis.analyze(npts=self.npts, check_stability=check_stability)
            self.analyses[combo_name] = beam_analysis
            self._stations[combo_name] = _span_stations(beam_analysis)
            self._reactions[combo_name] = _full_reactions(beam_analysis)

    # Aliases matching FEModel3D's analysis methods
    analyze_linear = analyze

    @property
    def solution(self) -> Optional[str]:
        """'Linear' once the model is analysed, otherwise None (as in Pynite)."""
        return "Linear" if self.analyses else None

    # ------------------------------------------------------------------
    # Pynite-shaped result views
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> dict[str, "NodeView"]:
        """{node name: NodeView} with Pynite-style node attributes and results."""
        if self._nodes is None:
            self._nodes = {
                name: NodeView(self, idx) for idx, name in enumerate(self.node_names)
            }
        return self._nodes

    @property
    def members(self) -> dict[str, "MemberView"]:
        """{member name: MemberView}: the whole beam as one Pynite-style physical member."""
        if self._members is None:
            self._members = {self.name: MemberView(self)}
        return self._members

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reset_results(self) -> None:
        self.analyses = {}
        self._stations = {}
        self._reactions = {}

    def _beam_analysis(self, LM: list[list]) -> BeamAnalysis:
        beam_analysis = BeamAnalysis(
            L=list(self.L),
            EI=list(self.EI),
            R=list(self.R),
            LM=[list(row) for row in LM],
            eletype=list(self.eletype),
            D=list(self.D),
            GAv=None if self.GAv is None else list(self.GAv),
            kf=None if self.kf is None else list(self.kf),
        )
        # Sample either side of concentrated loads so the shear (point load) and
        # moment (moment load) jumps are captured in the results
        shear_points: dict[int, list[float]] = {}
        for row in LM:
            if row[1] in (2, 4):
                shear_points.setdefault(int(row[0]) - 1, []).append(float(row[3]))
        if shear_points:
            beam_analysis.shear_points = {
                span: np.unique(points) for span, points in shear_points.items()
            }
        return beam_analysis

    def _check_within(self, x: float) -> None:
        X = self.node_locations
        if not (X[0] - _TOL <= x <= X[-1] + _TOL):
            raise ValueError(f"Location {x} is outside the beam ({X[0]} to {X[-1]}).")

    def _locate(self, x: float) -> tuple[int, float]:
        """Returns (0-based span index, local position) for global location 'x'."""
        x = float(x)
        self._check_within(x)
        span = self._span_index(x)
        a = min(max(x - self.node_locations[span], 0.0), self.L[span])
        return span, a

    def _span_index(self, x: float) -> int:
        """The 0-based span containing global 'x'; internal nodes belong to the left span."""
        idx = int(np.searchsorted(self.node_locations, x, side="left")) - 1
        return min(max(idx, 0), self.n_spans - 1)

    def _require_results(self, combo_name: str) -> list[dict[str, np.ndarray]]:
        if not self._stations:
            raise RuntimeError("The model has not been analysed. Run 'analyze()' first.")
        try:
            return self._stations[combo_name]
        except KeyError:
            raise KeyError(
                f"No results for load combination '{combo_name}'. "
                f"Available: {list(self._stations)}"
            ) from None

    def __repr__(self) -> str:
        return (
            f"CBAModel(name={self.name!r}, spans={self.L}, "
            f"load_cases={list(self.load_cases)}, load_combos={list(self.load_combos)})"
        )


# ----------------------------------------------------------------------
# Result views
# ----------------------------------------------------------------------

# Pynite result name -> (pycba station quantity, sign)
_ACTIONS = {
    "Fy": ("V", 1.0),
    "Mz": ("M", -1.0),
    "dy": ("D", 1.0),
}
# Pynite results that are always zero for a planar pycba beam
_ZERO_ACTIONS = {"Fx", "Fz", "Mx", "My", "dx", "dz"}


class NodeView:
    """
    A node of a CBAModel exposed with Pynite's Node3D attribute names:
    name, X/Y/Z, support_* flags, spring_* definitions, and the per-combination
    result dicts Rxn* (reactions) and DX..RZ (displacements).
    """

    def __init__(self, model: CBAModel, index: int):
        self.model = model
        self.index = index
        self.name = model.node_names[index]
        self.ID = index

    X = property(lambda self: self.model.node_locations[self.index])
    Y = property(lambda self: 0.0)
    Z = property(lambda self: 0.0)

    def _restraint(self, dof: int) -> float:
        return self.model.R[2 * self.index + dof]

    support_DY = property(lambda self: self._restraint(0) < 0)
    support_RZ = property(lambda self: self._restraint(1) < 0)
    support_DX = support_DZ = support_RX = support_RY = property(lambda self: False)

    def _spring(self, dof: int) -> list:
        k = self._restraint(dof)
        return [k, None, True] if k > 0 else [None, None, None]

    spring_DY = property(lambda self: self._spring(0))
    spring_RZ = property(lambda self: self._spring(1))
    spring_DX = spring_DZ = spring_RX = spring_RY = property(
        lambda self: [None, None, None]
    )

    def _reactions(self, dof: int) -> dict[str, float]:
        return {
            combo: float(reactions[2 * self.index + dof])
            for combo, reactions in self.model._reactions.items()
        }

    def _displacements(self, dof: int) -> dict[str, float]:
        return {
            combo: float(beam_analysis.beam_results.D[2 * self.index + dof])
            for combo, beam_analysis in self.model.analyses.items()
        }

    def _zeros(self) -> dict[str, float]:
        return {combo: 0.0 for combo in self.model.analyses}

    RxnFY = property(lambda self: self._reactions(0))
    RxnMZ = property(lambda self: self._reactions(1))
    RxnFX = RxnFZ = RxnMX = RxnMY = property(lambda self: self._zeros())
    DY = property(lambda self: self._displacements(0))
    RZ = property(lambda self: self._displacements(1))
    DX = DZ = RX = RY = property(lambda self: self._zeros())

    def __repr__(self) -> str:
        return f"NodeView({self.name!r}, X={self.X})"


class _SegmentView:
    """
    Shared Pynite-style result methods for a stretch of the beam between two
    nodes (the whole member, or a single span). 'x' is measured from the
    segment's i-node.
    """

    model: CBAModel
    i_index: int
    j_index: int
    name: str

    @property
    def i_node(self) -> NodeView:
        return self.model.nodes[self.model.node_names[self.i_index]]

    @property
    def j_node(self) -> NodeView:
        return self.model.nodes[self.model.node_names[self.j_index]]

    def L(self) -> float:
        X = self.model.node_locations
        return X[self.j_index] - X[self.i_index]

    # --- evaluation -----------------------------------------------------

    def _spans(self) -> range:
        return range(self.i_index, self.j_index)

    def _values(self, action: str, x_local: np.ndarray, combo_name: str) -> np.ndarray:
        x_local = np.asarray(x_local, dtype=float)
        if action in _ZERO_ACTIONS:
            self.model._require_results(combo_name)
            return np.zeros_like(x_local)
        quantity, sign = _ACTIONS[action]
        stations = self.model._require_results(combo_name)
        X = self.model.node_locations
        x_global = x_local + X[self.i_index]
        spans = np.searchsorted(X, x_global, side="left") - 1
        spans = np.clip(spans, self.i_index, self.j_index - 1)
        out = np.empty_like(x_global)
        for span in np.unique(spans):
            mask = spans == span
            s = stations[span]
            out[mask] = np.interp(x_global[mask], s["x"], s[quantity])
        return sign * out

    def _extreme(self, action: str, combo_name: str, func) -> float:
        if action in _ZERO_ACTIONS:
            self.model._require_results(combo_name)
            return 0.0
        quantity, sign = _ACTIONS[action]
        stations = self.model._require_results(combo_name)
        values = np.concatenate([sign * stations[span][quantity] for span in self._spans()])
        return float(func(values))

    def _array(self, action: str, n_points: int, combo_name: str, x_array=None) -> np.ndarray:
        if x_array is None:
            x_array = np.linspace(0, self.L(), n_points)
        x_array = np.asarray(x_array, dtype=float)
        return np.array([x_array, self._values(action, x_array, combo_name)])

    # --- Pynite Member3D/PhysMember3D API -----------------------------------

    def shear(self, Direction: str, x: float, combo_name: str = DEFAULT_COMBO) -> float:
        return float(self._values(Direction, np.array([x]), combo_name)[0])

    def moment(self, Direction: str, x: float, combo_name: str = DEFAULT_COMBO) -> float:
        return float(self._values(Direction, np.array([x]), combo_name)[0])

    def deflection(self, Direction: str, x: float, combo_name: str = DEFAULT_COMBO) -> float:
        return float(self._values(Direction, np.array([x]), combo_name)[0])

    def axial(self, x: float, combo_name: str = DEFAULT_COMBO) -> float:
        return float(self._values("Fx", np.array([x]), combo_name)[0])

    def torque(self, x: float, combo_name: str = DEFAULT_COMBO) -> float:
        return float(self._values("Mx", np.array([x]), combo_name)[0])

    def shear_array(self, Direction: str, n_points: int, combo_name: str = DEFAULT_COMBO, x_array=None):
        return self._array(Direction, n_points, combo_name, x_array)

    def moment_array(self, Direction: str, n_points: int, combo_name: str = DEFAULT_COMBO, x_array=None):
        return self._array(Direction, n_points, combo_name, x_array)

    def deflection_array(self, Direction: str, n_points: int, combo_name: str = DEFAULT_COMBO, x_array=None):
        return self._array(Direction, n_points, combo_name, x_array)

    def axial_array(self, n_points: int, combo_name: str = DEFAULT_COMBO, x_array=None):
        return self._array("Fx", n_points, combo_name, x_array)

    def torque_array(self, n_points: int, combo_name: str = DEFAULT_COMBO, x_array=None):
        return self._array("Mx", n_points, combo_name, x_array)

    def max_shear(self, Direction: str, combo_name: str = DEFAULT_COMBO) -> float:
        return self._extreme(Direction, combo_name, np.max)

    def min_shear(self, Direction: str, combo_name: str = DEFAULT_COMBO) -> float:
        return self._extreme(Direction, combo_name, np.min)

    def max_moment(self, Direction: str, combo_name: str = DEFAULT_COMBO) -> float:
        return self._extreme(Direction, combo_name, np.max)

    def min_moment(self, Direction: str, combo_name: str = DEFAULT_COMBO) -> float:
        return self._extreme(Direction, combo_name, np.min)

    def max_deflection(self, Direction: str, combo_name: str = DEFAULT_COMBO) -> float:
        return self._extreme(Direction, combo_name, np.max)

    def min_deflection(self, Direction: str, combo_name: str = DEFAULT_COMBO) -> float:
        return self._extreme(Direction, combo_name, np.min)

    def max_axial(self, combo_name: str = DEFAULT_COMBO) -> float:
        return self._extreme("Fx", combo_name, np.max)

    def min_axial(self, combo_name: str = DEFAULT_COMBO) -> float:
        return self._extreme("Fx", combo_name, np.min)

    def max_torque(self, combo_name: str = DEFAULT_COMBO) -> float:
        return self._extreme("Mx", combo_name, np.max)

    def min_torque(self, combo_name: str = DEFAULT_COMBO) -> float:
        return self._extreme("Mx", combo_name, np.min)


class SpanView(_SegmentView):
    """One span of a CBAModel, exposed like a Pynite Member3D (a sub-member)."""

    def __init__(self, model: CBAModel, span: int, name: str):
        self.model = model
        self.i_index = span
        self.j_index = span + 1
        self.name = name

    def __repr__(self) -> str:
        return f"SpanView({self.name!r}, L={self.L()})"


class MemberView(_SegmentView):
    """The whole CBAModel beam, exposed like a Pynite PhysMember3D."""

    def __init__(self, model: CBAModel):
        self.model = model
        self.i_index = 0
        self.j_index = model.n_spans
        self.name = model.name

    @property
    def sub_members(self) -> dict[str, SpanView]:
        """{name: SpanView}, one per span, named like Pynite's sub-members (M1a, M1b, ...)."""
        return {
            f"{self.name}{_span_suffix(span)}": SpanView(
                self.model, span, f"{self.name}{_span_suffix(span)}"
            )
            for span in range(self.model.n_spans)
        }

    def __repr__(self) -> str:
        return f"MemberView({self.name!r}, L={self.L()})"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _span_suffix(idx: int) -> str:
    """0 -> 'a', 25 -> 'z', 26 -> 'aa', ... (as Pynite names sub-members)."""
    suffix = ""
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        suffix = chr(97 + rem) + suffix
    return suffix


def _per_span(value: Any, n_spans: int, label: str, allow_none: bool = True) -> list:
    if isinstance(value, SectionEI) or (
        isinstance(value, (list, tuple)) and any(isinstance(v, SectionEI) for v in value)
    ):
        raise TypeError(
            f"Non-prismatic (SectionEI) '{label}' is not supported by pycba-tools yet."
        )
    if np.ndim(value) == 0:
        values = [value] * n_spans
    else:
        values = list(value)
    if len(values) != n_spans:
        raise ValueError(f"'{label}' needs 1 or {n_spans} values, got {len(values)}.")
    out = []
    for v in values:
        if v is None:
            if not allow_none:
                raise ValueError(f"'{label}' values cannot be None.")
            out.append(None)
        else:
            out.append(float(v))
    return out


def _validate_load_row(load: Sequence[float], n_spans: int) -> list:
    row = list(load)
    if len(row) < 3:
        raise ValueError(f"A pycba load row needs at least 3 entries, got {row}.")
    span, load_type = int(row[0]), int(row[1])
    if not 1 <= span <= n_spans:
        raise ValueError(f"Load span {span} is outside 1..{n_spans} (spans are 1-based).")
    if load_type not in (1, 2, 3, 4, 5, 6):
        raise ValueError(f"Unknown pycba load type {load_type}; expected 1-6.")
    if load_type != 6 and len(row) not in LOAD_ROW_LENGTHS[load_type]:
        raise ValueError(
            f"A type {load_type} load row needs {LOAD_ROW_LENGTHS[load_type]} entries, got {row}."
        )
    return [span, load_type] + [float(v) for v in row[2:]]


def _span_stations(beam_analysis: BeamAnalysis) -> list[dict[str, np.ndarray]]:
    """
    Per-span station arrays (global x, V, M, D, R) from an analysed BeamAnalysis.
    pycba pads each span with duplicated end stations; the interior stations are
    the evaluation grid.
    """
    stations = []
    for res in beam_analysis.beam_results.vRes:
        stations.append(
            {
                "x": np.asarray(res.x[1:-1], dtype=float),
                "V": np.asarray(res.V[1:-1], dtype=float),
                "M": np.asarray(res.M[1:-1], dtype=float),
                "D": np.asarray(res.D[1:-1], dtype=float),
                "R": np.asarray(res.R[1:-1], dtype=float),
            }
        )
    return stations


def _full_reactions(beam_analysis: BeamAnalysis) -> np.ndarray:
    """
    The reactions at every DOF (vertical, moment per node; zero at free DOFs),
    combining pycba's fixed-support reactions (R) and spring forces (Rs).
    """
    restraints = beam_analysis.beam.restraints
    results = beam_analysis.beam_results
    full = np.zeros(len(restraints))
    fixed = iter(np.atleast_1d(results.R))
    springs = iter(np.atleast_1d(results.Rs))
    for idx, restraint in enumerate(restraints):
        if restraint < 0:
            full[idx] = next(fixed)
        elif restraint > 0:
            full[idx] = next(springs)
    return full
