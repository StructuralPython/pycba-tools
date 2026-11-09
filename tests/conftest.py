"""
Pairs of equivalent models: one built in PyniteFEA, one built as a pycba_tools.CBAModel.

Pynite beams run along global X and are loaded in Y. pycba loads are positive
downward, so a Pynite 'Fy' load of -w is a pycba load of +w; moment loads have the
same sign in both.
"""
from dataclasses import dataclass

import pytest
from Pynite import FEModel3D

import pycba_tools as pt

E = 200e6  # kPa
I = 1.0e-4  # m4


@dataclass
class BeamSpec:
    """A beam definition that can be built in both Pynite and pycba-tools."""

    name: str
    nodes: list[tuple[str, float]]
    supports: dict[str, str]  # node name -> "pin" | "roller" | "fixed"
    springs: dict[str, float]  # node name -> vertical spring stiffness
    point_loads: list[tuple[float, float, str]]  # (P (Pynite sign), x, case)
    moment_loads: list[tuple[float, float, str]]  # (M, x, case)
    dist_loads: list[tuple[float, float, float, float, str]]  # (w1, w2, x1, x2, case) Pynite sign
    combos: dict[str, dict[str, float]]

    def to_pynite(self) -> FEModel3D:
        model = FEModel3D()
        for node_name, x in self.nodes:
            model.add_node(node_name, x, 0, 0)
        model.add_material("mat", E, E / 2.6, 0.3, 1.0)
        model.add_section("sec", 0.01, I, I, I)
        model.add_member(self.name, self.nodes[0][0], self.nodes[-1][0], "mat", "sec")
        for node_name, support in self.supports.items():
            restraints = {
                # Out-of-plane DOFs are restrained so the 3D model is stable, like
                # loadtracer's frame model
                "pin": (True, True, True, True, True, False),
                "roller": (False, True, True, True, True, False),
                "fixed": (True, True, True, True, True, True),
            }[support]
            model.def_support(node_name, *restraints)
        for node_name, k in self.springs.items():
            model.def_support(node_name, False, False, True, True, True, False)
            model.def_support_spring(node_name, "DY", k)
        for P, x, case in self.point_loads:
            model.add_member_pt_load(self.name, "Fy", P, x, case)
        for M, x, case in self.moment_loads:
            model.add_member_pt_load(self.name, "Mz", M, x, case)
        for w1, w2, x1, x2, case in self.dist_loads:
            model.add_member_dist_load(self.name, "Fy", w1, w2, x1, x2, case)
        for combo_name, factors in self.combos.items():
            model.add_load_combo(combo_name, factors)
        return model

    def to_pycba(self) -> pt.CBAModel:
        names = [name for name, _ in self.nodes]
        R = []
        for name in names:
            if name in self.springs:
                R.extend([self.springs[name], 0])
            else:
                R.extend({"pin": [-1, 0], "roller": [-1, 0], "fixed": [-1, -1], None: [0, 0]}[
                    self.supports.get(name)
                ])
        model = pt.CBAModel(
            node_locations=[x for _, x in self.nodes],
            EI=E * I,
            R=R,
            name=self.name,
            node_names=names,
        )
        for P, x, case in self.point_loads:
            model.add_point_load(-P, x, case=case)
        for M, x, case in self.moment_loads:
            model.add_moment_load(M, x, case=case)
        for w1, w2, x1, x2, case in self.dist_loads:
            model.add_distributed_load(-w1, -w2, x1, x2, case=case)
        for combo_name, factors in self.combos.items():
            model.add_load_combo(combo_name, factors)
        return model


COMBOS = {"1.4D": {"D": 1.4}, "1.2D+1.6L": {"D": 1.2, "L": 1.6}, "D": {"D": 1.0}, "L": {"L": 1.0}}

SPECS = {
    "simple": BeamSpec(
        name="M1",
        nodes=[("N1", 0.0), ("N2", 14.0)],
        supports={"N1": "pin", "N2": "roller"},
        springs={},
        point_loads=[(-5.0, 7.0, "D"), (-8.0, 7.0, "L")],
        moment_loads=[],
        dist_loads=[],
        combos=COMBOS,
    ),
    "continuous_cantilever": BeamSpec(
        name="B1",
        nodes=[("N0", 0.0), ("N1", 4.0), ("N2", 10.0), ("N3", 12.5)],
        supports={"N0": "pin", "N1": "roller", "N2": "roller"},
        springs={},
        point_loads=[(-10.0, 7.0, "D"), (-6.0, 12.5, "L"), (-3.0, 1.5, "L")],
        moment_loads=[(3.0, 7.0, "D")],
        dist_loads=[
            (-5.0, -5.0, 0.0, 12.5, "D"),
            (-2.0, -6.0, 2.0, 9.0, "L"),
            (-4.0, -4.0, 5.0, 11.0, "L"),
        ],
        combos=COMBOS,
    ),
    "propped_spring": BeamSpec(
        name="P1",
        nodes=[("A", 0.0), ("B", 5.0), ("C", 8.0)],
        supports={"A": "fixed", "C": "roller"},
        springs={"B": 5000.0},
        point_loads=[(-10.0, 3.0, "D"), (-12.0, 6.5, "L")],
        moment_loads=[(-4.0, 2.0, "L")],
        dist_loads=[(-1.0, -3.0, 0.0, 8.0, "D")],
        combos=COMBOS,
    ),
}


@pytest.fixture(params=list(SPECS))
def model_pair(request) -> tuple[FEModel3D, pt.CBAModel]:
    """An analysed (Pynite model, CBAModel) pair."""
    spec = SPECS[request.param]
    pynite_model = spec.to_pynite()
    pynite_model.analyze_linear()
    cba_model = spec.to_pycba()
    cba_model.analyze()
    return pynite_model, cba_model


@pytest.fixture
def beam_model() -> pt.CBAModel:
    """The pynite-tools 'beam_model' test fixture as an (unanalysed) CBAModel."""
    return SPECS["simple"].to_pycba()


@pytest.fixture
def continuous_model() -> pt.CBAModel:
    return SPECS["continuous_cantilever"].to_pycba()
