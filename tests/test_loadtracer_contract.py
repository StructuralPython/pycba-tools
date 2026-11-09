"""
The contract that loadtracer relies on (loadtracer/analysis_models/frame_model.py).

loadtracer builds one beam per horizontal papermodel ("pm"), analyses it, reads
the support reactions from the model's nodes and serializes the model. This test
copies loadtracer's Pynite implementation verbatim and checks that a pycba-tools
implementation gives identical reactions and survives the same serialization.
"""
import io
from typing import Optional

import pytest
from Pynite import FEModel3D

import pycba_tools as pt

# ---------------------------------------------------------------------------
# Verbatim from loadtracer (UnstableMemberError replaced by ValueError)
# ---------------------------------------------------------------------------

SUPPORT_CONSTRAINTS = {
    "fixed": [1, 1, 1, 1, 1, 1],
    "pin": [1, 1, 1, 1, 1, 0],
    "roller": [0, 1, 0, 0, 0, 0],
}
REACTION_DIRECTIONS = ("FX", "FY", "FZ", "MX", "MY", "MZ")


def get_node_locations(pm: dict) -> list[tuple[float, Optional[list[int]]]]:
    length = pm["element_attributes"]["length"]
    supports = [s for s in pm["element_geometry"]["supports"] if not s.get("coincident")]
    fixities = [s.get("fixity", "pin" if idx == 0 else "roller") for idx, s in enumerate(supports)]
    if not ("fixed" in fixities or (len(supports) >= 2 and "pin" in fixities)):
        raise ValueError("unstable")
    nodes = [(s["location"], SUPPORT_CONSTRAINTS[f]) for s, f in zip(supports, fixities)]
    if supports[0]["location"] > 0.0:
        nodes.insert(0, (0.0, None))
    if supports[-1]["location"] < length:
        nodes.append((length, None))
    return nodes


def get_support_reactions(model, pm: dict) -> dict[int, dict]:
    load_combos = list(model.load_combos)
    nodes_by_location = {node.X: node for node in model.nodes.values()}
    acc = {}
    for idx, support in enumerate(pm["element_geometry"]["supports"]):
        if support.get("coincident"):
            acc[idx] = {}
            continue
        node = nodes_by_location[support["location"]]
        acc[idx] = {
            direction: {
                combo: _clean(getattr(node, f"Rxn{direction}")[combo]) for combo in load_combos
            }
            for direction in REACTION_DIRECTIONS
        }
    return acc


def _clean(value: float, abs_tol: float = 1e-9) -> float:
    value = float(value)
    return 0.0 if abs(value) < abs_tol else value


def build_pynite_frame_model(pm: dict) -> FEModel3D:
    model = FEModel3D()
    model.add_material(name="default", E=1, G=1, nu=0.5, rho=1.0, fy=1.0)
    model.add_section(name="default", A=1, Iz=1, Iy=1, J=1)
    for idx, (location, constraints) in enumerate(get_node_locations(pm)):
        node_name = f"N{idx}"
        model.add_node(node_name, X=location, Y=0, Z=0)
        if constraints is not None:
            model.def_support(node_name, *constraints)
    tag = pm["element_attributes"]["tag"]
    model.add_member(tag, "N0", node_name, "default", "default", rotation=0.0)
    load_cases = set()
    for dist_load in pm["loads"]["distributed_loads"]:
        for e in dist_load.get("expanded", []):
            load_cases.add(e["case"])
            model.add_member_dist_load(
                tag, direction=e["dir"], w1=e["start_magnitude"], w2=e["end_magnitude"],
                x1=e["start_loc"], x2=e["end_loc"], case=e["case"],
            )
    for point_load in pm["loads"]["point_loads"]:
        for e in point_load.get("expanded", []):
            load_cases.add(e["case"])
            model.add_member_pt_load(
                tag, direction=e["dir"], P=e["magnitude"], x=e["location"], case=e["case"]
            )
    for load_case in sorted(load_cases):
        model.add_load_combo(name=load_case, factors={load_case: 1.0})
    return model


# ---------------------------------------------------------------------------
# The pycba-tools equivalent of build_pynite_frame_model
# ---------------------------------------------------------------------------

TRANSVERSE = {"Fy", "FY"}
MOMENT = {"Mz", "MZ"}


def build_pycba_frame_model(pm: dict, EI: float = 1.0) -> pt.CBAModel:
    """
    Returns a CBAModel populated with the data from 'pm'. loadtracer's loads use
    Pynite's sign convention (negative Fy is downward), pycba's positive is
    downward, so transverse magnitudes are negated.
    """
    nodes = get_node_locations(pm)
    R = []
    for _, constraints in nodes:
        constraints = constraints or [0] * 6
        R.extend([-1 if constraints[1] else 0, -1 if constraints[5] else 0])
    model = pt.CBAModel(
        node_locations=[location for location, _ in nodes],
        EI=EI,
        R=R,
        name=pm["element_attributes"]["tag"],
    )
    load_cases = set()
    for dist_load in pm["loads"]["distributed_loads"]:
        for e in dist_load.get("expanded", []):
            if e["dir"] not in TRANSVERSE:
                raise ValueError(f"pycba cannot model distributed loads in '{e['dir']}'.")
            load_cases.add(e["case"])
            model.add_distributed_load(
                -e["start_magnitude"], -e["end_magnitude"], e["start_loc"], e["end_loc"], case=e["case"]
            )
    for point_load in pm["loads"]["point_loads"]:
        for e in point_load.get("expanded", []):
            load_cases.add(e["case"])
            if e["dir"] in TRANSVERSE:
                model.add_point_load(-e["magnitude"], e["location"], case=e["case"])
            elif e["dir"] in MOMENT:
                model.add_moment_load(e["magnitude"], e["location"], case=e["case"])
            else:
                raise ValueError(f"pycba cannot model point loads in '{e['dir']}'.")
    for load_case in sorted(load_cases):
        model.add_load_combo(name=load_case, factors={load_case: 1.0})
    return model


# ---------------------------------------------------------------------------
# Papermodels
# ---------------------------------------------------------------------------


def make_pm(tag, length, supports, point_loads=(), dist_loads=()):
    return {
        "element_attributes": {"tag": tag, "length": length, "orientation": "horizontal"},
        "element_geometry": {"supports": [s if isinstance(s, dict) else {"location": s} for s in supports]},
        "loads": {
            "point_loads": [{"expanded": list(point_loads)}],
            "distributed_loads": [{"expanded": list(dist_loads)}],
        },
    }


def pl(magnitude, location, case, direction="FY"):
    return {"magnitude": magnitude, "location": location, "case": case, "dir": direction}


def dl(w1, w2, x1, x2, case, direction="Fy"):
    return {
        "start_magnitude": w1, "end_magnitude": w2, "start_loc": x1, "end_loc": x2,
        "case": case, "dir": direction,
    }


PMS = {
    "simple": make_pm("J1", 4.2, [0.0, 4.2], dist_loads=[dl(-1.2, -1.2, 0, 4.2, "D"), dl(-1.9, -1.9, 0, 4.2, "L")]),
    # Point load over the middle of three supports: the end supports have zero
    # reaction but must still be reported in support order (loadtracer audit F5)
    "load_over_middle_support": make_pm("B1", 6.0, [0.0, 3.0, 6.0], point_loads=[pl(-10.0, 3.0, "D")]),
    "overhangs": make_pm(
        "B2", 9.3, [0.6, 4.1, 8.0],
        point_loads=[pl(-5.0, 0.0, "D"), pl(-7.5, 9.3, "L"), pl(-3.3, 2.2, "S")],
        dist_loads=[dl(-0.5, -2.5, 0.0, 4.1, "D"), dl(-1.0, -1.0, 3.0, 5.5, "L")],
    ),
    "fixed_and_coincident": make_pm(
        "B3", 5.0,
        [{"location": 0.0, "fixity": "fixed"}, {"location": 0.0, "coincident": True}, {"location": 5.0}],
        point_loads=[pl(-4.0, 2.0, "D"), pl(1.5, 3.0, "W", direction="Mz")],
        dist_loads=[dl(-2.0, -2.0, 0, 5.0, "D")],
    ),
    "fixed_cantilever": make_pm(
        "C1", 2.5, [{"location": 0.0, "fixity": "fixed"}], point_loads=[pl(-6.0, 2.5, "L")]
    ),
}


@pytest.mark.parametrize("pm", PMS.values(), ids=PMS.keys())
def test_support_reactions_match(pm):
    pynite_model = build_pynite_frame_model(pm)
    pynite_model.analyze_linear()
    cba_model = build_pycba_frame_model(pm)
    cba_model.analyze_linear()

    expected = get_support_reactions(pynite_model, pm)
    actual = get_support_reactions(cba_model, pm)
    assert expected.keys() == actual.keys()
    for idx in expected:
        assert expected[idx].keys() == actual[idx].keys()
        for direction in expected[idx]:
            for combo, value in expected[idx][direction].items():
                assert actual[idx][direction][combo] == pytest.approx(value, rel=1e-6, abs=1e-9), (
                    idx, direction, combo
                )


def test_zero_reactions_are_reported():
    pm = PMS["load_over_middle_support"]
    model = build_pycba_frame_model(pm)
    model.analyze()
    reactions = get_support_reactions(model, pm)
    assert [reactions[idx]["FY"]["D"] for idx in range(3)] == pytest.approx([0.0, 10.0, 0.0], abs=1e-9)


@pytest.mark.parametrize("pm", PMS.values(), ids=PMS.keys())
def test_dump_and_load_model(pm):
    """loadtracer's dump_model/load_model/from_papermodel use these functions."""
    model = build_pycba_frame_model(pm)
    model_dict = pt.dump_dict(model)

    buffer = io.StringIO()
    pt.dump(model, buffer, indent=2)
    buffer.seek(0)
    for remodel in (pt.load(buffer), pt.load_dict(model_dict)):
        remodel.analyze()
        model.analyze()
        assert get_support_reactions(remodel, pm) == get_support_reactions(model, pm)


def test_unsupported_directions_are_rejected():
    pm = make_pm("X1", 4.0, [0.0, 4.0], dist_loads=[dl(-1.0, -1.0, 0, 4.0, "D", direction="Fx")])
    with pytest.raises(ValueError, match="Fx"):
        build_pycba_frame_model(pm)


# A trapezoidal load over three or more spans. PyniteFEA (<= 3.2.1) mis-interpolates
# the load on the middle sub-members (PhysMember.descritize re-binds 'w1' before
# evaluating 'w(xj)'), so its reactions do not balance the applied load.
TRAPEZOID_PM = make_pm(
    "T1", 9.3, [0.6, 4.1, 8.0], dist_loads=[dl(-0.5, -2.5, 0.0, 9.3, "D")]
)


def test_trapezoid_over_many_spans_is_in_equilibrium():
    model = build_pycba_frame_model(TRAPEZOID_PM)
    model.analyze()
    reactions = get_support_reactions(model, TRAPEZOID_PM)
    applied = (0.5 + 2.5) / 2 * 9.3
    assert sum(r["FY"]["D"] for r in reactions.values()) == pytest.approx(applied)


@pytest.mark.xfail(reason="PyniteFEA <= 3.2.1 sub-member trapezoid interpolation bug", strict=False)
def test_trapezoid_over_many_spans_matches_pynite():
    test_support_reactions_match(TRAPEZOID_PM)
