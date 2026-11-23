"""
Beams on a Winkler foundation: pycba's linear foundation ('kf'), and the
foundations pycba-tools analyses by contact iteration, stiffer in compression
than in tension ('kf_tension'): compression only (0) or bilinear.
"""
import numpy as np
import pynite_tools
import pytest
from pycba import BeamAnalysis

import pycba_tools as pt
from pycba_tools.model import _Mesh

EI = 2e8  # N m^2
KF = 2.5e6  # N/m per m (5 MPa/m under a 0.5 m wide beam)
LAMBDA = (KF / (4 * EI)) ** 0.25  # Hetényi's lambda, 1/m


def beam(L=(3.0, 3.0), kf_tension=None, supports=None, **kwargs) -> pt.CBAModel:
    n = len(L) + 1
    return pt.CBAModel(
        L=list(L),
        EI=EI,
        supports=supports or ["free"] * n,
        kf=KF,
        kf_tension=kf_tension,
        name="beam",
        **kwargs,
    )


def analysed(model: pt.CBAModel, add) -> pt.CBAModel:
    add(model)
    model.add_load_combo("D", {"D": 1.0})
    model.analyze()
    return model


def curve(model, action, x, combo="D"):
    member = model.members["beam"]
    method = {"dy": member.deflection_array, "Mz": member.moment_array, "Fy": member.shear_array}[action]
    return method(action, 0, combo, x_array=np.asarray(x, dtype=float))[1]


def reaction(model, n=3001, combo="D"):
    return model.members["beam"].foundation_reaction_array(n, combo)


# --- Linear foundation ---------------------------------------------------------


def test_linear_foundation_matches_hetenyi():
    """An infinite beam under a point load: w = P lambda / 2k, M = P / 4 lambda."""
    L = 16 / LAMBDA
    model = analysed(beam((L / 2, L / 2)), lambda m: m.add_point_load(100e3, L / 2, case="D"))
    assert model.solution == "Linear"
    w0 = -model.members["beam"].deflection("dy", L / 2, "D")
    assert w0 == pytest.approx(100e3 * LAMBDA / (2 * KF), rel=1e-4)
    m0 = abs(model.members["beam"].moment("Mz", L / 2, "D"))
    assert m0 == pytest.approx(100e3 / (4 * LAMBDA), rel=1e-4)


@pytest.mark.parametrize("kf_tension", [None, 0.0])
def test_free_ends_carry_no_shear_or_moment(kf_tension):
    """
    pycba 1.0.2 recovers a foundation span's shear and moment from bending
    alone: on this footing a free end showed 45 kN of shear (9% of the load).
    pycba-tools' recovery includes the foundation's reaction.
    """
    def add(m):
        m.add_point_load(400e3, 1.2, case="D")
        m.add_distributed_load(10e3, 30e3, 2.0, 6.0, case="D")

    model = pt.CBAModel(L=[6.0], EI=225e6, supports=["free", "free"], kf=KF, kf_tension=kf_tension, name="beam")
    member = analysed(model, add).members["beam"]
    for end in (0.0, 6.0):
        assert abs(member.shear("Fy", end, "D")) < 1e-8 * 480e3
        assert abs(member.moment("Mz", end, "D")) < 1e-8 * 480e3


def test_shear_is_the_slope_of_the_moment():
    """With the foundation's reaction in the recovery, dM/dx = V along the span."""
    model = analysed(beam((6.0,)), lambda m: m.add_point_load(100e3, 1.5, case="D"))
    for x in (np.linspace(0.05, 1.45, 200), np.linspace(1.55, 5.95, 600)):  # Either side of the load.
        moment = -curve(model, "Mz", x)  # Sagging positive: dM/dx = V.
        shear = curve(model, "Fy", x)
        slope = np.gradient(moment, x)
        np.testing.assert_allclose(slope[2:-2], shear[2:-2], atol=2e-3 * 100e3)


def test_the_recovery_is_only_for_pycba_tools_beams():
    """A plain pycba BeamAnalysis keeps pycba's own recovery."""
    plain = BeamAnalysis(L=[6.0], EI=[EI], R=[0, 0, 0, 0], LM=[[1, 2, 100e3, 1.5]], kf=[KF])
    plain.analyze(npts=200)
    assert abs(plain.beam_results.vRes[0].V[1]) > 1e3  # pycba 1.0.2's free-end shear.


def test_recovery_probe(monkeypatch):
    """The recovery is used only when pycba needs it and it passes the probe."""
    from pycba_tools import _recovery

    def probe(stock, patched):
        def result(patched_recovery):
            value = patched if patched_recovery else stock
            if isinstance(value, Exception):
                raise value
            return value
        return result

    for stock, patched, needed, warns in [
        (0.0, 0.0, False, False),  # pycba already exact: nothing changes.
        (0.06, 5e-7, True, False),  # pycba 1.0.2.
        (0.06, 0.06, False, True),  # The recovery no longer works.
        (0.06, AttributeError("Kcc_inv"), False, True),  # pycba's internals have changed.
    ]:
        monkeypatch.setattr(_recovery, "_needed", None)
        monkeypatch.setattr(_recovery, "_probe_end_shear", lambda patched, f=probe(stock, patched): f(patched))
        if warns:
            with pytest.warns(UserWarning, match="could not correct"):
                assert _recovery._recovery_needed() is needed
        else:
            assert _recovery._recovery_needed() is needed


def test_divided_beam_matches_the_beam():
    """The loads pycba takes on a foundation, split onto the pieces of a divided beam."""
    model = beam((4.0, 5.0), supports=["free", "roller", "free"])
    LM = [[1, 1, 10e3], [1, 2, 50e3, 1.3], [1, 3, 20e3, 0.7, 2.1], [2, 2, 30e3, 2.2], [2, 3, 4e3, 1.1, 2.5]]
    ref = BeamAnalysis(L=model.L, EI=model.EI, R=model.R, LM=LM, kf=model.kf)
    ref.analyze(npts=2000)
    mesh = _Mesh.divide(model, [0, 1])
    assert len(mesh.parent) > 2
    divided = BeamAnalysis(**mesh.beam_kwargs(model, [KF] * len(mesh.parent)), LM=mesh.split_loads(model, LM))
    divided.analyze(npts=100)

    def deflection(analysis):
        res = analysis.beam_results.vRes
        return np.concatenate([r.x[1:-1] for r in res]), np.concatenate([r.D[1:-1] for r in res])

    x = np.linspace(0.01, 8.99, 400)
    expected = np.interp(x, *deflection(ref))
    np.testing.assert_allclose(np.interp(x, *deflection(divided)), expected, atol=1e-5 * np.abs(expected).max())


def test_moment_and_trapezoid_on_a_foundation():
    """pycba's foundation spans take neither; pycba-tools divides the span to carry them."""
    def add(m):
        m.add_point_load(60e3, 1.0, case="D")
        m.add_moment_load(40e3, 4.2, case="D")
        m.add_distributed_load(5e3, 30e3, 0.5, 5.5, case="D")

    model = analysed(beam((2.5, 3.5)), add)
    member = model.members["beam"]
    x, p = reaction(model)
    assert np.trapezoid(p, x) == pytest.approx(60e3 + 0.5 * (5e3 + 30e3) * 5.0, rel=1e-3)
    # The moment's jump, and a free end's shear and moment.
    jump = member.moment("Mz", 4.2 + 1e-4, "D") - member.moment("Mz", 4.2 - 1e-4, "D")
    assert jump == pytest.approx(40e3, rel=2e-3)
    for end in (0.0, 6.0):
        assert abs(member.moment("Mz", end, "D")) < 1e-3 * 40e3
    # Moment and trapezoid alone are linear: no contact iteration.
    assert model.solution == "Linear"


# --- Contact iteration ---------------------------------------------------------


def test_no_uplift_is_the_linear_answer():
    """A compression-only foundation that stays in contact: the linear foundation's answer, in one pass."""
    def load(m):
        m.add_distributed_load(30e3, 30e3, case="D")
        m.add_point_load(60e3, 3.0, case="D")

    linear = analysed(beam(), load)
    _, p = reaction(linear)
    assert (p > 0).all()  # The linear foundation is in compression everywhere.
    contact = analysed(beam(kf_tension=0.0), load)
    assert contact.solution == "Nonlinear"
    x = np.linspace(0, 6, 61)
    np.testing.assert_allclose(curve(contact, "dy", x), curve(linear, "dy", x), rtol=1e-4)
    moment = curve(linear, "Mz", x)
    np.testing.assert_allclose(curve(contact, "Mz", x), moment, atol=5e-3 * np.abs(moment).max())


def test_compression_only_lifts_off():
    load = lambda m: m.add_point_load(100e3, 0.5, case="D")
    linear = analysed(beam(), load)
    contact = analysed(beam(kf_tension=0.0), load)
    x, p = reaction(contact)
    assert (p >= -1e-9).all()  # Never pulls the beam down.
    assert np.trapezoid(p, x) == pytest.approx(100e3, rel=1e-4)
    assert p[-1] == 0.0  # The far end has lifted off...
    assert curve(contact, "dy", [6.0])[0] > curve(linear, "dy", [6.0])[0] > 0  # ...further than the linear beam's.
    _, p_linear = reaction(linear)
    assert (p_linear < 0).any()  # The linear foundation holds the far end down.


def test_bilinear_lies_between():
    load = lambda m: m.add_point_load(100e3, 0.5, case="D")
    lift = {kt: curve(analysed(beam(kf_tension=kt), load), "dy", [6.0])[0] for kt in (None, 1e6, 0.0)}
    assert lift[None] < lift[1e6] < lift[0.0]


def test_compression_only_matches_pynite_springs():
    """Pynite with closely spaced compression-only springs: the same beam, lumped."""
    model = analysed(beam(kf_tension=0.0), lambda m: m.add_point_load(100e3, 0.5, case="D"))
    n, L = 300, 6.0
    h = L / n
    material = {"name": "m", "E": EI, "G": EI, "nu": 0.3, "rho": 0.0}
    section = {"name": "s", "A": 1.0, "Iy": 1.0, "Iz": 1.0, "J": 1.0}
    nodes = {
        f"N{i}": {
            "name": f"N{i}", "X": x, "Y": 0.0, "Z": 0.0,
            "support_DX": i == 0, "support_DY": False, "support_DZ": True,
            "support_RX": True, "support_RY": True, "support_RZ": False,
            "spring_DY": [KF * h * (0.5 if i in (0, n) else 1.0), "-", True],
        }
        for i, x in enumerate(np.linspace(0, L, n + 1))
    }
    pynite = pynite_tools.load_dict({
        "nodes": nodes, "materials": {"m": material}, "sections": {"s": section},
        "members": {"beam": {
            "name": "beam", "i_node": "N0", "j_node": f"N{n}", "material": material, "section": section,
            "PtLoads": [["FY", -100e3, 0.5, "D"]], "DistLoads": [],
        }},
        "load_combos": {"D": {"name": "D", "factors": {"D": 1.0}}},
    })
    pynite.analyze(check_statics=False)
    x = np.linspace(0, L, 61)
    expected = np.array([pynite.members["beam"].deflection("dy", xi, "D") for xi in x])
    np.testing.assert_allclose(curve(model, "dy", x), expected, atol=2e-3 * np.abs(expected).max())


def test_supports_and_foundation_share_the_load():
    """Node results are the model's own nodes, though the beam is divided."""
    model = analysed(
        beam(kf_tension=0.0, supports=["pin", "free", "free"], node_names=["A", "B", "C"]),
        lambda m: m.add_point_load(80e3, 4.5, case="D"),
    )
    assert list(model.nodes) == ["A", "B", "C"]
    x, p = reaction(model)
    support = model.nodes["A"].RxnFY["D"]
    assert support + np.trapezoid(p, x) == pytest.approx(80e3, rel=1e-4)
    assert model.nodes["B"].RxnFY["D"] == 0.0
    assert model.nodes["B"].DY["D"] == pytest.approx(curve(model, "dy", [3.0])[0], abs=1e-9)
    assert model.nodes["A"].DY["D"] == 0.0


def test_combinations_are_analysed_on_their_own():
    model = beam(kf_tension=0.0)
    model.add_point_load(100e3, 0.5, case="D")
    model.add_point_load(100e3, 5.5, case="L")
    model.add_load_combo("D", {"D": 1.0})
    model.add_load_combo("D+L", {"D": 1.0, "L": 1.0})
    model.analyze()
    # D alone lifts the far end; D+L is symmetric and stays down.
    assert curve(model, "dy", [6.0], "D")[0] > 0
    assert curve(model, "dy", [6.0], "D+L")[0] < 0


def test_net_uplift_is_unstable():
    model = beam(kf_tension=0.0)
    model.add_point_load(-50e3, 3.0, case="D")
    model.add_load_combo("D", {"D": 1.0})
    with pytest.raises(ValueError, match="lifts off its compression-only foundation"):
        model.analyze()


# --- Validation and serialization -----------------------------------------------


def test_kf_tension_cannot_be_negative():
    with pytest.raises(ValueError, match="cannot be negative"):
        beam(kf_tension=-1.0)


def test_kf_tension_needs_a_foundation():
    with pytest.raises(ValueError, match="no foundation"):
        pt.CBAModel(L=[3.0, 3.0], EI=EI, supports=["pin", "roller", "roller"], kf=[KF, None], kf_tension=[0.0, 0.0])


def test_kf_tension_round_trips():
    model = beam(kf_tension=[0.0, 1e6])
    model.add_point_load(100e3, 0.5, case="D")
    model.add_load_combo("D", {"D": 1.0})
    data = pt.dump_dict(model)
    assert data["kf_tension"] == [0.0, 1e6]
    loaded = pt.load_dict(data)
    model.analyze()
    loaded.analyze()
    x = np.linspace(0, 6, 31)
    np.testing.assert_allclose(curve(loaded, "dy", x), curve(model, "dy", x))


def test_old_models_load_as_linear():
    data = pt.dump_dict(beam())
    del data["kf_tension"]
    assert pt.load_dict(data).kf_tension is None


def test_contact_edge_does_not_depend_on_the_pieces(monkeypatch):
    """A node goes where the deflection changes sign, so the pieces' size hardly matters."""
    import pycba_tools.model as model_module

    tips = []
    for per in (10, 15, 20, 30):
        monkeypatch.setattr(model_module, "_PIECES_PER_LENGTH", per)
        model = analysed(beam((6.0,), kf_tension=0.0), lambda m: m.add_point_load(60e3, 1.0, case="D"))
        tips.append(model.nodes["N1"].DY["D"])
    assert max(tips) - min(tips) < 1e-4 * abs(tips[0])


def test_short_spans_with_a_moment_and_a_support():
    """Short spans get fewer pieces rather than pieces too short for pycba's stability check."""
    model = beam((0.3, 5.0, 0.4), kf_tension=0.0, supports=["free", "pin", "free", "free"])
    model.add_moment_load(10e3, 0.15, case="D")
    model.add_point_load(80e3, 4.0, case="D")
    model.add_distributed_load(0.0, 10e3, 0.0, 5.7, case="D")
    model.add_load_combo("D", {"D": 1.0})
    model.analyze()
    x, p = reaction(model, 4001)
    assert np.trapezoid(p, x) + model.nodes["N1"].RxnFY["D"] == pytest.approx(80e3 + 0.5 * 10e3 * 5.7, rel=1e-4)


def test_a_point_load_on_a_foundation_jumps_at_its_location():
    """A node goes under each point load, so the shear jumps exactly there."""
    model = pt.CBAModel(L=[6.0], EI=225e6, supports=["free", "free"], kf=KF, kf_tension=0.0, name="beam")
    model.add_point_load(400e3, 1.2, case="D")
    model.add_distributed_load(10e3, 30e3, 2.0, 6.0, case="D")
    model.add_load_combo("D", {"D": 1.0})
    model.analyze()
    member = model.members["beam"]
    jump = member.shear("Fy", 1.2 - 1e-6, "D") - member.shear("Fy", 1.2 + 1e-6, "D")
    assert jump == pytest.approx(400e3, rel=1e-6)


@pytest.mark.parametrize("x", [0.8, 1.2, 3.8, 6.8])
def test_the_shear_jump_is_on_the_right_side_of_a_load(x):
    """Wherever a point load falls, the shear just left of it is before the jump."""
    model = pt.CBAModel(L=[14.0], EI=9e9, supports=["free", "free"], kf=5e7, kf_tension=0.0, name="beam")
    model.add_point_load(1500e3, x, case="D")
    model.add_point_load(1500e3, 14.0 - x, case="D")
    model.add_load_combo("D", {"D": 1.0})
    model.analyze()
    member = model.members["beam"]
    before = member.shear("Fy", x - 1e-5, "D")
    after = member.shear("Fy", x + 1e-5, "D")
    assert before > 0 and after < 0  # The soil left of the load pushes up; past it the load wins.
    assert before - after == pytest.approx(1500e3, rel=1e-4)


def raft_strip() -> pt.CBAModel:
    """
    A raft strip wholly in contact: 18.2 m, 600 mm deep, 7.8 m wide on 5 MPa/m,
    with column loads as beamz9000 passes them (11.7 m from 11700 mm).
    """
    model = pt.CBAModel(
        L=[18.2],
        EI=26622e6 * 7.8 * 0.6**3 / 12,
        supports=["free", "free"],
        kf=5e6 * 7.8,
        kf_tension=0.0,
        name="beam",
    )
    for case, (edge, inner) in {"D": (1000e3, 1875e3), "L": (700e3, 1550e3), "S": (150e3, 300e3)}.items():
        for p, x in ((edge, 0.0), (inner, 6.5), (inner, 11.700000000000001), (edge, 18.2)):
            model.add_point_load(p, x, case=case)
        model.add_load_combo(case, {case: 1.0})
    return model


def test_a_beam_wholly_in_contact_settles():
    """
    The pads pycba leaves between a foundation's sub-elements read zero
    deflection (or +1e-10 of noise) and used to look like edges of the
    contact that came and went, so the contact never settled.
    """
    model = raft_strip()
    model.analyze()
    x = np.linspace(0, 18.2, 365)
    for combo, total in (("D", 5750e3), ("L", 4500e3), ("S", 900e3)):
        assert (curve(model, "dy", x, combo) < 0).all()
        xr, p = reaction(model, combo=combo)
        assert np.trapezoid(p, xr) == pytest.approx(total, rel=1e-4)


def test_no_padding_stations_on_a_foundation():
    """Shear, moment and deflection don't drop to zero where sub-elements meet."""
    from pycba_tools.model import _span_stations

    model = raft_strip()
    model.analyze()
    for s in _span_stations(model.analyses["D"]):
        assert (s["D"] < 0).all()
        runs = np.flatnonzero(np.isclose(np.diff(s["x"]), 0.0, atol=1e-9))
        assert not np.any(np.diff(runs) == 1)  # no 3 stations at one x
