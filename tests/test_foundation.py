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


def test_free_ends_carry_no_moment():
    """
    pycba recovers a foundation span's shear and moment from its bending
    alone. Undivided, this beam's free end shows 3% of the load as shear and
    1% of the peak moment; divided, 0.2% and nothing to speak of.
    """
    def add(m):
        m.add_distributed_load(30e3, 30e3, case="D")
        m.add_point_load(60e3, 3.0, case="D")

    member = analysed(beam(), add).members["beam"]
    peak = max(abs(member.max_moment("Mz", "D")), abs(member.min_moment("Mz", "D")))
    for end in (0.0, 6.0):
        assert abs(member.moment("Mz", end, "D")) < 1e-4 * peak
        assert abs(member.shear("Fy", end, "D")) < 3e-3 * 240e3  # Total load 240 kN.


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
