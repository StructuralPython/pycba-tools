import copy

import numpy as np
import pytest
from pycba import BeamAnalysis

import pycba_tools as pt


def two_span(**kwargs) -> pt.CBAModel:
    return pt.CBAModel(L=[4.0, 6.0], EI=1e4, supports=["pin", "roller", "roller"], **kwargs)


def test_defaults():
    model = two_span()
    assert model.node_names == ["N0", "N1", "N2"]
    assert model.node_locations == [0.0, 4.0, 10.0]
    assert model.eletype == [1, 1]
    assert model.R == [-1, 0, -1, 0, -1, 0]
    assert model.length == 10.0
    assert list(model.members) == ["M1"]
    assert list(model.members["M1"].sub_members) == ["M1a", "M1b"]


def test_node_locations_are_kept_exactly():
    locations = [0.1, 0.1 + 0.2, 1.7]
    model = pt.CBAModel(node_locations=locations, supports=["pin", "roller", "roller"])
    assert [node.X for node in model.nodes.values()] == locations


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),  # neither L nor node_locations
        dict(L=[4.0], R=[-1, 0, -1, 0], supports=["pin", "roller"]),  # both R and supports
        dict(L=[4.0]),  # neither R nor supports
        dict(L=[4.0, -1.0], supports=["pin", "roller", "roller"]),
        dict(L=[4.0], R=[-1, 0, -1]),
        dict(L=[4.0], EI=[1.0, 2.0], supports=["pin", "roller"]),
        dict(L=[4.0], supports=["pin", "roller"], node_names=["A", "A"]),
        dict(L=[4.0], supports=["pin", "roller"], node_locations=[0, 5]),
        dict(L=[4.0], supports=["pin", "roller"], D=[None, 0.1, None, None]),  # D at a free DOF
    ],
)
def test_invalid_models(kwargs):
    with pytest.raises(ValueError):
        pt.CBAModel(**kwargs)


@pytest.mark.parametrize(
    "row", [[0, 1, 5.0], [3, 1, 5.0], [1, 7, 5.0], [1, 2, 5.0], [1, 5, 1.0, 2.0, 3.0]]
)
def test_invalid_load_rows(row):
    with pytest.raises(ValueError):
        two_span().add_load(row)


def test_point_load_location():
    model = two_span()
    model.add_point_load(10.0, 2.5, case="D")
    model.add_point_load(10.0, 4.0, case="D")  # on the internal node: end of span 1
    model.add_point_load(10.0, 10.0, case="D")
    model.add_moment_load(2.0, 7.0, case="D")
    assert model.load_cases["D"] == [
        [1, 2, 10.0, 2.5],
        [1, 2, 10.0, 4.0],
        [2, 2, 10.0, 6.0],
        [2, 4, 2.0, 3.0],
    ]
    with pytest.raises(ValueError):
        model.add_point_load(1.0, 10.5)


def test_distributed_load_split_across_spans():
    model = two_span()
    model.add_distributed_load(2.0, 2.0, case="full")
    model.add_distributed_load(1.0, 1.0, 3.0, 5.0, case="partial")
    model.add_distributed_load(0.0, 10.0, 0.0, 10.0, case="trap")
    assert model.load_cases["full"] == [[1, 1, 2.0], [2, 1, 2.0]]
    assert model.load_cases["partial"] == [[1, 3, 1.0, 3.0, 1.0], [2, 3, 1.0, 0.0, 1.0]]
    assert model.load_cases["trap"] == [[1, 5, 0.0, 4.0], [2, 5, 4.0, 10.0]]
    for case, total in (("full", 20.0), ("partial", 2.0), ("trap", 50.0)):
        model.add_load_combo(case, {case: 1.0})
    model.analyze()
    for case, total in (("full", 20.0), ("partial", 2.0), ("trap", 50.0)):
        reactions = sum(node.RxnFY[case] for node in model.nodes.values())
        assert reactions == pytest.approx(total)


def test_default_combo_like_pynite():
    model = two_span()
    model.add_distributed_load(1.0, 1.0)
    model.analyze()
    assert list(model.load_combos) == ["Combo 1"]
    assert model.load_combos["Combo 1"].factors == {"Case 1": 1.0}
    assert model.solution == "Linear"


def test_combo_with_missing_case_has_no_loads():
    model = two_span()
    model.add_load_combo("Empty", {"X": 1.0})
    model.analyze()
    assert all(node.RxnFY["Empty"] == 0.0 for node in model.nodes.values())
    assert pt.extract_node_reactions(model) == {}


def test_combined_loads_are_factored():
    model = two_span()
    model.add_load([1, 5, 1.0, 2.0, 0.5, 1.0], case="D")
    model.add_load([2, 2, 3.0, 1.0], case="L")
    model.add_load_combo("C", {"D": 2.0, "L": 0.5, "W": 1.0})
    assert model.combined_loads("C") == [[1, 5, 2.0, 4.0, 0.5, 1.0], [2, 2, 1.5, 1.0]]


def test_results_need_analysis():
    model = two_span()
    model.add_load_combo("D", {"D": 1.0})
    with pytest.raises(RuntimeError):
        model.members["M1"].max_moment("Mz", "D")
    model.analyze()
    with pytest.raises(KeyError):
        model.members["M1"].max_moment("Mz", "nope")


def test_changing_the_model_clears_results():
    model = two_span()
    model.add_distributed_load(1.0, 1.0, case="D")
    model.add_load_combo("D", {"D": 1.0})
    model.analyze()
    model.add_point_load(1.0, 2.0, case="D")
    assert model.solution is None


def test_pycba_interop():
    beam_analysis = BeamAnalysis(
        [5.0, 5.0], 1e4, supports=["fixed", "roller", "roller"], LM=[[1, 1, 2.0], [2, 2, 5.0, 2.5]]
    )
    model = pt.CBAModel.from_beam_analysis(beam_analysis, case="D", name="B2")
    assert model.name == "B2"
    assert model.load_cases == {"D": [[1, 1, 2.0], [2, 2, 5.0, 2.5]]}
    model.add_load_combo("1.5D", {"D": 1.5})
    model.analyze()
    combo_analysis = model.to_beam_analysis(combo_name="1.5D")
    combo_analysis.analyze()
    np.testing.assert_allclose(combo_analysis.beam_results.R, 1.5 * _reactions(beam_analysis))
    # The pycba analysis of every combination is kept for pycba's own tools
    np.testing.assert_allclose(
        model.analyses["1.5D"].beam_results.R, combo_analysis.beam_results.R
    )


def _reactions(beam_analysis: BeamAnalysis) -> np.ndarray:
    beam_analysis.analyze()
    return beam_analysis.beam_results.R


def test_cantilever_detection(continuous_model):
    continuous_model.analyze()
    envelopes = pt.extract_span_envelopes(continuous_model, actions=["Mz"], n_points=50)
    spans = envelopes["B1"]["frame_span_envelopes"]["Mz"]
    assert [span["is_cantilever"] for span in spans.values()] == [False, False, True]


def test_model_add_combos(beam_model):
    combos = [{"name": "A", "factors": {"D": 1.0}}, {"name": "B", "factors": {"L": 2.0}}]
    copied = pt.model_add_combos(combos, beam_model, as_copy=True)
    assert {"A", "B"} <= set(copied.load_combos)
    assert "A" not in beam_model.load_combos
    assert pt.model_add_combos(combos, beam_model) is None
    assert {"A", "B"} <= set(beam_model.load_combos)


def test_deepcopy_analysed_model(beam_model):
    beam_model.analyze()
    clone = copy.deepcopy(beam_model)
    assert clone.nodes["N1"].RxnFY == beam_model.nodes["N1"].RxnFY
    assert clone.nodes["N1"].model is clone


def test_merge_result_trees(beam_model):
    beam_model.analyze()
    merged = pt.merge_result_trees(
        [pt.extract_member_envelopes(beam_model), pt.extract_node_reactions(beam_model)]
    )
    assert {"M1", "N1", "N2"} <= set(merged)
    assert "action_envelopes" in merged["M1"]


def test_sectionei_not_supported():
    from pycba import SectionEI

    with pytest.raises(TypeError):
        pt.CBAModel(L=[4.0], EI=SectionEI(lambda x: 1.0 + 0 * x), supports=["pin", "roller"])
