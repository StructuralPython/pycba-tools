"""
Round-trip tests for pycba_tools.serialize: every model is serialized,
deserialized, and both models are analyzed. Results must match exactly.
"""
import io
import json

import numpy as np
import pytest
from pycba import BeamAnalysis

import pycba_tools as pt


def hinged_model() -> pt.CBAModel:
    """Covers element types, springs, prescribed displacements and every load type."""
    model = pt.CBAModel(
        L=[5.0, 3.0, 4.0],
        EI=[30e3, 30e3, 20e3],
        R=[-1, -1, 0, 0, 2e3, 0, -1, 0],
        eletype=["FF", "FP", 1],
        D=[None, None, None, None, None, None, -0.002, None],
        name="G1",
        node_names=["A", "B", "C", "D"],
        npts=200,
    )
    model.add_load([1, 1, 4.0], case="D")
    model.add_load([1, 2, 10.0, 2.0], case="L")
    model.add_load([2, 3, 2.0, 0.5, 1.5], case="L")
    model.add_load([3, 4, 3.0, 2.0], case="W")
    model.add_load([3, 5, 1.0, 3.0], case="D")
    model.add_load([1, 5, 1.0, 3.0, 1.0, 2.0], case="W")
    model.add_load_combo("1.25D+1.5L", {"D": 1.25, "L": 1.5}, combo_tags=["strength"])
    model.add_load_combo("D+W", {"D": 1.0, "W": 1.0}, combo_tags="service")
    model.add_load_combo("D", {"D": 1.0})
    return model


def timoshenko_model() -> pt.CBAModel:
    model = pt.CBAModel(L=[4.0, 4.0], EI=1e4, supports=["pin", "roller", "roller"], GAv=[None, 5e3])
    model.add_distributed_load(3.0, 3.0, case="D")
    model.add_load_combo("D", {"D": 1.0})
    return model


MODELS = {
    "hinged": hinged_model,
    "timoshenko": timoshenko_model,
}


def results_of(model: pt.CBAModel) -> dict:
    model.analyze()
    return {
        "reactions": pt.extract_node_reactions(model),
        "displacements": pt.extract_node_displacements(model),
        "arrays": pt.extract_member_arrays(model, n_points=50, as_lists=True),
    }


@pytest.mark.parametrize("builder", MODELS.values(), ids=MODELS.keys())
def test_round_trip_dict(builder):
    model = builder()
    model_dict = pt.dump_dict(model)
    json.dumps(model_dict)  # JSON-safe
    remodel = pt.load_dict(model_dict)
    assert pt.dump_dict(remodel) == model_dict
    assert results_of(remodel) == results_of(model)


@pytest.mark.parametrize("builder", MODELS.values(), ids=MODELS.keys())
def test_round_trip_str(builder):
    model = builder()
    remodel = pt.loads(pt.dumps(model))
    assert pt.dump_dict(remodel) == pt.dump_dict(model)
    assert results_of(remodel) == results_of(model)


def test_round_trip_file_handles():
    model = hinged_model()
    buffer = io.StringIO()
    pt.dump(model, buffer, indent=4)
    buffer.seek(0)
    remodel = pt.load(buffer)
    assert pt.dump_dict(remodel) == pt.dump_dict(model)


def test_round_trip_json_file(tmp_path):
    model = hinged_model()
    filepath = tmp_path / "model.json"
    pt.to_json(model, filepath)
    remodel = pt.from_json(filepath)
    assert pt.dump_dict(remodel) == pt.dump_dict(model)


def test_serialize_dict(beam_model):
    """The pynite-tools serialize test, against a CBAModel."""
    model_dict = pt.dump_dict(beam_model)
    model = pt.load_dict(model_dict)
    for node_name in model.nodes:
        bmn = beam_model.nodes[node_name]
        smn = model.nodes[node_name]
        assert bmn.X == smn.X
        assert bmn.Y == smn.Y
        assert bmn.Z == smn.Z
        assert bmn.name == smn.name


def test_serialized_content():
    model_dict = pt.dump_dict(hinged_model())
    assert model_dict["backend"] == "pycba"
    assert model_dict["name"] == "G1"
    assert model_dict["node_names"] == ["A", "B", "C", "D"]
    assert model_dict["eletype"] == [1, 2, 1]
    assert model_dict["D"][6] == -0.002
    assert model_dict["load_cases"]["L"] == [[1, 2, 10.0, 2.0], [2, 3, 2.0, 0.5, 1.5]]
    assert model_dict["load_combos"]["D+W"] == {
        "name": "D+W",
        "factors": {"D": 1.0, "W": 1.0},
        "combo_tags": "service",
    }


def test_results_are_not_serialized():
    model = hinged_model()
    model.analyze()
    remodel = pt.load_dict(pt.dump_dict(model))
    assert remodel.solution is None
    assert remodel.analyses == {}


def test_dump_beam_analysis():
    beam_analysis = BeamAnalysis(
        [6.0, 4.0], 1e4, supports=["fixed", "roller", "free"], LM=[[1, 1, 5.0], [2, 2, 8.0, 4.0]]
    )
    remodel = pt.load_dict(pt.dump_dict(beam_analysis))
    assert list(remodel.load_cases) == ["Case 1"]
    remodel.analyze()
    beam_analysis.analyze(npts=remodel.npts)
    np.testing.assert_allclose(remodel.analyses["Combo 1"].beam_results.R, beam_analysis.beam_results.R)


def test_empty_load_case_survives():
    model = timoshenko_model()
    model.load_cases["Empty"] = []
    assert "Empty" in pt.load_dict(pt.dump_dict(model)).load_cases


def test_dump_rejects_other_objects():
    with pytest.raises(TypeError):
        pt.dump_dict({"not": "a model"})
