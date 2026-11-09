"""
Parity tests: the same beam analysed with PyniteFEA + pynite-tools and with
pycba + pycba-tools must give the same result trees (same keys, same values to
within the accuracy of pycba's station sampling).
"""
import math

import numpy as np
import pynite_tools
import pytest

import pycba_tools as pt

REL = 2e-3  # pycba results are sampled/interpolated; Pynite's are closed form


def scale_of(value) -> float:
    """The largest absolute number in a (nested) result tree."""
    if isinstance(value, dict):
        return max([scale_of(v) for v in value.values()] + [0.0])
    if isinstance(value, (list, tuple)) and any(isinstance(v, dict) for v in value):
        return max([scale_of(v) for v in value] + [0.0])
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float)
        return float(np.nanmax(np.abs(arr))) if arr.size else 0.0
    if value is None or isinstance(value, bool):
        return 0.0
    return abs(float(value))


def assert_trees_close(expected, actual, path="", abs_tol=None, skip_keys=()):
    """Recursively compares two result trees."""
    if abs_tol is None:
        abs_tol = REL * max(scale_of(expected), 1e-12)
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert set(expected) == set(actual), f"{path}: {set(expected) ^ set(actual)}"
        for key in expected:
            if key in skip_keys:
                continue
            assert_trees_close(expected[key], actual[key], f"{path}/{key}", abs_tol, skip_keys)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(expected) == len(actual), path
        for idx, (e, a) in enumerate(zip(expected, actual)):
            assert_trees_close(e, a, f"{path}[{idx}]", abs_tol, skip_keys)
    elif isinstance(expected, np.ndarray):
        assert_arrays_close(expected, actual, path)
    elif isinstance(expected, bool) or expected is None:
        assert expected == actual, path
    else:
        assert math.isclose(expected, actual, rel_tol=REL, abs_tol=abs_tol), (
            f"{path}: {expected} != {actual}"
        )


def assert_arrays_close(expected: np.ndarray, actual: np.ndarray, path: str = ""):
    """
    Compares two (2, n) result arrays. A handful of stations right next to a
    concentrated load may differ (pycba interpolates across the jump), so allow
    a few mismatches.
    """
    assert expected.shape == actual.shape, path
    np.testing.assert_allclose(actual[0], expected[0], err_msg=path)
    tol = REL * max(np.abs(expected[1]).max(), 1e-12)
    mismatches = ~np.isclose(actual[1], expected[1], rtol=REL, atol=tol)
    assert mismatches.sum() <= 6, f"{path}: {mismatches.sum()} stations differ"


def test_node_reactions(model_pair):
    pynite_model, cba_model = model_pair
    assert_trees_close(
        pynite_tools.extract_node_reactions(pynite_model),
        pt.extract_node_reactions(cba_model),
    )


def test_node_reactions_selected_combos_no_key(model_pair):
    pynite_model, cba_model = model_pair
    kwargs = dict(load_combinations=["1.4D"], results_key=None)
    assert_trees_close(
        pynite_tools.extract_node_reactions(pynite_model, **kwargs),
        pt.extract_node_reactions(cba_model, **kwargs),
    )


def test_node_displacements(model_pair):
    pynite_model, cba_model = model_pair
    assert_trees_close(
        pynite_tools.extract_node_displacements(pynite_model),
        pt.extract_node_displacements(cba_model),
    )


def test_member_arrays(model_pair):
    pynite_model, cba_model = model_pair
    assert_trees_close(
        pynite_tools.extract_member_arrays(pynite_model, n_points=200),
        pt.extract_member_arrays(cba_model, n_points=200),
    )


def test_member_arrays_as_lists(model_pair):
    _, cba_model = model_pair
    arrays = pt.extract_member_arrays(cba_model, n_points=10, as_lists=True, actions=["Mz"])
    member = next(iter(arrays.values()))
    for combo_array in member["action_arrays"]["Mz"].values():
        assert isinstance(combo_array, list) and len(combo_array) == 2


def test_member_envelopes(model_pair):
    pynite_model, cba_model = model_pair
    assert_trees_close(
        pynite_tools.extract_member_envelopes(pynite_model),
        pt.extract_member_envelopes(cba_model),
    )


def test_span_envelopes(model_pair):
    pynite_model, cba_model = model_pair
    expected = pynite_tools.extract_span_envelopes(pynite_model, n_points=400)
    actual = pt.extract_span_envelopes(cba_model, n_points=400)
    # Locations of flat extremes (e.g. constant shear) are arbitrary; the span to
    # delta ratio of an all-zero action is None in one and huge (from 1e-17 noise)
    # in the other.
    assert_trees_close(
        expected,
        actual,
        skip_keys=("loc_max_delta", "loc_min_delta", "span_to_delta_max", "delta_abs_max"),
    )
    for member, actions in expected.items():
        for action, spans in actions["frame_span_envelopes"].items():
            for idx, span in spans.items():
                other = actual[member]["frame_span_envelopes"][action][idx]
                assert math.isclose(span["delta_abs_max"], other["delta_abs_max"], rel_tol=REL, abs_tol=1e-6)


def test_actions_by_location(model_pair):
    pynite_model, cba_model = model_pair
    member_name = next(iter(cba_model.members))
    length = cba_model.length
    # Avoid locations exactly at concentrated loads, where the value is one-sided
    locations = {member_name: [0.37 * length, 0.61 * length, 0.93 * length]}
    assert_trees_close(
        pynite_tools.extract_member_actions_by_location(
            pynite_model,
            force_extraction_locations=locations,
            # pynite-tools <= 0.8.1 fails when a member has locations but no ratios
            force_extraction_ratios={member_name: []},
        ),
        pt.extract_member_actions_by_location(cba_model, force_extraction_locations=locations),
    )


def test_actions_by_ratio_by_span(model_pair):
    pynite_model, cba_model = model_pair
    member_name = next(iter(cba_model.members))
    ratios = {member_name: [0.13, 0.52, 0.88]}
    assert_trees_close(
        pynite_tools.extract_member_actions_by_location(
            pynite_model, force_extraction_ratios=ratios, by_span=True
        ),
        pt.extract_member_actions_by_location(
            cba_model, force_extraction_ratios=ratios, by_span=True
        ),
    )


def test_load_combinations(model_pair):
    pynite_model, cba_model = model_pair
    assert pynite_tools.extract_load_combinations(pynite_model) == pt.extract_load_combinations(cba_model)


def test_node_views_match_pynite_nodes(model_pair):
    """The Pynite-style attributes that loadtracer reads directly."""
    pynite_model, cba_model = model_pair
    assert list(pynite_model.nodes) == list(cba_model.nodes)
    for name, pynite_node in pynite_model.nodes.items():
        node = cba_model.nodes[name]
        assert node.X == pynite_node.X
        assert node.support_DY == pynite_node.support_DY
        for attr in ("RxnFX", "RxnFY", "RxnFZ", "RxnMX", "RxnMY", "RxnMZ", "DY", "RZ"):
            expected = getattr(pynite_node, attr)
            actual = getattr(node, attr)
            assert set(expected) == set(actual)
            for combo in expected:
                assert actual[combo] == pytest.approx(expected[combo], rel=REL, abs=1e-8), (name, attr, combo)


def test_member_views_match_pynite_members(model_pair):
    pynite_model, cba_model = model_pair
    assert list(pynite_model.members) == list(cba_model.members)
    for name, pynite_member in pynite_model.members.items():
        member = cba_model.members[name]
        assert member.L() == pytest.approx(pynite_member.L())
        assert len(member.sub_members) == len(pynite_member.sub_members)
        for pynite_sub, sub in zip(pynite_member.sub_members.values(), member.sub_members.values()):
            assert sub.L() == pytest.approx(pynite_sub.L())
            assert sub.i_node.name == pynite_sub.i_node.name
            assert sub.j_node.name == pynite_sub.j_node.name
