# pycba-tools

> Super-charge your pycba workflows (and swap them with Pynite)

[pycba](https://github.com/ccaprani/pycba) is a fast, accurate continuous beam analysis library.
The tools in this package give pycba the model-level features that
[pynite-tools](https://github.com/StructuralPython/pynite-tools) gives PyniteFEA: named load
cases and load combinations, JSON serialization of analysis-ready models, and bulk export of
structured results.

The public API mirrors pynite-tools: the same function names, the same signatures, and **the same
result trees, with the same names and sign conventions**. Code written against pynite-tools can
switch to pycba-tools for continuous beams by changing the import.

Modules included:

- `model`: `CBAModel`, a pycba beam with named load cases, load combinations and named nodes,
  and Pynite-shaped result views (`model.nodes`, `model.members`)
- `reports`: Quickly export node and member results in structured dictionaries
- `serialize`: Export and import `CBAModel` objects to JSON
- `combos`: Convenience function for bulk-adding load combinations to `CBAModel` objects

## Installation

```
pip install pycba-tools
```

## Dependencies

- Python >= 3.11
- `pycba` >= 1.0.2
- `numpy` >= 2.0.0
- `deepmerge` >= 2.0.0
- `pydantic` >= 2.0.0

## Examples: `model`

pycba's `BeamAnalysis` analyses a single load matrix. `CBAModel` holds the beam definition, any
number of named load cases (stored as ordinary pycba load matrix rows) and load combinations,
and analyses every combination.

```python
import pycba_tools as pt

model = pt.CBAModel(
    L=[4.0, 6.0, 2.5],                          # span lengths
    EI=30e3,                                    # one value, or one per span
    supports=["pin", "roller", "roller", "free"],  # or a raw pycba R vector
    name="B1",                                  # member name used in result trees
    # node_names=["N0", "N1", "N2", "N3"],      # default
    # node_locations=[0.0, 4.0, 10.0, 12.5],    # optional, instead of L
    # eletype=["FF", "FF", "FF"], D=[...], GAv=..., kf=...   # all pycba options
)

# Loads use pycba's convention: positive is downward, moments positive counter-clockwise.
# Global-coordinate helpers split loads at the nodes into pycba load matrix rows...
model.add_distributed_load(5.0, 5.0, case="D")             # full length
model.add_distributed_load(2.0, 6.0, 2.0, 9.0, case="L")   # trapezoidal, x1 to x2
model.add_point_load(10.0, 7.0, case="D")
model.add_moment_load(3.0, 7.0, case="D")
# ...or add raw pycba rows (1-based span number)
model.add_load([3, 2, 6.0, 2.5], case="L")

model.add_load_combo("1.25D+1.5L", {"D": 1.25, "L": 1.5})
model.add_load_combo("D+L", {"D": 1.0, "L": 1.0})

model.analyze()

# Pynite-style results...
model.nodes["N1"].RxnFY["1.25D+1.5L"]
model.members["B1"].max_moment("Mz", "D+L")

# ...and the pycba BeamAnalysis of every combination, for pycba's own tools
model.analyses["D+L"].plot_results()
```

Existing pycba objects convert both ways:

```python
from pycba import BeamAnalysis

ba = BeamAnalysis([5.0, 5.0], 30e3, supports=["fixed", "roller", "roller"], LM=[[1, 1, 2.0]])
model = pt.CBAModel.from_beam_analysis(ba, case="D")   # the load matrix becomes case "D"
combo_ba = model.to_beam_analysis(combo_name="1.25D+1.5L")  # a fresh BeamAnalysis
```

### Sign conventions

Inputs use pycba's conventions. The result views and every `reports` function use Pynite's
conventions for a member along global X loaded in Y, so result trees can be swapped:

| Pynite name   | pycba            | Relationship             |
|---------------|------------------|--------------------------|
| `RxnFY`       | vertical reaction | same sign (up positive) |
| `RxnMZ`       | moment reaction  | same sign (CCW positive) |
| `DY`, `RZ`    | nodal displacement, rotation | same sign    |
| `Fy` (shear)  | `V`              | same sign                |
| `Mz` (moment) | `M`              | `Mz = -M`                |
| `dy`          | `D`              | same sign                |

A Pynite `Fy` load of `-w` (downward) is a pycba load of `+w`. pycba is a planar beam solver, so
the axial, torsional and out-of-plane results are always zero and are dropped from result trees
exactly as pynite-tools drops all-zero results.

## Examples: `reports`

```python
import pycba_tools as pr

model = pr.CBAModel(...) # Build and analyze your model here

# All the below functions optionally take a list of load combos
# and a results_key, exactly as in pynite-tools.

reactions = pr.extract_node_reactions(model)
node_displacements = pr.extract_node_displacements(model)
force_arrays = pr.extract_member_arrays(model, n_points=1000)
forces_minmax = pr.extract_member_envelopes(model)
forces_minmax_spans = pr.extract_span_envelopes(model)
forces_at_locations = pr.extract_member_actions_by_location(
    model,
    force_extraction_locations={"B1": [0, 2.0, 3.6]},
)
forces_at_location_ratios = pr.extract_member_actions_by_location(
    model,
    force_extraction_ratios={"B1": [0.25, 0.5, 0.75]},
    by_span=True,
)
merged_tree = pr.merge_result_trees([force_arrays, forces_minmax, forces_at_locations])
```

pycba results are evaluated at `npts` stations per span (default 500, set on `CBAModel` or
`analyze(npts=...)`) and interpolated between them; stations are added either side of every
concentrated load so shear and moment jumps are captured.

## Examples: `serialize`

```python
import pycba_tools as ps

## Dumping/serializing functions
with open('model.json', 'w') as file:
    ps.dump(model, file)
json_str = ps.dumps(model)
json_dict = ps.dump_dict(model)
ps.to_json(model, "model.json")

## Loading/de-serializing functions
with open('model.json', 'r') as file:
    remodel = ps.load(file)
remodel = ps.loads(json_str)
remodel = ps.load_dict(json_dict)
remodel = ps.from_json("model.json")

remodel.analyze()
```

A pycba `BeamAnalysis` can also be passed to the dump functions; its load matrix is stored as load
case `"Case 1"`.

### What gets serialized

The JSON contains the model's *input* data: the member name, node names and locations, spans,
`EI`, the restraint vector `R` (supports and springs), element types, prescribed displacements,
`GAv` and `kf`, the number of evaluation points, load cases (as pycba load matrix rows) and load
combinations. It is tagged `"backend": "pycba"`.

Analysis results are not serialized: re-run the analysis after loading. Non-prismatic
(`SectionEI`) members are not supported yet.

## Examples: `combos`

```python
import json
import pycba_tools as pc

with open("load_combo_library/current_combos.json", 'r') as file:
    load_combos = json.load(file)

pc.model_add_combos(load_combos, model)                      # in place
model = pc.model_add_combos(load_combos, model, as_copy=True)  # on a copy
```

## Using pycba-tools in place of Pynite in loadtracer

loadtracer's frame model only needs a model it can build, analyse, read support reactions from
and serialize. `CBAModel` provides `load_combos`, `analyze_linear()` and nodes with `X` and
`RxnFX`...`RxnMZ`, so loadtracer's `get_support_reactions` works unchanged, and `pycba_tools`
provides the `dump`/`dump_dict`/`load`/`load_dict`/`from_json` functions that `PyniteFrame` calls.
The only new code is the model builder:

```python
import pycba_tools as pt

def build_pycba_frame_model(pm: dict, EI: float = 1.0) -> pt.CBAModel:
    nodes = get_node_locations(pm)  # loadtracer's helper
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
            if e["dir"] not in ("Fy", "FY"):
                raise ValueError(f"pycba cannot model distributed loads in '{e['dir']}'.")
            load_cases.add(e["case"])
            # loadtracer uses Pynite's sign convention: negate transverse loads
            model.add_distributed_load(
                -e["start_magnitude"], -e["end_magnitude"], e["start_loc"], e["end_loc"], case=e["case"]
            )
    for point_load in pm["loads"]["point_loads"]:
        for e in point_load.get("expanded", []):
            load_cases.add(e["case"])
            if e["dir"] in ("Fy", "FY"):
                model.add_point_load(-e["magnitude"], e["location"], case=e["case"])
            elif e["dir"] in ("Mz", "MZ"):
                model.add_moment_load(e["magnitude"], e["location"], case=e["case"])
            else:
                raise ValueError(f"pycba cannot model point loads in '{e['dir']}'.")
    for load_case in sorted(load_cases):
        model.add_load_combo(name=load_case, factors={load_case: 1.0})
    return model
```

`tests/test_loadtracer_contract.py` runs loadtracer's own `get_support_reactions` against this
builder and against loadtracer's Pynite builder and checks they agree.

pycba models in-plane bending only: gravity (`Fy`/`FY`) loads and `Mz` moments. In-plane
(`Fx`) and out-of-plane (`Fz`) loads on beams are rejected rather than silently dropped.

## Testing

```
uv sync
uv run pytest
```

The test suite includes parity tests that analyse the same beams with PyniteFEA + pynite-tools and
with pycba + pycba-tools and compare the result trees.

## License

pycba-tools is licensed under the Apache License 2.0. Note that it depends on pycba, which is
licensed under the AGPL-3.0-or-later.
