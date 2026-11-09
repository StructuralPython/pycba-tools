"""
pycba-tools: Serialization and bulk results extraction for pycba continuous beam models, interchangeable with pynite-tools.

Modules included:

- model: CBAModel, a pycba beam with named load cases, load combinations and
  Pynite-shaped result views
- reports: Functions for bulk exporting structured analysis results
- serialize: Functions for serializing/deserializing CBAModel objects
- combos: Functions for bulk importing load combinations
"""

__version__ = "0.1.0"

from .model import CBAModel, LoadCombo

from .reports import (
    extract_node_reactions,
    extract_node_displacements,
    extract_member_arrays,
    extract_member_envelopes,
    extract_member_actions_by_location,
    extract_span_envelopes,
    extract_load_combinations,
    merge_result_trees,
)

from .serialize import (
    to_json,
    from_json,
    dump,
    dumps,
    dump_dict,
    load,
    loads,
    load_dict,
)

from .combos import (
    model_add_combos,
)
