"""
Serialization of CBAModel objects to/from JSON-compatible data.

The function names and signatures mirror pynite_tools.serialize so that the two
packages can be used interchangeably.

The JSON contains the model's *input* data: the beam (spans, stiffnesses,
restraints, element types, prescribed displacements, shear rigidities, Winkler
moduli), node names and locations, load cases (as pycba load matrix rows) and
load combinations. Analysis results are not serialized; re-run the analysis
after loading.

A pycba BeamAnalysis may also be passed to the dump functions; it is serialized
as a CBAModel whose load matrix is load case "Case 1".
"""
import json
import pathlib
from typing import Literal, Optional, TextIO

from pydantic import BaseModel, ConfigDict, TypeAdapter, model_validator
from pycba import BeamAnalysis

from .model import CBAModel


def to_json(model: CBAModel | BeamAnalysis, filepath: str | pathlib.Path) -> None:
    """
    Serializes the model to a new JSON file at 'filepath'.
    """
    filepath = pathlib.Path(filepath)
    with open(filepath, "w") as file:
        dump(model, file)


def from_json(filepath: str | pathlib.Path) -> CBAModel:
    """
    Reads the JSON file at 'filepath' and returns the CBAModel.
    """
    with open(filepath, "r") as file:
        model = load(file)
    return model


def dump(model: CBAModel | BeamAnalysis, file_io: TextIO, indent: int = 2) -> None:
    """
    Writes the 'model' as a JSON data to the file-handler object, 'file_io'.

    'indent': the number of spaces to indent in the file.
    """
    model_dict = dump_dict(model)
    json.dump(model_dict, fp=file_io, indent=indent)


def dumps(model: CBAModel | BeamAnalysis, indent: int = 2) -> str:
    """
    Returns the model as JSON string.
    """
    model_schema = get_model_schema(model)
    return model_schema.model_dump_json(indent=indent)


def dump_dict(model: CBAModel | BeamAnalysis) -> dict:
    """
    Returns a Python dictionary representing the model.

    The Python dictionary is serializable to JSON.
    """
    model_schema = get_model_schema(model)
    return model_schema.model_dump(mode="json")


def load(file_io: TextIO) -> CBAModel:
    """
    Returns a CBAModel from the json data contained within the file.
    """
    json_data = json.load(file_io)
    return load_dict(json_data)


def loads(model_json: str) -> CBAModel:
    """
    Returns a CBAModel based on the provided 'model_json'.

    'model_json': a JSON-serialized str representing a CBAModel
    """
    model_schema = TypeAdapter(CBAModelSchema).validate_json(model_json)
    return model_schema.to_cbamodel()


def load_dict(model_dict: dict) -> CBAModel:
    """
    Returns a CBAModel based on the provided 'model_dict'.

    'model_dict': A JSON-serializable dict representing a CBAModel
    """
    model_schema = TypeAdapter(CBAModelSchema).validate_python(model_dict)
    return model_schema.to_cbamodel()


def get_model_schema(model: CBAModel | BeamAnalysis) -> "CBAModelSchema":
    """
    Returns a CBAModelSchema based on the supplied model.
    """
    if isinstance(model, BeamAnalysis):
        model = CBAModel.from_beam_analysis(model)
    if not isinstance(model, CBAModel):
        raise TypeError(
            f"Expected a pycba_tools.CBAModel or pycba.BeamAnalysis, got {type(model).__name__}."
        )
    return TypeAdapter(CBAModelSchema).validate_python(model, from_attributes=True)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class _Schema(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class LoadComboSchema(_Schema):
    name: str
    factors: dict[str, float]
    combo_tags: Optional[list[str] | str] = None


class CBAModelSchema(_Schema):
    """
    A continuous beam model: the pycba beam definition plus named nodes, load
    cases and load combinations.
    """

    backend: Literal["pycba"] = "pycba"
    name: str = "M1"
    node_names: list[str]
    node_locations: list[float]
    L: list[float]
    EI: list[float]
    R: list[float]
    eletype: list[int]
    D: list[Optional[float]]
    GAv: Optional[list[Optional[float]]] = None
    kf: Optional[list[Optional[float]]] = None
    npts: int = 500
    load_cases: dict[str, list[list[int | float]]] = {}  # pycba load matrix rows
    load_combos: dict[str, LoadComboSchema] = {}

    @model_validator(mode="after")
    def _check_lengths(self) -> "CBAModelSchema":
        n_spans = len(self.L)
        if len(self.node_names) != n_spans + 1 or len(self.node_locations) != n_spans + 1:
            raise ValueError("'node_names' and 'node_locations' need one entry per node.")
        return self

    def to_cbamodel(self) -> CBAModel:
        model = CBAModel(
            L=self.L,
            EI=self.EI,
            R=self.R,
            eletype=self.eletype,
            D=self.D,
            GAv=self.GAv,
            kf=self.kf,
            name=self.name,
            node_names=self.node_names,
            node_locations=self.node_locations,
            npts=self.npts,
        )
        for case, rows in self.load_cases.items():
            model.load_cases.setdefault(case, [])
            for row in rows:
                model.add_load(row, case=case)
        for combo in self.load_combos.values():
            model.add_load_combo(combo.name, dict(combo.factors), combo.combo_tags)
        return model
