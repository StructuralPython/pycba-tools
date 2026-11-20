"""
Exact shear and moment on pycba's foundation spans.

pycba 1.0.2's FoundationElement divides a foundation span into sub-elements and
recovers the shear and moment along each from bending alone: each
sub-element's end forces leave out the foundation's stiffness, and the
foundation's reaction along the sub-element is not among the loads its
diagrams are built from. On a free-free footing under a 400 kN column, a free
end then shows 45 kN of shear, where there is none.

`exact_recovery(beam)` gives the foundation elements of one pycba Beam a
recovery that includes both: end forces from the bending and foundation
stiffness together, and the foundation's reaction (the modulus times the
deflection, linear between the sub-element's ends) as a load on each
sub-element. Nodal displacements and reactions are unchanged; only the
diagrams along the span are.

It applies only to beams pycba-tools builds, and only if pycba needs it: on
first use, a probe (a free-free beam on a foundation, whose ends carry no
shear) is analysed with pycba's own recovery and, if that is not exact, with
this one. If pycba is already exact, nothing changes; if this recovery does
not pass the probe (pycba's internals have changed), pycba's own is kept and
a warning says so.
"""
from __future__ import annotations

import types
import warnings
from typing import Optional

import numpy as np
from pycba import BeamAnalysis

_MARK = "_pycba_tools_exact_recovery"
# The probe's end shear over its load: pycba 1.0.2's own recovery gives 6e-2,
# this one 5e-7 (the foundation's reaction is taken as linear along each
# sub-element, where the deflection is cubic).
_EXACT = 1e-5
_needed: Optional[bool] = None  # None: not probed yet.


def exact_recovery(beam) -> None:
    """Gives the foundation elements of the pycba Beam 'beam' the exact recovery, if pycba needs it."""
    if not _recovery_needed():
        return
    _install(beam)


def _install(beam) -> None:
    original = beam._foundation

    def foundation(i_span: int):
        element = original(i_span)
        if not getattr(element, _MARK, False):
            element.recover = types.MethodType(_recover, element)
            setattr(element, _MARK, True)
        return element

    beam._foundation = foundation


def _recover(self, dr, loads, npts: int):
    """FoundationElement.recover, with the foundation in the sub-elements' forces and loads."""
    from pycba import foundation as fd
    from pycba.load import LoadTrapez

    dr = np.asarray(dr, dtype=float)
    Rfull, per, ref_e_list = self._rfull(loads)
    # Internal-node displacements, as pycba solves them: Kcc dc = Fc - Kcr dr.
    dc = self.Kcc_inv @ (-Rfull[self.c] - self.Kcr @ dr)
    d_mesh = np.zeros(self.ndof)
    d_mesh[self.r] = dr
    d_mesh[self.c] = dc

    npts_sub = max(2, int(round(npts / self.n)))
    k_e = fd._k_ebeam(self.EI, self.h) + fd._k_found(self.kf, self.h)
    results = []
    for e in range(self.n):
        d_e = d_mesh[2 * e : 2 * e + 4]
        f_e = k_e @ d_e + ref_e_list[e]
        # The foundation's reaction on the sub-element, as a load (positive
        # down): kf times the deflection, linear between its ends.
        soil = LoadTrapez(0, self.kf * d_e[0], self.kf * d_e[2], 0.0, self.h)
        res = fd._eb_member_values(self.h, self.EI, f_e, d_e, list(per[e]) + [soil], npts_sub)
        res.x = res.x + e * self.h
        results.append(res)
    return fd._concat_results(results)


def _probe_end_shear(patched: bool) -> float:
    """The larger end shear of a free-free beam on a foundation, over its load (exactly: 0)."""
    load = 100.0
    analysis = BeamAnalysis(
        L=[4.0], EI=[1e4], R=[0, 0, 0, 0], LM=[[1, 2, load, 1.0], [1, 1, 10.0]], kf=[2e3]
    )
    if patched:
        _install(analysis.beam)
    analysis.analyze(npts=200)
    res = analysis.beam_results.vRes[0]
    return max(abs(res.V[1]), abs(res.V[-2])) / (load + 40.0)


def _recovery_needed() -> bool:
    global _needed
    if _needed is None:
        if _probe_end_shear(patched=False) < _EXACT:
            _needed = False  # pycba recovers the foundation exactly.
        else:
            try:
                works = _probe_end_shear(patched=True) < _EXACT
            except Exception:  # pycba's internals have changed
                works = False
            if not works:
                warnings.warn(
                    "pycba-tools could not correct the shear and moment pycba recovers on "
                    "foundation spans with this version of pycba; they are pycba's own, which "
                    "leave out the foundation's reaction within each sub-element.",
                    stacklevel=3,
                )
            _needed = works
    return _needed
