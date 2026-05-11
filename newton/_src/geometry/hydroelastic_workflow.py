"""Hydroelastic contact workflow policy."""

from __future__ import annotations

import warp as wp

from .flags import HydroelasticContactWorkflow

HYDROELASTIC_WORKFLOW_CLASSIC = wp.int32(int(HydroelasticContactWorkflow.CLASSIC))
HYDROELASTIC_WORKFLOW_PRESSURE = wp.int32(int(HydroelasticContactWorkflow.PRESSURE))
HYDROELASTIC_MODE_COMPLIANT = wp.int32(2)


@wp.func
def resolve_pair_contact_workflow(
    mode_a: wp.int32,
    mode_b: wp.int32,
    workflow_a: wp.int32,
    workflow_b: wp.int32,
) -> wp.int32:
    """Resolve per-pair hydroelastic workflow from per-shape settings.

    Rule: if any compliant participant chooses pressure workflow, the pair uses
    pressure workflow; otherwise classic workflow.
    """
    if mode_a == HYDROELASTIC_MODE_COMPLIANT and workflow_a == HYDROELASTIC_WORKFLOW_PRESSURE:
        return HYDROELASTIC_WORKFLOW_PRESSURE
    if mode_b == HYDROELASTIC_MODE_COMPLIANT and workflow_b == HYDROELASTIC_WORKFLOW_PRESSURE:
        return HYDROELASTIC_WORKFLOW_PRESSURE
    return HYDROELASTIC_WORKFLOW_CLASSIC
