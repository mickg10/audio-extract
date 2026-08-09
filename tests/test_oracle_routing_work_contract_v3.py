from copy import deepcopy

import pytest

from audio_extract.oracle_routing_work_contract_v3 import (
    HALL_REFERENCE_WORK_ID,
    NO_VOCAL_CONTROL_WORK_ID,
    REQUIRED_BASIS_ALIASES,
    REQUIRED_WORKS,
    WORK_CONTRACT_SHA256,
    identity_dict,
    semantic_sha256,
    validate_identity,
)


def test_work_contract_is_closed_hall_bearing_and_identity_hashed():
    value = identity_dict()
    assert tuple(value["required_voiced_works"]) == REQUIRED_WORKS
    assert tuple(value["required_basis_aliases"]) == REQUIRED_BASIS_ALIASES
    assert HALL_REFERENCE_WORK_ID in REQUIRED_WORKS
    assert value["source_roles"]["hall_reference"] == [HALL_REFERENCE_WORK_ID]
    assert value["source_roles"]["no_vocal_accompaniment"] == [NO_VOCAL_CONTROL_WORK_ID]
    assert semantic_sha256(value) == WORK_CONTRACT_SHA256
    validate_identity(value)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["required_voiced_works"].remove(HALL_REFERENCE_WORK_ID),
        lambda value: value["required_basis_aliases"].reverse(),
        lambda value: value["source_roles"].update(
            hall_reference=[NO_VOCAL_CONTROL_WORK_ID]
        ),
        lambda value: value["source_roles"].update(no_vocal_accompaniment=[]),
    ),
)
def test_work_contract_refuses_scope_or_order_mutation(mutation):
    value = deepcopy(identity_dict())
    mutation(value)
    with pytest.raises(ValueError, match="work contract"):
        validate_identity(value)
