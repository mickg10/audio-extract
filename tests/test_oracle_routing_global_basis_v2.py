from audio_extract.oracle_routing_basis_v2 import DeduplicatedCandidate
from audio_extract.oracle_routing_global_basis_v2 import (
    GlobalBasisError,
    build_global_basis,
    materialize_scene_basis,
)


ALIASES = ("a", "b", "c")


def row(scene, name, aliases, digit):
    return DeduplicatedCandidate(
        work_id=scene,
        canonical_name=name,
        aliases=tuple(aliases),
        canonical_recipe_id="sha256:" + digit * 64,
        alias_recipe_ids=("sha256:" + digit * 64,),
        resolved_path=f"/{scene}/{name}.wav",
        artifact_pcm_sha256="sha256:" + digit * 64,
        container_sha256="sha256:" + digit * 64,
        frames=100,
        sample_rate_hz=44_100,
        channels=("FL", "FR"),
        subtype="FLOAT",
        parent_recipe_ids=(),
        executed_model_bundle_hashes=(),
    )


def test_scene_local_duplicate_does_not_collapse_global_vertices():
    scenes = {
        "voiced": [
            row("voiced", "ab", ("a", "b"), "1"),
            row("voiced", "c", ("c",), "3"),
        ],
        "no_vocal": [
            row("no_vocal", "a", ("a",), "1"),
            row("no_vocal", "b", ("b",), "2"),
            row("no_vocal", "c", ("c",), "3"),
        ],
    }
    basis = build_global_basis(
        scenes,
        required_aliases=ALIASES,
        alias_order=ALIASES,
    )
    assert [vertex.aliases for vertex in basis.vertices] == [
        ("a",), ("b",), ("c",)
    ]
    voiced = materialize_scene_basis(
        basis, "voiced", scenes["voiced"]
    )
    assert voiced[0] is voiced[1]
    assert voiced[0].artifact_pcm_sha256 == "sha256:" + "1" * 64


def test_aliases_collapse_only_when_equal_on_every_scene():
    scenes = {
        "s1": [
            row("s1", "ab", ("a", "b"), "1"),
            row("s1", "c", ("c",), "3"),
        ],
        "s2": [
            row("s2", "ab", ("a", "b"), "1"),
            row("s2", "c", ("c",), "3"),
        ],
    }
    basis = build_global_basis(
        scenes,
        required_aliases=ALIASES,
        alias_order=ALIASES,
    )
    assert [vertex.aliases for vertex in basis.vertices] == [
        ("a", "b"), ("c",)
    ]


def test_scene_mapping_order_cannot_change_identity_or_vertex_order():
    first = {
        "a_scene": [
            row("a_scene", "a", ("a",), "1"),
            row("a_scene", "b", ("b",), "2"),
            row("a_scene", "c", ("c",), "3"),
        ],
        "z_scene": [
            row("z_scene", "a", ("a",), "4"),
            row("z_scene", "b", ("b",), "5"),
            row("z_scene", "c", ("c",), "6"),
        ],
    }
    second = dict(reversed(tuple(first.items())))
    left = build_global_basis(
        first,
        required_aliases=ALIASES,
        alias_order=ALIASES,
    )
    right = build_global_basis(
        second,
        required_aliases=ALIASES,
        alias_order=ALIASES,
    )
    assert left == right


def test_missing_or_ambiguous_alias_is_refused():
    missing = {
        "s1": [
            row("s1", "a", ("a",), "1"),
            row("s1", "b", ("b",), "2"),
        ],
        "s2": [
            row("s2", "a", ("a",), "1"),
            row("s2", "b", ("b",), "2"),
            row("s2", "c", ("c",), "3"),
        ],
    }
    try:
        build_global_basis(
            missing,
            required_aliases=ALIASES,
            alias_order=ALIASES,
        )
    except GlobalBasisError as exc:
        assert "lacks required aliases" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing alias accepted")

    ambiguous = {
        "s1": [
            row("s1", "a1", ("a",), "1"),
            row("s1", "a2", ("a", "b"), "2"),
            row("s1", "c", ("c",), "3"),
        ],
        "s2": [
            row("s2", "a", ("a",), "1"),
            row("s2", "b", ("b",), "2"),
            row("s2", "c", ("c",), "3"),
        ],
    }
    try:
        build_global_basis(
            ambiguous,
            required_aliases=ALIASES,
            alias_order=ALIASES,
        )
    except GlobalBasisError as exc:
        assert "more than once" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("ambiguous alias accepted")
