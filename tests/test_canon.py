import math

import pytest

from audio_extract import canon


def test_key_order_invariance():
    a = canon.canonicalize({"b": 1, "a": 2, "c": 3})
    b = canon.canonicalize({"c": 3, "a": 2, "b": 1})
    assert a == b == b'{"a":2,"b":1,"c":3}'


def test_nested_key_order_invariance():
    a = canon.canonicalize({"z": {"y": 1, "x": 2}})
    b = canon.canonicalize({"z": {"x": 2, "y": 1}})
    assert a == b == b'{"z":{"x":2,"y":1}}'


def test_array_order_preserved():
    assert canon.canonicalize([3, 1, 2]) == b"[3,1,2]"


def test_integer_valued_float_serializes_as_integer():
    assert canon.canonicalize({"x": 2.0}) == b'{"x":2}'
    assert canon.canonicalize(-0.0) == b"0"


def test_reject_non_integer_float():
    with pytest.raises(canon.CanonError):
        canon.canonicalize({"x": 0.1})


def test_reject_nonfinite():
    for bad in (math.inf, -math.inf, math.nan):
        with pytest.raises(canon.CanonError):
            canon.canonicalize({"x": bad})


def test_bool_is_not_number():
    assert canon.canonicalize({"t": True, "f": False}) == b'{"f":false,"t":true}'


def test_null():
    assert canon.canonicalize({"x": None}) == b'{"x":null}'


def test_string_escaping():
    assert canon.canonicalize('a"b\\c\n') == b'"a\\"b\\\\c\\n"'
    # control char without a short escape uses \u00xx (lowercase)
    assert canon.canonicalize("\x01") == b'"\\u0001"'


def test_unicode_emitted_verbatim_as_utf8():
    # Non-ASCII is NOT \u-escaped; it is carried by the UTF-8 encoding.
    assert canon.canonicalize("é") == '"é"'.encode("utf-8")


def test_non_string_key_rejected():
    with pytest.raises(canon.CanonError):
        canon.canonicalize({1: "x"})
