"""Tests for the error-correction layer."""

from itertools import combinations

import numpy as np
import pytest

from helioq.qec import (
    CSSCode,
    LookupDecoder,
    memory_experiment,
    repetition_code,
    rotated_surface_code,
    steane_code,
)

CODES = [repetition_code(3), repetition_code(5), steane_code(), rotated_surface_code(3)]


def x_distance(code: CSSCode) -> int:
    """Smallest X error that no Z-check sees but that flips logical Z."""
    for weight in range(1, code.n + 1):
        for support in combinations(range(code.n), weight):
            e = np.zeros(code.n, dtype=np.uint8)
            e[list(support)] = 1
            if not (code.hz @ e % 2).any() and e @ code.logical_z % 2:
                return weight
    raise AssertionError("no logical operator found")


# --- codes -------------------------------------------------------------------


def test_code_sizes():
    surface = rotated_surface_code(3)
    assert (surface.n, len(surface.hx), len(surface.hz)) == (9, 4, 4)
    surface5 = rotated_surface_code(5)
    assert (surface5.n, len(surface5.hx), len(surface5.hz)) == (25, 12, 12)
    assert steane_code().n == 7


@pytest.mark.parametrize(
    "code, distance",
    [(repetition_code(3), 3), (repetition_code(5), 5), (steane_code(), 3),
     (rotated_surface_code(3), 3), (rotated_surface_code(5), 5)],
)  # fmt: skip
def test_code_distance_against_x_errors(code, distance):
    assert x_distance(code) == distance


def test_invalid_code_is_rejected():
    with pytest.raises(ValueError, match="anticommutes"):
        CSSCode("bad", hx=[[1, 0]], hz=[[1, 0]], logical_x=[1, 1], logical_z=[1, 0])


# --- decoder -----------------------------------------------------------------


def test_lookup_decoder_fixes_every_single_error():
    for code in CODES:
        decoder = LookupDecoder(code.hz)
        for q in range(code.n):
            error = np.zeros(code.n, dtype=np.uint8)
            error[q] = 1
            leftover = error ^ decoder.decode(code.hz @ error % 2)
            assert not (code.hz @ leftover % 2).any()  # back in the code space
            assert leftover @ code.logical_z % 2 == 0, code.name  # no logical flip


def test_lookup_decoder_returns_minimum_weight():
    decoder = LookupDecoder(repetition_code(5).hz)
    # Errors on qubits 0 and 1 and errors on 2, 3, 4 have the same syndrome.
    np.testing.assert_array_equal(decoder.decode([0, 1, 0, 0]), [1, 1, 0, 0, 0])


# --- memory experiments ------------------------------------------------------


@pytest.mark.parametrize("code", CODES, ids=lambda c: c.name)
@pytest.mark.parametrize("basis", ["Z", "X"])
def test_no_noise_means_no_logical_errors(code, basis):
    if basis == "X" and len(code.hx) == 0:
        pytest.skip("the repetition code does not protect the X basis")
    assert memory_experiment(code, p=0.0, basis=basis).logical_error_rate(30, 0) == 0


def test_repetition_code_matches_the_exact_formula():
    # d=3 fails when 2 or 3 of the qubits flip: 3 p^2 (1-p) + p^3.
    p = 0.1
    exp = memory_experiment(repetition_code(3), p, noise="x_error")
    assert exp.logical_error_rate(6000, seed=0) == pytest.approx(0.028, abs=0.008)


def test_below_threshold_bigger_codes_are_better():
    rates = [
        memory_experiment(repetition_code(d), 0.1, noise="x_error").logical_error_rate(
            3000, seed=0
        )
        for d in (3, 5, 7)
    ]
    assert rates[0] > rates[1] > rates[2]


def test_decoding_beats_not_decoding_on_the_surface_code():
    exp = memory_experiment(rotated_surface_code(3), p=0.03)
    decoded = exp.logical_error_rate(1500, seed=0)
    raw = exp.logical_error_rate(1500, seed=0, decode=False)
    assert decoded < raw / 3


def test_record_layout():
    code = rotated_surface_code(3)
    exp = memory_experiment(code, p=0.01)
    assert exp.circuit.n_qubits == 9 + 8
    assert exp.circuit.n_measurements == 8 + 8 + 9
