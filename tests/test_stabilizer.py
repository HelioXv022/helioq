"""Tests for the stabilizer (tableau) simulator."""

import numpy as np
import pytest

from helioq import Circuit, simulate
from helioq.stabilizer import CLIFFORD_GATES, Tableau, run, run_once, stabilizers

# --- states ------------------------------------------------------------------


def test_start_state_is_stabilized_by_z_on_every_qubit():
    assert stabilizers(Circuit(3)) == ["+ZII", "+IZI", "+IIZ"]


def test_x_flips_the_sign_of_z():
    assert stabilizers(Circuit(1).x(0)) == ["-Z"]


def test_h_turns_z_into_x():
    assert stabilizers(Circuit(1).h(0)) == ["+X"]


def test_s_turns_x_into_y_and_sdg_undoes_it():
    assert stabilizers(Circuit(1).h(0).s(0)) == ["+Y"]
    assert stabilizers(Circuit(1).h(0).sdg(0)) == ["-Y"]
    assert stabilizers(Circuit(1).h(0).s(0).sdg(0)) == ["+X"]


def test_bell_and_ghz():
    assert stabilizers(Circuit(2).h(0).cx(0, 1)) == ["+XX", "+ZZ"]
    assert stabilizers(Circuit(3).h(0).cx(0, 1).cx(1, 2)) == ["+XXX", "+ZIZ", "+IZZ"]


def test_cnot_sign_rule_x_z_becomes_minus_y_y():
    # Prepare the state stabilized by +X on q0 and +Z on q1, whose product is
    # X(0)Z(1). After CNOT the generators are XX and ZZ, and their product
    # XX * ZZ = -YY: the minus sign must come out of the tableau arithmetic.
    t, _ = run_once(Circuit(2).h(0).cx(0, 1), np.random.default_rng(0))
    t._rowsum(2, 3)  # stabilizer 0 := stabilizer 0 * stabilizer 1
    assert t.stabilizers()[0] == "-YY"


def test_same_state_from_different_circuits_has_same_canonical_form():
    a = Circuit(2).h(0).cx(0, 1)
    b = Circuit(2).h(1).cx(1, 0)
    c = Circuit(2).h(0).h(1).cz(0, 1).h(1)
    assert stabilizers(a) == stabilizers(b) == stabilizers(c)


def test_swap_moves_the_state():
    assert stabilizers(Circuit(2).x(0).swap(0, 1)) == ["+ZI", "-IZ"]


def test_non_clifford_gate_is_rejected():
    with pytest.raises(ValueError, match="not a Clifford"):
        run(Circuit(1).t(0))


# --- measurement -------------------------------------------------------------


def test_deterministic_measurements():
    assert run(Circuit(2).x(1).measure_all(), shots=5).tolist() == [[0, 1]] * 5


def test_plus_state_gives_random_bits():
    bits = run(Circuit(1).h(0).measure(0), shots=2000, seed=0)
    assert 0.45 < bits.mean() < 0.55


def test_bell_pair_outcomes_always_agree():
    bits = run(Circuit(2).h(0).cx(0, 1).measure_all(), shots=200, seed=1)
    assert (bits[:, 0] == bits[:, 1]).all()
    assert 0 < bits[:, 0].sum() < 200  # and both values do occur


def test_measurement_collapses_the_state():
    # Measuring the same qubit twice must give the same answer twice.
    bits = run(Circuit(1).h(0).measure(0).measure(0), shots=100, seed=2)
    assert (bits[:, 0] == bits[:, 1]).all()


def test_reset_returns_qubit_to_zero():
    circ = Circuit(2).h(0).cx(0, 1).reset(0).measure(0)
    assert run(circ, shots=50, seed=3).sum() == 0


def test_same_seed_same_record():
    circ = Circuit(3).h(0).h(1).h(2).measure_all()
    np.testing.assert_array_equal(run(circ, 20, seed=7), run(circ, 20, seed=7))


def test_ghz_on_200_qubits():
    # 2**200 amplitudes would never fit in memory; the tableau is 400 x 200.
    n = 200
    circ = Circuit(n).h(0)
    for q in range(n - 1):
        circ.cx(q, q + 1)
    bits = run(circ.measure_all(), shots=1, seed=0)[0]
    assert bits.sum() in (0, n)


# --- noise -------------------------------------------------------------------


def test_x_error_flips_with_the_given_probability():
    bits = run(Circuit(1).x_error(0.2, 0).measure(0), shots=4000, seed=0)
    assert 0.17 < bits.mean() < 0.23


def test_z_error_is_invisible_in_the_z_basis_but_not_in_x():
    assert run(Circuit(1).z_error(1.0, 0).measure(0), shots=20).sum() == 0
    assert run(Circuit(1).h(0).z_error(1.0, 0).h(0).measure(0), shots=20).all()


def test_depolarize_flips_z_measurement_two_thirds_of_the_time():
    # X and Y flip the bit, Z does not: flip probability = 2p/3.
    bits = run(Circuit(1).depolarize1(0.3, 0).measure(0), shots=6000, seed=0)
    assert 0.17 < bits.mean() < 0.23


def test_probability_must_be_between_0_and_1():
    with pytest.raises(ValueError):
        Circuit(1).x_error(1.5, 0)


# --- cross-check against the state-vector simulator (layer 2) ----------------

_PAULI = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1, -1]),
}


def pauli_matrix(text: str) -> np.ndarray:
    """'+XZ' or '-YI' as a dense matrix (big-endian, like design.md D3)."""
    out = np.eye(1)
    for ch in text[1:]:
        out = np.kron(out, _PAULI[ch])
    return -out if text[0] == "-" else out


@pytest.mark.xfail(raises=NotImplementedError, reason="layer 2 not written yet")
def test_stabilizers_agree_with_state_vector_on_random_clifford_circuits():
    rng = np.random.default_rng(0)
    names = sorted(CLIFFORD_GATES)
    for _ in range(50):
        n = int(rng.integers(2, 6))
        circ = Circuit(n)
        for _ in range(30):
            name = names[rng.integers(len(names))]
            if name in ("cx", "cz", "swap"):
                a, b = rng.choice(n, size=2, replace=False)
                getattr(circ, name)(int(a), int(b))
            else:
                getattr(circ, name)(int(rng.integers(n)))
        state = simulate(circ)
        for generator in stabilizers(circ):
            np.testing.assert_allclose(
                pauli_matrix(generator) @ state, state, atol=1e-9
            )


def test_tableau_copy_is_independent():
    t = Tableau(1)
    u = t.copy()
    u.h(0)
    assert t.stabilizers() == ["+Z"] and u.stabilizers() == ["+X"]
