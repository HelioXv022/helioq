"""Specification for layer 2 (the simulator).

These tests are marked xfail until ``helioq.simulator`` is implemented.
When all of them pass, delete the ``pytestmark`` line below.
"""

import numpy as np
import pytest

from helioq import Circuit, simulate
from helioq.circuit import GATE_SPECS, Gate
from helioq.simulator import gate_matrix

pytestmark = pytest.mark.xfail(raises=NotImplementedError, reason="not written yet")

S2 = 1 / np.sqrt(2)


def basis(bits: str) -> np.ndarray:
    """|bits⟩ as a state vector, big-endian: basis('10') means q0=1, q1=0."""
    v = np.zeros(2 ** len(bits), dtype=complex)
    v[int(bits, 2)] = 1
    return v


# --- starting point ---------------------------------------------------------


def test_empty_circuit_is_all_zeros_state():
    np.testing.assert_allclose(simulate(Circuit(3)), basis("000"))


# --- single gates -----------------------------------------------------------


def test_x_flips_qubit():
    np.testing.assert_allclose(simulate(Circuit(1).x(0)), basis("1"))


def test_h_makes_plus_state():
    np.testing.assert_allclose(simulate(Circuit(1).h(0)), [S2, S2])


def test_u3_on_zero():
    theta, phi, lam = 0.7, 0.4, -1.1
    expected = [np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)]
    np.testing.assert_allclose(simulate(Circuit(1).u3(theta, phi, lam, 0)), expected)


# --- qubit ordering (D3: big-endian) ----------------------------------------


def test_x_on_qubit_0_sets_leftmost_bit():
    np.testing.assert_allclose(simulate(Circuit(2).x(0)), basis("10"))


def test_x_on_qubit_1_sets_rightmost_bit():
    np.testing.assert_allclose(simulate(Circuit(2).x(1)), basis("01"))


def test_x_on_middle_qubit_of_three():
    np.testing.assert_allclose(simulate(Circuit(3).x(1)), basis("010"))


# --- two-qubit gates ---------------------------------------------------------


def test_cx_flips_target_only_when_control_is_one():
    np.testing.assert_allclose(simulate(Circuit(2).cx(0, 1)), basis("00"))
    np.testing.assert_allclose(simulate(Circuit(2).x(0).cx(0, 1)), basis("11"))


def test_cx_control_and_target_order_matters():
    # control = qubit 1, target = qubit 0
    np.testing.assert_allclose(simulate(Circuit(2).x(1).cx(1, 0)), basis("11"))
    np.testing.assert_allclose(simulate(Circuit(2).x(0).cx(1, 0)), basis("10"))


def test_cx_on_non_adjacent_qubits():
    np.testing.assert_allclose(simulate(Circuit(3).x(0).cx(0, 2)), basis("101"))


def test_swap():
    np.testing.assert_allclose(simulate(Circuit(2).x(0).swap(0, 1)), basis("01"))


# --- famous states -----------------------------------------------------------


def test_bell_state():
    state = simulate(Circuit(2).h(0).cx(0, 1))
    np.testing.assert_allclose(state, S2 * (basis("00") + basis("11")))


def test_ghz_state_three_qubits():
    state = simulate(Circuit(3).h(0).cx(0, 1).cx(1, 2))
    np.testing.assert_allclose(state, S2 * (basis("000") + basis("111")))


# --- physics sanity checks ---------------------------------------------------


@pytest.mark.parametrize("name", sorted(GATE_SPECS))
def test_every_gate_matrix_is_unitary(name):
    n_qubits, n_params = GATE_SPECS[name]
    gate = Gate(
        name, tuple(range(n_qubits)), tuple(0.37 * (k + 1) for k in range(n_params))
    )
    U = gate_matrix(gate)
    assert U.shape == (2**n_qubits, 2**n_qubits)
    np.testing.assert_allclose(U.conj().T @ U, np.eye(2**n_qubits), atol=1e-12)


def test_random_circuit_keeps_norm_one():
    rng = np.random.default_rng(0)
    c = Circuit(4)
    for _ in range(40):
        a, b = rng.choice(4, size=2, replace=False)
        c.u3(*rng.uniform(-np.pi, np.pi, size=3), int(a)).cx(int(a), int(b))
    assert np.isclose(np.linalg.norm(simulate(c)), 1.0)


def test_simulate_does_not_change_the_circuit():
    c = Circuit(2).h(0).cx(0, 1)
    before = c.gates
    simulate(c)
    assert c.gates == before
