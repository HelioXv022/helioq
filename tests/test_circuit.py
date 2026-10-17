"""Tests for layer 1: building and validating circuits."""

import pytest

from helioq import GATE_SPECS, Circuit, Gate


def test_bell_circuit_records_gates_in_order():
    bell = Circuit(2).h(0).cx(0, 1)
    assert bell.gates == (Gate("h", (0,)), Gate("cx", (0, 1)))
    assert len(bell) == 2


def test_builders_return_the_same_circuit_for_chaining():
    c = Circuit(1)
    assert c.h(0) is c


def test_parameters_are_stored_in_order():
    c = Circuit(1).u3(0.3, 0.1, 0.2, 0)
    assert c.gates[0].params == (0.3, 0.1, 0.2)


def test_gates_property_is_read_only_copy():
    c = Circuit(1).h(0)
    assert isinstance(c.gates, tuple)


def test_circuit_equality():
    assert Circuit(2).h(0).cx(0, 1) == Circuit(2).h(0).cx(0, 1)
    assert Circuit(2).h(0) != Circuit(2).h(1)
    assert Circuit(2) != Circuit(3)


def test_str_lists_gates():
    text = str(Circuit(2).h(0).rz(0.5, 1).cx(0, 1))
    assert text.splitlines() == [
        "Circuit(2 qubits, 3 gates)",
        "  h q[0]",
        "  rz(0.5) q[1]",
        "  cx q[0], q[1]",
    ]


def test_every_spec_has_a_builder_method():
    for name in GATE_SPECS:
        assert callable(getattr(Circuit, name)), name


@pytest.mark.parametrize(
    "make, error",
    [
        (lambda: Gate("foo", (0,)), ValueError),  # unknown gate
        (lambda: Gate("h", (0, 1)), ValueError),  # wrong qubit count
        (lambda: Gate("rx", (0,)), ValueError),  # missing parameter
        (lambda: Gate("cx", (1, 1)), ValueError),  # same qubit twice
        (lambda: Circuit(0), ValueError),  # empty circuit
        (lambda: Circuit(2).h(2), IndexError),  # qubit out of range
        (lambda: Circuit(2).cx(0, -1), IndexError),  # negative index
    ],
)
def test_invalid_input_is_rejected(make, error):
    with pytest.raises(error):
        make()
