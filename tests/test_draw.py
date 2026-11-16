"""Tests for circuit drawing."""

from helioq import Circuit
from helioq.draw import depth, layout


def test_bell_text_diagram():
    assert Circuit(2).h(0).cx(0, 1).draw().splitlines() == [
        "q0: ─H──●─",
        "        │",
        "q1: ────⊕─",
    ]


def test_vertical_line_crosses_the_wire_in_between():
    lines = Circuit(3).cx(0, 2).draw().splitlines()
    assert lines[2] == "q1: ─┼─"


def test_gates_on_different_qubits_share_a_column():
    assert len(layout(Circuit(2).h(0).h(1))) == 1
    assert len(layout(Circuit(2).h(0).h(0))) == 2


def test_gate_under_a_long_cnot_waits_for_the_next_column():
    # h(1) cannot sit in the column where the 0-2 CNOT line crosses wire 1.
    assert len(layout(Circuit(3).cx(0, 2).h(1))) == 2


def test_parameters_and_wide_labels_keep_wires_aligned():
    lines = Circuit(2).rx(0.5, 0).h(1).cx(0, 1).draw().splitlines()
    assert lines[0] == "q0: ─RX(0.5)──●─"
    assert len(lines[0]) == len(lines[2])


def test_measure_reset_and_noise_have_symbols():
    text = Circuit(1).x_error(0.1, 0).measure(0).reset(0).draw()
    assert text == "q0: ─Xerr(0.1)──M──|0>─"


def test_depth_counts_parallel_time_steps():
    assert depth(Circuit(3).h(0).h(1).h(2)) == 1
    assert depth(Circuit(3).h(0).cx(0, 1).cx(1, 2)) == 3
    assert depth(Circuit(2)) == 0


def test_svg_is_well_formed_xml():
    import xml.etree.ElementTree as ET

    circ = Circuit(3).h(0).cx(0, 1).swap(1, 2).cz(0, 2).rz(0.25, 1).reset(0)
    root = ET.fromstring(circ.measure_all().to_svg())
    assert root.tag.endswith("svg")
