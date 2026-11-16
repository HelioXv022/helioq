"""Layer 1: circuit representation (helioq's IR).

A circuit is an ordered list of gates acting on ``n_qubits`` qubits. Every
other layer reads this representation. See ``docs/design.md`` (D2, D5, D6)
for why it looks the way it does.

Example:
    >>> from helioq import Circuit
    >>> bell = Circuit(2).h(0).cx(0, 1)
    >>> print(bell)
    Circuit(2 qubits, 2 gates)
      h q[0]
      cx q[0], q[1]
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

# name -> (number of qubits, number of parameters)
GATE_SPECS: dict[str, tuple[int, int]] = {
    # single-qubit, no parameters
    "id": (1, 0),
    "x": (1, 0),
    "y": (1, 0),
    "z": (1, 0),
    "h": (1, 0),
    "s": (1, 0),
    "sdg": (1, 0),
    "t": (1, 0),
    "tdg": (1, 0),
    # single-qubit rotations
    "rx": (1, 1),
    "ry": (1, 1),
    "rz": (1, 1),
    "u3": (1, 3),  # u3(theta, phi, lam), OpenQASM convention
    # two-qubit
    "cx": (2, 0),  # qubits = (control, target)
    "cz": (2, 0),
    "swap": (2, 0),
}

# Operations that are NOT unitary gates (design.md D7). They share the same
# ``Gate`` container, but only some backends understand them: the stabilizer
# simulator runs all of them, the state-vector simulator runs none (yet).
# name -> (number of qubits, number of parameters)
NON_UNITARY_SPECS: dict[str, tuple[int, int]] = {
    "measure": (1, 0),  # Z-basis measurement; appends one bit to the record
    "reset": (1, 0),  # put the qubit back in |0>
    # Pauli noise channels; the single parameter is a probability
    "x_error": (1, 1),  # apply X with probability p
    "z_error": (1, 1),  # apply Z with probability p
    "depolarize1": (1, 1),  # apply X, Y or Z, each with probability p/3
}

# Everything a circuit may contain.
OP_SPECS: dict[str, tuple[int, int]] = {**GATE_SPECS, **NON_UNITARY_SPECS}

NOISE_OPS = frozenset({"x_error", "z_error", "depolarize1"})


@dataclass(frozen=True)
class Gate:
    """One operation: which gate, on which qubits, with which parameters.

    ``Gate`` only checks itself (known name, right counts). Whether the qubit
    indices fit a circuit is checked by ``Circuit.append``.
    """

    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.name not in OP_SPECS:
            raise ValueError(f"unknown gate {self.name!r}")
        n_qubits, n_params = OP_SPECS[self.name]
        if len(self.qubits) != n_qubits:
            raise ValueError(
                f"{self.name} acts on {n_qubits} qubit(s), got {len(self.qubits)}"
            )
        if len(self.params) != n_params:
            raise ValueError(
                f"{self.name} takes {n_params} parameter(s), got {len(self.params)}"
            )
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError(f"{self.name} uses the same qubit twice: {self.qubits}")
        if self.name in NOISE_OPS and not 0.0 <= self.params[0] <= 1.0:
            raise ValueError(f"{self.name} needs a probability, got {self.params[0]}")

    @property
    def is_unitary(self) -> bool:
        return self.name in GATE_SPECS

    def __str__(self) -> str:
        args = f"({', '.join(f'{p:g}' for p in self.params)})" if self.params else ""
        targets = ", ".join(f"q[{q}]" for q in self.qubits)
        return f"{self.name}{args} {targets}"


class Circuit:
    """An ordered list of gates on ``n_qubits`` qubits.

    Gate methods append in place and return ``self``, so they can be chained.
    """

    def __init__(self, n_qubits: int) -> None:
        if n_qubits < 1:
            raise ValueError("a circuit needs at least one qubit")
        self.n_qubits = n_qubits
        self._gates: list[Gate] = []

    # --- core -----------------------------------------------------------

    def append(self, gate: Gate) -> Circuit:
        for q in gate.qubits:
            if not 0 <= q < self.n_qubits:
                raise IndexError(
                    f"qubit {q} out of range for a {self.n_qubits}-qubit circuit"
                )
        self._gates.append(gate)
        return self

    @property
    def gates(self) -> tuple[Gate, ...]:
        """Read-only view, so passes cannot change a circuit by accident."""
        return tuple(self._gates)

    def __len__(self) -> int:
        return len(self._gates)

    def __iter__(self) -> Iterator[Gate]:
        return iter(self._gates)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Circuit):
            return NotImplemented
        return self.n_qubits == other.n_qubits and self._gates == other._gates

    @property
    def n_measurements(self) -> int:
        """How many bits one run of this circuit writes to the record."""
        return sum(g.name == "measure" for g in self._gates)

    def __str__(self) -> str:
        lines = [f"Circuit({self.n_qubits} qubits, {len(self)} gates)"]
        lines += [f"  {g}" for g in self._gates]
        return "\n".join(lines)

    __repr__ = __str__

    # --- convenience builders ---------------------------------------------

    def _add(self, name: str, *qubits: int, params: tuple[float, ...] = ()):
        return self.append(Gate(name, tuple(qubits), tuple(params)))

    def id(self, q: int) -> Circuit:
        return self._add("id", q)

    def x(self, q: int) -> Circuit:
        return self._add("x", q)

    def y(self, q: int) -> Circuit:
        return self._add("y", q)

    def z(self, q: int) -> Circuit:
        return self._add("z", q)

    def h(self, q: int) -> Circuit:
        return self._add("h", q)

    def s(self, q: int) -> Circuit:
        return self._add("s", q)

    def sdg(self, q: int) -> Circuit:
        return self._add("sdg", q)

    def t(self, q: int) -> Circuit:
        return self._add("t", q)

    def tdg(self, q: int) -> Circuit:
        return self._add("tdg", q)

    def rx(self, theta: float, q: int) -> Circuit:
        return self._add("rx", q, params=(theta,))

    def ry(self, theta: float, q: int) -> Circuit:
        return self._add("ry", q, params=(theta,))

    def rz(self, theta: float, q: int) -> Circuit:
        return self._add("rz", q, params=(theta,))

    def u3(self, theta: float, phi: float, lam: float, q: int) -> Circuit:
        return self._add("u3", q, params=(theta, phi, lam))

    def cx(self, control: int, target: int) -> Circuit:
        return self._add("cx", control, target)

    def cz(self, a: int, b: int) -> Circuit:
        return self._add("cz", a, b)

    def swap(self, a: int, b: int) -> Circuit:
        return self._add("swap", a, b)

    # --- non-unitary operations (design.md D7) ----------------------------

    def measure(self, *qubits: int) -> Circuit:
        """Measure each qubit in the Z basis, in the order given."""
        for q in qubits:
            self._add("measure", q)
        return self

    def measure_all(self) -> Circuit:
        return self.measure(*range(self.n_qubits))

    def reset(self, *qubits: int) -> Circuit:
        for q in qubits:
            self._add("reset", q)
        return self

    def x_error(self, p: float, *qubits: int) -> Circuit:
        for q in qubits:
            self._add("x_error", q, params=(p,))
        return self

    def z_error(self, p: float, *qubits: int) -> Circuit:
        for q in qubits:
            self._add("z_error", q, params=(p,))
        return self

    def depolarize1(self, p: float, *qubits: int) -> Circuit:
        for q in qubits:
            self._add("depolarize1", q, params=(p,))
        return self

    # --- drawing (see helioq.draw) ----------------------------------------

    def draw(self) -> str:
        """The circuit as a text diagram. ``print(circ.draw())``."""
        from helioq.draw import draw_text

        return draw_text(self)

    def to_svg(self) -> str:
        """The circuit as an SVG picture (a string you can save to a file)."""
        from helioq.draw import draw_svg

        return draw_svg(self)

    def _repr_svg_(self) -> str:
        # Jupyter calls this, so a circuit shows up as a picture in notebooks.
        return self.to_svg()
