"""Stabilizer (tableau) simulation of Clifford circuits.

A state-vector simulator stores 2**n amplitudes. A stabilizer simulator
instead stores n Pauli operators that all have the state as a +1 eigenvector
("stabilizers"); for states made by Clifford gates this is a complete
description, and it costs n*n bits instead of 2**n numbers. That is why
error-correction circuits with hundreds of qubits can be simulated at all.

The data structure is the tableau of Aaronson and Gottesman, "Improved
simulation of stabilizer circuits" (2004), arXiv:quant-ph/0406196:

    row            x bits      z bits      sign r
    0 .. n-1       destabilizers  (helpers, needed only for measurement)
    n .. 2n-1      stabilizers

Row i stands for the Pauli string  (-1)**r[i] * P_0 P_1 ... P_{n-1}  where
qubit j carries I, X, Z or Y when (x[i,j], z[i,j]) is (0,0), (1,0), (0,1)
or (1,1). The start state |0...0> has stabilizers Z_0 .. Z_{n-1} and
destabilizers X_0 .. X_{n-1}.

A gate U replaces every row P by U P U†. For example H swaps X and Z, and
CNOT copies X from control to target and Z from target to control. Design
notes: docs/design.md D8-D9.

Example:
    >>> from helioq import Circuit
    >>> from helioq.stabilizer import stabilizers, run
    >>> bell = Circuit(2).h(0).cx(0, 1)
    >>> stabilizers(bell)
    ['+XX', '+ZZ']
    >>> run(bell.measure_all(), shots=5, seed=0)   # the two bits always agree
"""

from __future__ import annotations

import numpy as np

from helioq.circuit import Circuit, Gate

CLIFFORD_GATES = frozenset({"id", "x", "y", "z", "h", "s", "sdg", "cx", "cz", "swap"})

_PAULI_CHAR = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}


class Tableau:
    """The stabilizer state of ``n`` qubits. Starts in |0...0>."""

    def __init__(self, n_qubits: int) -> None:
        n = self.n = n_qubits
        # One extra row at the bottom (index 2n) is scratch space for measure.
        self.x = np.zeros((2 * n + 1, n), dtype=np.uint8)
        self.z = np.zeros((2 * n + 1, n), dtype=np.uint8)
        self.r = np.zeros(2 * n + 1, dtype=np.uint8)
        self.x[np.arange(n), np.arange(n)] = 1  # destabilizer i = X_i
        self.z[np.arange(n, 2 * n), np.arange(n)] = 1  # stabilizer i = Z_i

    def copy(self) -> Tableau:
        new = Tableau.__new__(Tableau)
        new.n = self.n
        new.x, new.z, new.r = self.x.copy(), self.z.copy(), self.r.copy()
        return new

    # --- Clifford gates: update every row at once (one column per qubit) ----

    def h(self, q: int) -> None:
        # X <-> Z, and Y -> -Y (so the sign flips where the row has a Y).
        self.r ^= self.x[:, q] & self.z[:, q]
        self.x[:, q], self.z[:, q] = self.z[:, q].copy(), self.x[:, q].copy()

    def s(self, q: int) -> None:
        # X -> Y, Y -> -X, Z -> Z.
        self.r ^= self.x[:, q] & self.z[:, q]
        self.z[:, q] ^= self.x[:, q]

    def sdg(self, q: int) -> None:
        # X -> -Y, Y -> X, Z -> Z.
        self.r ^= self.x[:, q] & (1 - self.z[:, q])
        self.z[:, q] ^= self.x[:, q]

    def pauli_x(self, q: int) -> None:
        # X anticommutes with Z and Y: flip the sign of rows that have them.
        self.r ^= self.z[:, q]

    def pauli_z(self, q: int) -> None:
        self.r ^= self.x[:, q]

    def pauli_y(self, q: int) -> None:
        self.r ^= self.x[:, q] ^ self.z[:, q]

    def cx(self, control: int, target: int) -> None:
        xc, zc = self.x[:, control], self.z[:, control]
        xt, zt = self.x[:, target], self.z[:, target]
        # The sign rule. Check it on X(c)Z(t) -> -Y(c)Y(t): xc=1, zt=1, and
        # xt == zc, so the bracket is 1 and the sign flips.
        self.r ^= xc & zt & (xt ^ zc ^ 1)
        xt ^= xc  # X spreads forward: control -> target
        zc ^= zt  # Z spreads backward: target -> control

    def cz(self, a: int, b: int) -> None:
        self.h(b)
        self.cx(a, b)
        self.h(b)

    def swap(self, a: int, b: int) -> None:
        self.x[:, [a, b]] = self.x[:, [b, a]]
        self.z[:, [a, b]] = self.z[:, [b, a]]

    def apply(self, gate: Gate) -> None:
        """Apply one unitary Clifford gate."""
        name, qubits = gate.name, gate.qubits
        if name not in CLIFFORD_GATES:
            raise ValueError(
                f"{name} is not a Clifford gate; the stabilizer simulator cannot run it"
            )
        if name == "id":
            return
        method = {"x": "pauli_x", "y": "pauli_y", "z": "pauli_z"}.get(name, name)
        getattr(self, method)(*qubits)

    # --- measurement ---------------------------------------------------------

    def _rowsum(self, rows, i: int) -> None:
        """Each row h in ``rows`` := row h times row i (Pauli strings), with sign.

        Multiplying single-qubit Paulis gives factors of i (XZ = -iY, ...).
        ``g`` counts those powers of i on each qubit; the total, plus twice
        the two old sign bits, is 0 or 2 mod 4 for commuting rows, i.e. the
        new sign is + or -. ``rows`` may be one index or an array of them
        (several rows are updated in one NumPy operation).
        """
        x1, z1 = self.x[i].astype(np.int8), self.z[i].astype(np.int8)
        x2, z2 = self.x[rows].astype(np.int8), self.z[rows].astype(np.int8)
        g = (
            x1 * z1 * (z2 - x2)  # row i has Y here
            + x1 * (1 - z1) * z2 * (2 * x2 - 1)  # row i has X here
            + (1 - x1) * z1 * x2 * (1 - 2 * z2)  # row i has Z here
        )
        total = 2 * self.r[rows].astype(int) + 2 * int(self.r[i]) + g.sum(axis=-1)
        self.r[rows] = (total % 4) // 2
        self.x[rows] ^= self.x[i]
        self.z[rows] ^= self.z[i]

    def measure(self, q: int, rng: np.random.Generator) -> int:
        """Measure qubit ``q`` in the Z basis; the state collapses. Returns 0/1."""
        n = self.n
        anticommuting = np.flatnonzero(self.x[n : 2 * n, q]) + n
        if anticommuting.size:
            # Some stabilizer anticommutes with Z_q: the outcome is random.
            p = int(anticommuting[0])
            others = np.flatnonzero(self.x[: 2 * n, q])
            self._rowsum(others[others != p], p)
            # The old stabilizer becomes the destabilizer; Z_q (with the
            # outcome as its sign) becomes the new stabilizer.
            self.x[p - n], self.z[p - n], self.r[p - n] = (
                self.x[p],
                self.z[p],
                self.r[p],
            )
            self.x[p], self.z[p] = 0, 0
            self.z[p, q] = 1
            self.r[p] = rng.integers(2)
            return int(self.r[p])
        # Otherwise Z_q is already (plus or minus) a product of stabilizers;
        # the destabilizers say which ones. Multiply them into a scratch row
        # and read the sign.
        return int(self._product_of_stabilizers(np.flatnonzero(self.x[:n, q])))

    def _product_of_stabilizers(self, which: np.ndarray) -> int:
        """Sign bit of the product of stabilizers number ``which`` (0-based)."""
        scratch = 2 * self.n
        self.x[scratch], self.z[scratch], self.r[scratch] = 0, 0, 0
        for i in which:
            self._rowsum(scratch, int(i) + self.n)
        return int(self.r[scratch])

    def reset(self, q: int, rng: np.random.Generator) -> None:
        if self.measure(q, rng):
            self.pauli_x(q)

    # --- reading the state ---------------------------------------------------

    def stabilizers(self) -> list[str]:
        """The n stabilizer generators as strings like '+XX' or '-ZIZ'."""
        n = self.n
        return [
            ("-" if self.r[i] else "+")
            + "".join(_PAULI_CHAR[self.x[i, j], self.z[i, j]] for j in range(n))
            for i in range(n, 2 * n)
        ]

    def canonical_stabilizers(self) -> list[str]:
        """Generators in a standard form: equal states give equal lists.

        The same state has many generator sets ({XX, ZZ} and {XX, -YY} both
        describe the Bell state), so never compare ``stabilizers()`` directly.
        This does Gaussian elimination on a copy, X bits first, then Z bits.
        """
        t = self.copy()
        n = t.n
        row = n
        for bits in (t.x, t.z):
            for q in range(n):
                pivots = np.flatnonzero(bits[row : 2 * n, q]) + row
                if pivots.size == 0:
                    continue
                p = int(pivots[0])
                if p != row:
                    for arr in (t.x, t.z):
                        arr[[row, p]] = arr[[p, row]]
                    t.r[[row, p]] = t.r[[p, row]]
                for i in range(n, 2 * n):
                    if i != row and bits[i, q]:
                        t._rowsum(i, row)
                row += 1
                if row == 2 * n:
                    return t.stabilizers()
        return t.stabilizers()


# --- running circuits -----------------------------------------------------------


def _apply_noise(t: Tableau, gate: Gate, rng: np.random.Generator) -> None:
    (q,), (p,) = gate.qubits, gate.params
    if rng.random() >= p:
        return
    if gate.name == "x_error":
        t.pauli_x(q)
    elif gate.name == "z_error":
        t.pauli_z(q)
    else:  # depolarize1: X, Y or Z with equal probability
        (t.pauli_x, t.pauli_y, t.pauli_z)[rng.integers(3)](q)


def run_once(circuit: Circuit, rng: np.random.Generator) -> tuple[Tableau, list[int]]:
    """Run the circuit one time. Returns the final tableau and the record."""
    t = Tableau(circuit.n_qubits)
    record: list[int] = []
    for gate in circuit:
        if gate.is_unitary:
            t.apply(gate)
        elif gate.name == "measure":
            record.append(t.measure(gate.qubits[0], rng))
        elif gate.name == "reset":
            t.reset(gate.qubits[0], rng)
        else:
            _apply_noise(t, gate, rng)
    return t, record


def run(circuit: Circuit, shots: int = 1, seed: int | None = None) -> np.ndarray:
    """Sample the measurement record ``shots`` times.

    Returns an array of shape (shots, circuit.n_measurements) of 0/1; column
    k is the k-th ``measure`` in the circuit.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros((shots, circuit.n_measurements), dtype=np.uint8)
    for shot in range(shots):
        out[shot] = run_once(circuit, rng)[1]
    return out


def stabilizers(circuit: Circuit, seed: int | None = None) -> list[str]:
    """Canonical stabilizer generators of the state the circuit prepares."""
    tableau, _ = run_once(circuit, np.random.default_rng(seed))
    return tableau.canonical_stabilizers()
