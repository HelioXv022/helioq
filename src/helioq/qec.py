"""Quantum error correction basics: codes, syndrome circuits, decoding.

The pieces, in the order an experiment uses them:

1. ``CSSCode``            – a code, given by its X-type and Z-type checks.
2. ``memory_experiment``  – a circuit that stores a logical state, lets noise
                            act on it, and measures the checks (the syndrome).
3. ``LookupDecoder``      – syndrome -> most likely error.
4. ``logical_error_rate`` – run many shots on the stabilizer simulator,
                            decode each one, count how often the logical
                            qubit is wrong.

Example (the classic "bigger code = fewer errors, below threshold" curve):
    >>> from helioq.qec import repetition_code, memory_experiment
    >>> for d in (3, 5, 7):
    ...     exp = memory_experiment(repetition_code(d), p=0.05)
    ...     print(d, exp.logical_error_rate(shots=2000, seed=0))

Scope (design.md D10): noise acts only on data qubits, between two perfect
rounds of syndrome measurement ("code-capacity" noise). Noisy gates and noisy
measurements need detectors over many rounds and a matching decoder; that is
the next layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from helioq.circuit import Circuit
from helioq.stabilizer import run

# --- codes ----------------------------------------------------------------------


@dataclass(frozen=True)
class CSSCode:
    """A CSS stabilizer code encoding one logical qubit.

    ``hx`` and ``hz`` are 0/1 matrices with one row per check and one column
    per data qubit: row i of ``hz`` marks the qubits on which Z-check i acts.
    ``logical_x`` / ``logical_z`` mark the qubits of the logical operators.

    Z-checks detect X errors (they anticommute); X-checks detect Z errors.
    """

    name: str
    hx: np.ndarray
    hz: np.ndarray
    logical_x: np.ndarray
    logical_z: np.ndarray

    def __post_init__(self) -> None:
        for attr in ("hx", "hz", "logical_x", "logical_z"):
            arr = np.asarray(getattr(self, attr), dtype=np.uint8)
            if attr in ("hx", "hz"):
                arr = arr.reshape(-1, len(self.logical_x))
            object.__setattr__(self, attr, arr)
        # Two Paulis commute iff they overlap on an even number of X-Z pairs.
        if (self.hx @ self.hz.T % 2).any():
            raise ValueError("some X-check anticommutes with some Z-check")
        if (self.hx @ self.logical_z % 2).any() or (self.hz @ self.logical_x % 2).any():
            raise ValueError("a logical operator anticommutes with a check")
        if self.logical_x @ self.logical_z % 2 != 1:
            raise ValueError("logical X and logical Z must anticommute")

    @property
    def n(self) -> int:
        """Number of data qubits."""
        return len(self.logical_x)


def repetition_code(distance: int) -> CSSCode:
    """|0>_L = |00...0>, |1>_L = |11...1>. Corrects X errors only.

    Checks: Z_i Z_{i+1} for neighbouring qubits. It has no X-checks, so it
    cannot see Z errors at all; it is a classical code in quantum clothing,
    and the simplest place to see error correction work.
    """
    d = distance
    hz = np.zeros((d - 1, d), dtype=np.uint8)
    for i in range(d - 1):
        hz[i, i] = hz[i, i + 1] = 1
    logical_z = np.zeros(d, dtype=np.uint8)
    logical_z[0] = 1  # Z on any single qubit reads the logical bit
    return CSSCode(
        f"repetition-{d}",
        hx=np.zeros((0, d), dtype=np.uint8),
        hz=hz,
        logical_x=np.ones(d, dtype=np.uint8),  # flip every qubit
        logical_z=logical_z,
    )


def steane_code() -> CSSCode:
    """The [[7,1,3]] Steane code: X- and Z-checks are both the Hamming code."""
    hamming = np.array(
        [[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]],
        dtype=np.uint8,
    )
    ones = np.ones(7, dtype=np.uint8)
    return CSSCode("steane", hx=hamming, hz=hamming, logical_x=ones, logical_z=ones)


def rotated_surface_code(distance: int) -> CSSCode:
    """The rotated surface code [[d*d, 1, d]] on a d x d grid of data qubits.

    Data qubit (row, col) has index row*d + col. Checks live on the squares
    between data qubits, X-type and Z-type in a checkerboard. Squares that
    stick out of the grid keep only two qubits: X-type ones on the top and
    bottom edges, Z-type ones on the left and right edges.

        d = 3:        X
                 0 --- 1 --- 2
                 |  Z  |  X  |  Z
                 3 --- 4 --- 5
              Z  |  X  |  Z  |
                 6 --- 7 --- 8
                          X
    """
    d = distance
    if d < 3 or d % 2 == 0:
        raise ValueError("distance must be odd and at least 3")
    hx, hz = [], []
    for r in range(-1, d):  # (r, c) is the square's top-left corner
        for c in range(-1, d):
            is_x = (r + c) % 2 == 1
            on_top_bottom = r in (-1, d - 1)
            on_left_right = c in (-1, d - 1)
            if on_top_bottom and on_left_right:
                continue  # corner: would touch one qubit only
            if (on_top_bottom and not is_x) or (on_left_right and is_x):
                continue
            row = np.zeros(d * d, dtype=np.uint8)
            for rr, cc in ((r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)):
                if 0 <= rr < d and 0 <= cc < d:
                    row[rr * d + cc] = 1
            (hx if is_x else hz).append(row)
    logical_z = np.zeros(d * d, dtype=np.uint8)
    logical_z[:d] = 1  # Z along the top row
    logical_x = np.zeros(d * d, dtype=np.uint8)
    logical_x[::d] = 1  # X down the left column
    return CSSCode(f"surface-{d}", np.array(hx), np.array(hz), logical_x, logical_z)


# --- decoding -------------------------------------------------------------------


class LookupDecoder:
    """Minimum-weight decoding by table: syndrome -> smallest error causing it.

    Built by listing errors of weight 0, 1, 2, ... and recording the first
    one seen for each syndrome, so every entry is a lowest-weight error. This
    is exact but only works for small codes: the table has up to 2**(number
    of checks) entries.
    """

    def __init__(self, checks: np.ndarray, max_weight: int | None = None) -> None:
        checks = np.asarray(checks, dtype=np.uint8)
        n_checks, n = checks.shape
        self.n = n
        self.table: dict[bytes, np.ndarray] = {}
        n_syndromes = 2 ** np.linalg.matrix_rank(checks) if n_checks else 1
        if n_checks > 24:
            raise ValueError("code too large for a lookup table")
        for weight in range(n + 1 if max_weight is None else max_weight + 1):
            for support in combinations(range(n), weight):
                error = np.zeros(n, dtype=np.uint8)
                error[list(support)] = 1
                self.table.setdefault((checks @ error % 2).tobytes(), error)
            if len(self.table) >= n_syndromes:
                break

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Return the correction (0/1 per data qubit) for one syndrome."""
        key = np.asarray(syndrome, dtype=np.uint8).tobytes()
        return self.table.get(key, np.zeros(self.n, dtype=np.uint8))


# --- the memory experiment ------------------------------------------------------


def _measure_checks(circ: Circuit, code: CSSCode) -> None:
    """Append one round of syndrome extraction: Z-checks first, then X-checks.

    Each check has its own ancilla qubit, numbered after the data qubits.
    Z-check: CNOTs from the data qubits into the ancilla collect their parity.
    X-check: the same with the ancilla in the X basis (H before and after)
    and the CNOTs pointing the other way.
    """
    ancilla = code.n
    for check in code.hz:
        for q in np.flatnonzero(check):
            circ.cx(int(q), ancilla)
        circ.measure(ancilla).reset(ancilla)
        ancilla += 1
    for check in code.hx:
        circ.h(ancilla)
        for q in np.flatnonzero(check):
            circ.cx(ancilla, int(q))
        circ.h(ancilla)
        circ.measure(ancilla).reset(ancilla)
        ancilla += 1


@dataclass
class MemoryExperiment:
    """A memory circuit plus what is needed to interpret its record.

    The record of one shot is laid out as
        [round-1 syndrome | round-2 syndrome | final data measurement]
    where each syndrome lists the Z-checks, then the X-checks.
    """

    code: CSSCode
    basis: str
    circuit: Circuit
    _decoder: LookupDecoder | None = field(default=None, repr=False)

    @property
    def decoder(self) -> LookupDecoder:
        if self._decoder is None:
            self._decoder = LookupDecoder(self._checks)
        return self._decoder

    @property
    def _checks(self) -> np.ndarray:
        # A Z-basis experiment is hurt by X errors, which Z-checks detect.
        return self.code.hz if self.basis == "Z" else self.code.hx

    def split_record(self, record: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(syndrome change of the relevant checks, data bits) per shot."""
        n_z, n_x = len(self.code.hz), len(self.code.hx)
        m = n_z + n_x
        first, second, data = record[:, :m], record[:, m : 2 * m], record[:, 2 * m :]
        change = first ^ second  # a check that changed has "fired"
        relevant = change[:, :n_z] if self.basis == "Z" else change[:, n_z:]
        return relevant, data

    def logical_errors(self, shots: int, seed: int | None = None, decode: bool = True):
        """Run ``shots`` shots; return a 0/1 array, 1 = logical qubit wrong."""
        syndromes, data = self.split_record(run(self.circuit, shots, seed))
        logical = self.code.logical_z if self.basis == "Z" else self.code.logical_x
        if decode:
            corrections = np.array([self.decoder.decode(s) for s in syndromes])
            data = data ^ corrections.reshape(shots, self.code.n)
        return data @ logical % 2

    def logical_error_rate(
        self, shots: int = 1000, seed: int | None = None, decode: bool = True
    ) -> float:
        return float(self.logical_errors(shots, seed, decode).mean())


def memory_experiment(
    code: CSSCode, p: float, basis: str = "Z", noise: str = "depolarize1"
) -> MemoryExperiment:
    """Store logical |0> (basis "Z") or |+> (basis "X") through one noisy step.

    Circuit: prepare the data qubits, measure all checks (reference round),
    apply ``noise`` with probability ``p`` to every data qubit, measure all
    checks again, then measure the data qubits in ``basis``.
    """
    if basis not in ("Z", "X"):
        raise ValueError("basis must be 'Z' or 'X'")
    data = range(code.n)
    circ = Circuit(code.n + len(code.hz) + len(code.hx))
    if basis == "X":
        for q in data:
            circ.h(q)
    _measure_checks(circ, code)
    getattr(circ, noise)(p, *data)
    _measure_checks(circ, code)
    if basis == "X":
        for q in data:
            circ.h(q)
    circ.measure(*data)
    return MemoryExperiment(code, basis, circ)
