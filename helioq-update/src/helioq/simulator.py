"""Layer 2: state-vector simulation.  ← Qiqi's part to write

Goal: ``simulate(circuit)`` returns the final state vector (2**n complex
amplitudes), starting from |00...0⟩. The tests in ``tests/test_simulator.py``
describe exactly what it must do; they currently "xfail" (expected to fail)
and will turn green as you implement the functions below.

Conventions you must follow (docs/design.md):
- D2: a gate only has a name, qubits and params. Its matrix comes from a
  lookup table you write here (``gate_matrix``).
- D3: big-endian. Qubit 0 is the leftmost bit, so for 2 qubits the amplitude
  of |q0=1, q1=0⟩ is at index 0b10 = 2. ``state.reshape((2,) * n)`` then has
  axis k = qubit k.
- Two-qubit gate matrices are written in the basis |a b⟩ where
  ``(a, b) = gate.qubits``. For ``cx``, a is the control and b the target.
- ``u3(theta, phi, lam)`` uses the OpenQASM convention:
      [[cos(θ/2),           -e^{iλ} sin(θ/2)],
       [e^{iφ} sin(θ/2),  e^{i(φ+λ)} cos(θ/2)]]

Scope: only the unitary gates in ``GATE_SPECS``. Circuits may also contain
``measure``, ``reset`` and noise (``NON_UNITARY_SPECS``, design.md D7); for
now ``simulate`` should raise ``ValueError`` when it meets one of those.

Suggested order:
1. ``gate_matrix``: start with x, h, cx; add the rest once Bell works.
2. ``apply_gate``: do NOT build a 2**n x 2**n matrix. Reshape the state to
   one axis per qubit, contract the gate with only the axes it acts on
   (``np.tensordot`` or ``np.einsum``), then put the axes back in order
   (``np.moveaxis``). This is the one tricky step; draw the axes on paper.
3. ``simulate``: start from |0...0⟩ and apply every gate in order.

Run ``pytest tests/test_simulator.py -v`` after each step.
"""

from __future__ import annotations

import numpy as np

from helioq.circuit import Circuit, Gate


def gate_matrix(gate: Gate) -> np.ndarray:
    """Return the gate's matrix: 2x2 for one qubit, 4x4 for two qubits."""
    raise NotImplementedError("TODO(Qiqi)")


def apply_gate(state: np.ndarray, gate: Gate, n_qubits: int) -> np.ndarray:
    """Return a new state vector with ``gate`` applied. Do not modify ``state``."""
    raise NotImplementedError("TODO(Qiqi)")


def simulate(circuit: Circuit) -> np.ndarray:
    """Run ``circuit`` from |0...0⟩ and return the final state vector."""
    raise NotImplementedError("TODO(Qiqi)")
