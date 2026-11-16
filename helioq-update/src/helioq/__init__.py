"""helioq: a small quantum computing library, built layer by layer.

Layers (see ``docs/design.md`` for the decisions behind each one):

1. ``helioq.circuit``   – how gates and circuits are represented (the IR)
2. ``helioq.simulator`` – state-vector simulation
3. ``helioq.passes``    – optimization passes that rewrite circuits
4. ``helioq.qasm``      – export to OpenQASM 2.0

Fault-tolerance track (works without layers 2-4):

- ``helioq.draw``       – text and SVG circuit diagrams
- ``helioq.stabilizer`` – tableau simulation of Clifford circuits with
  measurement, reset and Pauli noise
- ``helioq.qec``        – codes, syndrome circuits, decoding, logical error rates
"""

from helioq.circuit import GATE_SPECS, Circuit, Gate
from helioq.simulator import simulate

__version__ = "0.0.1"

__all__ = ["GATE_SPECS", "Circuit", "Gate", "simulate", "__version__"]
