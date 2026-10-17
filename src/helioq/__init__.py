"""helioq: a small quantum computing library, built layer by layer.

Layers (see ``docs/design.md`` for the decisions behind each one):

1. ``helioq.circuit``   – how gates and circuits are represented (the IR)
2. ``helioq.simulator`` – state-vector simulation
3. ``helioq.passes``    – optimization passes that rewrite circuits
4. ``helioq.qasm``      – export to OpenQASM 2.0
"""

from helioq.circuit import GATE_SPECS, Circuit, Gate
from helioq.simulator import simulate

__version__ = "0.0.1"

__all__ = ["GATE_SPECS", "Circuit", "Gate", "simulate", "__version__"]
