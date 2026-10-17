"""minq: a small quantum computing library, built layer by layer.

Layers (see ``docs/design.md`` for the decisions behind each one):

1. ``minq.circuit``   – how gates and circuits are represented (the IR)
2. ``minq.simulator`` – state-vector simulation
3. ``minq.passes``    – optimization passes that rewrite circuits
4. ``minq.qasm``      – export to OpenQASM 2.0
"""

__version__ = "0.0.1"
