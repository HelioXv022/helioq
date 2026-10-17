# helioq

A small quantum computing library, written from scratch to understand every
layer of a quantum software stack: how circuits are represented, how they are
simulated, how compilers rewrite them, and how they are exported.

This is a learning project, not a replacement for Qiskit or PennyLane. The
goal is that every line can be explained.

## Roadmap

| Layer | Module | Milestone |
|---|---|---|
| 1. Circuit representation | `helioq.circuit` | Build a Bell-state circuit |
| 2. State-vector simulator | `helioq.simulator` | Bell and GHZ states match the textbook amplitudes |
| 3. Optimization passes | `helioq.passes` | Cancel inverse pairs; merge single-qubit gates into U3 |
| 4. Export | `helioq.qasm` | Output OpenQASM 2.0 that Qiskit can load |
| 5. MPS simulator | `helioq.mps` | Simulate 30+ qubits of a low-entanglement circuit |

Status: layer 1 done. Layer 2 is next: `tests/test_simulator.py` is its
specification.

Design decisions are recorded in [`docs/design.md`](docs/design.md).

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check . && ruff format .
```
