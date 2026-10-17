"""Layer 3: optimization passes.

A pass takes a circuit and returns an equivalent, usually smaller, circuit.
Planned first passes:

- cancel adjacent inverse pairs (H·H, X·X, CNOT·CNOT on the same qubits)
- merge consecutive single-qubit gates into one U3 (the problem in
  bloqade-circuit issue #690)

Every pass must be tested by checking that the circuit's unitary is
unchanged up to a global phase.

TODO(Qiqi): implement after layers 1 and 2.
"""
