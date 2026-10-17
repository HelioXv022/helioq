# Design decisions

Each decision records what was chosen, what the alternatives were, and why.
When a decision changes, add a new entry rather than editing the old one.

## D1. Name: `helioq`

Chosen by Qiqi, after her GitHub handle HelioXv022.

## D2. A gate holds a name, target qubits and parameters, not a matrix

```python
Gate(name="u3", qubits=(0,), params=(0.3, 0.1, 0.2))
```

Matrices live in a separate lookup table owned by the simulator.

- **Why:** optimization passes need to recognize gates ("is this an H?").
  Comparing names is exact and cheap; comparing matrices needs a tolerance and
  is ambiguous (two different gate sequences can have the same matrix).
  Exporting to OpenQASM also needs the name.
- **Alternative rejected:** storing the matrix on the gate. Simplest for the
  simulator, but every other layer would have to reverse-engineer what the
  gate is.
- **Consequence:** the circuit layer has no NumPy dependency. Only the
  simulator knows what the gates do physically.

## D3. Qubit ordering: big-endian, q0 is the leftmost bit

In `|q0 q1 q2⟩`, qubit 0 is written first. For two qubits, `|10⟩` means
q0 = 1, q1 = 0, and its amplitude sits at index `0b10 = 2` of the state
vector.

- **Why:** matches textbooks and PennyLane, and is the natural order for
  `numpy.reshape(state, (2,) * n)`: axis `k` is qubit `k`.
- **Alternative rejected:** little-endian (Qiskit). Comparing results with
  Qiskit later will need the bit order reversed.

## D4. License: Apache-2.0

Chosen by Qiqi. Same license family as PennyLane and bloqade.

## D5. Circuits are mutable builders (provisional, chosen by Claude)

`circ.h(0)` appends to the circuit in place and returns the circuit, so calls
can be chained: `Circuit(2).h(0).cx(0, 1)`. Gates themselves are immutable
(frozen dataclasses).

- **Why:** this is the most familiar style (Qiskit, Cirq) and the simplest
  to write.
- **Alternative:** immutable circuits where every change returns a new
  circuit. Safer when passes share circuits, but more code.
- **Revisit:** when writing optimization passes. A pass must never modify
  its input circuit; it should build and return a new one.

## D6. Validate early

`Circuit.append` rejects unknown gate names, the wrong number of qubits or
parameters, out-of-range qubit indices, and a gate acting twice on the same
qubit. An invalid circuit can never be built, so the simulator and passes do
not need to check again.
