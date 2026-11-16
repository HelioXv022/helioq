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

## D7. Measurement, reset and noise are operations in the same list (chosen by Claude)

```python
Circuit(2).h(0).cx(0, 1).x_error(0.01, 0).measure(0, 1)
```

`measure`, `reset`, `x_error`, `z_error` and `depolarize1` are stored as
`Gate` objects next to the unitary gates. Their specs live in
`NON_UNITARY_SPECS`; `GATE_SPECS` still means "unitary gates" and
`OP_SPECS` is the union.

- **Why:** error correction measures in the middle of a circuit, so
  measurement has to have a position in the gate order. One list keeps that
  order explicit. This is also how Stim circuits are written.
- **Why two tables:** the state-vector simulator, the optimization passes and
  the layer-2 tests iterate over `GATE_SPECS` and assume every entry has a
  unitary matrix. Keeping that table unchanged means none of them broke.
- **Measurement results** are not stored in classical registers. Each
  `measure` appends one bit to the *measurement record*, in circuit order;
  `run(circuit, shots)` returns the records as an array of shape
  `(shots, n_measurements)`.
- **Alternative rejected:** named classical bits (`measure(q, c)`, as in
  OpenQASM). More flexible, but more IR to explain, and nothing here needs
  it yet.
- **Consequence:** a pass must not move a gate past a `measure` or `reset`
  on the same qubit, and must never remove noise. Check `gate.is_unitary`.

## D8. The stabilizer simulator is a second backend, not a replacement

`helioq.stabilizer` runs the same `Circuit` objects as `helioq.simulator`,
but only Clifford gates (`CLIFFORD_GATES`); it raises `ValueError` on `t`,
`rx` and friends. Because gates are names (D2), "is this Clifford?" is a set
lookup.

- **Why:** error-correction circuits are Clifford circuits with too many
  qubits for a state vector. A tableau costs about `2n × 2n` bits.
- **Data structure:** the Aaronson–Gottesman tableau with destabilizer rows.
  Stabilizer rows alone are enough to apply gates; the destabilizers are what
  make measurement cost O(n²) instead of a Gaussian elimination each time.
- **Not done:** Stim's "Pauli frame" trick (simulate once without noise, then
  only track how errors change the measurement results). It is the reason
  Stim is thousands of times faster per shot. Here every shot re-runs the
  whole tableau, which is simple and slow.

## D9. Never compare stabilizer generators directly

`{+XX, +ZZ}` and `{+XX, -YY}` describe the same Bell state. Tests compare
`Tableau.canonical_stabilizers()` (Gaussian elimination into a standard
form) or check against the state vector, never the raw rows. Same idea as
D2: do not identify things by a representation that is not unique.

## D10. The QEC layer starts with code-capacity noise

`helioq.qec.memory_experiment` builds: prepare → measure all checks → noise
on the data qubits → measure all checks again → measure the data. The
decoder sees which checks changed between the two rounds.

- **Why this noise model:** gates and measurements are perfect, so one
  syndrome is enough and a lookup table is an exact minimum-weight decoder.
  Every part can be checked by hand (for the d=3 repetition code the logical
  error rate must be `3p²(1−p) + p³`, and it is).
- **What it leaves out:** errors during syndrome extraction. With those, a
  single syndrome can lie, so the experiment needs many rounds, *detectors*
  (parities of consecutive measurements), a *detector error model*, and a
  decoder that matches detection events in space and time (MWPM, union-find,
  BP+OSD). That is the next layer, and where the 1% surface-code threshold
  lives.
- **Codes are check matrices.** `CSSCode(hx, hz, logical_x, logical_z)`
  validates the commutation rules on construction (same spirit as D6), and
  the syndrome circuit is generated from the matrices, so adding a new CSS
  code needs no new circuit code.
