"""Logical error rate of the repetition code versus physical error rate.

Run:  python examples/repetition_code_threshold.py

Below p = 0.5 a longer code is always better for this simple noise model;
watch the columns shrink from left to right when p is small and stop
shrinking as p approaches 0.5.
"""

from helioq.qec import memory_experiment, repetition_code

SHOTS = 3000
DISTANCES = (3, 5, 7, 9)

print("Memory circuit for d = 3:\n")
print(memory_experiment(repetition_code(3), p=0.05, noise="x_error").circuit.draw())
print(f"\nLogical error rate ({SHOTS} shots each)\n")
print("   p    " + "".join(f"d={d:<8}" for d in DISTANCES))
for p in (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
    rates = [
        memory_experiment(repetition_code(d), p, noise="x_error").logical_error_rate(
            SHOTS, seed=0
        )
        for d in DISTANCES
    ]
    print(f"  {p:<5} " + "".join(f"{r:<10.4f}" for r in rates))
