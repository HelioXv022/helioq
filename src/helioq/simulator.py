"""Layer 2: state-vector simulation.

Holds the 2**n complex amplitudes of an n-qubit state and applies gates to
it. The key idea: never build the full 2**n x 2**n matrix. Reshape the state
to a tensor with one axis per qubit and contract the small gate matrix with
only the axes it acts on (``numpy.tensordot`` or ``numpy.einsum``).

Depends on: the qubit-ordering convention in docs/design.md.

TODO(Qiqi): implement after layer 1.
"""
