"""Layer 1: circuit representation (minq's IR).

A circuit is an ordered list of operations. Every other layer reads this
representation, so the choices made here shape the whole library.

Design decisions to make before writing code (record them in docs/design.md):

- What does a gate object hold? A name + target qubits + parameters,
  or its matrix directly?
- Where do gate matrices live? On the gate, or in a lookup table?
- Should circuits be mutable (``circ.h(0)`` appends in place) or
  immutable (every change returns a new circuit)?

TODO(Qiqi): implement after the design decisions are made.
"""
