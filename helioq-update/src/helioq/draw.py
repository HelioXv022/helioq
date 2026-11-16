"""Circuit drawing: text diagrams and SVG pictures.

Both drawers share one step, ``layout``: decide which column each operation
goes in. After that, drawing is only a matter of turning each (column, qubit)
cell into characters or shapes.

Example:
    >>> from helioq import Circuit
    >>> print(Circuit(3).h(0).cx(0, 1).cx(1, 2).measure_all().draw())
    q0: ─H──●──M────
            │
    q1: ────⊕──●──M─
               │
    q2: ───────⊕──M─
"""

from __future__ import annotations

from helioq.circuit import Circuit, Gate

# --- step 1: layout -----------------------------------------------------------


def layout(circuit: Circuit) -> list[list[Gate]]:
    """Group operations into columns ("moments"), each as far left as it can go.

    An operation occupies every wire from its lowest to its highest qubit: a
    CNOT between q0 and q2 also blocks q1, because its vertical line crosses
    that wire. Each operation goes in the first column after everything that
    already occupies one of its wires.
    """
    columns: list[list[Gate]] = []
    next_free = [0] * circuit.n_qubits  # first free column on each wire
    for gate in circuit:
        lo, hi = min(gate.qubits), max(gate.qubits)
        col = max(next_free[lo : hi + 1])
        if col == len(columns):
            columns.append([])
        columns[col].append(gate)
        for q in range(lo, hi + 1):
            next_free[q] = col + 1
    return columns


def depth(circuit: Circuit) -> int:
    """Number of time steps if operations on different qubits run in parallel.

    Unlike ``layout``, a two-qubit gate here only blocks its own two qubits.
    """
    busy_until = [0] * circuit.n_qubits
    for gate in circuit:
        step = max(busy_until[q] for q in gate.qubits) + 1
        for q in gate.qubits:
            busy_until[q] = step
    return max(busy_until)


# --- step 2: what to write on each wire ---------------------------------------

_NOISE_LABELS = {"x_error": "Xerr", "z_error": "Zerr", "depolarize1": "Dep"}


def cell_labels(gate: Gate) -> dict[int, str]:
    """What each qubit of ``gate`` shows on its wire: {qubit: label}."""
    name = gate.name
    if name == "cx":
        control, target = gate.qubits
        return {control: "●", target: "⊕"}
    if name == "cz":
        return dict.fromkeys(gate.qubits, "●")
    if name == "swap":
        return dict.fromkeys(gate.qubits, "×")
    if name == "measure":
        return {gate.qubits[0]: "M"}
    if name == "reset":
        return {gate.qubits[0]: "|0>"}
    label = _NOISE_LABELS.get(name, name.upper())
    if gate.params:
        label += "(" + ",".join(f"{p:.3g}" for p in gate.params) + ")"
    return {gate.qubits[0]: label}


# --- text ---------------------------------------------------------------------


def draw_text(circuit: Circuit) -> str:
    n = circuit.n_qubits
    # Row 2k is the wire of qubit k; row 2k+1 is the gap below it.
    label_width = len(f"q{n - 1}: ")
    rows = [f"q{k // 2}: ".ljust(label_width) if k % 2 == 0 else " " * label_width
            for k in range(2 * n - 1)]  # fmt: skip

    for column in layout(circuit):
        labels: dict[int, str] = {}
        crossed_rows: set[int] = set()  # rows a vertical line passes through
        for gate in column:
            labels.update(cell_labels(gate))
            lo, hi = min(gate.qubits), max(gate.qubits)
            crossed_rows.update(range(2 * lo + 1, 2 * hi))
        width = max(len(text) for text in labels.values())

        for k in range(2 * n - 1):
            if k % 2 == 0:  # a wire
                if k // 2 in labels:
                    body = labels[k // 2].center(width, "─")
                elif k in crossed_rows:
                    body = "┼".center(width, "─")
                else:
                    body = "─" * width
                rows[k] += "─" + body + "─"
            else:  # a gap
                body = "│".center(width) if k in crossed_rows else " " * width
                rows[k] += " " + body + " "
    return "\n".join(row.rstrip() for row in rows)


# --- SVG ----------------------------------------------------------------------

_COL = 68  # horizontal distance between columns
_ROW = 44  # vertical distance between wires
_BOX = 28  # height of a gate box
_INK = "#1f2937"
_BOX_FILL = {"measure": "#e5e7eb", "reset": "#e5e7eb"}
_NOISE_FILL = "#fde2e2"
_GATE_FILL = "#dbeafe"


def draw_svg(circuit: Circuit) -> str:
    n = circuit.n_qubits
    columns = layout(circuit)
    left = 44
    width = left + _COL * max(len(columns), 1) + 12
    height = _ROW * n

    def x_of(col: int) -> float:
        return left + _COL * col + _COL / 2

    def y_of(q: int) -> float:
        return _ROW * q + _ROW / 2

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="monospace" font-size="13">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
    ]
    for q in range(n):
        y = y_of(q)
        out.append(
            f'<text x="6" y="{y + 4}" fill="{_INK}">q{q}</text>'
            f'<line x1="{left - 8}" y1="{y}" x2="{width - 6}" y2="{y}" '
            f'stroke="{_INK}" stroke-width="1"/>'
        )

    for col, column in enumerate(columns):
        x = x_of(col)
        for gate in column:
            if len(gate.qubits) == 2:
                ya, yb = y_of(gate.qubits[0]), y_of(gate.qubits[1])
                out.append(
                    f'<line x1="{x}" y1="{ya}" x2="{x}" y2="{yb}" '
                    f'stroke="{_INK}" stroke-width="1.5"/>'
                )
            for q, label in cell_labels(gate).items():
                y = y_of(q)
                if label == "●":
                    out.append(f'<circle cx="{x}" cy="{y}" r="4.5" fill="{_INK}"/>')
                elif label == "⊕":
                    out.append(
                        f'<circle cx="{x}" cy="{y}" r="9" fill="white" '
                        f'stroke="{_INK}" stroke-width="1.5"/>'
                        f'<path d="M{x - 9} {y}H{x + 9}M{x} {y - 9}V{y + 9}" '
                        f'stroke="{_INK}" stroke-width="1.5"/>'
                    )
                elif label == "×":
                    out.append(
                        f'<path d="M{x - 6} {y - 6}L{x + 6} {y + 6}'
                        f'M{x - 6} {y + 6}L{x + 6} {y - 6}" '
                        f'stroke="{_INK}" stroke-width="1.8"/>'
                    )
                else:
                    if gate.name in _NOISE_LABELS:
                        fill = _NOISE_FILL
                    else:
                        fill = _BOX_FILL.get(gate.name, _GATE_FILL)
                    size = 13 if len(label) <= 4 else 9
                    box = min(max(_BOX, 6 * len(label) * size / 9 + 6), _COL - 4)
                    text = label.replace("<", "&lt;").replace(">", "&gt;")
                    out.append(
                        f'<rect x="{x - box / 2}" y="{y - _BOX / 2}" width="{box}" '
                        f'height="{_BOX}" rx="4" fill="{fill}" stroke="{_INK}"/>'
                        f'<text x="{x}" y="{y + size / 3}" text-anchor="middle" '
                        f'font-size="{size}" fill="{_INK}">{text}</text>'
                    )
    out.append("</svg>")
    return "\n".join(out)
