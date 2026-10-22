"""Symbolic FNN export in the JSON format read by PlaJA: atoms, rules
("L<layer>_<id>": {"AND"|"OR": [operands]}) and the linear output layer."""
from decimal import Decimal, ROUND_DOWN, localcontext

from model import negate_atom_name


def _truncate(value):
    """Truncate a coefficient to six decimal places, toward zero."""
    coefficient = Decimal(str(float(value)))
    if not coefficient.is_finite():
        raise ValueError("Cannot export a non-finite output coefficient")
    with localcontext() as context:
        context.prec = max(28, coefficient.adjusted() + 7)
        truncated = coefficient.quantize(Decimal("0.000001"), rounding=ROUND_DOWN)
    return float(truncated) if truncated else 0.0


def export_fnn(net, feature_names, label_names, loader, device, atoms_type):
    net.count_activations(loader, device)  # dead and always-true rules are dropped
    layers = net.layer_list
    atoms = layers[0].atom_names(feature_names)
    atom_set = set(atoms)

    def operand(layer_idx, key):
        k, rid = key
        if layer_idx > 0:
            name = f"L{layer_idx}_{rid}"
            return {"NOT": name} if k > 0 else name
        if k <= 0:
            return atoms[rid]
        folded = negate_atom_name(atoms[rid])  # NOT of an atom is exported as the negated atom
        if folded not in atom_set:
            raise ValueError(f"negative connection to atom '{atoms[rid]}' but '{folded}' is not an atom")
        return folded

    rules = {}
    prev = layers[0]
    for i, layer in enumerate(layers[1:-1], start=1):
        layer.get_rules(prev)
        base = 0
        for op, sub_rules in zip(("AND", "OR"), layer.rule_list):
            for j, rule in enumerate(sub_rules):
                rules[f"L{i}_{base + j}"] = {op: [operand(i - 1, k) for k in rule]}
            base += len(sub_rules)
        prev = layer

    lr_layer = layers[-1]
    lr_layer.get_rule2weights(prev)
    last = len(layers) - 2
    return {
        "atoms": {"type": atoms_type, "len": len(atoms), "content": atoms},
        "rules": rules,
        "linear": {
            "bias": {label_names[a]: _truncate(lr_layer.bl[a]) for a in range(len(label_names))},
            "weights": {label: {f"L{last}_{rid[1]}": _truncate(w[li]) for rid, w in lr_layer.rule2weights}
                        for li, label in enumerate(label_names)},
        },
    }

