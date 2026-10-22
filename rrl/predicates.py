"""Predicate atoms for the 'predicates' binarization: the linear comparisons that
occur in the JANI guards and in the goal/failure conditions of the property."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_CMP_OP_MAP = {"=": "eq", "≠": "neq", "<": "lt", "≤": "leq", ">": "gt", "≥": "geq"}
_LOGICAL_OPS = {"∧", "∨"}
_OP_SWAP = {"lt": "gt", "leq": "geq", "gt": "lt", "geq": "leq"}
_COMPLEMENT_OP = {"lt": "geq", "leq": "gt", "gt": "leq", "geq": "lt"}
_FLOAT_EPS = 1e-9


@dataclass(frozen=True)
class Predicate:
    op: str
    vars: tuple
    coeffs: tuple
    const: float


def _normalize_predicate(pred):
    # Make the leading coefficient positive; swap the direction of inequalities.
    if pred.coeffs[0] >= 0:
        return pred
    return Predicate(op=_OP_SWAP.get(pred.op, pred.op), vars=pred.vars,
                     coeffs=tuple(-c for c in pred.coeffs), const=-pred.const)


def _complement_predicate(pred):
    comp = _COMPLEMENT_OP.get(pred.op)  # inequalities only
    return None if comp is None else Predicate(op=comp, vars=pred.vars, coeffs=pred.coeffs, const=pred.const)


def _normalize_number(x):
    xf = float(x)
    if abs(xf) < _FLOAT_EPS:
        return 0
    nearest = round(xf)
    if abs(xf - nearest) < _FLOAT_EPS:
        return int(nearest)
    return round(xf, 12)


def _is_zero(x):
    return abs(float(x)) < _FLOAT_EPS


def _add_coeffs(a, b, scale=1.0):
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + scale * v
    return {k: _normalize_number(v) for k, v in out.items() if not _is_zero(v)}


def _expr_to_linear(exp, consts):
    """(coefficients, constant) of a linear expression."""
    if isinstance(exp, (int, float)):
        return {}, float(exp)
    if isinstance(exp, str):
        return ({}, float(consts[exp])) if exp in consts else ({exp: 1}, 0)
    left_c, left_k = _expr_to_linear(exp["left"], consts)
    right_c, right_k = _expr_to_linear(exp["right"], consts)
    if exp["op"] == "+":
        return _add_coeffs(left_c, right_c), _normalize_number(left_k + right_k)
    if exp["op"] == "-":
        return _add_coeffs(left_c, right_c, scale=-1), _normalize_number(left_k - right_k)
    raise ValueError(f"Unsupported expression: {exp}")


def _collect_atoms(exp, out, consts):
    if isinstance(exp, dict) and "op" in exp and "left" in exp and "right" in exp:
        op = exp["op"]
        if op in _LOGICAL_OPS:
            _collect_atoms(exp["left"], out, consts)
            _collect_atoms(exp["right"], out, consts)
            return
        if op in _CMP_OP_MAP:
            left_c, left_k = _expr_to_linear(exp["left"], consts)
            right_c, right_k = _expr_to_linear(exp["right"], consts)
            coeffs = _add_coeffs(left_c, right_c, scale=-1)  # variables left, constants right
            items = sorted(coeffs.items())
            out.append(_normalize_predicate(Predicate(
                op=_CMP_OP_MAP[op], vars=tuple(k for k, _ in items),
                coeffs=tuple(_normalize_number(v) for _, v in items),
                const=_normalize_number(right_k - left_k))))
            return
    if isinstance(exp, dict) and "exp" in exp:
        _collect_atoms(exp["exp"], out, consts)


def _iter_property_exprs(prop_obj, key):
    if isinstance(prop_obj, dict):
        if key in prop_obj and isinstance(prop_obj[key], dict):
            exp = prop_obj[key].get("exp")
            if exp is not None:
                yield exp
        for v in prop_obj.values():
            yield from _iter_property_exprs(v, key)
    elif isinstance(prop_obj, list):
        for item in prop_obj:
            yield from _iter_property_exprs(item, key)


def parse_problem_predicates(jani_path):
    """Sorted, deduplicated predicates with their kinds (guard, goal, fail).

    The property is read from compact-property.jani next to the model if it
    exists, otherwise from the model's own properties (as in the paper)."""
    jani_path = Path(jani_path)
    j = json.loads(jani_path.read_text())
    consts = {c["name"]: c["value"] for c in j["constants"]}
    atoms = []
    for automaton in j["automata"]:
        for edge in automaton["edges"]:
            if "guard" in edge:
                tmp = []
                _collect_atoms(edge["guard"]["exp"], tmp, consts)
                atoms.extend((p, "guard") for p in tmp)

    prop_path = jani_path.parent / "compact-property.jani"
    prop_obj = json.loads(prop_path.read_text()) if prop_path.exists() else j
    exprs = [("goal", e) for e in _iter_property_exprs(prop_obj, "goal")]
    exprs += [("fail", e) for key in ("reach", "avoid") for e in _iter_property_exprs(prop_obj, key)]
    for kind, exp in exprs:
        tmp = []
        _collect_atoms(exp, tmp, consts)
        atoms.extend((p, kind) for p in tmp)
        if kind == "fail":
            atoms.extend((c, "fail") for c in map(_complement_predicate, tmp) if c is not None)

    uniq, kinds = {}, {}
    for p, kind in atoms:
        key = (p.op, p.vars, p.coeffs, p.const)
        uniq[key] = p
        kinds.setdefault(key, [])
        if kind not in kinds[key]:
            kinds[key].append(kind)
    keys = sorted(uniq)
    return [uniq[k] for k in keys], {k: kinds[k] for k in keys}


def predicate_to_str(p, kind):
    parts = []
    for var, coeff in zip(p.vars, p.coeffs):
        parts.append(var if coeff == 1 else f"-{var}" if coeff == -1 else f"{_normalize_number(coeff)}*{var}")
    lhs = parts[0]
    for term in parts[1:]:
        lhs += f" - {term[1:]}" if term.startswith("-") else f" + {term}"
    expr = f"{lhs} {p.op} {_normalize_number(p.const)}"
    return f"{kind}:{expr}"


def evaluate_predicates(X, predicates, var_index):
    X = np.asarray(X, dtype=np.float64)
    out = np.zeros((X.shape[0], len(predicates)), dtype=np.float32)
    for i, p in enumerate(predicates):
        vals = np.zeros(X.shape[0], dtype=np.float64)
        for var, coeff in zip(p.vars, p.coeffs):
            vals = vals + coeff * X[:, var_index[var]]
        if p.op == "eq":
            mask = np.isclose(vals, p.const, atol=_FLOAT_EPS, rtol=_FLOAT_EPS)
        elif p.op == "neq":
            mask = ~np.isclose(vals, p.const, atol=_FLOAT_EPS, rtol=_FLOAT_EPS)
        elif p.op == "leq":
            mask = vals <= (p.const + _FLOAT_EPS)
        elif p.op == "geq":
            mask = vals >= (p.const - _FLOAT_EPS)
        elif p.op == "lt":
            mask = vals < (p.const - _FLOAT_EPS)
        elif p.op == "gt":
            mask = vals > (p.const + _FLOAT_EPS)
        else:
            raise ValueError(f"Unknown predicate op: {p.op}")
        out[:, i] = mask.astype(np.float32)
    return out
