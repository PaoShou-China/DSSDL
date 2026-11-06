"""Expression-evaluation utilities only; this is not a GPLearn experiment entry."""
from __future__ import annotations
from typing import Any, Sequence
import multisystem_closed_loop_symbolic_discovery as multisystem

def sympy_context(variable_names: Sequence[str]):
    import sympy as sp

    symbols = sp.symbols(" ".join(variable_names), seq=True)
    symbol_map = dict(zip(variable_names, symbols))
    local_map: dict[str, Any] = {
        **symbol_map,
        "add": lambda left, right: left + right,
        "sub": lambda left, right: left - right,
        "mul": lambda left, right: left * right,
        "div": lambda left, right: left / right,
        "sin": sp.sin,
        "cos": sp.cos,
        "neg": lambda value: -value,
    }
    return tuple(symbols), symbol_map, local_map



def additive_factors(expression) -> list[Any]:
    import sympy as sp

    expanded = sp.expand(expression)
    factors: list[Any] = []
    seen: set[str] = set()
    for term in sp.Add.make_args(expanded):
        coefficient, factor = term.as_coeff_Mul()
        if factor.is_number:
            factor = sp.Integer(1)
        factor = sp.factor_terms(sp.simplify(factor))
        if coefficient == 0:
            continue
        key = sp.srepr(factor)
        if key not in seen:
            seen.add(key)
            factors.append(factor)
    return factors



def evaluate_sympy_numpy(expression, variable_names: Sequence[str], features):
    import numpy as np
    import sympy as sp

    symbols, _symbol_map, _local_map = sympy_context(variable_names)
    function = sp.lambdify(symbols, expression, modules=[{"sin": np.sin, "cos": np.cos}, "numpy"])
    values = function(*[features[:, index] for index in range(features.shape[1])])
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        values = np.full(features.shape[0], float(values), dtype=float)
    return values.reshape(-1)



def evaluate_sympy_torch(expression, environment: dict[str, Any], reference):
    import sympy as sp
    import torch

    if expression.is_Number:
        return torch.full_like(reference, float(expression))
    if expression.is_Symbol:
        return environment[str(expression)]
    if expression.func is sp.Add:
        result = torch.zeros_like(reference)
        for argument in expression.args:
            result = result + evaluate_sympy_torch(argument, environment, reference)
        return result
    if expression.func is sp.Mul:
        result = torch.ones_like(reference)
        for argument in expression.args:
            result = result * evaluate_sympy_torch(argument, environment, reference)
        return result
    if expression.func is sp.Pow:
        base, exponent = expression.args
        return evaluate_sympy_torch(base, environment, reference).pow(float(exponent))
    if expression.func is sp.sin:
        return torch.sin(evaluate_sympy_torch(expression.args[0], environment, reference))
    if expression.func is sp.cos:
        return torch.cos(evaluate_sympy_torch(expression.args[0], environment, reference))
    raise TypeError(f"Unsupported SymPy node in injected prior: {expression.func} ({expression})")

