"""计算器工具：精确算术求值。

为什么需要它：LLM 心算大数/多步运算不可靠，把计算交给确定性代码是经典的
「用工具弥补模型短板」场景。
"""
from __future__ import annotations

import ast
import operator

from pydantic import BaseModel, Field

from .base import Tool

# 只允许这些运算符，杜绝任意代码执行（安全意识：工具是 Agent 的攻击面）
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}
_ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class CalculatorArgs(BaseModel):
    expression: str = Field(description="合法算术表达式，如 (84729 * 13647) - 9981")


def _safe_eval(node: ast.AST) -> float:
    """递归求值 AST，只放行白名单内的节点，其余一律拒绝。"""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("表达式包含不允许的语法")


def _run(args: CalculatorArgs) -> str:
    tree = ast.parse(args.expression, mode="eval")
    return str(_safe_eval(tree))


calculator_tool = Tool(
    name="calculator",
    description="执行一个算术表达式并返回精确结果。当需要精确计算（尤其大数或多步运算）时使用。",
    args_model=CalculatorArgs,
    func=_run,
    idempotent=True,
)
