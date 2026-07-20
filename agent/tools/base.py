"""工具基类与注册表。

设计目标（阶段二要建立的核心抽象）：
  - 每个工具自描述：名字、给模型看的说明、参数 schema、本地执行逻辑。
  - 用 pydantic 定义参数 schema，既能自动生成给模型的 JSON Schema，
    又能在执行前对模型填的参数做校验（非法时报错让模型重试）—— 这就是「约束不确定性」。
  - 注册表统一管理所有工具，Agent Loop 只跟注册表打交道，新增工具零侵入。
"""

from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar


from anthropic.types import ToolParam
from pydantic import BaseModel, ValidationError

A = TypeVar("A", bound=BaseModel)


class Tool(Generic[A]):
    """一个工具 = 元信息 + pydantic 参数模型 + 本地执行函数。

    Args:
        name: 工具名，必须与给模型的声明一致。
        description: 给模型看的说明，写清楚「什么时候该用它」。
        args_model: 描述参数的 pydantic 模型，用于生成 schema 和校验。
        func: 真正的本地实现，接收校验后的参数模型实例，返回字符串结果。
    """

    def __init__(
        self,
        name: str,
        description: str,
        args_model: type[A],
        func: Callable[[A], str],
    ) -> None:
        self.name = name
        self.description = description
        self.args_model = args_model
        self.func = func

    def to_anthropic_schema(self) -> ToolParam:
        """转成 Anthropic tools 数组里需要的声明格式。

        pydantic 的 model_json_schema() 直接产出 JSON Schema，省去手写。
        返回类型标注为 SDK 的 ToolParam（TypedDict），便于类型检查器校验。
        """
        schema = self.args_model.model_json_schema()
        # Anthropic 不需要 schema 里的 title 字段，去掉让声明更干净
        schema.pop("title", None)
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": schema,
        }

    def run(self, raw_input: dict[str, Any]) -> str:
        """校验模型填的参数后执行。

        校验失败不抛异常，而是返回错误字符串 —— 它会被当作 tool_result 回填给模型，
        模型据此自我修正后重试，这正是 Agent 健壮性的关键。
        """
        try:
            args = self.args_model(**raw_input)
        except ValidationError as e:
            return f"参数校验失败：{e.errors()}。请修正参数后重试。"
        try:
            return self.func(args)
        except Exception as e:  # 工具内部异常也回传给模型，而非让整个 Loop 崩溃
            return f"工具执行出错：{type(e).__name__}: {e}"


class ToolRegistry:
    """工具注册表：集中持有所有工具，供 Agent Loop 查询与调用。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool[Any]] = {}

    def register(self, tool: Tool[Any]) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool[Any] | None:
        return self._tools.get(name)

    def anthropic_schemas(self) -> list[ToolParam]:
        """所有工具的声明，直接传给 messages.create 的 tools 参数。"""
        return [t.to_anthropic_schema() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)
