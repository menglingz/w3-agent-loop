"""时间工具：返回当前日期时间。

最纯粹的「模型不知道、必须问外部」示例 —— 模型权重里没有"现在几点"。
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from .base import Tool


class NowArgs(BaseModel):
    # 网关对空参数工具调用的兼容性较差，因此要求模型显式传入格式。
    fmt: str = Field(
        description=(
            "strftime 格式串；用户未指定格式时必须传入 "
            "%Y-%m-%d %H:%M:%S"
        )
    )


def _run(args: NowArgs) -> str:
    return datetime.now().strftime(args.fmt)


now_tool = Tool(
    name="now",
    description=(
        "获取当前本地日期和时间。必须调用此工具，不能凭模型记忆回答。"
        "调用时必须提供 fmt 参数；用户未指定格式时使用 "
        "%Y-%m-%d %H:%M:%S。"
    ),
    args_model=NowArgs,
    func=_run,
    idempotent=True,
)
