"""时间工具：返回当前日期时间。

最纯粹的「模型不知道、必须问外部」示例 —— 模型权重里没有"现在几点"。
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from .base import Tool


class NowArgs(BaseModel):
    # 留一个可选参数演示带参工具；不传则用默认格式
    fmt: str = Field(
        default="%Y-%m-%d %H:%M:%S",
        description="strftime 格式串，默认 %Y-%m-%d %H:%M:%S",
    )


def _run(args: NowArgs) -> str:
    return datetime.now().strftime(args.fmt)


now_tool = Tool(
    name="now",
    description="获取当前本地日期和时间。当问题涉及『现在』『今天』『当前时间』等实时信息时使用。",
    args_model=NowArgs,
    func=_run,
)
