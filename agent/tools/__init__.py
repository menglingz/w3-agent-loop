"""工具包导出：组装一个注册好全部工具的 ToolRegistry。"""

from __future__ import annotations

from .base import Tool, ToolPermission, ToolRegistry
from .calculator import calculator_tool
from .clock import now_tool
from .files import list_dir_tool, read_file_tool, write_file_tool, delete_file_tool
from .search import search_tool


def build_default_registry() -> ToolRegistry:
    """构建并返回注册了全部 4 类工具的注册表。"""
    registry = ToolRegistry()
    for tool in (
        calculator_tool,
        now_tool,
        search_tool,
        list_dir_tool,
        read_file_tool,
        write_file_tool,
        delete_file_tool,
    ):
        registry.register(tool)
    return registry


__all__ = ["Tool", "ToolPermission", "ToolRegistry", "build_default_registry"]
