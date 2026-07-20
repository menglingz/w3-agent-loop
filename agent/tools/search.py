"""网络搜索工具。

设计：优先用 Tavily 真实搜索（需 TAVILY_API_KEY），未配置时自动降级为离线 mock，
保证项目在没有任何额外 key 的情况下也能开箱跑通 Agent Loop。
这种「真实实现 + 离线兜底」是写教学/演示项目的实用技巧。
"""
from __future__ import annotations

import os

import httpx
from pydantic import BaseModel, Field

from .base import Tool, ToolPermission


class SearchArgs(BaseModel):
    query: str = Field(description="搜索关键词")


def _mock_search(query: str) -> str:
    """离线兜底：返回一条固定的「假」搜索结果，仅用于跑通流程。"""
    return (
        f"[离线 mock 搜索结果] 关于「{query}」的摘要：\n"
        "1. 这是一条模拟结果，未联网。配置 TAVILY_API_KEY 后将返回真实搜索。\n"
        "2. Agent 仍能据此演示『拿到工具结果 → 继续推理』的闭环。"
    )


def _tavily_search(query: str, api_key: str) -> str:
    """调用 Tavily API 做真实搜索，拼接前几条结果摘要。"""
    resp = httpx.post(
        "https://api.tavily.com/search",
        json={"api_key": api_key, "query": query, "max_results": 3},
        timeout=20,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        return f"未找到关于「{query}」的结果。"
    lines = [f"{i+1}. {r.get('title')}\n   {r.get('content', '')[:200]}" for i, r in enumerate(results)]
    return "\n".join(lines)


def _run(args: SearchArgs) -> str:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return _mock_search(args.query)
    return _tavily_search(args.query, api_key)


search_tool = Tool(
    name="web_search",
    description="联网搜索实时或外部信息。当问题需要你不掌握的最新事实、新闻、文档时使用。",
    args_model=SearchArgs,
    func=_run,
    permission=ToolPermission.NETWORK,
)
