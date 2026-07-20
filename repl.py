"""交互式 REPL 入口：和你手写的 Agent 多轮对话。

运行：python repl.py
命令：/reset 清空记忆，/exit 退出。
"""
from __future__ import annotations

import sys

from dotenv import load_dotenv

load_dotenv()

from agent import Agent, ApprovalRequest, ToolPolicy


# 引入 readline 以启用行编辑（方向键、历史、退格等）。导入即生效，无需直接调用。
try:
    import readline  # noqa: F401
except ImportError:  # 某些平台（如 Windows 原生）无此模块，忽略即可
    readline = None

# readline 用「列宽」推算光标位置，但 input() 的 prompt 里的宽字符（emoji 🧑 占 2 列，
# readline 却按 1 列算）会让它把起始列算错，导致退格删到行首前一个字符就停住。
# 约定：用 \001..\002 包裹「不应计入列宽」的片段，readline 便会跳过其宽度计算。
# 这里把 emoji 整体包起来，再补一个普通空格作为真正占位的提示符。
PROMPT = "\001\002🧑\001\002 " if readline else "🧑 "


def console_approve(request: ApprovalRequest) -> bool:
    """在终端确认一次受保护的工具调用，任何异常都按拒绝处理。"""
    arguments = repr(request.arguments)
    if len(arguments) > 300:
        arguments = arguments[:300] + "..."
    print(
        f"\n🔐 工具 {request.tool_name} 请求 {request.permission.value} 权限\n"
        f"   参数：{arguments}"
    )
    try:
        answer = input("   允许执行？[y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n   已拒绝")
        return False
    return answer in {"y", "yes"}


def main() -> None:

    agent = Agent(verbose=True, policy=ToolPolicy(approver=console_approve))
    print(f"🤖 Agent 已就绪，注册了 {len(agent.registry)} 个工具。")
    print("   直接提问，Agent 会自行决定是否调用工具。输入 /exit 退出，/reset 清空记忆。\n")

    while True:
        try:
            user = input(PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user:
            continue
        if user in ("/exit", "/quit"):
            break
        if user == "/reset":
            agent.memory.messages.clear()
            print("（已清空对话记忆）\n")
            continue

        try:
            answer = agent.run(user)
            print(f"\n✅ {answer}\n")
        except Exception as e:  # 顶层兜底，单轮出错不退出程序
            print(f"\n⚠️ 出错：{type(e).__name__}: {e}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
