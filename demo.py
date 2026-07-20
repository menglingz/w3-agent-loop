"""预设任务演示：一次性跑几个会触发不同工具 / 多步推理的任务。

运行：python demo.py
目的：在不进 REPL 的情况下，直观看到 Agent Loop 如何针对不同问题
      自主选择并组合工具。每个任务用一个全新 Agent，互不干扰。
"""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from agent import Agent

# 精心设计的任务，分别诱导不同工具与多步组合
TASKS = [
    # 单工具：大数运算 → calculator
    "精确计算 (84729 * 13647) - 9981 是多少？",
    # 单工具：实时信息 → now
    "现在几点了？只告诉我时间。",
    # 多步组合：先写文件，再读回来确认 → write_file + read_file
    "把『阶段二完成：手写了 Agent Loop』这句话写入 notes/progress.txt，然后读取该文件确认写入成功。",
    # 需要外部信息 → web_search（未配 key 时走 mock，仍能演示闭环）
    "搜索一下 Model Context Protocol 是什么，用一句话总结。",
]


def main() -> None:

    for i, task in enumerate(TASKS, 1):
        print("=" * 70)
        print(f"任务 {i}：{task}")
        print("-" * 70)
        agent = Agent(verbose=True)  # 每个任务独立 Agent，避免上下文串味
        answer = agent.run(task)
        print(f"\n✅ 最终答案：{answer}\n")


if __name__ == "__main__":
    main()
