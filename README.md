# 手写 Agent Loop

## 架构

| 能力 | 在代码里的位置 |
| --- | --- |
| **Agent Loop**（think→act→observe 循环） | `agent/loop.py` ★ 最该读 |
| **Tool Use**（声明工具 / 执行 / 回填 tool_result） | `agent/loop.py` + `agent/tools/` |
| **工具抽象与注册表**（新增工具零侵入） | `agent/tools/base.py` |
| **参数校验 + 失败重试**（pydantic 约束不确定性） | `agent/tools/base.py` 的 `Tool.run` |
| **上下文记忆与压缩**（超长自动摘要） | `agent/memory.py` |
| **工具安全边界**（沙箱 / 白名单，工具是攻击面） | `tools/files.py`、`tools/calculator.py` |

## 目录结构

```
w3-agent-loop/
├── agent/
│   ├── loop.py            # ★ Agent Loop 主体
│   ├── memory.py          # 对话记忆 + 上下文压缩
│   └── tools/
│       ├── base.py        # Tool 基类 + ToolRegistry
│       ├── calculator.py  # 计算器（AST 安全求值，拒绝代码注入）
│       ├── clock.py       # 当前时间
│       ├── files.py       # 列目录/读/写（限制在 workspace/ 沙箱）
│       └── search.py      # 网络搜索（Tavily 真实 + 离线 mock 兜底）
├── repl.py                # 交互式对话入口
├── demo.py                # 一次性跑 4 个预设任务
└── requirements.txt
```

## 准备与运行

```bash
cd w3-agent-loop
python3 -m venv .venv && source .venv/bin/activate
- 生产安装：pip install .
- 开发安装：pip install ".[dev]"
cp .env.example .env        # 编辑 .env 填入 ANTHROPIC_API_KEY

python repl.py              # 交互式：直接提问，Agent 自行决定用不用工具
python demo.py              # 一次性演示：4 个任务触发不同工具与多步组合
```

> 网络搜索默认走**离线 mock**（无需任何额外 key 即可跑通闭环）；想要真实搜索，在 `.env` 填 `TAVILY_API_KEY`（tavily.com 有免费额度）。

## 4 个工具

- `calculator`：精确算术，用 AST 白名单求值，**拒绝任意代码执行**
- `now`：当前时间，最纯粹的「模型不知道、必须问外部」示例
- `web_search`：联网搜索，真实/mock 自动切换
- `list_dir` / `read_file` / `write_file` / `delete_file`：本地文件，**全部锁在 `workspace/` 沙箱内，阻止路径逃逸**
