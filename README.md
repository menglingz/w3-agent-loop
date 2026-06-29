# W3-W5 项目：手写 Agent Loop（阶段二核心）

Agent 全栈学习大纲 **阶段二** 的代码产出。**不依赖任何 Agent 框架**（没用 LangChain/LangGraph），
从零手写 Agent Loop，吃透它你才不会把后面的框架当黑盒。语言：Python。

## 这个项目教你什么

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
pip install -r requirements.txt
cp .env.example .env        # 编辑 .env 填入 ANTHROPIC_API_KEY

python repl.py              # 交互式：直接提问，Agent 自行决定用不用工具
python demo.py              # 一次性演示：4 个任务触发不同工具与多步组合
```

> 网络搜索默认走**离线 mock**（无需任何额外 key 即可跑通闭环）；想要真实搜索，在 `.env` 填 `TAVILY_API_KEY`（tavily.com 有免费额度）。

## 4 个工具

- `calculator`：精确算术，用 AST 白名单求值，**拒绝任意代码执行**
- `now`：当前时间，最纯粹的「模型不知道、必须问外部」示例
- `web_search`：联网搜索，真实/mock 自动切换
- `list_dir` / `read_file` / `write_file`：本地文件，**全部锁在 `workspace/` 沙箱内，阻止路径逃逸**

## 建议的学习路径（按顺序读 + 改）

1. **先读 `agent/loop.py` 的 `run()`**——把循环每一步对着注释走一遍：
   发请求 → 看 `stop_reason` → 是 `tool_use` 就执行并回填 → 否则给最终答案。
2. **跑 `demo.py`**，看 `verbose` 日志里 `💭 思考` / `🔧 调用工具` / `↳ 结果` 的完整轨迹。
3. **故意制造异常**理解健壮性：
   - 问一个需要工具但参数会非法的问题，看 pydantic 校验失败 → 错误回传 → 模型自我修正。
   - 让它读 `../../etc/passwd`，看沙箱拦截。
4. **加一个你自己的工具**：在 `tools/` 下照着 `clock.py` 写一个（如「生成 UUID」「查汇率」），
   在 `tools/__init__.py` 注册——体会「注册表 + 自描述工具」让扩展零侵入。
5. **触发记忆压缩**：在 REPL 里多聊几轮（超过 20 条），看 `🗜️ 自动摘要压缩` 是否触发，读 `memory.py` 理解策略。

## ✅ 出阶段标准（对照大纲）

- [ ] 能不靠框架，讲清 Agent Loop 每一步发生了什么
- [ ] Agent 能自主选择并**组合多个工具**完成一个多步任务（如先写文件再读回）
- [ ] 理解工具的**参数校验 + 失败重试**为什么是健壮性关键
- [ ] 知道上下文超长时**记忆压缩**的基本做法
- [ ] 给项目**新增过一个自己的工具**

做完进入 **W6 阶段三 RAG**——`web_search` 那套「检索外部信息再回答」的思路会自然延伸到向量检索。
