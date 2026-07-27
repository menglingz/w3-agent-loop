"""本地文件工具：列目录 / 读文件 / 写文件。

★ 安全是重点：所有路径都被限制在 workspace/ 沙箱目录内，
  任何试图越界（路径逃逸到 ../../ 之外）的访问都会被拒绝。
  这演示了「工具是 Agent 的攻击面，必须做权限边界」——阶段六会系统展开。
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from .base import Tool, ToolPermission

# 沙箱根目录：所有文件操作都被关在这里面
WORKSPACE = (Path(__file__).resolve().parents[2] / "workspace").resolve()


def _resolve_in_sandbox(rel_path: str) -> Path:
    """把相对路径解析到沙箱内，并阻止路径逃逸。

    Raises:
        ValueError: 当解析后的绝对路径落在 workspace/ 之外时。
    """
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    target = (WORKSPACE / rel_path).resolve()
    # 关键校验：解析后的真实路径必须仍在沙箱内
    if WORKSPACE not in target.parents and target != WORKSPACE:
        raise ValueError(f"拒绝越界访问：{rel_path} 超出 workspace 沙箱")
    return target


# ---------- list_dir ----------
class ListDirArgs(BaseModel):
    path: str = Field(default=".", description="相对 workspace 的目录路径，默认根目录")


def _list_dir(args: ListDirArgs) -> str:
    target = _resolve_in_sandbox(args.path)
    if not target.exists():
        return f"目录不存在：{args.path}"
    entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
    return "\n".join(entries) if entries else "(空目录)"


list_dir_tool = Tool(
    name="list_dir",
    description="列出 workspace 沙箱目录下的文件和子目录。",
    args_model=ListDirArgs,
    func=_list_dir,
    idempotent=True,
    retryable=True,
)


# ---------- read_file ----------
class ReadFileArgs(BaseModel):
    path: str = Field(description="相对 workspace 的文件路径")


def _read_file(args: ReadFileArgs) -> str:
    target = _resolve_in_sandbox(args.path)
    if not target.is_file():
        return f"文件不存在：{args.path}"
    # 截断超长内容，避免一次塞爆上下文
    text = target.read_text(encoding="utf-8", errors="replace")
    if len(text) > 4000:
        return text[:4000] + f"\n…（已截断，文件共 {len(text)} 字符）"
    return text


read_file_tool = Tool(
    name="read_file",
    description="读取 workspace 沙箱内某个文本文件的内容。",
    args_model=ReadFileArgs,
    func=_read_file,
    idempotent=True,
    retryable=True,
)


# ---------- write_file ----------
class WriteFileArgs(BaseModel):
    path: str = Field(description="相对 workspace 的文件路径，父目录会自动创建")
    content: str = Field(description="要写入的文本内容")


def _write_file(args: WriteFileArgs) -> str:
    target = _resolve_in_sandbox(args.path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(args.content, encoding="utf-8")
    return f"已写入 {args.path}（{len(args.content)} 字符）"


write_file_tool = Tool(
    name="write_file",
    description="把文本内容写入 workspace 沙箱内的文件（覆盖写）。",
    args_model=WriteFileArgs,
    func=_write_file,
    permission=ToolPermission.WRITE,
)


class DeleteFileArgs(BaseModel):
    path: str = Field(description="相对 workspace 的文件路径")
    confirm: bool = Field(
        default=False,
        description="二次确认是否删除，必须明确为True的时候才执行删除操作",
    )


def _delete_file(args: DeleteFileArgs) -> str:
    target = _resolve_in_sandbox(args.path)
    if target.is_dir():
        return f"该工具只能删除文件，不可以删除文件夹: {args.path}"

    if not args.confirm:
        return "请明确确认需要删除文件后再执行"

    if not target.is_file():
        return f"文件不存在: {args.path}"

    text = target.read_text(encoding="utf-8", errors="replace")
    content = (
        text[:4000] + f"\n…（已截断，文件共 {len(text)} 字符）"
        if len(text) > 4000
        else text
    )
    target.unlink()
    return f"已删除文件: {args.path}， 删除文件内容为: {content}"


delete_file_tool = Tool(
    name="delete_file",
    description=(
        "删除单个文件，必须明确让用户二次确认后才可真正执行，否则不执行删除操作"
        "此工具不能删除目录，也不应被用来清空目录内容作为删除目录的替代方案——"
        "如需删除目录请直接告知用户当前不支持，不要自行删除目录下的文件。"
    ),
    args_model=DeleteFileArgs,
    func=_delete_file,
    permission=ToolPermission.DELETE,
)
