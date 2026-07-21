"""工具权限策略：决定工具调用是允许、拒绝还是需要人工审批。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping

from .tools.base import ToolPermission


class PolicyAction(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass(frozen=True)
class ApprovalRequest:
    """一次与具体工具调用绑定的审批请求。"""

    tool_use_id: str
    tool_name: str
    permission: ToolPermission
    arguments: dict[str, Any]


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str


ApprovalCallback = Callable[[ApprovalRequest], bool]


class ToolPolicy:
    """集中执行工具权限规则；受保护权限默认需要审批。"""

    def __init__(
        self,
        rules: Mapping[ToolPermission, PolicyAction] | None = None,
        approver: ApprovalCallback | None = None,
    ) -> None:
        self.rules = {
            ToolPermission.READ: PolicyAction.ALLOW,
            ToolPermission.WRITE: PolicyAction.ASK,
            ToolPermission.DELETE: PolicyAction.ASK,
            ToolPermission.NETWORK: PolicyAction.ASK,
            ToolPermission.PROCESS: PolicyAction.ASK,
            ToolPermission.SENSITIVE: PolicyAction.ASK,
        }
        if rules is not None:
            self.rules.update(rules)
        self.approver = approver

    def action_for(self, permission: ToolPermission) -> PolicyAction:
        return self.rules.get(permission, PolicyAction.DENY)

    def authorize(
        self,
        *,
        tool_use_id: str,
        tool_name: str,
        permission: ToolPermission,
        arguments: dict[str, Any],
    ) -> PolicyDecision:
        action = self.action_for(permission)
        if action == PolicyAction.ALLOW:
            return PolicyDecision(True, f"策略允许 {permission.value} 权限")
        if action == PolicyAction.DENY:
            return PolicyDecision(False, f"策略拒绝 {permission.value} 权限")
        if self.approver is None:
            return PolicyDecision(False, f"工具 {tool_name} 需要审批，但未配置审批器")

        request = ApprovalRequest(
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            permission=permission,
            arguments=deepcopy(arguments),
        )
        try:
            approved = self.approver(request)
        except Exception as exc:
            return PolicyDecision(
                False,
                f"工具审批失败：{type(exc).__name__}: {exc}",
            )
        if approved is not True:
            return PolicyDecision(False, f"用户拒绝执行工具 {tool_name}")
        return PolicyDecision(True, f"用户已批准执行工具 {tool_name}")
