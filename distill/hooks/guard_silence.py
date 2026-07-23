#!/usr/bin/env python3
"""PreToolUse 守卫:沉默太久就拦下一次工具调用,逼它先开口。

对应 REMOTE_AGENT_HANDBOOK.md §5「沉默上限」。

原理:读 transcript,从末尾往回找**最近一次 assistant 输出的正文文本**,
统计从那以后攒了多少次工具调用、过了多久。超阈值就拒绝这次调用,
理由里直接告诉它要报什么。它一旦开口,计数自然归零,不需要额外状态文件。

只能覆盖"连跑一串命令"这种沉默。
"一条命令跑很久"那种黑盒沉默 hook 是看不见的(期间根本没有事件),
那个交给 guard_slow_bash.py 事前拦截。

设计原则:**fail-open**,任何异常一律放行。
"""
import json
import sys
from datetime import datetime, timezone

MAX_TOOL_CALLS = 12  # 攒够这么多次调用还没说话
MAX_MINUTES = 10.0  # 或者沉默超过这么久
MIN_TEXT_CHARS = 40  # 太短的一句(如"好的")不算"说过话"


def parse_ts(raw):
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        return None


def main() -> None:
    try:
        payload = json.load(sys.stdin)
        path = payload["transcript_path"]
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except Exception:
        sys.exit(0)  # fail-open

    tool_calls = 0
    last_spoke_ts = None
    try:
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            if entry.get("type") != "assistant":
                continue
            blocks = (entry.get("message") or {}).get("content") or []
            if not isinstance(blocks, list):
                continue
            spoke = any(
                b.get("type") == "text" and len((b.get("text") or "").strip()) >= MIN_TEXT_CHARS
                for b in blocks
                if isinstance(b, dict)
            )
            if spoke:
                last_spoke_ts = parse_ts(entry.get("timestamp"))
                break
            tool_calls += sum(
                1 for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"
            )
    except Exception:
        sys.exit(0)

    minutes = 0.0
    if last_spoke_ts is not None:
        try:
            minutes = (datetime.now(timezone.utc) - last_spoke_ts).total_seconds() / 60.0
        except Exception:
            minutes = 0.0

    if tool_calls < MAX_TOOL_CALLS and minutes < MAX_MINUTES:
        sys.exit(0)

    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    f"手册 §5 沉默上限:距你上次开口已经 {tool_calls} 次工具调用 / "
                    f"{minutes:.1f} 分钟。\n"
                    "用户在两台机器之间转达,看不到你的终端——沉默对他而言等同于挂掉,"
                    "他会 kill 掉你正在跑的东西。\n"
                    "**先给用户一段文字再继续**,内容要能独立看懂:\n"
                    "  1. 我刚才在做什么、结果是什么(有数字就给数字);\n"
                    "  2. 现在卡在哪 / 下一步要做什么;\n"
                    "  3. 有没有需要他或 Opus 拍板的事(红灯项按 §3 出诊断)。\n"
                    "说完再重新发起这次调用即可,本拦截会自动解除。"
                ),
            }
        },
        sys.stdout,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
