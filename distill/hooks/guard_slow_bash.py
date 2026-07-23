#!/usr/bin/env python3
"""PreToolUse(Bash) 守卫:禁止把耗时命令跑成前台黑盒。

对应 REMOTE_AGENT_HANDBOOK.md §4.0。
拦截两类写法:
  1. 已知耗时的命令(下载 / 装包 / 生成 / 训练)跑在前台 —— 一旦启动就再没有
     任何 hook 会触发,用户面对的是一个不动的终端,只能 kill(M0 的 wget 就是这么没的)。
  2. 耗时命令被 -q / --quiet / >/dev/null 关掉了进度输出 —— 进度是唯一的生命体征。

设计原则:**fail-open**。任何解析异常都放行。
这个 hook 的作用是纠正习惯,不是当安全边界;它自己出 bug 绝不能把远程 agent 卡死。
"""
import json
import re
import sys

# 已知会跑很久的命令。宁可漏判也别误判——误判会让 agent 陷入"改写命令绕过 hook"的死循环,
# 那比它跑个前台命令糟糕得多。
SLOW = re.compile(
    r"""(?x)
    # 命令词可能被 nohup / setsid / time / env 之类包一层,锚点必须认这些前缀,
    # 否则后台写法会整个绕过检查(实测踩过)
    (?: ^ | [;&|] | \b(?:nohup|setsid|time|env|stdbuf) \b )
    \s*
    (
      wget | curl\s+[^|]*\s-O | git\s+clone | git\s+lfs
    | (pip|pip3|uv)\s+(install|download)
    | accelerate\s+launch | torchrun | deepspeed
    | python3?\s+\S*(train|gen_data|filter_data|infer_|evaluate_)\S*\.py
    | huggingface-cli\s+download | hf\s+download
    )
    \b
    """
)

# 关掉进度输出的写法
MUTED = re.compile(r"(?:^|\s)(-q|--quiet|--silent|-s)(?:\s|$)|2?>\s*/dev/null")

# 已经在后台了的写法
BACKGROUNDED = re.compile(r"(?:^|\s|;)nohup\s|&\s*$|\bsetsid\b|\bdisown\b|\btmux\b|\bscreen\s+-d")


def deny(reason: str) -> None:
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        },
        sys.stdout,
    )
    sys.exit(0)


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # fail-open

    if payload.get("tool_name") != "Bash":
        sys.exit(0)

    tool_input = payload.get("tool_input") or {}
    cmd = tool_input.get("command") or ""
    if not SLOW.search(cmd):
        sys.exit(0)

    if tool_input.get("run_in_background") is not True and not BACKGROUNDED.search(cmd):
        deny(
            "手册 §4.0-1:这条命令属于已知耗时类型,不能在前台跑。\n"
            "前台长命令期间不会触发任何 hook,用户只看到一个不动的终端,"
            "无法区分'在跑'和'卡死',最后只会 kill 掉它(M0 的 wget 就是这么没的)。\n"
            "改法(二选一):\n"
            "  a) Bash 工具传 run_in_background=true;\n"
            "  b) 命令写成 `nohup <cmd> > logs/<name>.log 2>&1 &` 并 echo $!。\n"
            "然后**先给用户一行预告**(要干什么 / 预计多久 / 日志在哪),再靠轮询日志观察。"
        )

    if MUTED.search(cmd):
        deny(
            "手册 §4.0-2:耗时命令不许关掉进度输出。\n"
            f"检测到静音写法:{MUTED.search(cmd).group(0).strip()!r}\n"
            "进度信息是这个任务唯一的生命体征——没有它,慢和死是同一种表现。\n"
            "去掉 -q/--quiet/>/dev/null;wget 用 --progress=dot:giga,"
            "把输出重定向到日志文件而不是丢弃。"
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
