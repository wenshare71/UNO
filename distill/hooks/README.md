# 远程 agent 的行为约束(hooks 层)

`REMOTE_AGENT_HANDBOOK.md` 是**约定**——靠 agent 记得住才生效。
这里是**机制**——在工具调用层拦截,记不住也生效。两者配合,不互相替代。

## 装在哪

**装到 `UNO/.claude/settings.local.json`(远程机器上),不要提交。**
`settings.json` 会跟着仓库走,连带影响本地电脑上的会话;`settings.local.json`
是机器本地的,各管各的。安装这一步属于绿灯,自己做即可。

```bash
cd <repo>/UNO
mkdir -p .claude
chmod +x distill/hooks/*.py
cat distill/hooks/settings.snippet.json   # 按需合并进 .claude/settings.local.json
```

改完在 Claude Code 里跑 `/hooks` 确认加载成功。**hook 脚本改动不会热重载**,
要重启会话或重新 `/hooks`。

## 三层,由硬到软

| 层 | 机制 | 管什么 | 特点 |
|---|---|---|---|
| 1 | `permissions.deny` | 绝对禁止的命令(force push、kill 别人的进程、`git add -f`) | 零代码,拒绝彻底,但只能按前缀匹配 |
| 2 | `PreToolUse` hook | 前台长命令、静音下载、沉默过久 | 能看到完整参数并给出**为什么被拒**,agent 会照着改 |
| 3 | `UNO/CLAUDE.md` | 铁律摘要 + 指向手册 | 常驻上下文,影响判断而非阻断动作 |

## 两个 hook 分别解决什么

- **`guard_slow_bash.py`** —— 事前拦截。已知耗时的命令(wget/pip/训练/生成)
  必须后台 + 日志,且不许 `-q`/`>/dev/null`。
  **这条是不可替代的**:命令一旦在前台跑起来,期间**没有任何 hook 会触发**,
  没有第二次机会。M0 的 `wget -c -q` 被 kill 掉就是这个失效模式。
- **`guard_silence.py`** —— 事后追赶。读 transcript,若距上次开口超过
  12 次工具调用或 10 分钟,就拒掉下一次调用并说明要汇报什么。
  它开口后计数自然归零,不需要状态文件。

## 两条设计原则(改这些脚本时请保持)

1. **一律 fail-open。** 任何解析异常都 `sys.exit(0)` 放行。
   hook 的目的是纠正习惯,不是当安全边界——它自己出 bug 绝不能把远程 agent 卡死,
   而远程 agent 卡死我们要隔一整个转达来回才发现。
2. **拒绝时必须给出可执行的改法。** 只说"不许这样"会让 agent 去猜、去绕
   (比如把 `wget` 换个写法躲开正则),那比原来的问题更糟。
   每条 deny 理由都要写清楚:违反了哪条、为什么这条存在、具体改成什么。

## 调阈值

`guard_silence.py` 顶部的 `MAX_TOOL_CALLS` / `MAX_MINUTES`;
`guard_slow_bash.py` 的 `SLOW` 正则。**宁可漏判也别误判**——
误判会让 agent 陷入"改写命令绕过 hook"的循环,比它偶尔跑个前台命令糟糕得多。

## 本地自测(不用起会话)

```bash
echo '{"tool_name":"Bash","tool_input":{"command":"wget -c -q -O x.pth https://example/x"}}' \
  | python3 distill/hooks/guard_slow_bash.py
# 期望:输出 deny,理由里同时点出「前台」这一条
```
