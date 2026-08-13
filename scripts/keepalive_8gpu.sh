#!/usr/bin/env bash
# 同时占满 8 张卡保活:起 8 个独立 keepalive_train.py 进程,每卡一个。
#
# 为什么是 8 个独立进程而不是 1 个进程用 8 卡:
#   - 保活信号 = 显存被占 + GPU 在跑,8 进程各占一卡 = 8 卡全占,信号最强;
#   - 进程隔离:一张卡/一个进程挂了,其他 7 张卡还在保活,不会全军覆没;
#   - 不用改 keepalive_train.py(已支持 CUDA_VISIBLE_DEVICES + KEEPALIVE_SAVE_DIR);
#   - H800 每卡 143GB,装一份 ~30GB 的 flux-dev bf16 权重绰绰有余。
#
# 每张卡的产物(在 $SAVE_ROOT/gpuN/ 下):
#   keepalive.log  脚本自己写的单行心跳(tmux 丢失后查这个)
#   stdout.log     nohup 捕获的 stdout/stderr(含 tqdm 进度条、traceback)
#   pid            launcher 写的 PID 文件,stop/status 用
#
# 用法:
#   bash scripts/keepalive_8gpu.sh start    # 启动 8 卡(错开 30s 避免 ceph 读拥塞)
#   bash scripts/keepalive_8gpu.sh status   # 查看每卡进程状态
#   bash scripts/keepalive_8gpu.sh stop     # 停止全部
#   bash scripts/keepalive_8gpu.sh logs     # tail -f 所有 keepalive.log
#
# 可调环境变量:
#   NUM_GPUS                  占几张卡(默认 8)
#   GPU_OFFSET                从第几张卡开始占(默认 0;训练占了 0/1 时,占 2-7 用
#                             NUM_GPUS=6 GPU_OFFSET=2)
#   KEEPALIVE_SAVE_ROOT       输出根目录(默认 output/keepalive_train;不可写时自动 fallback 到 /tmp/keepalive_train)
#   STARTUP_STAGGER           启动间隔秒(默认 30,避免 8 个进程同时读 ceph 把读带宽打爆)
#   KEEPALIVE_INTERVAL        传给子进程:每轮间隔秒(默认 60)
#   KEEPALIVE_STEPS_PER_ROUND 传给子进程:每轮训练步数(默认 10)
#   KEEPALIVE_BATCH_SIZE      传给子进程:每步 batch size(默认 1)
#   KEEPALIVE_LORA_RANK       传给子进程:LoRA rank(默认 512)
#   KEEPALIVE_LR              传给子进程:AdamW 学习率(默认 1e-4)
#   KEEPALIVE_MAX_ROUNDS      传给子进程:跑 N 轮退出(默认 0=无限;设 1 可冒烟自检)

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

ACTION="${1:-}"
NUM_GPUS="${NUM_GPUS:-8}"
GPU_OFFSET="${GPU_OFFSET:-0}"
STARTUP_STAGGER="${STARTUP_STAGGER:-30}"
PY="${PY:-$REPO/.venv-uno/bin/python}"

# --- 确定输出根目录:默认 output/keepalive_train,不可写就 fallback 到 /tmp/keepalive_train ---
SAVE_ROOT="${KEEPALIVE_SAVE_ROOT:-output/keepalive_train}"
if ! mkdir -p "$SAVE_ROOT" 2>/dev/null || ! touch "$SAVE_ROOT/.write_probe" 2>/dev/null; then
  SAVE_ROOT="/tmp/keepalive_train"
  mkdir -p "$SAVE_ROOT"
  echo "⚠️  output/keepalive_train 不可写(可能是前一次 sudo 跑留下的 root:root),"
  echo "    已 fallback 到 $SAVE_ROOT。要改回 output/keepalive_train 请先 chown 修复所有权。"
fi
rm -f "$SAVE_ROOT/.write_probe" 2>/dev/null || true

# --- 前置检查 ---
if [ ! -x "$PY" ]; then
  echo "❌ $PY 不存在或不可执行。先在 H800 上跑 bash scripts/setup_env_h800.sh。" >&2
  exit 1
fi
GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
if [ "$GPU_COUNT" -lt "$NUM_GPUS" ]; then
  echo "❌ 机器只有 $GPU_COUNT 张卡,不够 $NUM_GPUS 张。用 NUM_GPUS=$GPU_COUNT 降配或换机器。" >&2
  exit 1
fi
if [ -z "$ACTION" ] || ! [[ "$ACTION" =~ ^(start|stop|status|logs)$ ]]; then
  echo "用法: bash $0 {start|stop|status|logs}" >&2
  echo "  start   启动 $NUM_GPUS 个进程,每卡一个,错开 ${STARTUP_STAGGER}s"
  echo "  stop    停止全部"
  echo "  status  查看每卡进程状态"
  echo "  logs    tail -f 所有 keepalive.log(Ctrl-C 退出)"
  exit 1
fi

# 传给子进程的环境变量(只在子进程里生效,不污染当前 shell)
pass_env() {
  local i="$1"
  # KEEPALIVE_SAVE_DIR 子进程自己用;HF_HUB_OFFLINE / PYTORCH_CUDA_ALLOC_CONF 子进程已 setdefault
  echo "CUDA_VISIBLE_DEVICES=$i KEEPALIVE_SAVE_DIR=$SAVE_ROOT/gpu$i"
}

case "$ACTION" in
  start)
    echo "=== 启动 $NUM_GPUS 个保活进程,输出根目录 $SAVE_ROOT ==="
    # venv python 必须能 import torch(本机 .venv-uno/bin/python 是 3.8 会挂在这里,
    # H800 上是 3.10 就没问题——在启动 8 个进程前就发现环境问题,而不是 8 个都失败后才发现)
    if ! "$PY" -c "import torch" >/dev/null 2>&1; then
      echo "❌ $PY 无法 import torch。" >&2
      echo "   在 H800 上:先 source .venv-uno/bin/activate 再跑本脚本,或检查 venv 是否完整。" >&2
      echo "   在本机调试:PY=\$(uv python find 3.10) PYTHONPATH=.venv-uno/lib/python3.10/site-packages bash $0 start" >&2
      exit 1
    fi
    for i in $(seq "$GPU_OFFSET" $((GPU_OFFSET + NUM_GPUS - 1))); do
      DIR="$SAVE_ROOT/gpu${i}"
      mkdir -p "$DIR"
      LOG="$DIR/stdout.log"
      # 幂等:已在运行就跳过
      if [ -f "$DIR/pid" ] && kill -0 "$(cat "$DIR/pid")" 2>/dev/null; then
        echo "GPU$i: 已在运行 (PID $(cat "$DIR/pid")),跳过"
        continue
      fi
      # 先检查这张卡是否空闲(显存占用 >1GB 认为被别人占了)
      MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$i" 2>/dev/null | tr -d ' ')
      if [ "${MEM_USED:-0}" -gt 1024 ]; then
        echo "GPU$i: ⚠️  显存已用 ${MEM_USED}MiB,可能被别人占用,仍继续启动(保活优先)"
      fi
      # 启动(HF_HUB_OFFLINE=1 等由子脚本自己 setdefault,这里不重复设)
      env CUDA_VISIBLE_DEVICES="$i" KEEPALIVE_SAVE_DIR="$DIR" \
          KEEPALIVE_INTERVAL="${KEEPALIVE_INTERVAL:-60}" \
          KEEPALIVE_STEPS_PER_ROUND="${KEEPALIVE_STEPS_PER_ROUND:-10}" \
          KEEPALIVE_BATCH_SIZE="${KEEPALIVE_BATCH_SIZE:-1}" \
          KEEPALIVE_LORA_RANK="${KEEPALIVE_LORA_RANK:-512}" \
          KEEPALIVE_LR="${KEEPALIVE_LR:-1e-4}" \
          KEEPALIVE_MAX_ROUNDS="${KEEPALIVE_MAX_ROUNDS:-0}" \
          nohup "$PY" "$REPO/scripts/keepalive_train.py" > "$LOG" 2>&1 &
      PID=$!
      echo "$PID" > "$DIR/pid"
      echo "GPU$i: 启动 PID=$PID | $DIR/keepalive.log | $DIR/stdout.log"
      # 错开启动,避免 8 个进程同时读 ceph 76GB 权重把读带宽打爆
      [ $i -lt $((NUM_GPUS - 1)) ] && sleep "$STARTUP_STAGGER"
    done
    echo ""
    echo "全部已启动。监控:"
    echo "  bash $0 status"
    echo "  bash $0 logs"
    echo "  nvidia-smi"
    echo "停止: bash $0 stop"
    ;;

  stop)
    echo "=== 停止 $NUM_GPUS 个保活进程 ==="
    for i in $(seq "$GPU_OFFSET" $((GPU_OFFSET + NUM_GPUS - 1))); do
      DIR="$SAVE_ROOT/gpu${i}"
      [ -f "$DIR/pid" ] || { echo "GPU$i: 无 PID 文件,跳过"; continue; }
      PID="$(cat "$DIR/pid")"
      if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        echo "GPU$i: 已发 SIGTERM 到 PID $PID"
      else
        echo "GPU$i: PID $PID 已不在,清理 PID 文件"
      fi
      rm -f "$DIR/pid"
    done
    echo "已发 SIGTERM。若进程没退,等 5s 后发 SIGKILL..."
    sleep 5
    for i in $(seq "$GPU_OFFSET" $((GPU_OFFSET + NUM_GPUS - 1))); do
      DIR="$SAVE_ROOT/gpu${i}"
      [ -f "$DIR/pid" ] && kill -9 "$(cat "$DIR/pid")" 2>/dev/null && echo "GPU$i: SIGKILL"
      rm -f "$DIR/pid" 2>/dev/null || true
    done
    ;;

  status)
    echo "=== $SAVE_ROOT 下 $NUM_GPUS 个保活进程状态 ==="
    for i in $(seq "$GPU_OFFSET" $((GPU_OFFSET + NUM_GPUS - 1))); do
      DIR="$SAVE_ROOT/gpu${i}"
      if [ -f "$DIR/pid" ] && kill -0 "$(cat "$DIR/pid")" 2>/dev/null; then
        PID="$(cat "$DIR/pid")"
        ELAPSED="$(ps -o etime= -p "$PID" 2>/dev/null | tr -d ' ')"
        LAST="$(tail -1 "$DIR/keepalive.log" 2>/dev/null || echo '<无日志>')"
        echo "GPU$i: ✅ PID $PID | 已运行 $ELAPSED | $LAST"
      else
        echo "GPU$i: ❌ 未运行"
      fi
    done
    ;;

  logs)
    echo "=== tail -f 所有 keepalive.log(Ctrl-C 退出) ==="
    # 用 tail -f 同时跟多个文件,文件名头会自动打印
    exec tail -f "$SAVE_ROOT"/gpu*/keepalive.log
    ;;
esac
