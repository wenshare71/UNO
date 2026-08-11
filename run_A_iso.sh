#!/usr/bin/env bash
# M6 P2 stage-1 腿 A(student/iso)—— 单机串行第一腿,见 distill/M6_STEP2_RUN.md §0.4。
# setsid + </dev/null(不用 nohup,DISTILL_PLAN §11.12(a):elastic 会盖掉 nohup 的 SIG_IGN)。
cd /kaimm-distill/wuwenxuan/UNO && mkdir -p logs
source .venv-uno/bin/activate
REF_ISOLATION=True PROJECT_DIR=log/stage1_official \
setsid bash scripts/train_stage1_official.sh \
  > logs/p2_iso.log 2>&1 < /dev/null &
echo "pid=$!  ← 记下来"
