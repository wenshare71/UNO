#!/usr/bin/env bash
# M6 P2 stage-1 腿 B(full baseline)—— 双机并行第二腿,见 distill/M6_STEP2_RUN.md §0.4。
# 在新申请到、且挂载 kaimm-distill 盘的机器上跑。setsid + </dev/null(不用 nohup)。
cd /kaimm-distill/wuwenxuan/UNO && mkdir -p logs
source .venv-uno/bin/activate
REF_ISOLATION=False PROJECT_DIR=log/stage1_official_full \
setsid bash scripts/train_stage1_official.sh \
  > logs/p2_full.log 2>&1 < /dev/null &
echo "pid=$!  ← 记下来"
