#!/usr/bin/env bash
# M6 P3 蒸馏 —— B 腿(full baseline)的蒸馏。见 distill/M6_STEP2_RUN.md 末尾。
# ⚠️ 三个变量一个都不能漏,train_distill.sh 的默认值全是旧的:
#   PROJECT_DIR 默认 log/ref_distill          → 会覆盖 M3 既有结果(该脚本没有拒启 guard)
#   RESUME_FROM_CHECKPOINT 默认 4090 旧底座   → 消融两腿就不成对了
#   REF_ISOLATION 默认 True                   → 会变成第二条 iso 腿
cd /kaimm-distill/wuwenxuan/UNO && mkdir -p logs
source .venv-uno/bin/activate
REF_ISOLATION=False \
RESUME_FROM_CHECKPOINT=log/stage1_official_full/checkpoint-100000/dit_lora.safetensors \
PROJECT_DIR=log/ref_distill_full \
setsid bash scripts/train_distill.sh > logs/p3_full.log 2>&1 < /dev/null &
echo "pid=$!  ← 记下来"
