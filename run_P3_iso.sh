#!/usr/bin/env bash
# M6 P3 蒸馏腿 A(iso)—— A 腿 stage-1 跑满后在同一台机器(ge90-95)上接跑,见 M6_STEP2_RUN.md。
# 三个变量一个都不能漏(PROJECT_DIR 默认 log/ref_distill 会覆盖 M3 结果;
# RESUME 默认是 4090 旧底座;REF_ISOLATION 默认 True 与 A 腿相符但显式写死)。
# MAX_TRAIN_STEPS 默认 4000(train_distill.sh:38),与 SPEC §3 一致,不填。
cd /kaimm-distill/wuwenxuan/UNO && mkdir -p logs
source .venv-uno/bin/activate
REF_ISOLATION=True \
RESUME_FROM_CHECKPOINT=log/stage1_official/checkpoint-100000/dit_lora.safetensors \
PROJECT_DIR=log/ref_distill_iso \
setsid bash scripts/train_distill.sh > logs/p3_iso.log 2>&1 < /dev/null &
echo "pid=$!  ← 记下来"
