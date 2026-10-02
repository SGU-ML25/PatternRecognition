#!/usr/bin/env bash
# Train tiếp ResNet18 từ checkpoint cuối (checkpoints/resnet18/last.pt).
# Giữ nguyên lịch learning rate, best AP và file train_log.csv.
cd "$(dirname "$0")/training" || exit 1
nohup ../.venv/bin/python train.py --backbone resnet18 --epochs 10 --bs 32 --workers 6 \
  --out ../checkpoints/resnet18 --resume >> ../logs/train_resnet18.log 2>&1 &
echo "Đang train tiếp (PID $!). Xem tiến độ: tail -f logs/train_resnet18.log"
