#!/usr/bin/env bash
# watch-train.sh - Theo dõi tiến trình huấn luyện DTNet-P2 thời gian thực trên Tesla V100
echo "======================================================================"
echo "   ĐANG KẾT NỐI VÀ THEO DÕI LIVE LOG TRÊN TESLA V100 (dutai-bkmaker)"
echo "   (Nhấn Ctrl + C để thoát khỏi màn hình theo dõi bất cứ lúc nào)"
echo "======================================================================"
ssh -t dutai-bkmaker "tail -f -n 30 /home/dutai/EAI-2026/qualifiers/weights/dtnet_p2_champion/train.log"
