python create_data/gpt_create_train_r2r_rxr_long.py \
  --input_json /mnt/data/vmo-ai-task/anhdh35/JanusVLN/train_r2r_rxr.json \
  --trajectory_root /mnt/data/vmo-ai-task/anhdh35/JanusVLN/data/trajectory_data \
  --output_json /mnt/data/vmo-ai-task/anhdh35/JanusVLN/data/labels/train_r2r_rxr_long.json \
  --stats_json /mnt/data/vmo-ai-task/anhdh35/JanusVLN/data/labels/train_r2r_rxr_long_stats.json \
  --max_frames 300 \
  --local_frames 32 \
  --keep_mode all \
  --num_workers 40