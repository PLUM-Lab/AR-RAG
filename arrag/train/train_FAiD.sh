torchrun --nproc_per_node=1 --master_port=29998 arrag/train/train_FAiD.py \
    --train_data data/training_data_w_retrieval \
    --model_name_or_path "deepseek-ai/Janus-Pro-1B" \
    --output_dir result/ckpts \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --learning_rate 2e-4 \
    --max_text_length 512 \
    --max_image_length 576 \
    --retrieved_key_num 10 \
    --num_blender 2 \
    --num_hop '1,2'
echo "========== b2h12 done =========="