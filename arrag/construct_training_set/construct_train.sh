# python arrag/construct_training_set/img2code_L.py \
#     --img_source lmms-lab/LLaVA-ReCap-CC12M \
#     --output_dir data/training_data \
#     --txt_field 'conversations'

python arrag/construct_training_set/code2train_L.py \
    --retriever_path data/retriever/index_L \
    --data_path data/training_data \
    --save_dir data/training_data_w_retrieval