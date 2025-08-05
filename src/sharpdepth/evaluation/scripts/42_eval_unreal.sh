export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH 
export TOKENIZERS_PARALLELISM

python evaluation/eval_sharpness.py \
        --prediction_dir ../sharpdepth_exp/inference/og/unrealstereo/predictions \
        --output_dir ../sharpdepth_exp/inference/og/unrealstereo/metrics \
        --dataset_config data_configs/data_unrealstereo.yaml \
        --base_data_dir $BASE_DATA_DIR