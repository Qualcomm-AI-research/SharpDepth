export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH 
export TOKENIZERS_PARALLELISM

python evaluation/eval_sharpness.py \
        --prediction_dir ../sharpdepth_exp/inference/og/spring/predictions \
        --output_dir ../sharpdepth_exp/inference/og/spring/metrics \
        --dataset_config data_configs/data_spring.yaml \
        --base_data_dir $BASE_DATA_DIR