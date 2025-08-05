export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH 
export TOKENIZERS_PARALLELISM

python evaluation/eval.py \
        --prediction_dir ../sharpdepth_exp/inference/og/nyu/predictions \
        --output_dir ../sharpdepth_exp/inference/og/nyu/metrics \
        --dataset_config data_configs/data_nyu_test.yaml \
        --base_data_dir $BASE_DATA_DIR
