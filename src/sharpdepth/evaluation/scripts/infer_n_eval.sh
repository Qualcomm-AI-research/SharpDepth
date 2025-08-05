export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
export TOKENIZERS_PARALLELISM=false

python src/sharpdepth/evaluation/inference.py \
        --checkpoint checkpoints/sharpdepth \
        --dataset_config src/sharpdepth/data/data_configs/data_kitti_eigen_test.yaml \
        --base_data_dir $BASE_DATA_DIR \
        --output_dir ./sharpdepth_exp/inference/og/kitti/predictions

python src/sharpdepth/evaluation/inference.py \
        --checkpoint checkpoints/sharpdepth \
        --dataset_config src/sharpdepth/data/data_configs/data_nyu_test.yaml \
        --base_data_dir $BASE_DATA_DIR \
        --output_dir ./sharpdepth_exp/inference/og/nyu/predictions

python src/sharpdepth/evaluation/inference.py \
        --checkpoint checkpoints/sharpdepth \
        --dataset_config src/sharpdepth/data/data_configs/data_unrealstereo.yaml \
        --base_data_dir $BASE_DATA_DIR \
        --output_dir ./sharpdepth_exp/inference/og/unrealstereo/predictions

python src/sharpdepth/evaluation/inference.py \
        --checkpoint checkpoints/sharpdepth \
        --dataset_config src/sharpdepth/data/data_configs/data_spring.yaml \
        --base_data_dir $BASE_DATA_DIR \
        --output_dir ./sharpdepth_exp/inference/og/spring/predictions

python src/sharpdepth/evaluation/eval.py \
        --prediction_dir ./sharpdepth_exp/inference/og/kitti/predictions \
        --output_dir ./sharpdepth_exp/inference/og/kitti/metrics \
        --dataset_config src/sharpdepth/data/data_configs/data_kitti_eigen_test.yaml \
        --base_data_dir $BASE_DATA_DIR

python src/sharpdepth/evaluation/eval.py \
        --prediction_dir ./sharpdepth_exp/inference/og/nyu/predictions \
        --output_dir ./sharpdepth_exp/inference/og/nyu/metrics \
        --dataset_config src/sharpdepth/data/data_configs/data_nyu_test.yaml \
        --base_data_dir $BASE_DATA_DIR


python src/sharpdepth/evaluation/eval_sharpness.py \
        --prediction_dir ./sharpdepth_exp/inference/og/spring/predictions \
        --output_dir ./sharpdepth_exp/inference/og/spring/metrics \
        --dataset_config src/sharpdepth/data/data_configs/data_spring.yaml \
        --base_data_dir $BASE_DATA_DIR

python src/sharpdepth/evaluation/eval_sharpness.py \
        --prediction_dir ./sharpdepth_exp/inference/og/unrealstereo/predictions \
        --output_dir ./sharpdepth_exp/inference/og/unrealstereo/metrics \
        --dataset_config src/sharpdepth/data/data_configs/data_unrealstereo.yaml \
        --base_data_dir $BASE_DATA_DIR
