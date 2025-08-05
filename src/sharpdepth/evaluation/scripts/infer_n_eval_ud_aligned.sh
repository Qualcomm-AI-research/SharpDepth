export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH 
export TOKENIZERS_PARALLELISM=false

python src/sharpdepth/evaluation/inference.py \
        --checkpoint jingheya/lotus-depth-g-v1-0 \
        --dataset_config src/sharpdepth/data/data_configs/data_kitti_eigen_test.yaml \
        --base_data_dir $BASE_DATA_DIR \
        --output_dir ./sharpdepth_exp/inference/og_ud_aligned/kitti/predictions_ud \

python src/sharpdepth/evaluation/inference.py \
        --checkpoint jingheya/lotus-depth-g-v1-0 \
        --dataset_config src/sharpdepth/data/data_configs/data_nyu_test.yaml \
        --base_data_dir $BASE_DATA_DIR \
        --output_dir ./sharpdepth_exp/inference/og_ud_aligned/nyu/predictions_ud

python src/sharpdepth/evaluation/inference_ud_aligned.py \
        --checkpoint jingheya/lotus-depth-g-v1-0 \
        --dataset_config src/sharpdepth/data/data_configs/data_spring.yaml \
        --base_data_dir $BASE_DATA_DIR \
        --output_dir ./sharpdepth_exp/inference/og_ud_aligned/spring/predictions_ud

python src/sharpdepth/evaluation/inference_ud_aligned.py \
       --checkpoint jingheya/lotus-depth-g-v1-0 \
       --dataset_config src/sharpdepth/data/data_configs/data_unrealstereo.yaml \
       --base_data_dir $BASE_DATA_DIR \
       --output_dir ./sharpdepth_exp/inference/og_ud_aligned/unrealstereo/predictions_ud

python src/sharpdepth/evaluation/eval.py \
        --prediction_dir ./sharpdepth_exp/inference/og_ud_aligned/kitti/predictions_ud \
        --output_dir ./sharpdepth_exp/inference/og_ud_aligned/kitti/metrics \
        --dataset_config src/sharpdepth/data/data_configs/data_kitti_eigen_test.yaml \
        --base_data_dir $BASE_DATA_DIR

python src/sharpdepth/evaluation/eval.py \
        --prediction_dir ./sharpdepth_exp/inference/og_ud_aligned/nyu/predictions_ud \
        --output_dir ./sharpdepth_exp/inference/og_ud_aligned/nyu/metrics \
        --dataset_config src/sharpdepth/data/data_configs/data_nyu_test.yaml \
        --base_data_dir $BASE_DATA_DIR

python src/sharpdepth/evaluation/eval_sharpness.py \
        --prediction_dir ./sharpdepth_exp/inference/og_ud_aligned/spring/predictions_ud \
        --output_dir ./sharpdepth_exp/inference/og_ud_aligned/spring/metrics \
        --dataset_config src/sharpdepth/data/data_configs/data_spring.yaml \
        --base_data_dir $BASE_DATA_DIR

python src/sharpdepth/evaluation/eval_sharpness.py \
        --prediction_dir ./sharpdepth_exp/inference/og_ud_aligned/unrealstereo/predictions_ud \
        --output_dir ./sharpdepth_exp/inference/og_ud_aligned/unrealstereo/metrics \
        --dataset_config src/sharpdepth/data/data_configs/data_unrealstereo.yaml \
        --base_data_dir $BASE_DATA_DIR
