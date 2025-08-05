export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH 
export TOKENIZERS_PARALLELISM=false

python app.py \
        --checkpoint checkpoints/sharpdepth \
        --output_dir out/ \
        --input_dir assets/in-the-wild_example

