# SharpDepth: Sharpening Metric Depth Predictions Using Diffusion Distillation (CVPR 2025)

**Official PyTorch implementation** of our CVPR 2025 paper:
**"SharpDepth: Sharpening Metric Depth Predictions Using Diffusion Distillation"**

Team:
[Duc-Hai Pham*](https://haiphamcse.github.io/),
[Tung Do*](https://itsthanhtung.github.io/),
[Phong Nguyen](https://phongnhhn.info/)
[Binh-Son Hua](https://sonhua.github.io/),
[Khoi Nguyen](https://www.khoinguyen.org/),
[Rang Nguyen](https://rangnguyen.github.io/)

![teaser_all](assets/sharpdepth.gif)

> **Abstract**: 
We propose SharpDepth, a novel approach to monocular metric depth estimation that combines the metric accuracy of discriminative depth estimation methods (e.g., Metric3D, UniDepth) with the fine-grained boundary sharpness typically achieved by generative methods (e.g., Marigold, Lotus). Traditional discriminative models trained on real-world data with sparse ground-truth depth can accurately predict metric depth but often produce over-smoothed or low-detail depth maps. Generative models, in contrast, are trained on synthetic data with dense ground truth, generating depth maps with sharp boundaries yet only providing relative depth with low accuracy. Our approach bridges these limitations by integrating metric accuracy with detailed boundary preservation, resulting in depth predictions that are both metrically precise and visually sharp. Our extensive zero-shot evaluations on standard depth estimation benchmarks confirm SharpDepth's effectiveness, showing its ability to achieve both high depth accuracy and detailed representation, making it well-suited for applications requiring high-quality depth perception across diverse, real-world environments.

## 📑 Getting Started


### 🛠️ Environment Setup

A Dockerfile is provided to ease environment setup and code reproducibility. To build the Docker image you can run:

```bash
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg USERNAME=$(id -un)  -t sharpdepth:v1 -f docker/Dockerfile .
```

Once you have built the image you can run it using the following command, replacing with the appropriate paths and working directory. You
then have a bash command-line from which you can run the code.

```bash
docker run -v </path/to/sharpdepth/repo>:</path/to/sharpdepth/repo> -w </path/to/sharpdepth/repo> --shm-size=8g -it sharpdepth:v1 bash
```

You will then need to configure 🤗 [Accelerate](https://github.com/huggingface/accelerate):

```bash
accelerate config  # interactive setup
# or
accelerate config default  # quick default setup
```

---
### 💾 Datasets
Please refer to [data preparation](docs/dataset.md).

---

### 🔐 Checkpoints

We have released the primary checkpoint used in the paper. To download, please follow the instructions below:

```bash
mkdir checkpoints
wget https://github.com/Qualcomm-AI-research/SharpDepth/releases/download/v1.0/sharpdepth.tar.gz.part-aa
wget https://github.com/Qualcomm-AI-research/SharpDepth/releases/download/v1.0/sharpdepth.tar.gz.part-ab 
wget https://github.com/Qualcomm-AI-research/SharpDepth/releases/download/v1.0/sharpdepth.tar.gz.part-ac
cat sharpdepth.tar.gz.part-* >sharpdepth.tar.gz
tar zxvf sharpdepth.tar.gz
```

This should create the following directory with checkpoints:

```bash
checkpoints/sharpdepth
```
---



## 📑 How to Run
### 🖼 Inference

To run SharpDepth on in-the-wild images:

```bash
bash app.sh
```

---

### 📊 Evaluation

Before running evaluations, export the required paths:
```bash
export BASE_DATA_DIR=/path/to/datasets
export HF_HOME=/path/to/huggingface/cache
export BASE_CKPT_DIR=/path/to/checkpoints
```
#### SharpDepth Evaluation
```bash
bash src/sharpdepth/evaluation/scripts/infer_n_eval.sh
```

#### UniDepth-Aligned Lotus Evaluation

```bash
bash src/sharpdepth/evaluation/scripts/infer_n_eval_ud_aligned.sh
```

Both models can be evaluated on:

* **Metric accuracy** (e.g., KITTI, NYUv2)
* **Sharpness metrics** (e.g., Sintel, UnrealStereo4K, Spring)

---

### 📈 Results
After running you should obtain a score similar to this
|                      | KITTI δ1 ↑ | NYUv2 δ1 ↑ | Sintel DBE_acc ↓ | Spring DBE_acc ↓ | Unreal4k DBE_acc ↓  |
|-----------------------|------------|-----------------|------------|---------------------|-----------------|
| UD-Aligned Lotus | 71.650      | 87.310           |  2.04      |  1.27              | 1.21            | 
| UniDepth         | 97.921      |  93.921          | 3.73     |  5.29                |   8.65          | 
| Ours             | 97.315      | 96.949           | 1.94      | 1.24               | 1.37            | 

---


### 🏋️ Training

⚠️ Note: The training code is provided for reference and is not runnable out-of-the-box. You may follow Marigold for dataset loading and adapt our training script (based on 🤗 Diffusers).


Example training command:

```bash
bash src/sharpdepth/training/train.sh
```

For how to configure the settings of the training, please take a look into `src/sharpdepth/training/train.sh`

---


📄 [Read the full paper](https://openaccess.thecvf.com/content/CVPR2025/html/Pham_SharpDepth_Sharpening_Metric_Depth_Predictions_Using_Diffusion_Distillation_CVPR_2025_paper.html)

If you use this code in your work, please **cite our paper**:

```bibtex
@InProceedings{Pham_2025_CVPR,
    author    = {Pham, Duc-Hai and Do, Tung and Nguyen, Phong and Hua, Binh-Son and Nguyen, Khoi and Nguyen, Rang},
    title     = {SharpDepth: Sharpening Metric Depth Predictions Using Diffusion Distillation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2025},
    pages     = {17060--17069}
}
```

## 📬 Contact

For questions or issues, please contact:
📧 [haipham@qti.qualcomm.com](mailto:haipham@qti.qualcomm.com)
or open an issue in this repository.

---
