# EPIC: Explanation of Pretrained Image Classification Networks via Prototypes
Piotr Borycki, Magdalena Trędowicz, Szymon Janusz, Jacek Tabor, Przemysław Spurek, Arkadiusz Lewicki, Łukasz Struski

This repository contains the official authors implementation associated
with the paper ["EPIC: Explanation of Pretrained Image Classification Networks via Prototypes"](https://arxiv.org/abs/2505.12897).

Abstract:

Explainable AI (XAI) methods generally fall into two categories. Post-hoc approaches generate explanations for pre-trained models and are compatible with various neural network architectures. These methods often use feature importance visualizations, such as saliency maps, to indicate which input regions influenced the model’s prediction. Unfortunately, they typically offer a coarse understanding of the model’s decision-making process. In contrast, ante-hoc (inherently explainable) methods rely on specially designed model architectures trained from scratch. A notable subclass of these methods provides explanations through prototypes, representative patches extracted from the training data. However, prototype-based approaches have limitations: they require dedicated architectures, involve specialized training procedures, and perform well only on specific datasets. In this work, we propose EPIC (Explanation of Pretrained Image Classification), a novel approach that bridges the gap between these two paradigms.

Like post-hoc methods, EPIC operates on pre-trained models without architectural modifications. Simultaneously, it delivers intuitive, prototype-based explanations inspired by ante-hoc techniques. To the best of our knowledge, EPIC is the first post-hoc method capable of fully replicating the core explanatory power of inherently interpretable models. We evaluate EPIC on benchmark datasets commonly used in prototype-based explanations, such as CUB-200-2011 and Stanford Cars, alongside large-scale datasets like ImageNet, typically employed by post-hoc methods. EPIC uses prototypes to explain model decisions, providing a flexible and easy-to-understand tool for creating clear, high-quality explanations.

<img src="imgs/architecture.jpg" width="600">

# Installation

The following installation instructions are provided for a Conda-based Python environment.

### Clone the Repository

```shell
# SSH
git clone git@github.com:piotr310100/EPIC.git
```
or
```shell
# HTTPS
git clone https://github.com/piotr310100/EPIC.git
```
### Environment
To install
```bash
# Create and activate the environment
conda create -y -n epic python=3.13
conda activate epic

# Install PyTorch
# To install with CUDA (if you have a compatible GPU), uncomment ONE of the following:

# For CUDA 12.8:
# pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# For CUDA 12.6:
# pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126

# For CUDA 11.8:
# pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118

# Otherwise, install CPU-only version:
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```
# Datasets 
We evaluate **EPIC** on four widely used benchmarks to assess its interpretability and generalization across a diverse range of domains and visual granularities:

- **ImageNet (ILSVRC-2012)**  
  A large-scale benchmark consisting of over 1.2 million images labeled across 1,000 object categories. ImageNet serves as the primary testbed for evaluating the global behavior of pretrained networks and the scalability of EPIC on complex, real-world visual concepts.

- **Stanford Cars**  
  A fine-grained dataset containing 16,185 images of 196 car classes. This benchmark is used to assess EPIC’s ability to localize class-representative prototypes within tightly clustered visual domains.

- **Stanford Dogs**  
  A dataset comprising 20,580 images spanning 120 dog breeds. Given the high intra-class variability and subtle inter-class differences, this dataset is ideal for testing EPIC's robustness and semantic alignment in explaining subtle visual distinctions.

- **CUB-200-2011**  
  A fine-grained bird species dataset with 11,788 images across 200 categories. We use CUB to further validate EPIC’s capacity to extract semantically meaningful and localized prototypes in challenging, high-resolution, biologically diverse settings.

All datasets are used in accordance with their respective licenses. For experimental reproducibility, we follow the official train/test splits and preprocessing pipelines as described in their original publications.

## Datasets structure

This project expects image datasets to follow the structure required by `torchvision.datasets.ImageFolder`:

```bash
dataset_root/
├── class_0/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── class_1/
│   ├── image3.png
│   ├── image4.png
│   └── ...
└── ...
```

For more details refer to PyTorch documentation of [ImageFolder](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html#imagefolder).

# Running the code

### Configuration

This project uses [Hydra](https://hydra.cc) YAML configs to define all training and evaluation settings.

All configs can be found in the [`configs/`](./configs) folder:

- `configs/config_base.yaml ` — base config containing overlapping parameters (e.g. model name)
- `configs/config_train.yaml` — for training the prototype disentanglement module.
- `configs/config_explain.yaml` — for generating explanations on validation samples.

### Example Commands

We provide example configs for training Disentanglement Module on ResNet34 and ImageNet dataset.

Make sure you run the commands from the project root directory.

To train using the default configuration:
```bash
# Train with base values included in the config_train.yaml
python src/main.py --config-name config_train
```

To change the model and save directory (e.g., use ResNet50):
```bash
python src/main.py --config-name config_train model.name resnet50 output_path outputs/resnet50
```

To generate explanations for random images from the validation set:
```bash
python src/main.py --config-name config_explain
```

**Note**: Ensure you have trained and saved model outputs in the specified output_path before running explanations.


### Saved Files and Outputs

After training, the script saves several artifacts to the directory specified by `output_path`. These include:

- **Disentanglement Matrix Weights**
  - `<matrix_type>.pt`: The learned disentanglement matrix (e.g., `orthogonal.pt`).
  - You can load this matrix later to apply the disentanglement transformation or for further experiments.

- **Purity Scores**
  - `purity_after_<matrix_type>_train.json`: A JSON file recording the prototype purity before and after training.

- **Prototype Visualizations**
  - `Base_prototypes`: Contains prototype images for randomly selected channels saved as `channel_{c}.jpg`.
  - `Trained_prototypes`: Contains prototype images before and after training for randomly selected channels saved as `channel_{c}_prototypes.jpg`

- **Purity Plot**
  - `<matrix_type>_train.png`: A plot showing how purity evolved across training epochs.

- **Hydra Configuration Files**
  - Hydra automatically saves the exact configuration used for the run under the output directory in the `./<exp>/.hydra/` folder.
  - This includes all parameter values and overrides.

- **Explanations**
  - `Explanations`: Contains explanations for images randomly selected from validation set `{idx}.jpg`.

**Note:** The names of files and folders are automatically derived from the `matrix.type` and `output_path` you specify in your config.


# Citations

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
<h3 class="title">EPIC: Explanation of Pretrained Image Classification Networks via Prototypes</h3>
    <pre><code>@misc{borycki2025epicexplanationpretrainedimage,
      title={EPIC: Explanation of Pretrained Image Classification Networks via Prototype},
      author={Piotr Borycki and Magdalena Trędowicz and Szymon Janusz and Jacek Tabor and Przemysław Spurek and Arkadiusz Lewicki and Łukasz Struski},
      year={2025},
      eprint={2505.12897},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.12897},
}
</code></pre>
</section>
