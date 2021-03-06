<div align="center">
    <h1>Formula Image Latex Recognition</h1>
    <img src="assets/logo.png" alt="logo"/>
    <br/>
    <img src="https://img.shields.io/github/stars/pstage-ocr-team6/ocr-teamcode?color=yellow" alt="Star"/>
    <img src="https://img.shields.io/github/forks/pstage-ocr-team6/ocr-teamcode?color=green" alt="Forks">
    <img src="https://img.shields.io/github/issues/pstage-ocr-team6/ocr-teamcode?color=red" alt="Issues"/>
    <img src="https://img.shields.io/github/license/pstage-ocr-team6/ocr-teamcode" alt="License"/>
</div>

---

## π Table of Contents

- [Latex Recognition Task](#-latex-recognition-task)
- [File Structure](#-file-structure)
  - [Code Folder](#code-folder)
  - [Dataset Folder](#dataset-folder)
- [Getting Started](#-getting-started)
  - [Installation](#installation)
  - [Download Dataset](#download-dataset)
  - [Dataset Setting](#dataset-setting)
  - [Create .env for wandb](#create-env-for-wandb)
  - [Config Setting](#config-setting)
- [Usage](#-usage)
  - [Train](#train)
  - [Inference](#inference)
- [Demo](#-demo)
- [References](#-references)
- [Contributors](#-contributors)
- [License](#-license)

---

## β Latex Recognition Task

<div align="center">
  <img src="assets/competition-overview.png" alt="Competition Overview"/>
</div>

μμ μΈμ(Latex Recognition)μ **μμ μ΄λ―Έμ§μμ LaTeX ν¬λ§·μ νμ€νΈλ₯Ό μΈμνλ νμ€ν¬**λ‘, λ¬Έμ μΈμ(Character Recognition)κ³Ό λ¬λ¦¬ μμ μΈμμ κ²½μ° `μ’ β μ°` λΏλ§ μλλΌ Multi-lineμ λν΄μ `μ β μλ`μ λν μμ ν¨ν΄ νμ΅λ νμνλ€λ νΉμ§μ κ°μ§λλ€.

<br/>

## π File Structure

### Code Folder

```text
ocr_teamcode/
β
βββ config/                   # train argument config file
β   βββ Attention.yaml
β   βββ SATRN.yaml
β
βββ data_tools/               # utils for dataset
β   βββ download.sh           # dataset download script
β   βββ extract_tokens.py     # extract tokens from token.txt
β   βββ make_dataset.py       # sample dataset
β   βββ parse_upstage.py      # convert JSON ground truth file to ICDAR15 format
β   βββ train_test_split.py   # split dataset into train and test dataset
β
βββ networks/                 # network, loss
β   βββ Attention.py
β   βββ SATRN.py
β   βββ loss.py
β   βββ spatial_transformation.py
β
βββ checkpoint.py             # save, load checkpoints
βββ pre_processing.py         # preprocess images with OpenCV
βββ custom_augment.py         # image augmentations
βββ transform.py
βββ dataset.py
βββ flags.py                  # parse yaml to FLAG format
βββ inference.py              # inference
βββ metrics.py                # calculate evaluation metrics
βββ scheduler.py              # learning rate scheduler
βββ train.py                  # train
βββ utils.py                  # utils for training
```

### Dataset Folder

```text
input/data/train_dataset
β
βββ images/                 # input image folder
β   βββ train_00000.jpg
β   βββ train_00001.jpg
β   βββ train_00002.jpg
β   βββ ...
|
βββ gt.txt                  # input data
βββ level.txt               # formula difficulty feature
βββ source.txt              # printed output / hand written feature
βββ tokens.txt              # vocabulary for training
```

<br/>

## β¨ Getting Started

### Installation

```shell
pip install -r requirements.txt
```

- scikit_image==0.14.1
- opencv_python==3.4.4.19
- tqdm==4.28.1
- torch==1.7.1+cu101
- torchvision==0.8.2+cu101
- scipy==1.2.0
- numpy==1.15.4
- pillow==8.2.0
- tensorboardX==1.5
- editdistance==0.5.3
- python-dotenv==0.17.1
- wandb==0.10.30
- adamp==0.3.0
- python-dotenv==0.17.1

### Download Dataset

```shell
sh filename.sh
```

### Dataset Setting

> <b>π νμ΅λ°μ΄ν°λ [Dataset Folder](#dataset-folder)μ κ°μ΄ λ£μ΄μ£ΌμΈμ!</b>

> <b>π λ¨μΌ μ»¬λΌμΌλ‘ κ΅¬μ±λ txtλ `\n`μ κΈ°μ€μΌλ‘ λ°μ΄ν°λ₯Ό κ΅¬λΆνλ©°, 2κ° μ΄μμ μ»¬λΌμΌλ‘ κ΅¬μ±λ txtλ `\t`λ‘ μ»¬λΌμ, `\n`μΌλ‘ λ°μ΄ν°λ₯Ό κ΅¬λΆν©λλ€.</b>

νμ΅λ°μ΄ν°λ `tokens.txt`, `gt.txt`, `level.txt`, `source.txt` μ΄ 4κ°μ νμΌκ³Ό μ΄λ―Έμ§ ν΄λλ‘ κ΅¬μ±λμ΄ μμ΅λλ€.

μ΄ μ€ `tokens.txt`μ `gt.txt`λ **λͺ¨λΈ νμ΅**μ κΌ­ νμν μλ ₯ νμΌμ΄λ©°, `level.txt`, `source.txt`λ μ΄λ―Έμ§μ λν λ©ν λ°μ΄ν°λ‘ **λ°μ΄ν°μ λΆλ¦¬**μμ μ¬μ©ν©λλ€.

- `tokens.txt`λ **νμ΅μ μ¬μ©λλ vocabulary νμΌ**λ‘μ λͺ¨λΈ νμ΅μ νμν tokenλ€μ΄ μ μλμ΄ μμ΅λλ€.

  ```text
  O
  \prod
  \downarrow
  ...
  ```

- `gt.txt`λ **μ€μ  νμ΅μ μ¬μ©νλ νμΌ**λ‘ μ΄λ―Έμ§ κ²½λ‘, LaTexλ‘ λ Ground Truthλ‘ κ° μ»¬λΌμ΄ κ΅¬μ±λμ΄ μμ΅λλ€.

  ```text
  train_00000.jpg	4 \times 7 = 2 8
  train_00001.jpg	a ^ { x } > q
  train_00002.jpg	8 \times 9
  ...
  ```

- `level.txt`λ **μμμ λμ΄λ μ λ³΄ νμΌ**λ‘ κ° μ»¬λΌμ κ²½λ‘μ λμ΄λλ‘ κ΅¬μ±λμ΄ μμ΅λλ€. κ° μ«μλ 1(μ΄λ±), 2(μ€λ±), 3(κ³ λ±), 4(λν), 5(λν μ΄μ)μ μλ―Έν©λλ€.

  ```text
  train_00000.jpg	1
  train_00001.jpg	2
  train_00002.jpg	2
  ...
  ```

- `source.txt`λ μ΄λ―Έμ§μ μΆλ ₯ νν μ λ³΄ νμΌλ‘, μ»¬λΌμ κ²½λ‘μ μμ€λ‘ κ΅¬μ±λμ΄ μμ΅λλ€. κ° μ«μλ 0(νλ¦°νΈ μΆλ ₯λ¬Ό), 1(μκΈμ¨)λ₯Ό λ»ν©λλ€.

  ```text
  train_00000.jpg	1
  train_00001.jpg	0
  train_00002.jpg	0
  ```

### Create .env for wandb

wandb loggingμ μ¬μ© μ wandbμ λκ²¨μ£Όμ΄μΌ νλ μΈμλ₯Ό `.env` νμΌμ μ μν©λλ€.

```
PROJECT="[wandb project name]"
ENTITY="[wandb nickname]"
```

### Config Setting

νμ΅ μ μ¬μ©νλ config νμΌμ `yaml`νμΌλ‘ νμ΅ λͺ©νμ λ°λΌ λ€μκ³Ό κ°μ΄ μ€μ ν΄μ£ΌμΈμ.

```yaml
network: SATRN
input_size: # resize image
  height: 48
  width: 192
SATRN:
  encoder:
    hidden_dim: 300
    filter_dim: 1200
    layer_num: 6
    head_num: 8

    shallower_cnn: True # shallow CNN
    adaptive_gate: True # A2DPE
    conv_ff: True # locality-aware feedforward
    separable_ff: True # only if conv_ff is True
  decoder:
    src_dim: 300
    hidden_dim: 300
    filter_dim: 1200
    layer_num: 3
    head_num: 8

checkpoint: "" # load checkpoint
prefix: "./log/satrn" # log folder name

data:
  train: # train dataset file path
    - "/opt/ml/input/data/train_dataset/gt.txt"
  test: # validation dataset file path
    -
  token_paths: # token file path
    - "/opt/ml/input/data/train_dataset/tokens.txt" # 241 tokens
  dataset_proportions: # proportion of data to take from train (not test)
    - 1.0
  random_split: True # if True, random split from train files
  test_proportions: 0.2 # only if random_split is True, create validation set
  crop: True # center crop image
  rgb: 1 # 3 for color, 1 for greyscale

batch_size: 16
num_workers: 8
num_epochs: 200
print_epochs: 1 # print interval
dropout_rate: 0.1
teacher_forcing_ratio: 0.5 # teacher forcing ratio
teacher_forcing_damp: 5e-3 # teacher forcing decay (0 to turn off)
max_grad_norm: 2.0 # gradient clipping
seed: 1234
optimizer:
  optimizer: AdamP
  lr: 5e-4
  weight_decay: 1e-4
  selective_weight_decay: True # no decay in norm and bias
  is_cycle: True # cyclic learning rate scheduler
label_smoothing: 0.2 # label smoothing factor (0 to off)

patience: 30 # stop train after waiting (-1 for off)
save_best_only: True # save best model only

fp16: True # mixed precision

wandb:
  wandb: True # wandb logging
  run_name: "sample_run" # wandb project run name
```

<br/>

## β© Usage

### Train

```shell
python train.py [--config_file]
```

- `--config_file`: config νμΌ κ²½λ‘

### Inference

```shell
python inference.py [--checkpoint] [--max_sequence] [--batch_size] [--file_path] [--output_dir]
```

- `--checkpoint`: checkpoint νμΌ κ²½λ‘
- `--max_sequence`: inference μ μ΅λ μνμ€ κΈΈμ΄
- `--batch_size`: λ°°μΉ ν¬κΈ°
- `--file_path`: test dataset κ²½λ‘
- `--output_dir`: inference κ²°κ³Ό μ μ₯ λλ ν λ¦¬

<br/>

## π Demo

<div align="center">
<img src="assets/demo.png" alt="demo">
</div>
<br/>

## π References

- <i>On Recognizing Texts of Arbitrary Shapes with 2D Self-Attention, Lee et al., 2019</i>
- <i>Bag of Tricks for Image Classiτcation with Convolutional Neural Networks, He et al., 2018</i>
- <i>Averaging Weights Leads to Wider Optima and Better Generalization, Izmailov et al., 2018</i>
- <i>CSTR: Revisiting Classification Perspective on Scene Text Recognition, Cai et al., 2021</i>
- <i>Improvement of End-to-End Offline Handwritten Mathematical Expression Recognition by Weakly Supervised
  Learning, Truong et al., 2020</i>
- <i>ELECTRA: Pre-training Text Encoders As Discriminators Rather Than Generators, Clark et al., 2020</i>
- <i>SEED: Semantics Enhanced Encoder-Decoder Framework for Scene Text Recognition, Qiao et al., 2020</i>
- <i>Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling for Scene Text Recognition,
  Fang et al., 2021</i>
- <i>Googleβs Neural Machine Translation System: Bridging the Gap between Human and Machine Translation, Wu et
  al., 2016</i>

<br/>

## π©βπ» Contributors

|                           **[κΉμ’μ](https://github.com/kjy93217)**                            |                           **[λ―Όμ§μ](https://github.com/peacecheejecake)**                            |                                                    **[λ°μν](https://github.com/CoodingPenguin)**                                                    |                              **[λ°°μλ―Ό](https://github.com/bsm8734)**                               |                           **[μ€μΈλ―Ό](https://github.com/osmosm7)**                            |                              **[μ΅μ¬ν](https://github.com/opijae)**                               |
| :--------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------: |
| [![Avatar](https://avatars.githubusercontent.com/u/39907037?v=4)](https://github.com/kjy93217) | [![Avatar](https://avatars.githubusercontent.com/u/29668380?v=4)](https://github.com/peacecheejecake) | [![Avatar](https://avatars.githubusercontent.com/u/37505775?s=460&u=44732fef53503e63d47192ce5c2de747eff5f0c6&v=4)](https://github.com/CoodingPenguin) | [![Avatar](https://avatars.githubusercontent.com/u/35002768?s=460&v=4)](https://github.com/bsm8734) | [![Avatar](https://avatars.githubusercontent.com/u/48181287?v=4)](https://github.com/osmosm7) | [![Avatar](https://avatars.githubusercontent.com/u/26226101?s=460&v=4)](https://github.com/opijae) |


<br/>

## β License

Distributed under the MIT License. See `LICENSE` for more information.
