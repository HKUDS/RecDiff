# Diffusion Model for Social Recommendation (RecDiff)
This is the PyTorch-based implementation for RecDiff model proposed in this paper:
>Diffusion Model for Social Recommendation
![model](./framework_00.png)
## Abstract
Social recommendation has emerged as a powerful approach to enhance personalized recommendations by leveraging the social connections among users, such as following and friend relations observed in online social platforms. The fundamental assumption of social recommendation is that socially-connected users exhibit homophily in their preference patterns. This means that users connected by social ties tend to have similar tastes in user-item activities, such as rating and purchasing. However, this assumption is not always valid due to the presence of irrelevant and false social ties, which can contaminate user embeddings and adversely affect recommendation accuracy. To address this challenge, we propose a novel diffusion-based social denoising framework for recommendation (RecDiff). Our approach utilizes a simple yet effective hidden-space diffusion paradigm to alleivate the noisy effect in the compressed and dense representation space. By performing multi-step noise diffusion and removal, RecDiff possesses a robust ability to identify and eliminate noise from the encoded user representations, even when the noise levels vary. The diffusion module is optimized in a downstream task-aware manner, thereby maximizing its ability to enhance the recommendation process. We conducted extensive experiments to evaluate the efficacy of our framework, and the results demonstrate its superiority in terms of recommendation accuracy, training efficiency, and denoising effectiveness.
## Environment
- python=3.8
- torch=1.12.1
- numpy=1.23.1
- scipy=1.9.1
- dgl=1.0.2+cu113
## Datasets
Our experiments are conducted on three benchmark datasets collected from Ciao, Epinions and Yelp online platforms. In those sites, social connections can be established among users in addition to their observed implicit feedback (e.g., rating, click) over different items.

| Dataset  | # Users | # Items | # Interactions | # Social Ties |
| :------: | :-----: |:-------:|:--------------:|:-------------:|
|   Ciao   |  1,925  | 1,5053  |     23,223     |    65,084     |
| Epinions | 14,680  | 233,261 |    447,312     |    632,144    |
|   Yelp   |  99,262 | 105,142 |    672,513     |   1,298,522   |
## Usage

Please unzip the datasets first. Also you need to create the `History/`+'dataset_name (e.g,ciao)' and the `Models/`+ 'dataset_name (e.g,ciao)' directories. The command lines to train SDR on the three datasets are as below. The hyperparameters in the commands are set as default.

- Ciao

  ```shell
  bash scripts/run_ciao.sh
  ```

- Epinions

  ```shell
  bash scripts/run_epinions.sh
  ```

- Yelp

  ```shell
  bash scripts/run_yelp.sh
  ```
