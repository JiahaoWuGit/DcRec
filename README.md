# DcRec-Pytorch
This is the pytorch implementation of our CIKM 2022 short paper:

>CIKM 2022. Jiahao Wu, Wenqi Fan, Jingfan Chen, Shengcai Liu, Qing Li, Ke Tang. Disentangled Contrastive Learning for Social Recommendation.  [Paper in arXiv](https://arxiv.org/abs/2208.08723).

# Introduction
In this work, we propose a novel disentangled contrastive learning framework for social Recommendations to model heterogeneous behavior patterns of users in item domain and social domain.

# Environment 
`pip install -r requirements.txt`

# Dataset
 We conduct experiments on two datasets: Ciao and Douban.

# Command
`cd code && python main.py`


*NOTE*:
1. The setting of hyperparameters could be found in the file of *code/main.py*.
2. The optimal hyperparameters could refer to the logs in fold of  *logs of optimal hyperparameters*.
3. If you find our work helpful, please cite: [Disentangled Contrastive Learning for Social Recommendation](https://dl.acm.org/doi/abs/10.1145/3511808.3557583)
```
@inproceedings{wu2022DcRec,
  author = {Wu, Jiahao and Fan, Wenqi and Chen, Jingfan and Liu, Shengcai and Li, Qing and Tang, Ke},
  title = {Disentangled Contrastive Learning for Social Recommendation},
  year = {2022},
  publisher = {ACM},
  booktitle = {Proc. of CIKM'2022}
}
```

# Acknowledgements
Thanks to the authors of [LightGCN](https://arxiv.org/abs/2002.02126) since our implementation is partially based on their [pytorch implementation](https://github.com/gusye1234/LightGCN-PyTorch/tree/master).

