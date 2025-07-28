# Fairer Benchmark of Group-Contribution and Machine-Learning Property Models

## Overview

This project provides a fair and reproducible framework for benchmarking classical group-contribution (GC) models and modern machine learning (ML) approaches for property prediction. It addresses the common criticism of GC models by ensuring proper validation and comparison with ML models across a wide range of properties.

The main innovation here is the construction of a hybrid-data splitting strategy that accounts for the presence of groups. the strategy first identify the smallest subset of data for which all available groups are present. Then it fills up untill a specific given split ratio using data samples either: randomly or using Butina splitting given a pecific cut-off. This way, one avoids loosing out on groups when developping models. that uses groups as features. 

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Features
- Fair comparison of GC and ML models
- Automated data splitting and validation
- Support for multiple properties and model types
- Reproducible experiments and results

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/arnaou/fair_gc.git
   cd fair_gc
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Example: Run a trained MLP for a property (e.g., Omega) with a specific model:
```bash
python evaluate_mlp.py --property Omega
```

Example: run hyperparameter optimization for a specific property:
```bash
python scripts/optuna_mlp.py  --property Tc --config_file mlp_hyperopt_config.yaml --model mlp --n_trials 27 --path_2_data data/ --path_2_result results/ --path_2_model models/ --seed 42 --n_jobs 3 --split_type fair_min
```  

See the `scripts/` and `src/` folders for more usage examples and available scripts.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the [GPLv3 License](LICENSE).

## Acknowledgements
- Inspired by the need for fair benchmarking in property prediction research.
- Built with Python and open-source libraries (see `requirements.txt`).
- Thanks to all contributors and the open-source community.

## Cite this work
If you would like to cite this work, please reference our paper as:

```bibtex
@article{AOUICHAOUI2025,
title = {Fairer benchmark of group contribution and machine learning models for property prediction: A new data splitting strategy},
journal = {Computers & Chemical Engineering},
volume = {202},
pages = {109271},
year = {2025},
issn = {0098-1354},
doi = {https://doi.org/10.1016/j.compchemeng.2025.109271},
url = {https://www.sciencedirect.com/science/article/pii/S009813542500273X},
author = {Adem R.N. Aouichaoui and Jingkang Liang and Jens Abildskov and Gürkan Sin},
keywords = {Graph Neural Networks, Ciritical Point Properties, Molecular Properties, Group Contribution, Machine Learning, Uncertainty Estimation},
abstract = {Accurate prediction of thermophysical properties is important in chemical engineering, where group-contribution models (GCM) have been used extensively. The norm when developing GCMs is to use all available data for parameter estimation, preventing a fair comparison with machine learning (ML) methods that require separate training, validation, and testing data. In this study, we first highlight the detrimental effect of missing groups resulting from using conventional split methods (random and cluster-based) in the development of GCM and ML models using groups as features. This was illustrated by developing property models for the critical point properties (critical temperature, critical pressure and critical volume) as well as the acentric factor. To alleviate this, we propose a novel hybrid splitting algorithm that first ensures that all available groups are represented using the smallest subset of compounds possible and then supplements the subset with molecules based on the Butina clustering of the remaining compounds. The methods show performance close to the "optimal" possible result produced from using all data for the model calibration, and provide a more fair basis for comparing GCM with ML-based methods. We further benchmark the GCM with seven ML techniques (random forest, decision tree, gradient boosting, extreme gradient boosting, Gaussian processes and support vector machines as well as deep neural networks) using groups as features and three graph neural network models (attentiveFP, MEGNet and GroupGAT). The results show that GroupGAT consistently outperforms other methods on the external test dataset, achieving lower errors than both traditional and ML-enhanced GC methods.}
}
```
If you use the GroupGAT model please cite the original works
```bibtex
@article{AOUICHAOUI2023a,
author = {Aouichaoui, Adem R. N. and Fan, Fan and Mansouri, Seyed Soheil and Abildskov, Jens and Sin, G{\"u}rkan},
title = {Combining Group-Contribution Concept and Graph Neural Networks Toward Interpretable Molecular Property Models},
journal = {Journal of Chemical Information and Modeling},
volume = {63},
number = {3},
pages = {725-744},
year = {2023},
doi = {10.1021/acs.jcim.2c01091},
note ={PMID: 36716461},
URL = { https://doi.org/10.1021/acs.jcim.2c01091},
eprint = { https://doi.org/10.1021/acs.jcim.2c01091},
abstract = { Quantitative structure–property relationships (QSPRs) are important tools to facilitate and accelerate the discovery of compounds with desired properties. While many QSPRs have been developed, they are associated with various shortcomings such as a lack of generalizability and modest accuracy. Albeit various machine-learning and deep-learning techniques have been integrated into such models, another shortcoming has emerged in the form of a lack of transparency and interpretability of such models. In this work, two interpretable graph neural network (GNN) models (attentive group-contribution (AGC) and group-contribution-based graph attention (GroupGAT)) are developed by integrating fundamentals using the concept of group contributions (GC). The interpretability consists of highlighting the substructure with the highest attention weights in the latent representation of the molecules using the attention mechanism. The proposed models showcased better performance compared to classical group-contribution models, as well as against various other GNN models describing the aqueous solubility, melting point, and enthalpies of formation, combustion, and fusion of organic compounds. The insights provided are consistent with insights obtained from the semiempirical GC models confirming that the proposed framework allows highlighting the important substructures of the molecules for a specific property. }
}
```
For benchmark results of GroupGAT consider citing
```bibtex
@article{AOUICHAOUI2023b,
title = {Application of interpretable group-embedded graph neural networks for pure compound properties},
journal = {Computers & Chemical Engineering},
volume = {176},
pages = {108291},
year = {2023},
issn = {0098-1354},
doi = {https://doi.org/10.1016/j.compchemeng.2023.108291},
url = {https://www.sciencedirect.com/science/article/pii/S0098135423001618},
author = {Adem R.N. Aouichaoui and Fan Fan and Jens Abildskov and Gürkan Sin},
keywords = {Deep-learning, Graph neural networks, Group-contribution models, Thermophysical properties, Interpretability, Pure compound properties},
abstract = {The ability to evaluate pure compound properties of various molecular species is an important prerequisite for process simulation in general and in particular for computer-aided molecular design (CAMD). Current techniques rely on group-contribution (GC) methods, which suffer from many drawbacks mainly the absence of contributions for specific groups. To overcome this challenge, in this work, we extended the range of interpretable graph neural network (GNN) models for describing a wide range of pure component properties. The new model library contains 30 different properties ranging from thermophysical, safety-related, and environmental properties. All of these have been modeled with a suitable level of accuracy for compound screening purposes compared to current GC models used within CAMD applications. Moreover, the developed models have been subjected to a series of sanity checks using logical and thermodynamic constraints. Results show the importance of evaluating the model across a range of properties to establish their thermodynamic consistency.}
}
```
