# This part looks alike this: identifying important parts of explained instances and prototypes

[![Preprint - arXiv](https://img.shields.io/badge/Preprint-arXiv-red)](https://arxiv.org/abs/2505.05597)
[![Preprint - arXiv](https://img.shields.io/badge/Accepted_to-3rd_World_XAI_Conference-blue)](https://xaiworldconference.com/2025/)

[**Jacek Karolczak**](https://github.com/jkarolczak),
[**Jerzy Stefanowski**](https://www.cs.put.poznan.pl/jstefanowski/) <br>
Poznan Universtiy of Technology

## Abstract

Although prototype-based explanations provide a human-understandable way of representing model predictions they often
fail to direct user attention to the most relevant features. We propose a novel approach to identify the most
informative features within prototypes, termed alike parts. Using feature importance scores derived from an agnostic
explanation method, it emphasizes the most relevant overlapping features between an instance and its nearest prototype.
Furthermore, the feature importance score is incorporated into the objective function of the prototype selection
algorithms to promote global prototypes diversity. Through experiments on six benchmark datasets, we demonstrate that
the proposed approach improves user comprehension while maintaining or even increasing predictive accuracy.

## High-level overview

This project introduces an enhanced approach for prototype-based explanations of black-box machine learning models (
especially Random Forests). Our main contributions:

- **Alike Parts**: Highlight the most important shared features between a prototype and the explained instance, based on
  SHAP-derived feature importance.
- **FI-Informed Prototype Selection**: Extend existing prototype selection methods (e.g., A-PETE, SM-A, G-KM) to
  incorporate feature importance directly into the optimization objective, promoting interpretability and diversity.

## Reproducibility

All results presented in the paper were produced using scripts shared in this repository (`experiments*.py`).
To reproduce results, you just have to clone the repository, install requirements:

```shell
pip install -r requirements.txt
```

Then you will be able to run scripts, for intance:

```shell
python main.py <DATASET> <EXPLAINER> [OPTIONS]
```

## Cite us!

> [!NOTE]  
> The BibTex entry will be updated, once the article is published in The 3rd World XAI Conference proceedings.

```bib
@misc{karolczak2025looksalikethisidentifying,
      title={This part looks alike this: identifying important parts of explained instances and prototypes}, 
      author={Jacek Karolczak and Jerzy Stefanowski},
      year={2025},
      eprint={2505.05597},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.05597}, 
}
```

## Acknowledgments

This research was funded in part by National Science Centre, Poland OPUS grant no. 2023/51/B/ST6/00545 and in part by
PUT SBAD 0311/SBAD/0752 grant.
