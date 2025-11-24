# Alike Parts: A Feature-Informed Approach to Local and Global Prototype Explanations

[![Preprint - arXiv](https://img.shields.io/badge/Accepted_to-3rd_World_XAI_Conference-blue)](https://xaiworldconference.com/2025/)

[**Jacek Karolczak**](https://github.com/jkarolczak),
[**Jerzy Stefanowski**](https://www.cs.put.poznan.pl/jstefanowski/) <br>
Poznan Universtiy of Technology

## Abstract

Prototype-based explanations offer an intuitive, example-based approach to support the interpretability of machine
learning black box classifiers but often lack feature-level granularity. We introduce a framework that integrates
feature importance at two levels to address this gap. First, for local explanations, we propose alike parts:
a method that uses feature importance scores to highlight the most relevant, shared feature subsets between a classified
instance and its nearest prototype, guiding user attention. Second, we augment the global prototype selection objective
function with a feature importance term to actively promote diversity in the feature attributions of the selected
prototypes. Experiments on six benchmark datasets show that this augmented selection process maintains or, in some
cases, increases the prediction fidelity of the surrogate model, suggesting that feature diversity does not compromise
model fidelity.

## High-level overview

This project introduces an enhanced approach for prototype-based explanations of black-box machine learning models (
especially Random Forests). Our main contributions:

- **Alike Parts**: Highlight the most important shared features between a prototype and the explained instance, based on
  SHAP-derived feature importance.
- **FI-Informed Prototype Selection**: Extend existing prototype selection methods (e.g., A-PETE, SM-A, G-KM) to
  incorporate feature importance directly into the optimization objective, promoting interpretability and diversity.

## Reproducibility

> [!NOTE]
> Are you here because of the xAI 2025 conference publication? If so, checkout to the
> [`xai2025`](https://github.com/jkarolczak/alike-parts/tree/xai2025) branch for the exact code used to produce the
> results in the paper.

All results presented in the paper were produced using scripts shared in this repository (`experiment_*.py`).
To reproduce results, you just have to clone the repository, install requirements:

```shell
pip install -r requirements.txt
```

Then you will be able to run scripts, for instance:

```shell
python main.py <DATASET> <EXPLAINER> [OPTIONS]
```

## Cite us!

```bib
@inproceedings{karolczak2025,
	author="{Karolczak, Jacek and Stefanowski, Jerzy}",
	title="{This part looks alike this: identifying important parts of explained instances and prototypes}",
	year="2025",
	booktitle="{Joint Proceedings of the xAI 2025 Late-breaking Work, Demos and Doctoral Consortium co-located with the 3rd World Conference on eXplainable Artificial Intelligence (xAI 2025), Istanbul, Turkey, July 9-11, 2025}",
	publisher="{CEUR}",
	series="{CEUR Workshop Proceedings}",
	pages="33--40",
	url="https://ceur-ws.org/Vol-4017/paper_05.pdf",
}
```

## Acknowledgments

This research was funded in part by National Science Centre, Poland OPUS grant no. 2023/51/B/ST6/00545 and in part by
PUT SBAD 0311/SBAD/0752 grant.
