import hashlib
import pickle
from functools import lru_cache

import click
import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from sklearn.ensemble import RandomForestClassifier

from lib.data import (get_dataset,
                      TDataset)
from lib.explainers import (get_explainer,
                            TExplainer)

params = {
    "APete": {
        "apple_quality": {"beta": 2.0, "alpha": 0.025},
        "australia_rain": {"beta": 2.5, "alpha": 0.025},
        "breast_cancer": {"beta": 3.0, "alpha": 0.05},
        "diabetes": {"beta": 2.0, "alpha": 0.025},
        "passenger_satisfaction": {"beta": 0.5, "alpha": 0.05},
        "wine_quality": {"beta": 2.5, "alpha": 0.075},
    },
    "G_KM": {
        "apple_quality": {"beta": 1.5, "n_prototypes": 8},
        "australia_rain": {"beta": 0.5, "n_prototypes": 8},
        "breast_cancer": {"beta": 4.0, "n_prototypes": 8},
        "diabetes": {"beta": 0.5, "n_prototypes": 8},
        "passenger_satisfaction": {"beta": 4.0, "n_prototypes": 5},
        "wine_quality": {"beta": 0.5, "n_prototypes": 5},
    },
    "SM_A": {
        "apple_quality": {"beta": 3.0, "n_prototypes": 10},
        "australia_rain": {"beta": 1.0, "n_prototypes": 10},
        "breast_cancer": {"beta": 1.5, "n_prototypes": 15},
        "diabetes": {"beta": 1.0, "n_prototypes": 10},
        "passenger_satisfaction": {"beta": 0.5, "n_prototypes": 20},
        "wine_quality": {"beta": 2.0, "n_prototypes": 10},
    }
}

rc("font", **{"family": "serif", "serif": ["CMU Serif"]})
rc("text", usetex=False)


def cache_study(objective_func, dataset_name, explainer_name, random_state):
    cache_key = hashlib.md5(
        f"study_{dataset_name}_{explainer_name}_{random_state}".encode()
    ).hexdigest()
    cache_file = f"cache/activating/{cache_key}.pkl"

    try:
        with open(cache_file, "rb") as f:
            study = pickle.load(f)
        print(f"Loaded cached study for {dataset_name}, {explainer_name}")
    except FileNotFoundError:
        print(f"Running study for {dataset_name}, {explainer_name}...")
        study = objective_func(dataset_name, explainer_name, random_state)
        with open(cache_file, "wb") as f:
            pickle.dump(study, f)
    return study


@lru_cache(maxsize=None)
def cached_get_dataset(dataset_name: TDataset, random_state: int):
    return get_dataset(dataset_name, random_state=random_state)


@lru_cache(maxsize=None)
def cached_objective_results(dataset_name: TDataset, explainer_name: TExplainer, random_state: int
                             ) -> dict[str, dict[str, float]]:
    x_train, y_train, x_valid, y_valid, x_test, y_test = cached_get_dataset(dataset_name, random_state)

    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(x_train, y_train)

    explainer_raw = get_explainer(explainer_name)(model, beta=0.0)
    explainer_fi = get_explainer(explainer_name)(model, beta=params[explainer_name][dataset_name]["beta"])

    match explainer_name:
        case "APete":
            prototypes_fi = explainer_fi.prototypes_fi(x_train, alpha=params[explainer_name][dataset_name]["alpha"])
            prototypes_raw = explainer_raw.prototypes_raw(x_train, alpha=params[explainer_name][dataset_name]["alpha"])
        case "G_KM":
            prototypes_fi = explainer_fi.prototypes_fi(x_train,
                                                       n_prototypes=params[explainer_name][dataset_name]["n_prototypes"])
            prototypes_raw = explainer_raw.prototypes_raw(x_train,
                                                          n_prototypes=params[explainer_name][dataset_name]["n_prototypes"])
        case "SM_A":
            prototypes_fi = explainer_fi.prototypes_fi(x_train,
                                                       n_prototypes=params[explainer_name][dataset_name]["n_prototypes"])
            prototypes_raw = explainer_raw.prototypes_raw(x_train,
                                                          n_prototypes=params[explainer_name][dataset_name]["n_prototypes"])

    results = {
        "fi": {col: 0 for col in x_valid.columns},
        "raw": {col: 0 for col in x_valid.columns}
    }

    for idx, x in x_valid.iterrows():
        proto_cls, proto_idx = explainer_fi.get_nearest_prototype(x_valid.iloc[[idx]], prototypes_fi)
        similar_parts = explainer_fi.similar_parts(x_valid.iloc[[idx]], prototypes_fi[proto_cls].iloc[[proto_idx]])
        for col in similar_parts.columns:
            if col != "obj":
                results["fi"][col] += 1

        proto_cls, proto_idx = explainer_raw.get_nearest_prototype(x_valid.iloc[[idx]], prototypes_raw)
        similar_parts = explainer_raw.similar_parts(x_valid.iloc[[idx]], prototypes_raw[proto_cls].iloc[[proto_idx]])
        for col in similar_parts.columns:
            if col != "obj":
                results["raw"][col] += 1

    for col in results["fi"]:
        results["fi"][col] /= x_valid.shape[0]
        results["raw"][col] /= x_valid.shape[0]

    return results


@click.command()
@click.option("--random_state", "-r", type=int, default=42, help="The random state.")
def main(random_state: int) -> None:
    explainers = ["APete", "G_KM", "SM_A"]
    datasets = ["apple_quality", "australia_rain", "breast_cancer", "diabetes", "passenger_satisfaction", "wine_quality"]

    fig = plt.figure(constrained_layout=True, figsize=(14.8, 6.5), dpi=300)
    subfigs = fig.subfigures(len(explainers), 1, wspace=0.01, hspace=0.01)
    fig.supylabel("Frequency of occurrence in masks", fontsize=15)

    colors = cmc.batlowS(np.linspace(0, 1, 10))
    color_raw = colors[0]
    color_fi = colors[1]

    for expl_idx, (explainer_name, subfig) in enumerate(zip(explainers, subfigs)):
        subfig.suptitle({"SM_A": "SM-A", "G_KM": "G-KM", "APete": "A-Pete"}[explainer_name], fontsize=20)
        axs = subfig.subplots(1, len(datasets), sharey=False)

        for data_idx, (dataset_name, ax) in enumerate(zip(datasets, axs)):
            result = cache_study(cached_objective_results, dataset_name, explainer_name, random_state)
            ticks = [2 * i for i in range(len(result["fi"]))]
            fi = ax.bar([t + 0.1 for t in ticks], list(result["fi"].values()), color=color_fi, label="FI")
            raw = ax.bar([t + 1 - 0.1 for t in ticks], list(result["raw"].values()), color=color_raw, label="Raw")

            ax.set_xticks([i + 0.5 for i in ticks], [""] * len(ticks))
            ticks = np.linspace(0, ax.get_ylim()[1], 5)[1:-1]
            ax.set_yticks(ticks, [f"{t:.2f}" for t in ticks])
            ax.tick_params(axis="x", labelsize=10)
            ax.tick_params(axis="y", labelsize=10)

            if expl_idx == 0:
                ax.set_title(" ".join(dataset_name.split("_")).title(), fontsize=15)
            else:
                ax.set_title(" ", fontsize=15)

    fig.legend(handles=[fi, raw], title="Strategy", bbox_to_anchor=(1.0, 0.70), ncols=2, fontsize=15
               ).get_title().set_fontsize(15)

    fig.supxlabel("Feature Index")
    plt.savefig("results_activating_features.png")


if __name__ == "__main__":
    main()
