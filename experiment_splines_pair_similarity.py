import hashlib
import pickle
from functools import lru_cache

import click
import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc
from scipy.interpolate import make_interp_spline
from sklearn.ensemble import RandomForestClassifier

from lib.data import (TDataset, get_dataset)
from lib.explainers import TExplainer, get_explainer

alpha = 0.05
beta_min = 0.0
beta_max = 4.0
n_prototypes = 20
n_prototypes_class = 10
n_samples = 20

rc("font", **{"family": "serif", "serif": ["CMU Serif"]})
rc("text", usetex=False)


def cache_study(objective_func, dataset_name, explainer_name, random_state):
    cache_key = hashlib.md5(
        f"study_{dataset_name}_{explainer_name}_{random_state}".encode()
    ).hexdigest()
    cache_file = f"cache/lines/{cache_key}.pkl"

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
def cached_objective_results(dataset_name: TDataset, explainer_name: TExplainer, random_state: int):
    x_train, y_train, x_valid, y_valid, x_test, y_test = cached_get_dataset(dataset_name, random_state)

    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(x_train, y_train)

    results = {}
    for beta in np.linspace(beta_min, beta_max, n_samples):
        explainer_raw = get_explainer(explainer_name)(model, beta=0.0)
        explainer_fi = get_explainer(explainer_name)(model, beta=beta)

        match explainer_name:
            case "APete":
                prototypes_fi = explainer_fi.prototypes_fi(x_train, alpha=alpha)
                prototypes_raw = explainer_raw.prototypes_raw(x_train, alpha=alpha)
            case "G_KM":
                prototypes_fi = explainer_fi.prototypes_fi(x_train, n_prototypes=n_prototypes_class)
                prototypes_raw = explainer_raw.prototypes_raw(x_train, n_prototypes=n_prototypes_class)
            case "SM_A":
                prototypes_fi = explainer_fi.prototypes_fi(x_train, n_prototypes=n_prototypes)
                prototypes_raw = explainer_raw.prototypes_raw(x_train, n_prototypes=n_prototypes)

        mean_fi, len_fi, mean_raw, len_raw = [], [], [], []

        for idx, x in x_valid.iloc[:200].iterrows():
            proto_cls, proto_idx = explainer_fi.get_nearest_prototype(x_valid.iloc[[idx]], prototypes_fi)
            similar_parts = explainer_fi.similar_parts(x_valid.iloc[[idx]], prototypes_fi[proto_cls].iloc[[proto_idx]])
            part = similar_parts.iloc[[-1], :-1]
            mean_fi.append(np.mean(part))
            len_fi.append(part.shape[1])

            proto_cls, proto_idx = explainer_raw.get_nearest_prototype(x_valid.iloc[[idx]], prototypes_raw)
            similar_parts = explainer_raw.similar_parts(x_valid.iloc[[idx]], prototypes_raw[proto_cls].iloc[[proto_idx]])
            part = similar_parts.iloc[[-1], :-1]
            mean_raw.append(np.mean(part))
            len_raw.append(part.shape[1])

        results[beta] = {
            "mean_fi": np.mean(mean_fi).item(),
            "len_fi": np.mean(len_fi).item(),
            "mean_raw": np.mean(mean_raw).item(),
            "len_raw": np.mean(len_raw).item()
        }

    return pd.DataFrame(results).T


@click.command()
@click.option("--random_state", "-r", type=int, default=42, help="The random state.")
def main(random_state: int) -> None:
    explainers = ["APete", "G_KM", "SM_A"]
    datasets = ["apple_quality", "australia_rain", "breast_cancer", "diabetes", "passenger_satisfaction", "wine_quality"]

    fig = plt.figure(constrained_layout=True, figsize=(14.8, 8), dpi=300)
    subfigs = fig.subfigures(len(explainers), 1, wspace=0.05, hspace=0.1)

    colors = cmc.batlowS(np.linspace(0, 1, 10))
    color_len = colors[0]
    color_fi = colors[2]
    results = []
    line_labels = []

    for expl_idx, (explainer_name, subfig) in enumerate(zip(explainers, subfigs)):
        subfig.suptitle({"SM_A": "SM-A", "G_KM": "G-KM", "APete": "A-Pete"}[explainer_name], fontsize=20)
        axs = subfig.subplots(1, len(datasets), sharey=False)

        for data_idx, (dataset_name, ax) in enumerate(zip(datasets, axs)):
            result = cache_study(cached_objective_results, dataset_name, explainer_name, random_state)

            mean_fi = result["mean_fi"]
            len_fi = result["len_fi"]
            mean_raw = result["mean_raw"]
            len_raw = result["len_raw"]

            results.append(result)
            x_grid = np.linspace(beta_min, beta_max, 100)

            spline = make_interp_spline(result.index, mean_fi)(x_grid)
            line_fi, = ax.plot(x_grid, spline, label="Mean FI (FI)", color=color_fi)
            line_raw = ax.axhline(y=mean_raw.iloc[0], xmin=beta_min, xmax=beta_max, color=color_fi, linestyle="--",
                                  label="Mean FI (raw)")
            ax.add_line(line_raw)

            twin_ax = ax.twinx()
            spline = make_interp_spline(result.index, len_fi)(x_grid)
            line_len_fi, = twin_ax.plot(x_grid, spline, label="Length (FI)", color=color_len)
            line_len_raw = twin_ax.axhline(y=len_raw.iloc[0], xmin=beta_min, xmax=beta_max, color=color_len,
                                           linestyle="--", label="Length (raw)")
            twin_ax.add_line(line_len_raw)

            ax_ticks = np.linspace(*ax.get_ylim(), 5)[1:-1]
            ax.set_yticks(ax_ticks, [f"{t:.2f}" for t in ax_ticks], color=color_fi)

            twin_ax_ticks = np.linspace(*twin_ax.get_ylim(), 6)[1:-1]
            twin_ax.set_yticks(twin_ax_ticks, [f"{t:.2f}" for t in twin_ax_ticks], color=color_len)

            if data_idx == 0:
                if expl_idx == 1:
                    ax.set_ylabel("Mean feature importance", color=color_fi, fontsize=15)
                else:
                    ax.set_ylabel(" ", fontsize=15)

            if data_idx == 5:
                if expl_idx == 1:
                    twin_ax.set_ylabel("Length of features", color=color_len, fontsize=15)
                else:
                    twin_ax.set_ylabel(" ", fontsize=15)

            if expl_idx == 0:
                ax.set_title(" ".join(dataset_name.split("_")).title(), fontsize=15)
            else:
                ax.set_title(" ", fontsize=15)

            if expl_idx == 0 and data_idx == 0:
                line_labels.extend([line_fi, line_raw, line_len_fi, line_len_raw])

    fig.supxlabel("$\\beta$", fontsize=15)
    fig.legend(handles=line_labels, title="Lines", ncols=2, bbox_to_anchor=(1.0, 0.73),
               fontsize=12).get_title().set_fontsize(12)
    plt.savefig("results_pair_similarity.png")


if __name__ == "__main__":
    main()
