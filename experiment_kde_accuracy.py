import hashlib
import pickle
from functools import lru_cache

import click
import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from matplotlib import rc
from scipy.stats import gaussian_kde
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from lib.data import (TDataset, get_dataset)
from lib.explainers import (TExplainer, get_explainer)

alpha_min = 0.0
alpha_max = 0.1
beta_min = 0.0
beta_max = 4.0
n_prototypes_min = 1
n_prototypes_max = 20
n_prototypes_max_class = 20
n_trials = 20
n_jobs = 1

rc("font", **{"family": "serif", "serif": ["CMU Serif"]})
rc("text", usetex=False)


# Cache dataset loading using lru_cache
@lru_cache(maxsize=None)
def cached_get_dataset(dataset_name: TDataset, random_state: int):
    return get_dataset(dataset_name, random_state=random_state)


def cache_study(objective_func, model, dataset_name, explainer_name, random_state, x1_name, x2_name):
    cache_key = hashlib.md5(
        f"acc_{dataset_name}_{explainer_name}_{random_state}_{x1_name}_{x2_name}".encode()
    ).hexdigest()
    cache_file = f"cache/acc/{cache_key}.pkl"

    try:
        with open(cache_file, "rb") as f:
            study = pickle.load(f)
        print(f"Loaded cached study for {dataset_name}, {explainer_name}")
    except FileNotFoundError:
        print(f"Optimizing {dataset_name}, {explainer_name}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective_func(trial, model, dataset_name, explainer_name, random_state),
            n_trials=n_trials,
            n_jobs=n_jobs
        )
        with open(cache_file, "wb") as f:
            pickle.dump(study, f)
    return study


def objective_alpha(
        trial,
        model: RandomForestClassifier,
        dataset_name: TDataset,
        explainer_name: TExplainer,
        random_state: int
) -> float:
    x_train, y_train, x_valid, y_valid, x_test, y_test = cached_get_dataset(dataset_name, random_state=random_state)

    alpha = trial.suggest_float("alpha", alpha_min, alpha_max)
    beta = trial.suggest_float("beta", beta_min, beta_max)
    explainer = get_explainer(explainer_name)(model, beta=beta)
    prototypes = explainer.prototypes_fi(x_train, alpha=alpha)
    y_hat = explainer.predict_with_prototypes(x_valid, prototypes)
    recall = accuracy_score(y_valid, y_hat)
    return recall


def objective_n_prototypes(
        trial,
        model: RandomForestClassifier,
        dataset_name: TDataset,
        explainer_name: TExplainer,
        random_state: int
) -> float:
    x_train, y_train, x_valid, y_valid, x_test, y_test = cached_get_dataset(dataset_name, random_state=random_state)

    n_prototypes = trial.suggest_int("n_prototypes", n_prototypes_min, n_prototypes_max)
    beta = trial.suggest_float("beta", beta_min, beta_max)
    explainer = get_explainer(explainer_name)(model, beta=beta)
    prototypes = explainer.prototypes_fi(x_train, n_prototypes=n_prototypes)
    y_hat = explainer.predict_with_prototypes(x_valid, prototypes)
    recall = accuracy_score(y_valid, y_hat)
    return recall


@click.command()
@click.option("--random_state", "-r", type=int, default=42, help="The random state.")
def main(random_state: int) -> None:
    explainers = ["APete", "G_KM", "SM_A"]
    datasets = ["apple_quality", "australia_rain", "breast_cancer", "diabetes", "passenger_satisfaction", "wine_quality"]
    x2_name = "beta"

    fig = plt.figure(constrained_layout=True, figsize=(14.8, 6.5), dpi=300)
    fig.supylabel("$\\beta$", fontsize=15)

    subfigs = fig.subfigures(len(explainers), 1, wspace=0.05)

    df = pd.DataFrame(columns=datasets,
                      index=pd.MultiIndex.from_product([explainers, ["fi", "raw"]], names=["explainer", "method"]))

    for expl_idx, (explainer_name, subfig) in enumerate(zip(explainers, subfigs)):
        x1_name = "alpha" if explainer_name == "APete" else "n_prototypes"

        subfig.suptitle({"SM_A": "SM-A", "G_KM": "G-KM", "APete": "A-Pete"}[explainer_name], fontsize=20)
        subfig.supxlabel(x1_name if x1_name != "alpha" else "$\\alpha$", fontsize=15)

        axs = subfig.subplots(1, len(datasets), sharey=True)

        for data_idx, (dataset_name, ax) in enumerate(zip(datasets, axs)):
            x_train, y_train, x_valid, y_valid, x_test, y_test = cached_get_dataset(dataset_name, random_state=random_state)
            model = RandomForestClassifier(n_estimators=100, random_state=random_state)
            model.fit(x_train, y_train)

            global n_prototypes_max
            global n_prototypes_max_class
            if explainer_name == "SM_A":
                n_prototypes_max = n_prototypes_max_class * 2
            elif explainer_name == "G_KM":
                n_prototypes_max = n_prototypes_max_class

            objective = objective_alpha if explainer_name == "APete" else objective_n_prototypes

            study = cache_study(objective, model, dataset_name, explainer_name, random_state, x1_name, x2_name)

            x1 = [trial.params[x1_name] for trial in study.trials]
            x2 = [trial.params[x2_name] for trial in study.trials]
            y = [trial.values[0] for trial in study.trials]
            y_min, y_max = min(y), max(y)

            xy = np.vstack([x1, x2])
            kde = gaussian_kde(xy, weights=y)

            x1_grid = np.linspace(alpha_min if x1_name == "alpha" else n_prototypes_min,
                                  alpha_max if x1_name == "alpha" else n_prototypes_max, 100)
            x2_grid = np.linspace(beta_min, beta_max, 100)
            x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid)
            if explainer_name == "G_KM":
                x1_mesh *= 2
            grid_points = np.vstack([x1_mesh.ravel(), x2_mesh.ravel()])

            kde_values = kde(grid_points).reshape(100, 100)
            kde_values = (kde_values - kde_values.min()) / (kde_values.max() - kde_values.min())
            kde_values = kde_values * (y_max - y_min) + y_min

            df.loc[(explainer_name, "fi"), dataset_name] = kde_values.max()
            df.loc[(explainer_name, "raw"), dataset_name] = kde_values[0].max()

            contour = ax.contourf(x1_mesh, x2_mesh, kde_values, levels=50, cmap=cmc.batlow_r)
            cbar = plt.colorbar(contour, ax=ax, format="%.3f")
            cbar.ax.set_title("Acc.", fontsize=8)
            cbar.ax.tick_params(labelsize=8)

            ax.set_xticks(np.linspace(0, ax.get_xlim()[1], 5)[1:-1])
            ax.tick_params(axis="x", labelsize=10)
            ax.tick_params(axis="y", labelsize=10)
            if expl_idx == 0:
                ax.set_title(" ".join(dataset_name.split("_")).title())

    df = df.rename(columns=lambda c: " ".join(c.split("_")).title())
    df = df.style.format(precision=3)
    print(df.to_latex())
    plt.savefig("results_acc_all_explainers.png")


if __name__ == "__main__":
    main()
