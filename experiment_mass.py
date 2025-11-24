from pprint import pprint

import click
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import wandb
from lib.data import TDataset, get_dataset
from lib.explainers import get_explainer, TExplainer


def run(x_train: pd.DataFrame, y_train: pd.Series,
        x_valid: pd.DataFrame, y_valid: pd.Series,
        x_test: pd.DataFrame, y_test: pd.Series,
        model: RandomForestClassifier, beta: float,
        sp_kwargs: dict, p_kwargs: dict,
        dataset_name: str, explainer_name: TExplainer,
        random_state: int, log: bool) -> float:
    if log:
        wandb.init(
            project="Important parts of prototypes",
            entity="jacek-karolczak",
            name=f"{dataset_name}-{explainer_name}-beta={beta}",
            config={
                "dataset": dataset_name,
                "explainer": explainer_name,
                "beta": beta,
                "random_state": random_state,
                **{"p_kwargs/" + k: v for k, v in p_kwargs.items()},
                **{"sp_kwargs/" + k: v for k, v in sp_kwargs.items()}
            }
        )

    explainer = get_explainer(explainer_name)(model, beta=beta, **sp_kwargs)

    prototypes_raw = explainer.prototypes_raw(x_train, **p_kwargs)
    prototypes_fi = explainer.prototypes_fi(x_train, **p_kwargs)

    y_train_hat = model.predict(x_train)
    y_valid_hat = model.predict(x_valid)
    y_test_hat = model.predict(x_test)

    y_train_hat_proto_raw = explainer.predict_with_prototypes(x_train, prototypes_raw)
    y_valid_hat_proto_raw = explainer.predict_with_prototypes(x_valid, prototypes_raw)
    y_test_hat_proto_raw = explainer.predict_with_prototypes(x_test, prototypes_raw)

    y_train_hat_proto_fi = explainer.predict_with_prototypes(x_train, prototypes_fi, fi=True)
    y_valid_hat_proto_fi = explainer.predict_with_prototypes(x_valid, prototypes_fi, fi=True)
    y_test_hat_proto_fi = explainer.predict_with_prototypes(x_test, prototypes_fi, fi=True)

    statistics = {
        "total_n_prototypes/raw": sum([len(c) for c in prototypes_raw.values()]),
        "total_n_prototypes/fi": sum([len(c) for c in prototypes_fi.values()]),

        # accuracy - random forest
        "score/random_forest/accuracy/train": accuracy_score(y_train, y_train_hat),
        "score/random_forest/accuracy/valid": accuracy_score(y_valid, y_valid_hat),
        "score/random_forest/accuracy/test": accuracy_score(y_test, y_test_hat),

        # accuracy - raw prototypes
        "score/prototypes/accuracy/train/raw": accuracy_score(y_train, y_train_hat_proto_raw),
        "score/prototypes/accuracy/valid/raw": accuracy_score(y_valid, y_valid_hat_proto_raw),
        "score/prototypes/accuracy/test/raw": accuracy_score(y_test, y_test_hat_proto_raw),

        # accuracy - fi prototypes
        "score/prototypes/accuracy/train/fi": accuracy_score(y_train, y_train_hat_proto_fi),
        "score/prototypes/accuracy/valid/fi": accuracy_score(y_valid, y_valid_hat_proto_fi),
        "score/prototypes/accuracy/test/fi": accuracy_score(y_test, y_test_hat_proto_fi),

        # fidelity - raw prototypes
        "score/prototypes/fidelity/train/raw": accuracy_score(y_train_hat, y_train_hat_proto_raw),
        "score/prototypes/fidelity/valid/raw": accuracy_score(y_valid_hat, y_valid_hat_proto_raw),
        "score/prototypes/fidelity/test/raw": accuracy_score(y_test_hat, y_test_hat_proto_raw),

        # fidelity - fi prototypes
        "score/prototypes/fidelity/train/fi": accuracy_score(y_train_hat, y_train_hat_proto_fi),
        "score/prototypes/fidelity/valid/fi": accuracy_score(y_valid_hat, y_valid_hat_proto_fi),
        "score/prototypes/fidelity/test/fi": accuracy_score(y_test_hat, y_test_hat_proto_fi),
    }

    if log:
        wandb.log(statistics)
        wandb.finish(quiet=True)
    else:
        pprint(statistics)

    return statistics["score/prototypes/fidelity/valid/fi"]


@click.command()
@click.option("--random_state", "-r", type=int, default=42, help="The random state.")
@click.option("--log", is_flag=True, help="A flag indicating whether to log the results to wandb.")
def main(log: bool, random_state: int) -> None:
    def objective(trial: optuna.Trial, data, model, dataset_name, explainer):
        beta = trial.suggest_float("beta", 0.0, 3.0)

        p_kwargs = {
            "APete": lambda t: {"alpha": t.suggest_float("alpha", 0.01, 0.1)},
            "SM_A": lambda t: {"n_prototypes": t.suggest_int("n_prototypes", 4, 30)},
            "G_KM": lambda t: {"n_prototypes": t.suggest_int("n_prototypes", 2, 15)},
        }[explainer](trial)

        sp_kwargs = {
            "ignore_direction": trial.suggest_categorical("ignore_direction", [True, False]),
            "normalize_fi": trial.suggest_categorical("normalize_fi", [True, False]),
            "masking_strategy": trial.suggest_categorical("masking_strategy", ["mean", "knee", "sqrt", "log"]),
            "similarity_metric": trial.suggest_categorical("similarity_metric", ["dot", "l1", "l2"]),
            "fi_method": trial.suggest_categorical("fi_method", ["treeinterpreter", "TreeSHAP"])
        }

        return run(
            *data, model=model, beta=beta,
            sp_kwargs=sp_kwargs, p_kwargs=p_kwargs,
            dataset_name=dataset_name, explainer_name=explainer,
            random_state=random_state, log=log
        )

    for dataset_name in TDataset.__args__:
        data = get_dataset(name=dataset_name)
        model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=random_state)
        x_train, y_train = data[0], data[1]
        model.fit(x_train, y_train)

        for explainer in TExplainer.__args__:
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: objective(trial, data, model, dataset_name, explainer),
                n_trials=200
            )


if __name__ == "__main__":
    main()
