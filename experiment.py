from pprint import pprint

import click
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import wandb
from lib.data import (TDataset,
                      get_dataset)
from lib.explainers import TExplainer, get_explainer


@click.command()
@click.argument("dataset_name", type=click.Choice(TDataset.__args__))
@click.argument("explainer_name", type=click.Choice(TExplainer.__args__))
@click.option("--beta", "-b", type=float, default=3.0, help="The beta parameter.")
@click.option("--fi_kwargs", "-fikw", type=str, default="",
              help="Additional, keyword arguments for the explainer for finding prototypes considering feature importance. "
                   "Must be in the form of key=value,key2=value2...")
@click.option("--p_kwargs", "-pkw", type=str, default="",
              help="Additional, keyword arguments for the explainer for finding prototypes. "
                   "Must be in the form of key=value,key2=value2...")
@click.option("--random_state", "-r", type=int, default=42, help="The random state.")
@click.option("--log", is_flag=True, help="A flag indicating whether to log the results to wandb.")
def main(dataset_name: TDataset, explainer_name: TExplainer, beta: float, fi_kwargs: str, p_kwargs: str, log: bool,
         random_state: int) -> None:
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_dataset(dataset_name, random_state=random_state)

    p_kwargs = dict([arg.split("=") for arg in (p_kwargs.split(",") if p_kwargs else [])])
    fi_kwargs = dict([arg.split("=") for arg in (fi_kwargs.split(",") if fi_kwargs else [])])

    if log:
        wandb.init(
            project="personalized-prototypes",
            entity="jacek-karolczak",
            config={
                "dataset": dataset_name,
                "beta": beta,
                "random_state": random_state,
                **{"p_kwargs/" + k: v for k, v in p_kwargs.items()},
                **{"fi_kwargs/" + k: v for k, v in fi_kwargs.items()}
            }
        )

    model = RandomForestClassifier(n_estimators=200, random_state=random_state)
    model.fit(x_train, y_train)

    explainer = get_explainer(explainer_name)(model, beta=beta)

    prototypes_raw = explainer.prototypes_raw(x_train, **p_kwargs)
    prototypes_fi = explainer.prototypes_fi(x_train, **fi_kwargs)

    proto_cls, proto_idx = explainer.get_nearest_prototype(x_test[:1], prototypes_fi)

    for cls in prototypes_fi:
        print(f"Class {cls}:")
        print(prototypes_fi[cls])

    similar_parts = explainer.similar_parts(x_test[:1], prototypes_fi[proto_cls].iloc[[proto_idx]])
    columns = [col for col in similar_parts.columns if col != "obj"]

    print(similar_parts)

    statistics = {
        "total_n_prototypes/raw": sum([len(c) for c in prototypes_raw.values()]),
        "total_n_prototypes/fi": sum([len(c) for c in prototypes_fi.values()]),
        "score/accuracy/train/random_forest": accuracy_score(y_train, model.predict(x_train)),
        "score/accuracy/valid/random_forest": accuracy_score(y_valid, model.predict(x_valid)),
        "score/accuracy/test/random_forest": accuracy_score(y_test, model.predict(x_test)),
        "score/accuracy/train/prototypes/raw": accuracy_score(
            y_train, explainer.predict_with_prototypes(x_train, prototypes_raw)),
        "score/accuracy/train/prototypes/fi": accuracy_score(
            y_train, explainer.predict_with_prototypes(x_train, prototypes_fi, fi=True)),
        "score/accuracy/valid/prototypes/raw": accuracy_score(
            y_valid, explainer.predict_with_prototypes(x_valid, prototypes_raw)),
        "score/accuracy/valid/prototypes/fi": accuracy_score(
            y_valid, explainer.predict_with_prototypes(x_valid, prototypes_fi, fi=True)),
        "score/accuracy/test/prototypes/raw": accuracy_score(
            y_test, explainer.predict_with_prototypes(x_test, prototypes_raw)),
        "score/accuracy/test/prototypes/fi": accuracy_score(
            y_test, explainer.predict_with_prototypes(x_test, prototypes_fi, fi=True))
    }

    pprint(statistics)

    if log:
        wandb.log(statistics)
        wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
