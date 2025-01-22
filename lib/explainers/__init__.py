from typing import (Literal,
                    Type,
                    TypeAlias)

from lib.explainers.interface import IExplainer

TExplainer: TypeAlias = Literal["G_KM", "SM_A", "APete"]


def get_explainer(name: TExplainer) -> Type[IExplainer]:
    match name:
        case "G_KM":
            from lib.explainers.gkm import G_KM
            return G_KM
        case "SM_A":
            from lib.explainers.sma import SM_A
            return SM_A
        case "APete":
            from lib.explainers.apete import APete
            return APete
        case _:
            raise ValueError(f"Unknown explainer: {name}")
