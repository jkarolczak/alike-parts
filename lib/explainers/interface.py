from abc import abstractmethod

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier


def cosine_similarity(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return np.dot(x1, x2.T) / (np.linalg.norm(x1) * np.linalg.norm(x2))


class IExplainer:
    """Interface for the explainers.

    Args:
        model (RandomForestClassifier): The tree-based model to be explained.
        beta (float): The beta parameter for the feature importance weights.
    """

    def __init__(self, model: RandomForestClassifier, beta: float) -> None:
        self.model = model
        self.beta = beta

    def _tree_similarity_matrix(self, x1: pd.DataFrame, x2: pd.DataFrame | pd.Series | None = None) -> np.ndarray:
        if x2 is None:
            x2 = x1

        if isinstance(x2, pd.Series):
            x2 = x2.to_frame().T

        nodes1 = self.model.apply(x1)
        nodes2 = self.model.apply(x2)

        return (nodes1[:, None] == nodes2).sum(axis=2) / self.model.n_estimators

    def _tree_distance_matrix(self, x1: pd.DataFrame, x2: pd.DataFrame | pd.Series | None = None) -> np.ndarray:
        return 1 - self._tree_similarity_matrix(x1, x2)

    def _fi_proximity_matrix(self, x1: pd.DataFrame, x2: pd.DataFrame | pd.Series | None = None) -> np.ndarray:
        if x2 is None:
            x2 = x1

        if isinstance(x2, pd.Series):
            x2 = x2.to_frame().T

        fi1 = shap.TreeExplainer(self.model)(x1).values[:, :, 1]
        fi2 = shap.TreeExplainer(self.model)(x2).values[:, :, 1]

        fi1 = np.power(fi1, 2)
        fi2 = np.power(fi2, 2)

        fi1 = fi1 / fi1.sum(axis=1)[:, None]
        fi2 = fi2 / fi2.sum(axis=1)[:, None]

        return np.dot(fi1, fi2.T)

    def predict_with_prototypes(self, x: pd.DataFrame, prototypes: dict[int, pd.DataFrame], fi: bool = False) -> np.ndarray:
        """Predict the class labels of the instances using the prototypes.

        Args:
            x (pd.DataFrame): The instances.
            prototypes (dict[int, pd.DataFrame]): The prototypes for each class.
            fi (bool): A flag indicating whether to use feature importance. Default is False.

        Returns:
            np.ndarray: The predicted class labels.
        """

        predictions = np.zeros((len(x), 2))
        for cls, class_prototypes in prototypes.items():
            if class_prototypes.shape[0] == 0:
                continue
            class_distances = self._tree_distance_matrix(x, class_prototypes)
            if fi:
                class_distances -= self.beta * self._fi_proximity_matrix(x, class_prototypes)
            predictions[:, cls] = np.minimum.reduce(class_distances, axis=1)
        return predictions.argmin(axis=1)

    @abstractmethod
    def prototypes_raw(self, x: pd.DataFrame, y: pd.Series | None = None, *args, **kwargs
                       ) -> dict[int, pd.DataFrame]:
        pass

    @abstractmethod
    def prototypes_fi(self, x: pd.DataFrame, interesting_prototypes: list[int], y: pd.Series | None = None,
                      *args, **kwargs) -> dict[int, pd.DataFrame]:
        pass

    def get_nearest_prototype(self, x: pd.DataFrame, prototypes: dict[int, pd.DataFrame], cls: int | None = None,
                              fi: bool = False) -> tuple[int, int]:
        """Get the nearest prototype for the given instance.

        Args:
            x (pd.DataFrame): The instance.
            prototypes (dict[int, pd.DataFrame]): The prototypes for each class.
            cls (int): The to choose the prototype from. If None, the prototype is chosen from the nearest class.
            fi (bool): A flag indicating whether to use feature importance. Default is False.

        Returns:
            tuple[int, int]: The class and the index of the nearest prototype for each instance.
        """

        distances = {cls: self._tree_distance_matrix(x, class_prototypes)
                     for cls, class_prototypes in prototypes.items()
                     if class_prototypes.shape[0] > 0}
        if cls:
            distances = {cls: distances[cls]}
        if fi:
            for cls in distances:
                distances[cls] += self.beta * self._fi_proximity_matrix(x, prototypes[cls])

        nearest_prototype = (-np.inf, -1, -1)

        for cls in distances:
            idx = np.argmin(distances[cls])
            improvement = 1 - distances[cls][0][idx]
            if improvement > nearest_prototype[0]:
                nearest_prototype = (improvement, cls, idx)

        return nearest_prototype[1], nearest_prototype[2].item()

    def similar_parts(self, x: pd.DataFrame, proto: pd.DataFrame) -> pd.DataFrame:
        """Find the parts that have high feature importance for both the instance and the prototype.

        Args:
            x (pd.DataFrame): The instance.
            proto (pd.DataFrame): The prototype.

        Returns:
            pd.DataFrame: The parts that have high feature importance for both the instance and the prototype.
        """

        x_fi = shap.TreeExplainer(self.model)(x).values[0, :, 1]
        x_fi = np.abs(x_fi)
        x_fi /= x_fi.sum()

        proto_fi = shap.TreeExplainer(self.model)(proto).values[0, :, 1]
        proto_fi = np.abs(proto_fi)
        proto_fi /= proto_fi.sum()

        similarity = x_fi * proto_fi
        similarity /= similarity.sum()

        mask = (similarity > similarity.mean()).astype(bool)

        x = x.iloc[:, mask]
        x["obj"] = "instance"
        proto = proto.iloc[:, mask]
        proto["obj"] = "prototype"
        result = pd.concat([x, proto, pd.DataFrame(
            {
                col: val for col, val in zip(x.columns, similarity[mask].tolist() + ["similarity"])
            }, index=[2]
        )]).reset_index(drop=True)

        return result
