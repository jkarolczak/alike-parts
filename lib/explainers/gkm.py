import numpy as np
import pandas as pd

from lib.explainers.interface import IExplainer


class G_KM(IExplainer):
    """Greedy K-Medoid (G-KM) explainer for tree-based models. The explainer is based on the similarity of the instances
    in the tree space. The similarity is computed as the proportion of trees that assign the instances to the same leaf node.

    Args:
        model (RandomForestClassifier): The tree-based model to be explained.
        beta (float): The beta parameter for the feature importance weights.
    """

    def _k_medoid_step(self, distance_matrix: np.ndarray, prototypes: list[int]) -> int:
        mask = np.isin(range(distance_matrix.shape[0]), prototypes)
        current_partial_distances = np.minimum.reduce(distance_matrix[mask], axis=0) if prototypes else np.inf
        candidate_distances = np.minimum(distance_matrix[~mask], current_partial_distances).sum(axis=1)
        return np.where(~mask)[0][np.argmin(candidate_distances)].item()

    def _greedy_k_medoid(self, distance_matrix: np.ndarray, k: int) -> list[int]:
        prototypes = []
        for _ in range(k):
            prototypes.append(self._k_medoid_step(distance_matrix=distance_matrix, prototypes=prototypes))
        return prototypes

    def _find_prototypes(self, x: pd.DataFrame, y: pd.Series | None = None, n_prototypes: int | str = 15, fi: bool = False
                         ) -> dict[int, pd.DataFrame]:
        if isinstance(n_prototypes, str):
            n_prototypes = int(n_prototypes)

        y = pd.Series(self.model.predict(x)) if y is None else y

        classes = y.unique()
        prototypes = {cls.item(): [] for cls in classes}

        for cls in classes:
            class_x = x[y == cls]
            distances = self._tree_distance_matrix(class_x)
            if fi:
                distances -= self.beta * self._fi_proximity_matrix(class_x)
            indices = self._greedy_k_medoid(distances, n_prototypes)
            prototypes[cls] = class_x.iloc[indices]
        return prototypes

    def prototypes_raw(self, x: pd.DataFrame, y: pd.Series | None = None, n_prototypes: int | str = 5
                       ) -> dict[int, pd.DataFrame]:
        """Select prototypes based on the tree distance only.

        Args:
            x (pd.DataFrame): The instances.
            y (pd.Series): The class labels. If None, the class labels are predicted by the model. Default is None.
            n_prototypes (int): The number of prototypes to select for each class. Default is 5.

        Returns:
            dict[int, pd.DataFrame]: The candidate prototypes for each class.
        """
        return self._find_prototypes(x, y, n_prototypes, fi=False)

    def prototypes_fi(self, x: pd.DataFrame, y: pd.Series | None = None, n_prototypes: int | str = 5
                      ) -> dict[int, pd.DataFrame]:
        """Select prototypes based on the tree distance and feature importance.

        Args:
            x (pd.DataFrame): The instances.
            y (pd.Series): The class labels. If None, the class labels are predicted by the model. Default is None.
            n_prototypes (int): The number of prototypes to select for each class. Default is 5.

        Returns:
            dict[int, pd.DataFrame]: The selected prototypes for each class.
        """
        return self._find_prototypes(x, y, n_prototypes, fi=True)
