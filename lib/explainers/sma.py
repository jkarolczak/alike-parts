import numpy as np
import pandas as pd

from lib.explainers.interface import IExplainer


class SM_A(IExplainer):
    """Adaptive greedy submodular prototype selection (SM-A) extended by considering the feature importance during the
    prototypes selection.

    Args:
        model (RandomForestClassifier): The tree-based model to be explained.
        beta (float): The beta parameter for the feature importance weights.
    """

    def _find_classwise_prototype(self, distance_matrix: np.ndarray, prototypes: pd.DataFrame) -> tuple[int, float]:
        if not prototypes:
            idx = np.argmin(distance_matrix.sum(axis=1)).item()
            improvement = distance_matrix.shape[0] - np.sum(distance_matrix[idx])
            return idx, improvement

        mask = np.isin(range(distance_matrix.shape[0]), prototypes)

        current_partial_distances = np.minimum.reduce(distance_matrix[mask], axis=0) if prototypes else np.inf
        original_distance = current_partial_distances.sum()

        candidates = distance_matrix[~mask]
        candidate_distances = np.minimum(candidates, current_partial_distances).sum(axis=1)

        idx = np.argmin(candidate_distances)
        improvement = original_distance - candidate_distances[idx]
        original_idx = np.where(~mask)[0][np.argmin(candidate_distances)].item()

        return original_idx, improvement

    def _find_next_prototype(self, distances: dict[int, np.ndarray], prototypes: dict[int, pd.DataFrame]) -> tuple[int, int]:
        prototype: tuple[float, int, int] = (-np.inf, -1, -1)
        for cls in distances:
            idx, improvement = self._find_classwise_prototype(distances[cls], prototypes[cls])
            if improvement > prototype[0]:
                prototype = (improvement, cls, idx)
        return prototype[1], prototype[2]

    def _find_prototypes(self, x: pd.DataFrame, y: pd.Series | None = None, n_prototypes: int | str = 10,
                         fi: bool = False) -> dict[int, pd.DataFrame]:
        if isinstance(n_prototypes, str):
            n_prototypes = int(n_prototypes)

        y = pd.Series(self.model.predict(x)) if y is None else y

        classes = y.unique()
        prototypes = {cls.item(): [] for cls in classes}
        distances = {cls: self._tree_distance_matrix(x[y == cls]) for cls in classes}

        if fi:
            for cls in classes:
                distances[cls] += self.beta * self._fi_proximity_matrix(x[y == cls])

        for _ in range(n_prototypes):
            cls, idx = self._find_next_prototype(distances, prototypes)
            prototypes[cls].append(idx)

        return {cls: x.iloc[indices] for cls, indices in prototypes.items()}

    def prototypes_raw(self, x: pd.DataFrame, y: pd.Series | None = None, n_prototypes: int | str = 10
                       ) -> dict[int, pd.DataFrame]:
        """Select prototypes based on the tree distance only.

        Args:
            x (pd.DataFrame): The instances.
            y (pd.Series): The class labels. If None, the class labels are predicted by the model. Default is None.
            n_prototypes (int): The number of prototypes to select. Default is 10.

        Returns:
            dict[int, pd.DataFrame]: The candidate prototypes for each class.
        """

        return self._find_prototypes(x, y, n_prototypes, fi=False)

    def prototypes_fi(self, x: pd.DataFrame, y: pd.Series | None = None, n_prototypes: int = 10) -> dict[int, pd.DataFrame]:
        """Select prototypes based on the tree distance and feature importance.

        Args:
           x (pd.DataFrame): The instances.
           y (pd.Series): The class labels. If None, the class labels are predicted by the model. Default is None.
           n_prototypes (int): The number of prototypes to select. Default is 10.

        Returns:
           dict[int, pd.DataFrame]: The selected prototypes for each class.
        """

        return self._find_prototypes(x, y, n_prototypes, fi=True)
