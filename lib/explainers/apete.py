import numpy as np
import pandas as pd

from lib.explainers.sma import SM_A


class APete(SM_A):
    def _find_next_prototype(self, distances: dict[int, np.ndarray], prototypes: dict[int, pd.DataFrame],
                             fi_score: dict[int, np.ndarray] | None = None) -> tuple[int, int, float]:
        delta_prim = -np.inf
        prototype_idx = -1
        prototype_cls = -1
        for cls in distances:
            idx, delta = self._find_classwise_prototype(distances[cls], prototypes[cls])
            if delta > delta_prim:
                delta_prim = delta
                prototype_idx = idx
                prototype_cls = cls
        return prototype_cls, prototype_idx, delta_prim

    def _find_prototypes(self, x: pd.DataFrame, y: pd.Series | None = None, alpha: float | str = 0.05,
                         fi: bool = False) -> dict[int, pd.DataFrame]:
        if isinstance(alpha, str):
            alpha = float(alpha)

        y = pd.Series(self.model.predict(x)) if y is None else y

        x, y = self._sample(x, y)
        
        prev_improvement = 0.0
        classes = y.unique()
        prototypes = {cls.item(): [] for cls in classes}
        distances = {cls: self._tree_distance_matrix(x[y == cls]) for cls in classes}

        if fi:
            for cls in classes:
                distances[cls] -= self.beta * self._fi_proximity_matrix(x[y == cls])

        while True:
            cls, idx, improvement = self._find_next_prototype(distances, prototypes)
            prototypes[cls].append(idx)

            if improvement == 0 or np.abs(prev_improvement - improvement) / improvement <= alpha:
                return {cls: x.iloc[indices] for cls, indices in prototypes.items()}

            prev_improvement = improvement

    def prototypes_raw(self, x: pd.DataFrame, y: pd.Series | None = None, alpha: float | str = 0.05
                       ) -> dict[int, pd.DataFrame]:
        """Select prototypes based on the tree distance only.

        Args:
            x (pd.DataFrame): The instances.
            y (pd.Series): The class labels. If None, the class labels are predicted by the model. Default is None.
            alpha (float): The convergence threshold. Default is 0.01.

        Returns:
            dict[int, pd.DataFrame]: The candidate prototypes for each class.
        """

        return self._find_prototypes(x, y, alpha, fi=False)

    def prototypes_fi(self, x: pd.DataFrame, y: pd.Series | None = None, alpha: float | str = 0.05) -> dict[int, pd.DataFrame]:
        """Select prototypes based on the tree distance and feature importance.

        Args:
           x (pd.DataFrame): The instances.
           y (pd.Series): The class labels. If None, the class labels are predicted by the model. Default is None.
           alpha (float): The convergence threshold. Default is 0.05.

        Returns:
           dict[int, pd.DataFrame]: The selected prototypes for each class.
        """

        return self._find_prototypes(x, y, alpha, fi=True)
