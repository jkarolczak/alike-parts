from abc import abstractmethod
from typing import (Literal,
                    TypeAlias)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

TSampleStrategy: TypeAlias = Literal["uniform", "compress"] | None
TFIMethod: TypeAlias = Literal["TreeSHAP", "treeinterpreter"]
TMaskStrategy: TypeAlias = Literal["mean", "knee", "sqrt", "log"]
TSimilarityMetric: TypeAlias = Literal["dot", "l1", "l2"]

EPS = 1e-6


class IExplainer:
    """Interface for the explainers.

    Args:
        model (RandomForestClassifier): The tree-based model to be explained.
        beta (float): The beta parameter for the feature importance weights.
        ignore_direction (bool): A flag indicating whether to ignore the direction of the feature importance. Default is
            True.
        normalize_fi (bool): A flag indicating whether to normalize the feature importance. Default is True.
        masking_strategy (TMaskStrategy): The strategy for selecting the important features. Default is "mean".
        similarity_metric (TSimilarityMetric): The metric for computing the similarity between the importance of
            features of the instance and the prototype. Default is "dot".
    """

    def __init__(self, model: RandomForestClassifier, beta: float, ignore_direction: bool = True, normalize_fi: bool = False,
                 masking_strategy: TMaskStrategy = "mean", similarity_metric: TSimilarityMetric = "dot",
                 fi_method: TFIMethod = "LIME", sampling_strategy: TSampleStrategy = "compress",
                 random_seed: int = 42) -> None:
        self.model = model
        self.beta = beta
        self.ignore_direction = ignore_direction
        self.normalize_fi = normalize_fi
        self.masking_strategy = masking_strategy
        self.sampling_strategy = sampling_strategy
        self.similarity_metric = similarity_metric
        self.fi_method = fi_method
        self.random_seed = random_seed

    def _sample(self, x: pd.DataFrame, y: pd.Series, n_points: int = 512) -> tuple[pd.DataFrame, pd.Series]:
        n_points = min(n_points, len(x))
        if self.sampling_strategy == "uniform":
            idcs = np.random.choice(len(x), size=n_points, replace=False)
        elif self.sampling_strategy == "compress":
            x_np = x.values.astype(np.float32)
            from goodpoints import (compress,
                                    kt)

            def kernel_gaussian(y_: np.ndarray, x_: np.ndarray, gamma: float = 1.0) -> np.ndarray:
                k_vals = np.sum((x_ - y_) ** 2, axis=1)
                return np.exp(-gamma * k_vals / 2)

            f_halve = lambda x: kt.thin(x, m=1, split_kernel=kernel_gaussian, swap_kernel=kernel_gaussian)
            idcs = compress.compress(x_np, halve=f_halve, g=0)

            if len(idcs) > n_points:
                idcs = np.random.choice(idcs, size=n_points, replace=False)
        else:
            idcs = np.arange(len(x))
        return (
            x.iloc[idcs].reset_index(drop=True),
            y.iloc[idcs].reset_index(drop=True)
        )

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

    def _fi(self, x: pd.DataFrame) -> np.ndarray:
        match self.fi_method:
            case "TreeSHAP":
                import shap

                fi = shap.TreeExplainer(self.model)(x).values[:, :, 1]
            case "KernelSHAP":
                import shap

                fi = shap.KernelExplainer(self.model.predict_proba, x).shap_values(x)[:, :, 1]
            case "LIME":
                import warnings

                from lime.lime_tabular import LimeTabularExplainer

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    explainer = LimeTabularExplainer(x.values, feature_names=x.columns.tolist(), class_names=["0", "1"],
                                                     discretize_continuous=True)
                    fi = np.array([explainer.explain_instance(xi.values, self.model.predict_proba, num_features=x.shape[1])
                                  .as_list(label=1) for _, xi in x.iterrows()])
                fi = fi[:, :, 1]
                fi = fi.astype(float)
            case "treeinterpreter":
                from treeinterpreter import treeinterpreter as ti

                predictions, bias, contributions = ti.predict(self.model, x.values)
                fi = contributions[:, :, 1]
            case _:
                raise ValueError(f"Unknown feature importance method: {self.fi_method}")

        return fi

    def _fi_proximity_matrix(self, x1: pd.DataFrame, x2: pd.DataFrame | pd.Series | None = None) -> np.ndarray:
        if x2 is None:
            x2 = x1.copy()

        if isinstance(x2, pd.Series):
            x2 = x2.to_frame().T

        fi1 = self._fi(x1)
        fi2 = self._fi(x2)

        if self.ignore_direction:
            fi1 = np.abs(fi1)
            fi2 = np.abs(fi2)

        if self.normalize_fi:
            fi1 /= max(np.abs(fi1).sum(), EPS)
            fi2 /= max(np.abs(fi2).sum(), EPS)

        similarity = self._compute_similarity(fi1, fi2)
        return similarity

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
                distances[cls] -= self.beta * self._fi_proximity_matrix(x, prototypes[cls])

        nearest_prototype = (-np.inf, -1, -1)

        for cls in distances:
            idx = np.argmin(distances[cls])
            improvement = 1 - distances[cls][0][idx]
            if improvement > nearest_prototype[0]:
                nearest_prototype = (improvement, cls, idx)
        return nearest_prototype[1], nearest_prototype[2].item()

    def _fi_similarity_to_mask(self, v: np.ndarray) -> np.ndarray:
        match self.masking_strategy:
            case "mean":
                mask = (v > v.mean())
            case "knee":
                from kneed import KneeLocator

                idcs = np.argsort(-1 * v)
                sorted_v = v[idcs]

                kneedle = KneeLocator(np.arange(len(sorted_v)), sorted_v, S=1.0, curve="concave", direction="decreasing")
                knee_point = kneedle.knee_y

                if knee_point is None:
                    mask = (v > v.mean())

                else:
                    mask = (v >= knee_point)
                    if mask.sum() == len(mask) or mask.sum() == 0:
                        mask = (v > v.mean())

            case "sqrt":
                top_n_features = int(np.sqrt(len(v)))
                indices = np.argsort(v)[-top_n_features:]
                mask = np.zeros_like(v, dtype=bool)
                mask[indices] = True
            case "log":
                top_n_features = int(np.log(len(v)))
                indices = np.argsort(v)[-top_n_features:]
                mask = np.zeros_like(v, dtype=bool)
                mask[indices] = True
            case _:
                raise ValueError(f"Unknown masking strategy: {self.masking_strategy}")
        mask = mask.astype(bool)
        return mask

    def _compute_similarity(self, fi1: np.ndarray, fi2: np.ndarray) -> np.ndarray:
        if fi1.shape[0] > 1 or fi2.shape[0] > 1:
            match self.similarity_metric:
                case "dot":
                    similarity = fi1 @ fi2.T
                case "l1":
                    l1_distance = np.abs(fi1[:, None, :] - fi2).sum(axis=2)
                    similarity = 1 - l1_distance
                case "l2":
                    l2_distance = ((fi1[:, None, :] - fi2) ** 2).sum(axis=2)
                    similarity = 1 - l2_distance
                case _:
                    raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        else:
            match self.similarity_metric:
                case "dot":
                    similarity = fi1 * fi2
                case "l1":
                    similarity = 1 - np.abs(fi1 - fi2)
                case "l2":
                    similarity = 1 - (fi1 - fi2) ** 2
                case _:
                    raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        return similarity

    def similar_parts(self, x: pd.DataFrame, proto: pd.DataFrame) -> pd.DataFrame:
        """Find the parts that have high feature importance for both the instance and the prototype.

        Args:
            x (pd.DataFrame): The instance.
            proto (pd.DataFrame): The prototype.

        Returns:
            pd.DataFrame: The parts that have high feature importance for both the instance and the prototype.
        """

        similarity = self._fi_proximity_matrix(x[:1], proto[:1])[0]
        mask = self._fi_similarity_to_mask(similarity)

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
