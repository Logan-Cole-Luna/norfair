from abc import ABC, abstractmethod
from functools import partial
from logging import warning
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Union

import numpy as np
import cupy as cp
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from .tracker import Detection, TrackedObject
from .utils import get_array_module

class Distance(ABC):
    """
    Abstract class representing a distance.

    Subclasses must implement the method `get_distances`
    """

    @abstractmethod
    def get_distances(
        self,
        objects: Sequence["TrackedObject"],
        candidates: Optional[Union[List["Detection"], List["TrackedObject"]]],
    ) -> np.ndarray:
        """
        Method that calculates the distances between new candidates and objects.

        Parameters
        ----------
        objects : Sequence[TrackedObject]
            Sequence of [TrackedObject][norfair.tracker.TrackedObject] to be compared with potential [Detection][norfair.tracker.Detection] or [TrackedObject][norfair.tracker.TrackedObject]
            candidates.
        candidates : Union[List[Detection], List[TrackedObject]], optional
            List of candidates ([Detection][norfair.tracker.Detection] or [TrackedObject][norfair.tracker.TrackedObject]) to be compared to [TrackedObject][norfair.tracker.TrackedObject].

        Returns
        -------
        np.ndarray
            A matrix containing the distances between objects and candidates.
        """


class ScalarDistance(Distance):
    """
    ScalarDistance class represents a distance that is calculated pointwise.

    Parameters
    ----------
    distance_function : Union[Callable[["Detection", "TrackedObject"], float], Callable[["TrackedObject", "TrackedObject"], float]]
        Distance function used to determine the pointwise distance between new candidates and objects.
        This function should take 2 input arguments, the first being a `Union[Detection, TrackedObject]`,
        and the second [TrackedObject][norfair.tracker.TrackedObject]. It has to return a `float` with the distance it calculates.
    """

    def __init__(
        self,
        distance_function: Union[
            Callable[["Detection", "TrackedObject"], float],
            Callable[["TrackedObject", "TrackedObject"], float],
        ],
        use_gpu: bool = False,
    ):
        self.distance_function = distance_function
        self.use_gpu = use_gpu

    def get_distances(
        self,
        objects: Sequence["TrackedObject"],
        candidates: Optional[Union[List["Detection"], List["TrackedObject"]]],
    ) -> np.ndarray:
        """
        Method that calculates the distances between new candidates and objects.

        Parameters
        ----------
        objects : Sequence[TrackedObject]
            Sequence of [TrackedObject][norfair.tracker.TrackedObject] to be compared with potential [Detection][norfair.tracker.Detection] or [TrackedObject][norfair.tracker.TrackedObject]
            candidates.
        candidates : Union[List[Detection], List[TrackedObject]], optional
            List of candidates ([Detection][norfair.tracker.Detection] or [TrackedObject][norfair.tracker.TrackedObject]) to be compared to [TrackedObject][norfair.tracker.TrackedObject].

        Returns
        -------
        np.ndarray
            A matrix containing the distances between objects and candidates.
        """
        xp = get_array_module(self.use_gpu)
        distance_matrix = xp.full(
            (len(candidates), len(objects)),
            fill_value=xp.inf,
            dtype=xp.float32,
        )
        if not objects or not candidates:
            return distance_matrix
        for c, candidate in enumerate(candidates):
            for o, obj in enumerate(objects):
                if candidate.label != obj.label:
                    if (candidate.label is None) or (obj.label is None):
                        print("\nThere are detections with and without label!")
                    continue
                distance = self.distance_function(candidate, obj)
                distance_matrix[c, o] = distance
        return distance_matrix


class VectorizedDistance(Distance):
    """
    VectorizedDistance class represents a distance that is calculated in a vectorized way. This means
    that instead of going through every pair and explicitly calculating its distance, VectorizedDistance
    uses the entire vectors to compare to each other in a single operation.

    Parameters
    ----------
    distance_function : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Distance function used to determine the distances between new candidates and objects.
        This function should take 2 input arguments, the first being a `np.ndarray` and the second
        `np.ndarray`. It has to return a `np.ndarray` with the distance matrix it calculates.
    """

    def __init__(
        self,
        distance_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
        use_gpu: bool = False,
    ):
        self.distance_function = distance_function
        self.use_gpu = use_gpu

    def get_distances(
        self,
        objects: Sequence["TrackedObject"],
        candidates: Optional[Union[List["Detection"], List["TrackedObject"]]],
    ) -> np.ndarray:
        """
        Method that calculates the distances between new candidates and objects.

        Parameters
        ----------
        objects : Sequence[TrackedObject]
            Sequence of [TrackedObject][norfair.tracker.TrackedObject] to be compared with potential [Detection][norfair.tracker.Detection] or [TrackedObject][norfair.tracker.TrackedObject]
            candidates.
        candidates : Union[List[Detection], List[TrackedObject]], optional
            List of candidates ([Detection][norfair.tracker.Detection] or [TrackedObject][norfair.tracker.TrackedObject]) to be compared to [TrackedObject][norfair.tracker.TrackedObject].

        Returns
        -------
        np.ndarray
            A matrix containing the distances between objects and candidates.
        """
        xp = get_array_module(self.use_gpu)
        distance_matrix = xp.full(
            (len(candidates), len(objects)),
            fill_value=xp.inf,
            dtype=xp.float32,
        )
        if not objects or not candidates:
            return distance_matrix

        object_labels = np.array([o.label for o in objects]).astype(str)
        candidate_labels = np.array([c.label for c in candidates]).astype(str)

        # iterate over labels that are present both in objects and detections
        for label in np.intersect1d(
            np.unique(object_labels), np.unique(candidate_labels)
        ):
            # generate masks of the subset of object and detections for this label
            obj_mask = object_labels == label
            cand_mask = candidate_labels == label

            stacked_objects = []
            for o in objects:
                if str(o.label) == label:
                    stacked_objects.append(o.estimate.ravel())
            stacked_objects = np.stack(stacked_objects)

            stacked_candidates = []
            for c in candidates:
                if str(c.label) == label:
                    if "Detection" in str(type(c)):
                        stacked_candidates.append(c.points.ravel())
                    else:
                        stacked_candidates.append(c.estimate.ravel())
            stacked_candidates = xp.stack(stacked_candidates)

            # calculate the pairwise distances between objects and candidates with this label
            # and assign the result to the correct positions inside distance_matrix
            distance_matrix[xp.ix_(cand_mask, obj_mask)] = self._compute_distance(
                stacked_candidates, stacked_objects
            )

        return distance_matrix

    def _compute_distance(
        self, stacked_candidates: np.ndarray, stacked_objects: np.ndarray
    ) -> np.ndarray:
        """
        Method that computes the pairwise distances between new candidates and objects.
        It is intended to use the entire vectors to compare to each other in a single operation.

        Parameters
        ----------
        stacked_candidates : np.ndarray
            np.ndarray containing a stack of candidates to be compared with the stacked_objects.
        stacked_objects : np.ndarray
            np.ndarray containing a stack of objects to be compared with the stacked_objects.

        Returns
        -------
        np.ndarray
            A matrix containing the distances between objects and candidates.
        """
        return self.distance_function(stacked_candidates, stacked_objects)


class ScipyDistance(VectorizedDistance):
    """
    ScipyDistance class extends VectorizedDistance for the use of Scipy's vectorized distances.

    This class uses `scipy.spatial.distance.cdist` to calculate distances between two `np.ndarray`.

    Parameters
    ----------
    metric : str, optional
        Defines the specific Scipy metric to use to calculate the pairwise distances between
        new candidates and objects.

    Other kwargs are passed through to cdist

    See Also
    --------
    [`scipy.spatial.distance.cdist`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
    """

    def __init__(self, metric: str = "euclidean", **kwargs):
        self.metric = metric
        super().__init__(distance_function=partial(cdist, metric=self.metric, **kwargs), use_gpu=use_gpu)

def frobenius(detection: "Detection", tracked_object: "TrackedObject") -> float:
    """
    Frobernius norm on the difference of the points in detection and the estimates in tracked_object.

    The Frobenius distance and norm are given by:

    $$
    d_f(a, b) = ||a - b||_F
    $$

    $$
    ||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}
    $$

    Parameters
    ----------
    detection : Detection
        A detection.
    tracked_object : TrackedObject
        A tracked object.

    Returns
    -------
    float
        The distance.

    See Also
    --------
    [`np.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)
    """
    return np.linalg.norm(detection.points - tracked_object.estimate)


def mean_euclidean(detection: "Detection", tracked_object: "TrackedObject") -> float:
    xp = get_array_module(detection.use_gpu)
    return xp.linalg.norm(detection.points - tracked_object.estimate, axis=1).mean()

def mean_manhattan(detection: "Detection", tracked_object: "TrackedObject") -> float:
    xp = get_array_module(detection.use_gpu)
    return xp.linalg.norm(
        detection.points - tracked_object.estimate, ord=1, axis=1
    ).mean()

def _boxes_area(boxes: np.ndarray) -> np.ndarray:
    return (boxes[2] - boxes[0]) * (boxes[3] - boxes[1])

def _validate_bboxes(bboxes: np.ndarray):
    assert (
        isinstance(bboxes, np.ndarray)
        and len(bboxes.shape) == 2
        and bboxes.shape[1] == 4
    ), f"Bounding boxes must be defined as np.array with (N, 4) shape, {bboxes} given"

    if not (all(bboxes[:, 0] < bboxes[:, 2]) and all(bboxes[:, 1] < bboxes[:, 3])):
        warning("Incorrect bounding boxes. Check that x_min < x_max and y_min < y_max.")

def iou(candidates: np.ndarray, objects: np.ndarray) -> np.ndarray:
    xp = get_array_module(candidates)
    _validate_bboxes(candidates)

    area_candidates = _boxes_area(candidates.T)
    area_objects = _boxes_area(objects.T)

    top_left = xp.maximum(candidates[:, None, :2], objects[:, :2])
    bottom_right = xp.minimum(candidates[:, None, 2:], objects[:, 2:])

    area_intersection = xp.prod(
        xp.clip(bottom_right - top_left, a_min=0, a_max=None), 2
    )
    return 1 - area_intersection / (
        area_candidates[:, None] + area_objects - area_intersection
    )

iou_opt = iou  # deprecated

_SCALAR_DISTANCE_FUNCTIONS = {
    "frobenius": frobenius,
    "mean_manhattan": mean_manhattan,
    "mean_euclidean": mean_euclidean,
}
_VECTORIZED_DISTANCE_FUNCTIONS = {
    "iou": iou,
    "iou_opt": iou,  # deprecated
}
_SCIPY_DISTANCE_FUNCTIONS = [
    "braycurtis",
    "canberra",
    "chebyshev",
    "cityblock",
    "correlation",
    "cosine",
    "dice",
    "euclidean",
    "hamming",
    "jaccard",
    "jensenshannon",
    "kulczynski1",
    "mahalanobis",
    "matching",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
]
AVAILABLE_VECTORIZED_DISTANCES = (
    list(_VECTORIZED_DISTANCE_FUNCTIONS.keys()) + _SCIPY_DISTANCE_FUNCTIONS
)

def get_distance_by_name(name: str, use_gpu: bool = False) -> Distance:
    if name in _SCALAR_DISTANCE_FUNCTIONS:
        warning(
            "You are using a scalar distance function. If you want to speed up the"
            " tracking process please consider using a vectorized distance function"
            f" such as {AVAILABLE_VECTORIZED_DISTANCES}."
        )
        distance = _SCALAR_DISTANCE_FUNCTIONS[name]
        distance_function = ScalarDistance(distance, use_gpu=use_gpu)
    elif name in _SCIPY_DISTANCE_FUNCTIONS:
        distance_function = ScipyDistance(name, use_gpu=use_gpu)
    elif name in _VECTORIZED_DISTANCE_FUNCTIONS:
        if name == "iou_opt":
            warning("iou_opt is deprecated, use iou instead")
        distance = _VECTORIZED_DISTANCE_FUNCTIONS[name]
        distance_function = VectorizedDistance(distance, use_gpu=use_gpu)
    else:
        raise ValueError(
            f"Invalid distance '{name}', expecting one of"
            f" {list(_SCALAR_DISTANCE_FUNCTIONS.keys()) + AVAILABLE_VECTORIZED_DISTANCES}"
        )
    return distance_function

def create_keypoints_voting_distance(
    keypoint_distance_threshold: float, detection_threshold: float, use_gpu: bool = False
) -> Callable[["Detection", "TrackedObject"], float]:
    def keypoints_voting_distance(
        detection: "Detection", tracked_object: "TrackedObject"
    ) -> float:
        xp = get_array_module(detection.use_gpu)
        distances = xp.linalg.norm(detection.points - tracked_object.estimate, axis=1)
        match_num = xp.count_nonzero(
            (distances < keypoint_distance_threshold)
            * (detection.scores > detection_threshold)
            * (tracked_object.last_detection.scores > detection_threshold)
        )
        return 1 / (1 + match_num)
    return keypoints_voting_distance

def create_normalized_mean_euclidean_distance(
    height: int, width: int, use_gpu: bool = False
) -> Callable[["Detection", "TrackedObject"], float]:
    def normalized__mean_euclidean_distance(
        detection: "Detection", tracked_object: "TrackedObject"
    ) -> float:
        xp = get_array_module(detection.use_gpu)
        difference = (detection.points - tracked_object.estimate).astype(float)
        difference[:, 0] /= width
        difference[:, 1] /= height
        return xp.linalg.norm(difference, axis=1).mean()
    return normalized__mean_euclidean_distance

__all__ = [
    "frobenius",
    "mean_manhattan",
    "mean_euclidean",
    "iou",
    "iou_opt",
    "get_distance_by_name",
    "create_keypoints_voting_distance",
    "create_normalized_mean_euclidean_distance",
]
