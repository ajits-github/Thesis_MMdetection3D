# nuScenes dev-kit.
# Code written by Holger Caesar & Oscar Beijbom, 2019.

import abc
from collections import defaultdict
from typing import List, Tuple, Union

import numpy as np


class EvalBox(abc.ABC):
    """ Abstract base class for data classes used during detection evaluation. Can be a prediction or ground truth."""

    # def __init__(self,
    #              sample_token: str = "",
    #              translation: Tuple[float, float, float] = (0, 0, 0),
    #              size: Tuple[float, float, float] = (0, 0, 0),
    #              rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
    #              velocity: Tuple[float, float] = (0, 0),
    #              ego_translation: Tuple[float, float, float] = (0, 0, 0),  # Translation to ego vehicle in meters.
    #              num_pts: int = -1):  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.

    def __init__(self,
                 sample_token: str = "",
                 translation: Tuple[float, float, float] = (0, 0, 0),
                 size: Tuple[float, float, float] = (0, 0, 0),
                 rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
                 velocity: Tuple[float, float] = (0, 0),
                 ego_translation: Tuple[float, float, float] = (0, 0, 0),  # Translation to ego vehicle in meters.
                 num_pts: int = -1, # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
                 time_to_coll_pred = 0.0,
                 time_to_coll_calc = 0.0,
                 sample_data_token: str = "",
                 sample_annotation_token: str = ""):
                
                 
        # print(".............EvalBox.init................")
        # Assert data for shape and NaNs.
        assert type(sample_token) == str, 'Error: sample_token must be a string!'
        assert type(sample_data_token) == str, 'Error: sample_data_token must be a string!'
        assert type(sample_annotation_token) == str, 'Error: sample_annotation_token must be a string!'

        assert len(translation) == 3, 'Error: Translation must have 3 elements!'
        assert not np.any(np.isnan(translation)), 'Error: Translation may not be NaN!'

        assert len(size) == 3, 'Error: Size must have 3 elements!'
        assert not np.any(np.isnan(size)), 'Error: Size may not be NaN!'

        assert len(rotation) == 4, 'Error: Rotation must have 4 elements!'
        assert not np.any(np.isnan(rotation)), 'Error: Rotation may not be NaN!'

        # Velocity can be NaN from our database for certain annotations.
        assert len(velocity) == 2, 'Error: Velocity must have 2 elements!'

        assert len(ego_translation) == 3, 'Error: Translation must have 3 elements!'
        assert not np.any(np.isnan(ego_translation)), 'Error: Translation may not be NaN!'

        assert type(num_pts) == int, 'Error: num_pts must be int!'
        assert not np.any(np.isnan(num_pts)), 'Error: num_pts may not be NaN!'

        assert type(time_to_coll_pred) == float, 'Error: time_to_coll_pred must be a float!'
        assert not np.any(np.isnan(time_to_coll_pred)), 'Error: time_to_coll_pred may not be NaN!'

        # print("..........time_to_coll_calc........", time_to_coll_calc)
        # assert type(time_to_coll_calc) == float or not(time_to_coll_calc != time_to_coll_calc), 'Error: time_to_coll_calc must be a float!'
        # assert not np.any(np.isnan(time_to_coll_calc)), 'Error: time_to_coll_calc may not be NaN!'

        # Assign.
        self.sample_token = sample_token
        self.sample_data_token = sample_data_token
        self.sample_annotation_token = sample_annotation_token
        self.translation = translation
        self.size = size
        self.rotation = rotation
        self.velocity = velocity
        self.ego_translation = ego_translation
        self.num_pts = num_pts
        self.time_to_coll_pred = time_to_coll_pred
        self.time_to_coll_calc = time_to_coll_calc


    @property
    def ego_dist(self) -> float:
        """ Compute the distance from this box to the ego vehicle in 2D. """
        return np.sqrt(np.sum(np.array(self.ego_translation[:2]) ** 2))

    def __repr__(self):
        return str(self.serialize())

    @abc.abstractmethod
    def serialize(self) -> dict:
        pass

    @classmethod
    @abc.abstractmethod
    def deserialize(cls, content: dict):
        pass


EvalBoxType = Union['DetectionBox', 'TrackingBox']


class EvalBoxes:
    """ Data class that groups EvalBox instances by sample. """

    def __init__(self):
        """
        Initializes the EvalBoxes for GT or predictions.
        """
        self.boxes = defaultdict(list)

    def __repr__(self):
        return "EvalBoxes with {} boxes across {} samples".format(len(self.all), len(self.sample_tokens))

    def __getitem__(self, item) -> List[EvalBoxType]:
        return self.boxes[item]

    def __eq__(self, other):
        if not set(self.sample_tokens) == set(other.sample_tokens):
            return False
        for token in self.sample_tokens:
            if not len(self[token]) == len(other[token]):
                return False
            for box1, box2 in zip(self[token], other[token]):
                if box1 != box2:
                    return False
        return True

    def __len__(self):
        return len(self.boxes)

    @property
    def all(self) -> List[EvalBoxType]:
        """ Returns all EvalBoxes in a list. """
        ab = []
        for sample_token in self.sample_tokens:
            ab.extend(self[sample_token])
        return ab

    @property
    def sample_tokens(self) -> List[str]:
        """ Returns a list of all keys. """
        return list(self.boxes.keys())

    def add_boxes(self, sample_token: str, boxes: List[EvalBoxType]) -> None:
        """ Adds a list of boxes. """
        self.boxes[sample_token].extend(boxes)

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        # for key, boxes in self.boxes.items():
        #     print("...........serialize.......boxes[0].....", boxes[0])
        #     print("...........key............", key)
        #     for box in boxes:
        #         print("...........box.serialize()....", box.serialize())
        #         exit()
        # exit()
        return {key: [box.serialize() for box in boxes] for key, boxes in self.boxes.items()}

    @classmethod
    def deserialize(cls, content: dict, box_cls):
        """
        Initialize from serialized content.
        :param content: A dictionary with the serialized content of the box.
        :param box_cls: The class of the boxes, DetectionBox or TrackingBox.
        """
        eb = cls()
        # print(".............content..........", content.keys())
        for sample_token, boxes in content.items():
            # print("...........boxes[0]..........", boxes[0])
            eb.add_boxes(sample_token, [box_cls.deserialize(box) for box in boxes])
            # for box in boxes:
            #     print(".......box......eval.common.data_classes.py......", box)
                # k = box_cls.deserialize(box)
            #     eb.add_boxes(sample_token, k)
            # #     # print(".......eb.....eval.common.data_classes.py.......", eb)
                # print(".......box......eval.common.data_classes.py......", k)
                # exit()
        # print(".......eb.len...", type(eb))
        # print(".......eb....", eb[0])
        # exit()
        return eb


class MetricData(abc.ABC):
    """ Abstract base class for the *MetricData classes specific to each task. """

    @abc.abstractmethod
    def serialize(self):
        """ Serialize instance into json-friendly format. """
        pass

    @classmethod
    @abc.abstractmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        pass
