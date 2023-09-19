import typing as tp

from pydantic import BaseModel


class BBox(BaseModel):
    top_left_x: int
    top_left_y: int
    bottom_right_x: int
    bottom_right_y: int
    confidence: float
    class_id: int


class DetectionResult(BaseModel):
    result: tp.List[BBox]
