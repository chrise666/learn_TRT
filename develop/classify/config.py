from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Config:
    data_root: str = "E:/datasets/shape_classification"
    save: str = "E:/workspace/learn_AI/save/shape_classification"
    classes: Optional[int] = None
    image_size: List[int] = field(default_factory=lambda: [128, 128])  # h,w
    epoch: int = 5
    valid_interval: int = 5
    batch_size: int = 8
    num_workers: int = 8
    lr: float = 0.0001  # 1e-4
    weight_decay: float = 0.01
    seed: int = 12
    split_seed: int = 8
    fold: int = 1
    valid_frac: float = 0.3  # valid/all, 1.0/fold if -1
    return_meta: bool = True  # dataloader returns all info in dataset