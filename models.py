
from pydantic import BaseModel
from typing import List

class ClusterRequest(BaseModel):
    texts: List[str]
    eps: float = 0.4
    min_samples: int = 2


ClusterRequest.model_rebuild()