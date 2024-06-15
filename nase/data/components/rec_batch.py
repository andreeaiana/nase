# Adapted from https://github.com/andreeaiana/newsreclib/blob/main/newsreclib/data/components/batch.py

from typing import Any, Dict, TypedDict

import torch


class RecommendationBatch(TypedDict):
    """Batch used for recommendation.

    Attributes:
        batch_hist:
            Batch of histories of users.
        batch_cand:
            Batch of candidates for each user.
        x_hist:
            Dictionary of news from a the users' history, mapping news features to values.
        x_cand
            Dictionary of news from a the users' candidates, mapping news features to values.
        labels:
            Ground truth specifying whether the news is relevant to the user.
    """

    batch_hist: torch.Tensor
    batch_cand: torch.Tensor
    x_hist: Dict[str, Any]
    x_cand: Dict[str, Any]
    labels: torch.Tensor
