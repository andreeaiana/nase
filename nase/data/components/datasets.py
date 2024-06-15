# Adapted from https://github.com/andreeaiana/newsreclib/blob/main/newsreclib/data/components/rec_dataset.py

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import random
import numpy as np
import pandas as pd
import torch
from omegaconf.dictconfig import DictConfig
from datasets.arrow_dataset import Dataset
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset as TorchDataset
from transformers.utils.import_utils import is_nltk_available, NLTK_IMPORT_ERROR

from nase.data.components.rec_batch import RecommendationBatch
from nase.data.components.xmind_dataframe import xMINDDataFrame


class DenoisingAutoEncoderDataset(TorchDataset):
    """
    The DenoisingAutoEncoderDataset returns a Tuple in the format: texts=[noise_fn(sentence), sentence]
    It is used in combination with the DenoisingAutoEncoderLoss: Here, a decoder tries to re-construct the
    sentence without noise.


    Adapted from https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/datasets/DenoisingAutoEncoderDataset.py
    """
    def __init__(
            self,
            parallel_data: bool,
            combined_objectives: bool,
            datasets: dict[str, Dataset], 
            S: float = 1 / 3, 
            noise_fn=lambda s: DenoisingAutoEncoderDataset.delete(s)
            ):
        if not is_nltk_available():
            raise ImportError(NLTK_IMPORT_ERROR.format(self.__class__.__name__))

        self.parallel_data = parallel_data
        self.combined_objectives = combined_objectives
        self.datasets = datasets
        self.S = S
        
        # Compute the total counts from all datasets
        self.lg2count = {lg: len(ds) for lg, ds in datasets.items()}
        self.sample_size = sum(self.lg2count.values())  # Set sample_size to cover all instances

        self.noise_fn = noise_fn

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx: tuple[str, int]):
        if not self.parallel_data:
            sentence = self.datasets[idx[0]][int(idx[1])]['text'] 
            return [self.noise_fn(sentence), sentence]
        else:
            src = self.datasets[idx[0]][int(idx[1])]['src'] 
            tgt = self.datasets[idx[0]][int(idx[1])]['tgt'] 
            
            if not self.combined_objectives:
                return [src, tgt] 
            else:
                choices = [
                        [self.noise_fn(src), src],
                        [src, tgt],
                        [self.noise_fn(tgt), tgt],
                        [tgt, src],
                        ]
                return random.choice(choices)

    # Deletion noise.
    @staticmethod
    def delete(text, del_ratio=0.6):
        from nltk import word_tokenize, TreebankWordDetokenizer

        words = word_tokenize(text)
        n = len(words)
        if n == 0:
            return text

        keep_or_not = np.random.rand(n) > del_ratio
        if sum(keep_or_not) == 0:
            keep_or_not[np.random.choice(n)] = True  # guarantee that at least one word remains
        words_processed = TreebankWordDetokenizer().detokenize(np.array(words)[keep_or_not])
        return words_processed


class RecommendationDatasetTest(xMINDDataFrame):
    def __init__(self, news: pd.DataFrame, behaviors: pd.DataFrame, max_history_len: int) -> None:
        self.news = news
        self.behaviors = behaviors
        self.max_history_len = max_history_len

    def __getitem__(self, idx: Any) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        bhv = self.behaviors.iloc[idx]

        history = np.array(bhv["history"])[: self.max_history_len]
        candidates = np.array(bhv["candidates"])
        labels = np.array(bhv["labels"]) 

        history = self.news.loc[history]
        candidates = self.news.loc[candidates]

        return history, candidates, labels

    def __len__(self) -> int:
        return len(self.behaviors)


@dataclass
class DatasetCollate:
    def __init__(
            self,
            tokenizer: Optional[PreTrainedTokenizer],
            ) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        source_features, target_features = zip(*batch)

        tokenized_source_features = self.tokenizer(
            source_features, return_tensors="pt", return_token_type_ids=False, padding=True, truncation=True
        )
        tokenized_target_features = self.tokenizer(
            target_features, return_tensors="pt", return_token_type_ids=False, padding=True, truncation=True
        )

        return [tokenized_source_features, tokenized_target_features]

@dataclass
class RecDatasetCollate:
    def __init__(
        self,
        dataset_attributes: List[str],
        tokenizer: Optional[PreTrainedTokenizer],
        concatenate_inputs: bool,
    ) -> None:
        self.dataset_attributes = dataset_attributes
        self.tokenizer = tokenizer
        self.concatenate_inputs = concatenate_inputs

    def __call__(self, batch) -> RecommendationBatch:
        histories, candidates, labels = zip(*batch)

        batch_hist = self._make_batch_asignees(histories)
        batch_cand = self._make_batch_asignees(candidates)

        x_hist = self._tokenize_df(pd.concat(histories))
        x_cand = self._tokenize_df(pd.concat(candidates))
        labels = torch.from_numpy(np.concatenate(labels)).float()

        return RecommendationBatch(
            batch_hist=batch_hist,
            batch_cand=batch_cand,
            x_hist=x_hist,
            x_cand=x_cand,
            labels=labels,
        )

    def _tokenize_plm(self, text: List[str]):
        return self.tokenizer(
            text, return_tensors="pt", return_token_type_ids=False, padding=True, truncation=True
        )

    def _tokenize_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        batch_out = {}

        if not self.concatenate_inputs:
            # prepare text
            batch_out["title"] = self._tokenize_plm(df["title"].values.tolist())
            if "abstract" in self.dataset_attributes:
                batch_out["abstract"] = self._tokenize_plm(df["abstract"].values.tolist())

        else:
            # prepare text
            if "abstract" in self.dataset_attributes:
                concatenated_text = list(map(' '.join, zip(df["title"].values.tolist(), df["abstract"].values.tolist())))
                text = self._tokenize_plm(concatenated_text)
            else:
                text = self._tokenize_plm(df["title"].values.tolist())
            
            batch_out["text"] = text

        return batch_out

    def _make_batch_asignees(self, items: Sequence[Sequence[Any]]) -> torch.Tensor:
        sizes = torch.tensor([len(x) for x in items])
        batch = torch.repeat_interleave(torch.arange(len(items)), sizes)

        return batch
