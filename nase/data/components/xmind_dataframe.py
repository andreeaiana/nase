# https://github.com/andreeaiana/newsreclib/blob/main/newsreclib/data/components/xmind_dataframe.py

import os
from typing import Any, List, Dict, Tuple

import numpy as np
import pandas as pd
from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset
from tqdm import tqdm

import nase.data.components.data_utils as data_utils
import nase.data.components.file_utils as file_utils
from nase.utils import RankedLogger

tqdm.pandas()
log = RankedLogger(__name__, rank_zero_only=True)


class xMINDDataFrame(Dataset):
    """Creates a dataframe for the xMIND dataset.

    Additionally:
        - Parses the news and behaviors data.
        - Split the behaviors into `train` and `validation` sets by time.

    Attributes:
        dataset_size:
            A string indicating the type of the dataset. Choose between `large` and `small`.
        mind_dataset_url:
            Dictionary of URLs for downloading the MIND `train` and `dev` datasets for the specified `dataset_size`.
        xmind_dataset_url:
            Dictionary of URLs for downloading the xMIND `train` and `dev` datasets for the specified `dataset_size`.
        data_dir:
            Path to the data directory.
        tgt_lang:
            Target language for the xMIND dataset.
        eval_tgt_lang: 
            Whether the user history and the candidates list is monolingual in target language or monolingual in source language.
        dataset_attributes:
            List of news features available in the used dataset (e.g., title, category, etc.).
        valid_time_split:
            A string with the date before which click behaviors are included in the train set. After this date, behaviors are included in the validation set.
        train:
            If ``True``, the data will be processed and used for training. If ``False``, it will be processed and used for validation or testing.
        validation:
            If ``True`` and `train` is also``True``, the data will be processed and used for validation. If ``False`` and `train` is `True``, the data will be processed ad used for training. If ``False`` and `train` is `False``, the data will be processed and used for testing.
        download_mind:
            Whether to download the MIND dataset, if not already downloaded.
        download_xmind:
            Whether to download the xMIND dataset, if not already downloaded.
    """

    def __init__(
        self,
        dataset_size: str,
        mind_dataset_url: DictConfig,
        xmind_dataset_url: Dict[str, Dict[str, str]],
        data_dir: str,
        tgt_lang: str,
        eval_tgt_lang: bool,
        dataset_attributes: List[str],
        valid_time_split: str,
        train: bool,
        validation: bool,
        download_mind: bool,
        download_xmind: bool,
    ) -> None:
        super().__init__()

        self.dataset_size = dataset_size
        self.tgt_lang = tgt_lang
        assert self._validate_tgt_lang()

        self.data_dir = data_dir

        self.dataset_attributes = dataset_attributes

        self.valid_time_split = valid_time_split

        self.validation = validation
        self.data_split = "train" if train else "dev"
        
        self.eval_tgt_lang = eval_tgt_lang
        
        self.mind_dst_dir = os.path.join(
            self.data_dir, "MIND" + self.dataset_size + "_" + self.data_split
        )
        self.xmind_dst_dir = os.path.join(
            self.data_dir, "xMIND", self.tgt_lang, "xMIND" + self.dataset_size + "_" + self.data_split
        )
        if not os.path.isdir(self.xmind_dst_dir):
            os.makedirs(self.xmind_dst_dir)

        # download the MIND dataset
        if download_mind:
            url = mind_dataset_url[dataset_size][self.data_split]
            log.info(
                f"Downloading MIND{self.dataset_size} dataset for {self.data_split} from {url}."
            )
            data_utils.download_and_extract_dataset(
                data_dir=self.data_dir,
                url=url,
                filename=url.split("/")[-1],
                extract_compressed=True,
                dst_dir=self.mind_dst_dir,
                clean_archive=False,
            )
        
        # download the xMIND dataset
        if download_xmind:
            filename = 'xMIND' + dataset_size + '_' + self.data_split
            zipped_filename = filename + '.zip'
            url = xmind_dataset_url + self.tgt_lang + '/' + zipped_filename
            log.info(
                f"Downloading xMIND{self.dataset_size} dataset for {self.data_split} from {url}."
            )
            data_utils.download_and_extract_dataset(
                data_dir=os.path.join(self.data_dir, 'xMIND', self.tgt_lang),
                url=url,
                filename=zipped_filename,
                extract_compressed=True,
                dst_dir=os.path.join(self.data_dir, 'xMIND', self.tgt_lang),
                clean_archive=False,
            )

        self.news, self.behaviors = self.load_data()

    def __getitem__(self, index) -> Tuple[Any, Any, Any]:
        user_bhv = self.behaviors.iloc[index]

        history = user_bhv["history"]
        cand = user_bhv["cand"]
        labels = user_bhv["labels"]

        history = self.news[history]
        cand = self.news.loc[cand]
        labels = np.array(labels)

        return history, cand, labels

    def __len__(self) -> int:
        return len(self.behaviors)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads the parsed news and user behaviors.

        Returns:
            Tuple of news and behaviors datasets.
        """
        news = self._load_news()
        log.info(f"News data size: {len(news)}")

        behaviors = self._load_behaviors()
        log.info(
            f"Behaviors data size for data split {self.data_split}, validation={self.validation}: {len(behaviors)}"
        )

        return news, behaviors

    def _load_news(self) -> pd.DataFrame:
        """Loads the parsed news. If not already parsed, loads and preprocesses the raw news data.

        Args:
            news:
                Dataframe of news articles.

        Returns:
            Parsed and annotated news data.
        """
        parsed_news_file = os.path.join(self.xmind_dst_dir, "parsed_news.tsv")

        if file_utils.check_integrity(parsed_news_file):
            # news already parsed
            log.info(f"News already parsed. Loading from {parsed_news_file}.")

            news = pd.read_table(filepath_or_buffer=parsed_news_file)
            news["abstract"].fillna("", inplace=True)
        else:
            log.info("News not parsed. Loading and parsing raw data.")

            # load MIND news
            log.info('Loading MIND news')
            mind_columns_names = [
                "nid",
                "category",
                "subcategory",
                "title",
                "abstract",
                "url",
                "title_entities",
                "abstract_entities",
            ]
            mind_news = pd.read_table(
                filepath_or_buffer=os.path.join(self.mind_dst_dir, "news.tsv"),
                header=None,
                names=mind_columns_names,
                usecols=range(len(mind_columns_names)),
            )
            mind_news = mind_news.drop(
                    columns=[
                        "url",
                        "category",
                        "subcategory",
                        "title_entities",
                        "abstract_entities",
                        ]
                    )

            # replace missing values
            mind_news["abstract"].fillna("", inplace=True)

            if self.eval_tgt_lang:
                # load xMIND news 
                log.info('Loading xMIND news')
                xmind_columns_names = [
                    "nid",
                    "title",
                    "abstract",
                ]
                xmind_news = pd.read_table(
                    filepath_or_buffer=os.path.join(self.xmind_dst_dir, "news.tsv"),
                    header=None,
                    names=xmind_columns_names,
                    usecols=range(len(xmind_columns_names)),
                )

                # join MIND news data and xMIND news data
                cols_from_mind_news = list(mind_news.columns)
                cols_from_mind_news.remove('title')
                cols_from_mind_news.remove('abstract')
                enhanced_xmind_news = pd.merge(
                       left=mind_news[cols_from_mind_news],
                       right=xmind_news,
                       how='inner',
                       on='nid'
                        )
                enhanced_xmind_news['nid'] = enhanced_xmind_news['nid'].apply(lambda x: x + '_' + self.tgt_lang)
                news = pd.concat([mind_news, enhanced_xmind_news], ignore_index=True)
            else:
                news = mind_news.copy()

            # cache parsed news
            log.info(f"Caching parsed news of size {len(news)} to {parsed_news_file}.")
            file_utils.to_tsv(news, parsed_news_file)

        news = news.set_index("nid", drop=True)

        return news

    def _load_behaviors(self) -> pd.DataFrame:
        """Loads the parsed user behaviors. If not already parsed, loads and parses the raw
        behavior data.

        Returns:
            Parsed and split user behavior data.
        """
        file_prefix = ""
        if self.data_split == "train":
            file_prefix = "train_" if not self.validation else "val_"
        parsed_bhv_file = os.path.join(
                self.xmind_dst_dir, 
                file_prefix + "parsed_behaviors_" + str(self.eval_tgt_lang) + ".tsv"
                )

        if file_utils.check_integrity(parsed_bhv_file):
            # behaviors already parsed
            log.info(f"User behaviors already parsed. Loading from {parsed_bhv_file}.")
            behaviors = pd.read_table(
                filepath_or_buffer=parsed_bhv_file,
                converters={
                    "history": lambda x: x.strip("[]").replace("'", "").split(", "),
                    "candidates": lambda x: x.strip("[]").replace("'", "").split(", "),
                    "labels": lambda x: list(map(int, x.strip("[]").split(", "))),
                },
            )
        else:
            log.info("User behaviors not parsed. Loading and parsing raw data.")

            # load behaviors
            column_names = ["impid", "uid", "time", "history", "impressions"]
            behaviors = pd.read_table(
                filepath_or_buffer=os.path.join(self.mind_dst_dir, "behaviors.tsv"),
                header=None,
                names=column_names,
                usecols=range(len(column_names)),
            )

            # parse behaviors
            log.info("Parsing behaviors.")
            behaviors["time"] = pd.to_datetime(behaviors["time"], format="%m/%d/%Y %I:%M:%S %p")
            behaviors["history"] = behaviors["history"].fillna("").str.split()
            behaviors["impressions"] = behaviors["impressions"].str.split()
            behaviors["candidates"] = behaviors["impressions"].apply(
                lambda x: [impression.split("-")[0] for impression in x]
            )
            behaviors["labels"] = behaviors["impressions"].apply(
                lambda x: [int(impression.split("-")[1]) for impression in x]
            )
            behaviors = behaviors.drop(columns=["impressions"])

            cnt_bhv = len(behaviors)
            behaviors = behaviors[behaviors["history"].apply(len) > 0]
            dropped_bhv = cnt_bhv - len(behaviors)
            log.info(
                f"Removed {dropped_bhv} ({dropped_bhv / cnt_bhv}%) behaviors without user history"
            )

            behaviors = behaviors.reset_index(drop=True)

            if self.data_split == "train":
                log.info("Splitting behavior data into train and validation sets.")
                if not self.validation:
                    # training set
                    behaviors = behaviors.loc[behaviors["time"] < self.valid_time_split]
                    behaviors = behaviors.reset_index(drop=True)

                else:
                    # validation set
                    behaviors = behaviors.loc[behaviors["time"] >= self.valid_time_split]
                    behaviors = behaviors.reset_index(drop=True)

            if self.eval_tgt_lang:
                behaviors['history'] = behaviors['history'].apply(lambda x: self._generate_bilingual_history(x))
                behaviors['candidates'] = behaviors.apply(lambda x: self._generate_bilingual_candidates(x['candidates'], x['labels']), axis=1)

            # cache parsed behaviors
            log.info(f"Caching parsed behaviors of size {len(behaviors)} to {parsed_bhv_file}.")
            behaviors = behaviors[["history", "candidates", "labels"]]
            file_utils.to_tsv(behaviors, parsed_bhv_file)

        return behaviors

    def _generate_bilingual_history(self, history: List[str]) -> List[str]:
        # sample news to replace 
        tgt_lang_ids = np.random.choice(
                np.random.permutation(np.array(history)),
                len(history),
                replace=False
                ).tolist()

        # build bilingual history
        bilingual_history = [nid if not nid in tgt_lang_ids else nid + '_' + self.tgt_lang for nid in history]

        return bilingual_history

    def _generate_bilingual_candidates(self, candidates: List[str], labels: List[int]) -> List[str]:
        # select positive and negative candidates
        pos_candidates = [candidates[i] for i in labels if i == 1]
        neg_candidates = [cand for cand in candidates if not cand in pos_candidates]

        # sample news to replace
        tgt_lang_pos_cand = np.random.choice(
                np.random.permutation(np.array(pos_candidates)),
                len(pos_candidates),
                replace=False
                ).tolist()
        tgt_lang_neg_cand = np.random.choice(
                np.random.permutation(np.array(neg_candidates)),
                len(neg_candidates),
                replace=False
                ).tolist()

        # build bilingual candidates
        bilingual_candidates = [cand if not ((cand in tgt_lang_pos_cand) or (cand in tgt_lang_neg_cand)) else cand + "_" + self.tgt_lang for cand in candidates]

        return bilingual_candidates     

    def _validate_tgt_lang(self):
        return self.tgt_lang in [
                'fin',
                'grn',
                'hat',
                'ind',
                'jpn',
                'kat',
                'ron',
                'som',
                'swh',
                'tam',
                'tha',
                'tur',
                'vie',
                'cmn',
                'eng'
                ]
