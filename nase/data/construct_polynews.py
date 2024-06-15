#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List, Dict, Tuple

import os
import gzip
import hydra
import fasttext
import rootutils
import pandas as pd
from tqdm import tqdm
from lxml import etree
from omegaconf import DictConfig
import xml.etree.ElementTree as ET
from huggingface_hub import hf_hub_download
from datasets import load_dataset, concatenate_datasets

from nase.utils.pylogger import RankedLogger
from nase.data.components import file_utils, download_utils


tqdm.pandas()
log = RankedLogger(__name__, rank_zero_only=True)
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def construct_wikinews_dataset(
        lang_id2iso_code: Dict[str, str],
        languages: List[str], 
        output_dir: str, 
        remove_text_len_pct: float,
        lang_identification_model
        ) -> Dict[str, pd.DataFrame]:
    """ Downloads the latest dump from Wikinews for the given languages and constructs a dataset from it.

    Args:
        lang_id2iso_code:
            Dictionary mapping language IDs from different datasets to ISO 693-3 Code
        languages: 
            List of languages available in Wikinews.
        output_dir:
            Path to the output directory where the Wikinews dumps will be downloaded.
        remove_text_len_pct:
            The percentage of texts with the shortest texts lengths to be removed from the final dataset.
        lang_identification_model:
            FastText model for language and script identification.

    Returns:
        Dictionary with language as the key and a DataFrame as the value, where the DataFrame contains the titles of the news from Wikinews and the provenance dataset.
    """
    wikinews = {}

    log.info('Download and read Wikinews dumps.')
    for lang in tqdm(languages):
        file = lang + 'wikinews-latest-abstract.xml.gz'
        url = 'https://dumps.wikimedia.org/' + lang + 'wikinews/latest/' + file

        # download Wikinews dump
        download_utils.maybe_download(
                url=url,
                filename=file,
                work_directory=output_dir
                )
        
        # read Wkinews dump
        f = gzip.open(os.path.join(output_dir, file), 'r')
        tree = ET.parse(f)
        root = tree.getroot()
        url_list, title_list, abstract_list = [], [], []

        for type_tag in tqdm(root.findall('doc')):
            title = type_tag.find('title').text
            abstract = type_tag.find('abstract').text
            url = type_tag.find('url').text

            if title == 'Wikinews: Main Page':
                continue

            url_list.append(url)
            title_list.append(title)
            abstract_list.append(abstract)
        f.close()

        df = pd.DataFrame({
            'title': title_list,
            'abstract': abstract_list,
            'url': url_list
            })

        # drop news without abstract
        init_len = len(df)
        df = df.dropna(subset=['abstract'])
        log.info(f'Dropped {init_len - len(df)} news without an abstract for language {lang}.')

        # remove news whose predicted script differs from the expected script for the given language 
        if lang != 'sr':
            script = lang_id2iso_code[lang].split('_')[-1]
            df = _drop_different_script_news(
                    script=script,
                    df=df,
                    df_subset=df,
                    text_column='title',
                    lang=lang_id2iso_code[lang],
                    lang_identification_model=lang_identification_model
                    )
            wikinews[lang_id2iso_code[lang]] = df[['title']]   
        else:
            # Serbian appeas with both Latn and Cyrl scripts
            df = _predict_language_script(
                    df, 
                    text_column='title',
                    lang_identification_model=lang_identification_model
                    )
            wikinews[lang_id2iso_code['sr_latn']] = df[df['pred_script']=='Latn'][['title']]
            wikinews[lang_id2iso_code['sr_cyrl']] = df[df['pred_script']=='Cyrl'][['title']]

    log.info('Dumps downloaded and read.')

    log.info('Processing dataset.')
    # normalize texts
    lang2wikinews_intro = {}
    for lang in wikinews:
        wikinews_intro =  wikinews[lang].iloc[0].title.split(':')[0] + ': '
        lang2wikinews_intro[lang] = wikinews_intro
        wikinews[lang]['title'] = wikinews[lang]['title'].apply(lambda x: x.lstrip(wikinews_intro))

    # remove news which are too short (i.e., text is among the x% shortest texts per language)
    for lang in tqdm(wikinews):
        df = _drop_short_texts(
                df=wikinews[lang],
                text_column='title',
                lang = lang,
                remove_text_len_pct=remove_text_len_pct
                )

        df['provenance'] = ['wikinews'] * len(df)
        df = df.rename(columns={'title': 'text'})
        wikinews[lang] = df[['text', 'provenance']]
    log.info('Finished')

    return wikinews


def construct_masakhanews_dataset(
        lang_id2iso_code: Dict[str, str],
        languages: List[str], 
        remove_text_len_pct: float,
        lang_identification_model
        ) -> Dict[str, pd.DataFrame]:
    """ Loads the MasakhaNews dataset from HuggingFace Hub and constructs a dataset from it.

    Reference: Adelani, David Ifeoluwa, Marek Masiak, Israel Abebe Azime, Jesujoba Alabi, Atnafu Lambebo Tonja, Christine Mwase, Odunayo Ogundepo et al. "MasakhaNEWS: News Topic Classification for African languages." In Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 144-159. 2023.

    For further details, please refer to the `paper <https://aclanthology.org/2023.ijcnlp-main.10/>`_

    Args:
        lang_id2iso_code:
            Dictionary mapping language IDs from different datasets to ISO 693-3 Code
        languages: 
            List of languages available in MasakhaNews.
        remove_text_len_pct:
            The percentage of texts with the shortest texts lengths to be removed from the final dataset. 
        lang_identification_model:
            FastText model for language and script identification.

    Returns:
        Dictionary with language as the key and a DataFrame as the value, where the DataFrame contains the text of the news from MasakhaNews and the provenance dataset.
    """
    masakha_news = {}

    # load dataset
    log.info('Loading and constructing MasakhaNews dataset.')
    for lang in tqdm(languages):
        dataset = load_dataset('masakhane/masakhanews', lang, trust_remote_code=True)
        dataset = dataset['train']
        dataset = dataset.remove_columns(['headline_text', 'text', 'url', 'label'])
        dataset = dataset.rename_column('headline', 'text')

        df = dataset.to_pandas()
        df['text'] = df['text'].apply(lambda x: x.replace('\n', ' '))
        
        # drop exact duplicates
        init_len = len(df)
        df = df.drop_duplicates()
        log.info(f'Dropped {init_len - len(df)} exact duplicates for {lang_id2iso_code[lang]}.')

        # remove news whose predicted script differs from the expected script for the given language 
        script = lang_id2iso_code[lang].split('_')[-1]
        df = _drop_different_script_news(
                script=script,
                df=df,
                df_subset=df,
                text_column='text',
                lang=lang_id2iso_code[lang],
                lang_identification_model=lang_identification_model
                )
       
        # remove news which are too short (i.e., text is among the x% shortest texts per language)
        df = _drop_short_texts(
                df,
                text_column='text',
                lang=lang_id2iso_code[lang],
                remove_text_len_pct=remove_text_len_pct
                )

        df['provenance'] = ['masakhanews'] * len(df)
        masakha_news[lang_id2iso_code[lang]] = df[['text', 'provenance']]
    log.info('Finished.')

    return masakha_news


def construct_mafand_dataset(
        lang_id2iso_code: Dict[str, str],
        language_pairs: List[str], 
        remove_text_len_pct: float,
        lang_identification_model
        ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """ Loads the MAFAND dataset from HuggingFace Hub and constructs a dataset from it.

    Reference: Adelani, David, Jesujoba Alabi, Angela Fan, Julia Kreutzer, Xiaoyu Shen, Machel Reid, Dana Ruiter et al. "A Few Thousand Translations Go a Long Way! Leveraging Pre-trained Models for African News Translation." In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 3053-3070. 2022.

    For further details, please refer to the `paper <https://aclanthology.org/2022.naacl-main.223/>`_

    Args:
        lang_id2iso_code:
            Dictionary mapping language IDs from different datasets to ISO 693-3 Code
        language_pairs: 
            List of language pairs available in MasakhaNews.
        remove_text_len_pct:
            The percentage of texts with the shortest texts lengths to be removed from the final dataset. 
        lang_identification_model:
            FastText model for language and script identification.

    Returns:
        Dictionary with language pairs as the key and a DataFrame as the value, where the DataFrame contains the text of the news from MAFAND and the provenance dataset.
    """
    dataset_list = []

    # load dataset
    log.info('Loading the MAFAND dataset.')
    for lang_pair in tqdm(language_pairs):
        src_lang, tgt_lang = lang_pair.split('-')

        dataset = load_dataset('masakhane/mafand', lang_pair, trust_remote_code=True)
        dataset = dataset['train']

        # create new columns from the translation pairs
        src = [(translation[src_lang], lang_id2iso_code[src_lang]) for translation in dataset['translation']]
        tgt = [(translation[tgt_lang], lang_id2iso_code[tgt_lang]) for translation in dataset['translation']]

        # append the new columns to the dataset
        dataset = dataset.add_column('src', src)
        dataset = dataset.add_column('tgt', tgt)
        dataset = dataset.remove_columns(['translation'])

        dataset_list.append(dataset)
    log.info('Loaded.')

    log.info('Processing dataset.')
    # create dataframe from the list of datasets
    concatenated_datasets = concatenate_datasets(dataset_list)
    provenance_column = ['mafand'] * len(concatenated_datasets)
    concatenated_datasets = concatenated_datasets.add_column('provenance', provenance_column)
    mafand_df = concatenated_datasets.to_pandas()

    # create new columns for the src (tgt) text and language
    mafand_df = pd.concat(
            [
                mafand_df.drop(['src'], axis=1), 
                mafand_df['src'].apply(pd.Series)
                ],
            axis=1)
    mafand_df = mafand_df.rename(columns={0: 'src', 1: 'src_lang'})

    mafand_df = pd.concat(
            [
                mafand_df.drop(['tgt'], axis=1), 
                mafand_df['tgt'].apply(pd.Series)
                ],
            axis=1)
    mafand_df = mafand_df.rename(columns={0: 'tgt', 1: 'tgt_lang'})

    mafand_df['src'] = mafand_df['src'].apply(lambda x: x.replace('\n', ' '))
    mafand_df['tgt'] = mafand_df['tgt'].apply(lambda x: x.replace('\n', ' '))

    mafand_df['lang_pair'] = mafand_df.apply(lambda x: x['src_lang'] + '-' + x['tgt_lang'], axis=1)
    init_len = len(mafand_df['lang_pair'].unique()) 
    log.info(f'Original MAFAND dataset contains {len(mafand_df)} news for {init_len} language pairs.')

    # remove news whose predicted script differs from the expected script for the given language 
    for src_lang in mafand_df['src_lang'].unique():
        script = src_lang.split('_')[-1]
        mafand_df = _drop_different_script_news(
                script=script,
                df=mafand_df,
                df_subset=mafand_df[mafand_df['src_lang']==src_lang],
                text_column='src',
                lang=src_lang,
                lang_identification_model=lang_identification_model
                )
    mafand_df.reset_index(drop=True, inplace=True) 
    
    for tgt_lang in mafand_df['tgt_lang'].unique():
        script = tgt_lang.split('_')[-1]
        mafand_df = _drop_different_script_news(
                script=script,
                df=mafand_df,
                df_subset=mafand_df[mafand_df['tgt_lang']==tgt_lang],
                text_column='tgt',
                lang=tgt_lang,
                lang_identification_model=lang_identification_model
                )
    mafand_df.reset_index(drop=True, inplace=True) 

    # remove news which are too short (i.e., text is among the x% shortest texts per source language)
    subset_dfs = []
    for src_lang in mafand_df['src_lang'].unique(): 
        df = _drop_short_texts(
                df=mafand_df[mafand_df['src_lang']==src_lang],
                text_column='src',
                lang=src_lang,
                remove_text_len_pct=remove_text_len_pct
                )
        subset_dfs.append(df)
    mafand_df = pd.concat(subset_dfs, axis=0)
    log.info('Finished.')

    # create monolingual MAFAND dataset
    log.info('Constructing monolingual MAFAND dataset.')
    mafand_mono = _construct_monolingual_dataset(df=mafand_df)
    log.info('Monolingual dataset constructed.')    

    # create parallel MAFAND dataset
    log.info('Constructing parallel MAFAND dataset.')
    mafand_parallel = _construct_parallel_dataset(df=mafand_df)
    log.info('Parallel dataset constructed.')

    return mafand_mono, mafand_parallel


def construct_dataset_from_opus(
        dataset_name: str,
        lang_id2iso_code: Dict[str, str],
        input_dir: str, 
        remove_text_len_pct: float,
        lang_identification_model
        ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """ Processes the downloaded OPUS corpus and constructs a dataset from it.

    Args:
        dataset_name:
            The name of the dataset from OPUS.
        lang_id2iso_code:
            Dictionary mapping language IDs from different datasets to ISO 693-3 Code
        input_dir:
            Path to the input directory where the OPUS corpus has been downloaded.
        remove_text_len_pct:
            The percentage of texts with the shortest texts lengths to be removed from the final dataset. 
        lang_identification_model:
            FastText model for language and script identification.
    
    Returns:
        Dictionary with language as the key and a DataFrame as the value, where the DataFrame contains the texts of the news from GlobalVoices and the provenance dataset.
    """
    # assert dataset is WMT-News or GlobalVoices    
    assert dataset_name in ['wmtnews', 'globalvoices']

    # assert that the OPUS dataset has been downloaded
    assert os.path.isdir(input_dir)
    if not os.listdir(input_dir):
        raise FileNotFoundError(f'The {input_dir} is empty! Download the dataset from OPUS and try again.')

    # load the dataset
    df_list = []
    log.info(f'Loading the dataset files for {dataset_name} from {input_dir}.')
    for file in tqdm(os.listdir(input_dir)):
        if 'tmx.gz' in file:
            lang_pair = file.split('tmx_')[-1].split('.tmx.gz')[0]
            src_lang, tgt_lang = lang_pair.split('-')
            
            if src_lang!='eo' and tgt_lang!='eo':
                sources, targets = [], []
                f = gzip.open(os.path.join(input_dir, file), 'r')
                parser = etree.XMLParser(remove_blank_text=True)
                tree = etree.parse(f, parser)
                root = tree.getroot()
                for tu in root.findall('.//tu'):
                    children = tu.getchildren()
                    src, tgt = [child.xpath('seg')[0].text for child in children]
                    if src != 'None' and tgt != None:
                        sources.append(src)
                        targets.append(tgt)

                if dataset_name == 'globalvoices':
                    if src_lang == 'sr':
                        src_lang = 'sr_latn'
                    
                    if tgt_lang == 'sr':
                        tgt_lang = 'sr_latn'

                df = pd.DataFrame({
                    'src': sources,
                    'tgt': targets,
                    'src_lang': [lang_id2iso_code[src_lang]] * len(sources),
                    'tgt_lang': [lang_id2iso_code[tgt_lang]] * len(targets),
                    'lang_pair': [lang_id2iso_code[src_lang] + '-' + lang_id2iso_code[tgt_lang]] * len(sources),
                    'provenance': [dataset_name] * len(sources)
                    })
                df_list.append(df)
    
    news_df = pd.concat(df_list, axis=0)
    
    log.info(f'Dataset {dataset_name} loaded.')
    init_len = len(news_df['lang_pair'].unique()) 
    log.info(f'Original {dataset_name} dataset contains {len(news_df)} news for {init_len} language pairs.')

    log.info('Processing dataset.')
    # remove news which are too short (i.e., text is among the x% shortest texts per source language)
    subset_dfs = []
    for src_lang in news_df['src_lang'].unique():
        df = _drop_short_texts(
                df=news_df[news_df['src_lang']==src_lang],
                text_column='src',
                lang=src_lang,
                remove_text_len_pct=remove_text_len_pct
                )
        subset_dfs.append(df)
    news_df = pd.concat(subset_dfs, axis=0)
    news_df.reset_index(drop=True, inplace=True)

    # remove news whose predicted script differs from the expected script for the given language 
    for src_lang in news_df['src_lang'].unique():
        script = src_lang.split('_')[-1]
        news_df = _drop_different_script_news(
                script=script,
                df=news_df,
                df_subset=news_df[news_df['src_lang']==src_lang],
                text_column='src',
                lang=src_lang,
                lang_identification_model=lang_identification_model
                )
    news_df.reset_index(drop=True, inplace=True) 
    
    for tgt_lang in news_df['tgt_lang'].unique():
        script = tgt_lang.split('_')[-1]
        news_df = _drop_different_script_news(
                script=script,
                df=news_df,
                df_subset=news_df[news_df['tgt_lang']==tgt_lang],
                text_column='tgt',
                lang=tgt_lang,
                lang_identification_model=lang_identification_model
                )
    news_df.reset_index(drop=True, inplace=True) 
    log.info('Finished.')
    
    # create monolingual dataset
    log.info(f'Constructing monolingual {dataset_name} dataset.')
    dataset_mono = _construct_monolingual_dataset(df=news_df)
    log.info('Monolingual dataset constructed.')    

    # create parallel dataset
    log.info(f'Constructing parallel {dataset_name} dataset.')
    dataset_parallel = _construct_parallel_dataset(df=news_df)
    log.info('Parallel dataset constructed.')

    return dataset_mono, dataset_parallel
    

def _construct_monolingual_dataset(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    dataset_dico = {}

    df_src = df[['src', 'src_lang', 'provenance']]
    df_src = df_src.rename(columns={'src': 'text', 'src_lang': 'language'})

    df_tgt = df[['tgt', 'tgt_lang', 'provenance']]
    df_tgt = df_tgt.rename(columns={'tgt': 'text', 'tgt_lang': 'language'})

    df_mono = pd.concat([df_src, df_tgt], axis=0)
    
    for lang in df_mono['language'].unique():
        df_subset = df_mono[df_mono['language']==lang]
        
        # drop exact duplicates
        init_len = len(df_subset)
        df_subset = df_subset.drop_duplicates()
        log.info(f'Dropped {init_len - len(df_subset)} exact duplicates for {lang}.')
        
        dataset_dico[lang] = df_subset[['text', 'provenance']]

    return dataset_dico


def _construct_parallel_dataset(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    dataset_dico = {}

    for lang_pair in df['lang_pair'].unique():
        df_subset = df[df['lang_pair']==lang_pair]
        
        # drop exact duplicates
        init_len = len(df_subset)
        df_subset = df_subset.drop_duplicates()
        log.info(f'Dropped {init_len - len(df_subset)} exact duplicates for {lang_pair}.')

        dataset_dico[lang_pair] = df_subset[['src', 'tgt', 'provenance']]

    return dataset_dico


def _predict_language_script(df: pd.DataFrame, text_column: str, lang_identification_model) -> pd.DataFrame:
    """ Predicts the language and script of a given text."""
    # predict the language
    df['predicted_lang'] = df[text_column].progress_apply(lambda x: lang_identification_model.predict(x))
    
    # extract predicted language and model confidence
    df[['pred_lang', 'pred_conf']] = df['predicted_lang'].apply(pd.Series)
    df.drop(columns=['predicted_lang'], inplace=True)

    # process language code, script, and model confidence
    df['pred_lang'] = df['pred_lang'].apply(lambda x: x[0].split('_label__')[-1])
    df['pred_script'] = df['pred_lang'].apply(lambda x: x.split('_')[-1])

    return df


def _drop_different_script_news(
        script: str,
        df: pd.DataFrame, 
        df_subset: pd.DataFrame, 
        text_column: str, 
        lang: str, 
        lang_identification_model
        ) -> pd.DataFrame:

    if script not in ['Hans', 'Hant', 'Jpan']:
        df_subset = _predict_language_script(
                df_subset, 
                text_column=text_column,
                lang_identification_model=lang_identification_model
                )
        drop_rows = df_subset[df_subset['pred_script']!=script].index
        df = df.drop(drop_rows)
        log.info(f'Dropped {len(drop_rows)} news in another script for language {lang}.')

    return df


def _drop_short_texts(df: pd.DataFrame, text_column: str, lang: str, remove_text_len_pct: float) -> pd.DataFrame:
    df['text_len'] = df[text_column].apply(lambda x: len(x))
    text_lens = sorted(df['text_len'].unique(), reverse=False)
    top_lowest_text_lens = int(len(text_lens)*remove_text_len_pct)
    init_len = len(df)
    df = df[~df['text_len'].isin(text_lens[:top_lowest_text_lens])].copy()
    log.info(f'Removed {init_len - len(df)} news which were too short for {lang}.')
    df = df.drop(columns=['text_len'])

    return df


@hydra.main(version_base="1.3", config_path="../../configs", config_name="construct_polynews.yaml")
def main(cfg: DictConfig):
   
    # initialize language id map
    lang_id2iso_code = cfg.get('lang_id2iso_code')

    # load language identification model
    log.info('Loading language identification model.')
    model_path = hf_hub_download(
            repo_id=cfg.get('lang_identification_model'),
            filename='model.bin'
            )
    lang_identification_model = fasttext.load_model(model_path)
    log.info(f'Model loaded from {model_path}.')

    # construct Wikinews dataset
    wikinews = construct_wikinews_dataset(
            lang_id2iso_code=lang_id2iso_code,
            languages=cfg.get('wikinews_languages'),
            output_dir=cfg.get('wikinews_dir'),
            remove_text_len_pct=cfg.get('wikinews_remove_text_len_pct'),
            lang_identification_model=lang_identification_model
            )
   
    # # construct MasakhaNews dataset
    masakha_news = construct_masakhanews_dataset(
            lang_id2iso_code=lang_id2iso_code,
            languages=cfg.get('masakhanews_languages'),
            remove_text_len_pct=cfg.get('masakhanews_remove_text_len_pct'),
            lang_identification_model=lang_identification_model
            )

    # # construct MAFAND datasets
    mafand_mono, mafand_parallel = construct_mafand_dataset(
            lang_id2iso_code=lang_id2iso_code,
            language_pairs=cfg.get('mafand_language_pairs'),
            remove_text_len_pct=cfg.get('mafand_remove_text_len_pct'),
            lang_identification_model=lang_identification_model
            )

    # construct WMT-News datasets
    wmtnews_mono, wmtnews_parallel = construct_dataset_from_opus(
            dataset_name='wmtnews',
            lang_id2iso_code=lang_id2iso_code,
            input_dir=cfg.get('wmtnews_dir'),
            remove_text_len_pct=cfg.get('wmtnews_remove_text_len_pct'),
            lang_identification_model=lang_identification_model
            )

    # construct GlobalVoices datasets
    globalvoices_mono, globalvoices_parallel = construct_dataset_from_opus(
            dataset_name='globalvoices',
            lang_id2iso_code=lang_id2iso_code,
            input_dir=cfg.get('globalvoices_dir'),
            remove_text_len_pct=cfg.get('globalvoices_remove_text_len_pct'),
            lang_identification_model=lang_identification_model
            )

    # construct Polynews
    log.info('Constructing PolyNews.')
    polynews_languages = list(wikinews.keys()) + list(masakha_news.keys()) + list(mafand_mono.keys()) + list(wmtnews_mono.keys()) + list(globalvoices_mono.keys())
    polynews_languages = list(set(polynews_languages))
    polynews_dir = cfg.get('polynews_dir')

    for lang in tqdm(polynews_languages):
        df_list = []
        if lang in wikinews:
            df_list.append(wikinews[lang])
        if lang in masakha_news:
            df_list.append(masakha_news[lang])
        if lang in mafand_mono:
            df_list.append(mafand_mono[lang])
        if lang in wmtnews_mono:
            df_list.append(wmtnews_mono[lang])
        if lang in globalvoices_mono:
            df_list.append(globalvoices_mono[lang])

        polynews = pd.concat(df_list, axis=0)
        init_len = len(polynews)
        polynews = polynews.drop_duplicates()
        log.info(f'Dropped {init_len - len(polynews)} exact duplicates for {lang}.')
        polynews = polynews.reset_index(drop=True)

        # replace tabs in text
        polynews['text'] = polynews['text'].apply(lambda x: x.replace('\t', '\\t'))

        # cache to disk
        fdir = os.path.join(polynews_dir, 'raw', lang)
        if not os.path.isdir(fdir):
            os.makedirs(fdir)
        fpath = os.path.join(fdir, 'train.tsv')
        file_utils.to_tsv(polynews, fpath, index=True)
    log.info('Finished.')

    # construct PolyNewsParallel
    log.info('Constructing PolyNews.')
    polynews_parallel_languages = list(mafand_parallel.keys()) + list(wmtnews_parallel.keys()) + list(globalvoices_parallel.keys())
    polynews_parallel_languages = list(set(polynews_parallel_languages))
    polynews_parallel_dir = cfg.get('polynews_parallel_dir')
    
    for lang_pair in tqdm(polynews_parallel_languages):
        df_list = []
        if lang_pair in mafand_parallel:
            df_list.append(mafand_parallel[lang_pair])
        if lang_pair in wmtnews_parallel:
            df_list.append(wmtnews_parallel[lang_pair])
        if lang_pair in globalvoices_parallel:
            df_list.append(globalvoices_parallel[lang_pair])

        polynews_parallel = pd.concat(df_list, axis=0)
        init_len = len(polynews_parallel)
        polynews_parallel = polynews_parallel.drop_duplicates()
        log.info(f'Dropped {init_len - len(polynews_parallel)} exact duplicates for {lang_pair}.')
        polynews_parallel = polynews_parallel.reset_index(drop=True)

        # replace tabs in text
        polynews_parallel['src'] = polynews_parallel['src'].apply(lambda x: x.replace('\t', '\\t'))
        polynews_parallel['tgt'] = polynews_parallel['tgt'].apply(lambda x: x.replace('\t', '\\t'))
    
        # cache to disk
        fdir = os.path.join(polynews_parallel_dir, 'raw', lang_pair)
        if not os.path.isdir(fdir):
            os.makedirs(fdir)
        fpath = os.path.join(fdir, 'train.tsv')
        file_utils.to_tsv(polynews_parallel, fpath, index=True)
    log.info('Finished.')


if __name__ == '__main__':
    main()
