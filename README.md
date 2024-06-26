# NaSE: News-adapted Sentence Embeddings

NaSE is a news-adapted sentence encoder, domain-specialized from a pretrained massively multilingual sentence encoder (e.g, [LaBSE](https://aclanthology.org/2022.acl-long.62)) leveraging the [PolyNews](https://huggingface.co/datasets/aiana94/polynews) and [PolyNewsParallel](https://huggingface.co/datasets/aiana94/polynews-parallel) datasets.

# Using the pre-trained model
NaSE can be loaded directly from [HuggingFace Hub Models](https://huggingface.co/models).


Here is how to use this model to get the sentence embeddings of a given text in PyTorch:

```python
    from transformers import BertModel, BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained('aiana94/NaSE')
    model = BertModel.from_pretrained('aiana94/NaSE')

    # pepare input
    sentences = ["This is an example sentence", "Dies ist auch ein Beispielsatz in einer anderen Sprache."]
    encoded_input = tokenizer(sentences, return_tensors='pt', padding=True)

    # forward pass
    with torch.no_grad():
        output = model(**encoded_input)

    # to get the sentence embeddings, use the pooler output
    sentence_embeddings = output.pooler_output
```

and in Tensorflow:

```python
    from transformers import TFBertModel, BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained('aiana94/NaSE')
    model = TFBertModel.from_pretrained('aiana94/NaSE')

    # pepare input
    sentences = ["This is an example sentence", "Dies ist auch ein Beispielsatz in einer anderen Sprache."]
    encoded_input = tokenizer(sentences, return_tensors='tf', padding=True)

    # forward pass
    with torch.no_grad():
        output = model(**encoded_input)

    # to get the sentence embeddings, use the pooler output
    sentence_embeddings = output.pooler_output
```

For similarity between sentences, an L2-norm is recommended before calculating the similarity:

```python
  import torch
  import torch.nn.functional as F

  def cos_sim(a: torch.Tensor, b: torch.Tensor):
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)

    return torch.mm(a_norm, b_norm.transpose(0, 1))
```

# Pretraining NaSE

## Installation

Pretraining NaSE requires Python version 3.9 or later.

It also requires PyTorch, PyTorch Lightning, and TorchMetrics version 2.0 or later.
If you want to use this repository with GPU, please ensure CUDA or cudatoolkit version of 11.8.

### Install from source

#### CONDA

```bash
   git clone https://github.com/andreeaiana/nase.git
   cd nase
   conda create --name nase_env python=3.9
   conda activate nase_env
   pip install -e .
```

## Training
NaSE was adapted to the news domain using two training objectives:
1. **DAE**: reconstructing the input sentence from its corrupted version obtained by adding discrete noise (cf. [TSDAE](https://aclanthology.org/2021.findings-emnlp.59/))
2. **MT**: domain-specific sequence-to-sequence machine translation, i.e., we treat the input in the source language to be the 'corruption' of the target sentence in the target language which is to be 'reconstructed'. 

The final NaSE model was trained by sequentially combining the two objectives:

1. Train with the DAE objective on [PolyNews](https://huggingface.co/datasets/aiana94/polynews):

```python
    python nase/train.py experiment=train_nase_dae
```

2. Extract the best checkpoint:

```python
    NASE_DAE_CKPT_PATH = ${nase_dae_ckpt_path}
    NASE_DAE_SAVE_PATH = ${nase_dae_save_path}

    from nase.models.nase import NASE

    nase_model = NASE.load_from_checkpoint(NASE_DAE_CKPT_PATH)

    encoder = nase_model.loss_fct.encoder
    tokenizer = nase_model.loss_fct.encoder.tokenizer

    encoder.save_pretrained(NASE_DAE_SAVE_PATH)
    tokenizer.save_pretrained(NASE_DAE_SAVE_PATH)
```

3. Continue training with the MT objective on [PolyNewsParallel](https://huggingface.co/datasets/aiana94/polynews-parallel):

```python
    python nase/train.py experiment=train_nase_mt data.tokenizer.petrained_model_name_or_path=${NASE_DAE_CKPT_PATH}
```

4. Save the final model:

```python
    NASE_FINAL_CKPT_PATH = ${nase_final_ckpt_path}
    NASE_FINAL_SAVE_PATH = ${nase_final_save_path}

    from nase.models.nase import NASE

    nase_model = NASE.load_from_checkpoint(NASE_FINAL_CKPT_PATH)
    
    encoder = nase_model.loss_fct.encoder
    tokenizer = nase_model.loss_fct.encoder.tokenizer

    encoder.save_pretrained(NASE_FINAL_SAVE_PATH)
    tokenizer.save_pretrained(NASE_FINAL_SAVE_PATH)
```

### Other variants of NaSE

1. NaSE (DAE only), it uses only [PolyNews](https://huggingface.co/datasets/aiana94/polynews):

```python
    python nase/train.py experiment=train_nase_dae
```


2. NaSE (MT only), it uses only [PolyNewsParallel](https://huggingface.co/datasets/aiana94/polynews-parallel):


```python
    python nase/train.py experiment=train_nase_mt
```

3. NaSE (DAE + MT in parallel), it uses only [PolyNewsParallel](https://huggingface.co/datasets/aiana94/polynews-parallel):


```python
    python nase/train.py experiment=train_nase_dae_plus_mt
```


### Use a subset of the data
If you want to use only a subset of the PolyNews dataset, you can specify the list of languages:

```python
    python nase/train.py experiment=train_nase_dae data.training_languages=${language_1, language_2}.
```

Similarly, you can train the model only on a subset of the language pairs of PolyNewsParallel, by specifying which languages should be contained as the `source` or `target` language of the language pair:

```python
    python nase/train.oy experiment=train_nase_mt data.parallel_data=True data.training_languages=${language_1, language_2}.
```

You can include only language or language pairs with a `minimum number of instances`:

```python
    python nase/train.oy experiment=train_nase_dae data.min_examples=${THRESHOLD}.
```


### Validation
During both steps, the model is validated every `n_steps` on the [xMIND](https://github.com/andreeaiana/xMIND/tree/main) *small* dataset. In the validation step, the frozen NASE is used to embed both the clicked and the candidate news. We adopt a [late fusion](https://dl.acm.org/doi/abs/10.1145/3539618.3592062) strategy and compute the final score as the mean-pooling of dot-product scores between the candidate and the clicked news embeddings. 

You can change the version of xMIND and choose from `{small, large}`:
```python
    python nase/train.oy experiment=train_nase_dae data.xmind_dataset_size=${SIZE}.
```

You can also select a subset of the [xMIND languages](https://github.com/andreeaiana/xMIND):
```python
    python nase/train.oy experiment=train_nase_dae data.tgt_languages=${language_1, language_2}.
```


# Reproducing PolyNews & PolyNewsParallel

Run [construct_polynews.sh](scripts/construct_polynews.sh) to constrct the Polynews and PolyNewsParallel datasets.

Alternatively, you can follow the next steps: 
1. Install [opustools](https://github.com/Helsinki-NLP/OpusTools/blob/master/opustools_pkg/README.md), if not already installed.

2. Download WMT-News corpus from OPUS

```python
    mkdir data/wmtnews
    opus_get -d 'WMT-News' -p 'tmx' -dl 'data/wmtnews/'
```

3. Download GlobalVoices corpus from OPUS
```python
    mkdir data/globalvoices
    opus_get -d 'GlobalVoices' -p 'tmx' -dl 'data/gobalvoices/'
```

4. Create raw datasets
```
    python -m nase.data.polynews
```

5. Deduplicate PolyNews and save as TSV
```
    bash deduplicate_mono.sh
```

6. Deduplicate PolyNews-Parallel and save as TSV
```
    bash deduplicate_parallel.sh
```

## Citation

If you use NaSE, PolyNews or PolyNewsParallel, please cite:

```bibtex
@misc{iana2024news,
      title={News Without Borders: Domain Adaptation of Multilingual Sentence Embeddings for Cross-lingual News Recommendation}, 
      author={Andreea Iana and Fabian David Schmidt and Goran Glavaš and Heiko Paulheim},
      year={2024},
      eprint={2406.12634},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2406.12634}
}
```
