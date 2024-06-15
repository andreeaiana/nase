#!/bin/bash
# Run from root folder with: bash scripts/construct_polynews.sh

source $HOME/.bashrc
conda activate nase_env

# download WMT-News corpus from OPUS
mkdir data/wmtnews/
opus_get -d 'WMT-News' -p 'tmx' -dl 'data/wmtnews/'

# download GlobalVoices corpus from OPUS
mkdir data/globalvoices/
opus_get -d 'GlobalVoices' -p 'tmx' -dl 'data/globalvoices/'

# create raw PolyNews and PolyNewsParallel dataset files (in TSV format)
python -m nase.data.construct_polynews

# deduplicate PolyNews 
polynews_directory="data/polynews/raw"
find "$polynews_directory" -mindepth 1 -maxdepth 1 -type d -print | while read subdir; do
    subdir_name=$(basename "$subdir")
    echo "Deduplicating $subdir_name"
    python -m nase.data.deduplicate lang=$subdir_name
done

# deduplicate PolyNewsParallel 
polynews_parallel_directory="data/polynews-parallel/raw"
find "$polynews_parallel_directory" -mindepth 1 -maxdepth 1 -type d -print | while read subdir; do
    subdir_name=$(basename "$subdir")
    echo "Deduplicating $subdir_name"
    python -m nase.data.deduplicate lang=$subdir_name polynews_dir=data/polynews-parallel column=src
done
