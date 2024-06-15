#!/bin/bash
source $HOME/.bashrc
conda activate nase_env

polynews_parallel_directory="data/polynews-parallel/raw"
find "$polynews_parallel_directory" -mindepth 1 -maxdepth 1 -type d -print | while read subdir; do
    subdir_name=$(basename "$subdir")
    echo "Deduplicating $subdir_name"
    python -m nase.data.deduplicate lang=$subdir_name polynews_dir=data/polynews-parallel column=src
done