# Adapted from https://github.com/andreeaiana/newsreclib/blob/main/newsreclib/data/components/data_utils.py

from typing import Optional

from nase.utils import RankedLogger
from nase.data.components.download_utils import (
    download_path,
    extract_file,
    maybe_download,
)

log = RankedLogger(__name__, rank_zero_only=True)


def download_and_extract_dataset(
    data_dir: str,
    url: str,
    filename: str,
    extract_compressed: bool,
    dst_dir: Optional[str],
    clean_archive: Optional[bool],
) -> None:
    """Downloads a dataset from the specified url and extracts the compessed data file.

    Args:
        data_dir:
            Path where to download data.
        url:
            URL of the file to download.
        filename:
            Name of the file to download.
        extract_compressed:
            Whether to extract the compressed downloaded file.
        dst_dir:
            Destination directory for the extracted file.
        clean_archive:
            Whether to delete the compressed file after extraction.
    """
    with download_path(data_dir) as path:
        path = maybe_download(url=url, filename=filename, work_directory=path)
        log.info("Compressed dataset downloaded")

        if extract_compressed:
            assert isinstance(dst_dir, str) and isinstance(clean_archive, bool)
            # extract the compressed dataset
            log.info(f"Extracting dataset from {path} into {dst_dir}.")
            extract_file(archive_file=path, dst_dir=dst_dir, clean_archive=clean_archive)
            log.info("Dataset extraction completed.")
