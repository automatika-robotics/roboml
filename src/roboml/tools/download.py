import logging
import os
from pathlib import Path

import requests
from tqdm import tqdm


class DownloadManager:
    """
    A simple resumable download manager
    """

    def __init__(self, url: str, save_path: str | Path, chunk_size: int = 4096):
        """
        Init with params

        :param url: URL of the hosted file
        :type url: str
        :param save_path: Path where the download file with be saved
        :type save_path: str | Path
        :param chunk_size: size of download chunk,
        depends on network, memory availability, defaults to 4096
        :type chunk_size: int, optional
        """
        self.url = url
        self.save_path = str(save_path)
        self.chunk_size = chunk_size
        self.total_size = 0
        self.downloaded_size = 0
        self.resumable = False
        self.headers = {}
        self.session = (
            requests.Session()
        )  # TODO(ajay): authentication/session management

    def _get_file_info(self):
        """
        Retrieve file information from the server, including total size and headers.
        """
        try:
            response = self.session.head(self.url)
            response.raise_for_status()  # Raise exception for non-2xx status codes
            self.total_size = int(response.headers.get("content-length", 0))
            self.resumable = (
                "accept-ranges" in response.headers
            )  # support for partial requests
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch file information: {e}")
            raise

    def _download(self):
        """
        Download the file in chunks and update the progress bar.
        """
        try:
            response = self.session.get(self.url, stream=True)
            response.raise_for_status()  # Raise exception for non-2xx status codes

            with open(self.save_path, "ab") as file:
                with tqdm(
                    desc=os.path.basename(self.save_path),
                    total=self.total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    initial=self.downloaded_size,
                    colour="#32cd32",  # progress bar colour
                ) as pbar:
                    for data in response.iter_content(chunk_size=self.chunk_size):
                        if not data:
                            break
                        file.write(data)
                        pbar.update(len(data))
                        self.downloaded_size += len(data)
        except requests.exceptions.RequestException as e:
            logging.error(f"Download failed: {e}")
            raise

    def download(self):
        """
        Start the download from scratch.
        """
        if os.path.exists(self.save_path):
            self.downloaded_size = os.path.getsize(self.save_path)
        else:
            self._get_file_info()

        self._download()
        logging.info(f"{os.path.basename(self.save_path)} downloaded!")

    def resume(self):
        """
        Resume the download if resumable, or log a message if not supported.
        """
        if not self.resumable:
            # TODO: handle this in a better way
            logging.warning("Resuming is not supported for this URL.")
            return

        self._get_file_info()
        self._download()
        logging.info(f"{os.path.basename(self.save_path)} resumed!")
