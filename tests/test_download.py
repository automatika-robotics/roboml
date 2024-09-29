import os
import shutil

import pytest

from roboml.tools.download import DownloadManager

save_folder: str = ""


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    """Fixture to execute asserts before and after a test is run"""
    global save_folder

    if os.name == "nt":  # Windows
        downloads_folder = f"{os.getenv('USERPROFILE')}\\Downloads"
    else:  # PORT: For *Nix systems
        downloads_folder = f"{os.getenv('HOME')}/Downloads"

    save_folder = f"{downloads_folder}/test"
    os.makedirs(save_folder, exist_ok=True)

    yield

    # Teardown
    shutil.rmtree(save_folder)


def test_download():
    """A test case for Download Manager. Downloads a file of size 1Mb from OVH servers"""

    global save_folder
    file_url: str = "https://proof.ovh.net/files/1Mb.dat"
    file_name: str = os.path.basename(file_url)
    save_path: str = os.path.join(save_folder, file_name)

    manager = DownloadManager(file_url, save_path)
    manager.download()

    assert manager.downloaded_size == manager.total_size
