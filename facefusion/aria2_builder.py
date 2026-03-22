import itertools
import shutil
from typing import List

from facefusion import metadata
from facefusion.types import Command


def run(commands : List[Command]) -> List[Command]:
	user_agent = metadata.get('name') + '/' + metadata.get('version')

	return [ shutil.which('aria2c'), '--user-agent', user_agent, '--stderr=true', '--console-log-level=error', '--download-result=hide' ] + commands


def chain(*commands : List[Command]) -> List[Command]:
	return list(itertools.chain(*commands))


def download(url : str, download_directory_path : str, download_file_name : str) -> List[Command]:
	return [ '--continue=true', '--split=16', '--max-connection-per-server=16', '--min-split-size=1M', '--dir', download_directory_path, '--out', download_file_name, url ]


def set_timeout(timeout : int) -> List[Command]:
	return [ '--connect-timeout', str(timeout) ]


def set_retry(retry : int) -> List[Command]:
	return [ '--max-tries', str(retry) ]
