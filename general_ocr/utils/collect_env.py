# Copyright (c) GeneralOCR. All rights reserved.
from general_ocr.utils import collect_env as collect_base_env
from general_ocr.utils import get_git_hash

import general_ocr


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['GENERAL_OCR'] = general_ocr.__version__ + '+' + get_git_hash()[:7]
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
