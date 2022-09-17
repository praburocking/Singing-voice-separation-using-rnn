#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from typing import Tuple, Optional, Union, Dict
import numpy as np
from pathlib import Path
import librosa as lb

from torch.utils.data import Dataset
import numpy

from utils import get_files_from_dir_with_pathlib

__docformat__ = 'reStructuredText'
__all__ = ['MyDataset']


class MyDataset(Dataset):
    def __init__(self, mix_files,mix_path, source_path, source_prefix,mix_prefix,test_file_path=None) -> None:
        self.mix_path=mix_path
        self.source_path=source_path
        self.mix_files = mix_files
        self.test_file_path=test_file_path
        self.source_prefix=source_prefix
        self.mix_prefix=mix_prefix

    @staticmethod
    def _load_file(file_path) -> Dict[str, Union[int, numpy.ndarray]]:

        with open(file_path, 'rb') as f:
            try:
                a = np.load(f)
                return a
            except Exception as e:
                print(file_path)
                raise(e)



    def __len__(self) -> int:
        return len(self.mix_files)

    def __getitem__(self, item: int):
        mix_item = self._load_file(self.mix_path+self.mix_files[item])
        if self.test_file_path is None:
            source_item = self._load_file(self.source_path+self.mix_files[item].replace(self.mix_prefix,self.source_prefix))
            return mix_item, source_item
        else:
            raw_data, sr = lb.load(self.test_file_path, sr=None)
            return mix_item, (raw_data,sr)

# EOF
