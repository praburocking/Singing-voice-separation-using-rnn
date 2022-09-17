#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Union
import os
import pathlib
from pathlib import Path
from typing import Optional, Union
from torch.utils.data import DataLoader

__docformat__ = 'reStructuredText'
__all__ = ['get_files_from_dir_with_os', 'get_files_from_dir_with_pathlib']


def get_files_from_dir_with_os(dir_name: str) -> List[str]:
    """Returns the files in the directory `dir_name` using the os package.

    :param dir_name: The name of the directory.
    :type dir_name: str
    :return: The filenames of the files in the directory `dir_name`.
    :rtype: list[str]
    """
    return os.listdir(dir_name)


def get_files_from_dir_with_pathlib(dir_name: Union[str, pathlib.Path])  -> List[pathlib.Path]:
    """Returns the files in the directory `dir_name` using the pathlib package.

    :param dir_name: The name of the directory.
    :type dir_name: str
    :return: The filenames of the files in the directory `dir_name`.
    :rtype: list[pathlib.Path]
    """
    return list(pathlib.Path(dir_name).iterdir())





def get_dataset(data_dir: Union[str, Path],
                data_parent_dir: Optional[str] = '',
                key_features: Optional[str] = 'features',
                key_class: Optional[str] = 'class',
                load_into_memory: Optional[bool] = True):
    """Creates and returns a dataset, according to `MyDataset` class.

    :param data_dir: Directory to read data from.
    :type data_dir: str|pathlib.Path
    :param data_parent_dir: Parent directory of the data, defaults\
                            to ``.
    :type data_parent_dir: str
    :param key_features: Key to use for getting the features,\
                         defaults to `features`.
    :type key_features: str
    :param key_class: Key to use for getting the class, defaults\
                      to `class`.
    :type key_class: str
    :param load_into_memory: Load the data into memory? Default to True
    :type load_into_memory: bool
    :return: Dataset.
    :rtype: dataset_class.MyDataset
    """
    from data_handling_prabu_mohan import MyDataset
    return MyDataset(data_dir=data_dir,
                     data_parent_dir=data_parent_dir,
                     key_features=key_features,
                     key_class=key_class,
                     load_into_memory=load_into_memory)


def get_data_loader(dataset,
                    batch_size: int,
                    shuffle: bool,
                    drop_last: bool)  -> DataLoader:
    """Creates and returns a data loader.

    :param dataset: Dataset to use.
    :type dataset: dataset_class.MyDataset
    :param batch_size: Batch size to use.
    :type batch_size: int
    :param shuffle: Shuffle the data?
    :type shuffle: bool
    :return: Data loader, using the specified dataset.
    :rtype: torch.utils.data.DataLoader
    """
    from data_handling_prabu_mohan import MyDataset
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, 
                      drop_last=drop_last, num_workers=1)
    
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path+"/train")
        os.makedirs(path+"/test")
        os.makedirs(path+"/val")
        return True
    return False

def data_loading(train_dir,test_dir,val_dir,batch=10):
    train_loader=get_data_loader(get_dataset(train_dir,key_class='label'),batch,True,True)
    test_loader=get_data_loader(get_dataset(test_dir,key_class='label'),batch,True,True)
    val_loader=get_data_loader(get_dataset(val_dir,key_class='label'),batch,True,True)
    return {"train":train_loader,"test":test_loader,"val":val_loader}

# EOF
