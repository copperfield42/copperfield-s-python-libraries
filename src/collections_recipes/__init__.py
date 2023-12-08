import typing as _typing

from collections import *
from collections import abc
from . import abc_recipes, mapping_recipes
from .filedict import *
from .collections_recipes import *

# 5/6/2022-14/7/2022 tested with mypy


if _typing.TYPE_CHECKING:  # to please mypy and so it correctly see the content of the module
    __all__ = [
        'LastUpdatedOrderedDict',
        'SerializerDictConfig',
        'collections_recipes',
        'FileSerializerDict',
        'LineSeekableFile',
        'MultiHitLRUCache',
        'constant_factory',
        'mapping_recipes',
        'FileDictPickle',
        'OrderedCounter',
        'SortedSequence',
        'TimeBoundedLRU',
        'moving_average',
        'SQLPickleDict',
        'BaseFileDict',
        'DeepChainMap',
        'FileDictJson',
        'ListBasedSet',
        'FileDictExt',
        'OrderedDict',
        'SQLJsonDict',
        'abc_recipes',
        'defaultdict',
        'FolderDict',
        'OrderedSet',
        'UserString',
        'delete_nth',
        'namedtuple',
        'roundrobin',
        'RangedSet',
        'chr_range',
        'cr_typing',
        'ChainMap',
        'ChainSet',
        'FileDict',
        'UserDict',
        'UserList',
        'filedict',
        'Counter',
        'SQLDict',
        'BitSet',
        'deque',
        'tail',
        'LRU',
        'abc'
    ]
