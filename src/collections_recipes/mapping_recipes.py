from abc import abstractmethod
from collections.abc import MutableMapping, Mapping, Hashable
from contextlib import AbstractContextManager
from .cr_typing import (
    SerializerToString,
    SerializerToFile,
    TracebackType,
    Serializer,
    Iterator,
    Callable,
    Any,
    KT,
    VT,
    IO,
)



__exclude_from_all__=set(dir())

class BufferDict(MutableMapping[KT,VT], AbstractContextManager):
    """
    Helper class to use the FileDict or SQLDict from collections_recipes.filedict
    (or similar class that can't reflect inside it changes to its mutables
    elements after they were stored but changed after that ) more like a
    regular dict such that changes to the elements in those by mutating
    the elements are preserverd in the FileDict or SQLDict

    >>> fdj=FileDictJson(".")
    >>> fdj["test"]=[]
    >>> lst=fdj["test"]
    >>> lst.extend(range(10))
    >>> fdj["test"]
    []
    >>> lst
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>>
    >>> with BufferDict(fdj) as mydict:
           mydict["test2"]=[]
           lst=mydict[key]
           lst.extent(range(10))

    >>> fdj["test2"]
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>>
    """

    def __init__(self, data:MutableMapping[KT,VT]) -> None:
        if not isinstance(data,MutableMapping):
            raise ValueError("A dictionary-like object is required")
        self._data   = data
        self._buffer:dict[KT,VT] = {}

    def __len__(self) -> int:
        return len(self._data.keys() | self._buffer.keys())

    def __getitem__(self, key:KT) -> VT:
        try:
            return self._buffer[key]
        except KeyError:
            pass
        try:
            value = self._data[key]
            self._buffer[key]=value
            return value
        except KeyError:
            raise KeyError(key) from None

    def __setitem__(self, key:KT, value:VT) -> None:
        self._data[key]=value
        self._buffer[key]=value

    def __delitem__(self, key:KT) -> None:
        errors=0
        try:
            del self._data[key]
        except KeyError:
            errors += 1
        try:
            del self._buffer[key]
        except KeyError:
            errors += 1
        if errors>1:
            raise KeyError(key) from None

    def __contains__(self, key:Any) -> bool:
        return key in self._buffer or key in self._data

    def sync(self) -> None:
        """
        save into the buffered dict the content on the buffer
        """
        self._data.update(self._buffer)

    def __exit__(self, exc_type:type[BaseException]|None, exc_value:BaseException|None, traceback:TracebackType|None) -> None:
        self.sync()

    def __iter__(self) -> Iterator[KT]:
        return iter(self._data.keys() | self._buffer.keys())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._data!r})"

    def clearbuffer(self) -> None:
        self._buffer.clear()


class MissingDict(Mapping[KT,VT]):
    """
    Coperative class to add the __missing__ interface
    """

    @abstractmethod
    def __missing__(self, key:KT) -> VT:
        raise KeyError(key)

    def __getitem__(self, key:KT) -> VT:
        try:
            return super().__getitem__(key)
        except KeyError:
            return self.__missing__(key)


class ValueMapDict(MutableMapping[KT,VT]):
    """
    Coperative class that apply a transformation to the store value
    before deliveriong/store it

    Example:
    >>> class LenValueDict(ValueMapDict, collections.UserDict):
           "dict that only store the first n elements of value"
           def __init__(self, valuelen, *arg, **karg):
               super().__init__(*arg, **karg)
               self.valuelen = valuelen
           def valuemapwrite(self, value):
               return value[:self.valuelen]
           def valuemapread(self, value):
               return value


    >>> a = LenValueDict(3)
    >>> a[1] = [1,2,3,4,5,6]
    >>> a[1]
    [1, 2, 3]
    >>>
    """

    @abstractmethod
    def valuemapwrite(self, value:VT) -> Any:
        """
        Transformation function for the value to store it
        """
        return value

    @abstractmethod
    def valuemapread(self, value:Any) -> VT:
        """
        Transformation function for the value to read it
        """
        return value

    def __getitem__(self,key:KT) -> VT:
        return self.valuemapread( super().__getitem__(key) )

    def __setitem__(self, key:KT, value:VT) -> None:
        super().__setitem__(key, self.valuemapwrite(value) )


class KeyMapDict(MutableMapping[KT,VT]):
    """
    Coperative class that apply a transformation to the key
    before store/consult it

    the __iter__ method should apply the inverse of keymap if nessesary

    Example:
    >>> class CaseInsensitiveDict(KeyMapDict, collections.UserDict):
           def keymap(self,key:str):
                 return key.lower()


    >>> a = CaseInsensitiveDict()
    >>> a["fun"] = 1
    >>> "FUN" in a
    True
    >>> a
    {'fun': 1}
    >>>


    """

    @abstractmethod
    def keymap(self, key:KT) -> Any:
        """
        Transformation function for the key to store/consult
        """
        return key

    def __getitem__(self, key:KT) -> VT:
        return super().__getitem__( self.keymap(key) )

    def __setitem__(self, key:KT, value:VT) -> None:
        super().__setitem__( self.keymap(key), value )

    def __delitem__(self, key:KT) -> None:
        super().__delitem__( self.keymap(key) )

    def __contains__(self, key:Any) -> bool:
        try:
            tkey = self.keymap(key)
        except (TypeError, ValueError):
            return False
        return super().__contains__( tkey )


class SerializerDict(MutableMapping[KT,VT]):
    """
    Cooperative class to store its values in a serialize way
    """

    def __init__(self, *arg, serializer:Serializer|None=None, read_config:dict[str,Any]|None=None, write_config:dict[str,Any]|None=None, **karg) -> None:
        self.serializer = serializer
        self.read_config = read_config    #type: ignore
        self.write_config = write_config  #type: ignore
        super().__init__(*arg, **karg)

    @property
    def serializer(self) -> Serializer | None:
        """
        modulo/class to serialize data,
        it must have .dumps and .loads functions/methods.
        """
        return self._serializer

    @serializer.setter
    def serializer(self, value:Serializer|None) -> None:
        if value and not isinstance(value, SerializerToString):
            raise ValueError(f"Isn't a valid {SerializerToString}")
        self._serializer = value

    @property
    def read_config(self) -> dict[str,Any]:
        """
        keyword arguments for self.serializer.loads
        """
        return self._read_config

    @read_config.setter
    def read_config(self, value:dict[str,Any]|None) -> None:
        if value and not isinstance(value,Mapping):
            raise TypeError("Not a valid configuration dictionary")
        self._read_config:dict[str,Any] = dict(value or () )

    @property
    def write_config(self) -> dict[str,Any]:
        """
        keyword arguments for self.serializer.dumps
        """
        return self._write_config

    @write_config.setter
    def write_config(self, value:dict[str,Any]|None) -> None:
        if value and not isinstance(value,Mapping):
            raise TypeError("Not a valid configuration dictionary")
        self._write_config:dict[str,Any] = dict(value or () )

    def __getitem__(self, key:KT) -> VT:
        value = super().__getitem__(key)
        if (s:=self.serializer):
            return s.loads(value, **self.read_config) #type: ignore
        return value

    def __setitem__(self, key:KT, value:VT) -> None:
        if (s:=self.serializer):
            value = s.dumps(value, **self.write_config)  #type: ignore
        super().__setitem__(key, value)


class SerializerDictToFile(SerializerDict[KT,VT]):
    """
    Cooperative class to store its values in a serialize way,
    that support doing so directly into files.
    """

    @abstractmethod
    def _open_key(self, key:KT, read:bool) -> Callable[[],IO[str]|IO[bytes]] | None:
        """
        Return a function to open a new or existing file to read or write
        if self.serializer is a SerializerToFile, otherwise return None
        """
        if not isinstance(self.serializer, SerializerToFile):
            return None
        raise NotImplementedError("SerializerDictToFile._open_key")

    def __getitem__(self, key:KT) -> VT:
        if (key_open:=self._open_key(key, True)):
            try:
                with key_open() as file:
                    return self.serializer.load(file, **self.read_config) #type: ignore #not going to worsen the code with an unnesesary check for non-None just to please mypy
            except FileNotFoundError:
                raise KeyError(key) from None
        return super().__getitem__(key)

    def __setitem__(self, key:KT, value:VT) -> None:
        if (key_open:= self._open_key(key, False)):
            with key_open() as file:
                return self.serializer.dump(value, file, **self.write_config) #type: ignore
        super().__setitem__(key, value)




class HashableDict(Mapping, Hashable):
    """A Hashable dict by its keys"""
    #https://stackoverflow.com/questions/1151658/python-hashable-dicts

    def __hash__(self) -> int:
        return hash(frozenset(self))


class HashableValueDict(Mapping, Hashable):
    """A Hashable dict by its keys and values"""
    #https://stackoverflow.com/questions/1151658/python-hashable-dicts

    def __hash__(self) -> int:
        return hash((frozenset(self),frozenset(self.values())))


__all__ = [ x for x in dir() if not (x.startswith("_") or x in __exclude_from_all__) ]
del __exclude_from_all__