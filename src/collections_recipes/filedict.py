import sqlite3, json, pickle    #, dbm
from abc import abstractmethod
from pathlib import Path
from operator import itemgetter
from functools import partial
from contextlib import suppress
from collections.abc import Mapping

from abc_recipes import ConfigClass, PropertyConfig
from contextlib_recipes import AbstractClosableContextManager

from .mapping_recipes import KeyMapDict, SerializerDict, SerializerDictToFile
from .abc_recipes import AutoSized, MutableMappingExtended
from .cr_typing import (
    SerializerToFile,
    PathType,
    Iterator,
    Callable,
    overload,
    PathStr,
    AnyStr,
    Any,
    IO,
    KT,
    VT,
)


__exclude_from_all__=set(dir())


class BaseFileDict(AutoSized, Mapping[KT,VT], ConfigClass, AbstractClosableContextManager):

    def __init__(self, path:PathType=PathStr("."), text_mode:bool=True) -> None:
        self._path = p = Path(path)
        if str(p)==".":
            self._path = p.resolve()
        self.text_mode = text_mode

    @property
    def path(self) -> Path:
        """Folder/File to hold this dict data"""
        return self._path

    @property
    def text_mode(self) -> bool:
        """Determine if the files are read/write in text or byte mode"""
        return self._text_mode

    @text_mode.setter
    def text_mode(self, value:bool):
        self._text_mode = bool(value)

    @property
    def config(self) -> dict[str,Any]:
        """configuration of this dict"""
        config = super().config
        config.update( path = self.path, text_mode=self.text_mode )
        return config

    @abstractmethod
    def close(self) -> None:
        """Close all open files that this class holds"""
        pass

    def _get_path(self, key:PathType) -> Path:
        """Return the path to key"""
        dirpath = self.path
        path = dirpath / key
        if path.parent == dirpath: #not allow paths shenaigants of accesing files in other folders
            return path
        else:
            raise KeyError(key)



class SQLDict(PropertyConfig, BaseFileDict[str,AnyStr], MutableMappingExtended):
    """
    Dictionary-like object with database back-end store.

    Concurrent and persistent.
    Easy to share with other programs
    Queryable
    Single file (easy to email and backup).

    This is a variation on Raymond Hettinger's original idea
    as presented in this video:
    https://www.youtube.com/watch?v=S_ipdVNSFlo
    """

    def __init__(self, *argv, **kwarg) -> None:
        super().__init__(*argv, **kwarg)
        self._conn = conn = sqlite3.connect(self.path)
        with conn as c:
            with suppress(sqlite3.OperationalError):
                c.execute( "CREATE TABLE Dict (key text, value text)" )
                c.execute( "CREATE UNIQUE INDEX key_index ON Dict (key)" )

    @property
    def conn(self) -> sqlite3.Connection:
        """The database conection"""
        return self._conn

    def __setitem__(self, key:str, value:AnyStr) -> None:
        if key in self:
            del self[key]
        with self.conn as c:
            c.execute( "INSERT INTO Dict VALUES (?,?)", (key, value) )

    def __getitem__(self, key:str) -> AnyStr:
        c = self.conn.execute( "SELECT value FROM Dict WHERE key=?", (key,) )
        row = c.fetchone()
        if row is None:
            raise KeyError(key)
        return row[0]

    def __delitem__(self, key:str) -> None:
        if key not in self:
            raise KeyError(key)
        with self.conn as c:
            c.execute( "DELETE FROM Dict WHERE key=?", (key,) )

    def __len__(self) -> int:
        return next(self.conn.execute("SELECT COUNT(*) FROM Dict"))[0]

    def __iter__(self) -> Iterator[str]:
        c = self.conn.execute("SELECT key FROM Dict")
        return map(itemgetter(0), c.fetchall())

    def close(self) -> None:
        self.conn.close()



class FileDict(PropertyConfig, BaseFileDict[str, str|bytes], MutableMappingExtended):
    """
    File based dictionary

    A dictionary-like object based on the file system rather than
    in-memory hash tables. It is persistent and sharable between
    proceses

    This is a variation on Raymond Hettinger's original idea
    as presented in this video:
    https://www.youtube.com/watch?v=S_ipdVNSFlo
    """

    def __init__(self, *argv, encoding:str="utf8", errors:str=None, **kwarg) -> None:
        super().__init__(*argv, **kwarg)
        self.path.mkdir(exist_ok=True)
        self.encoding   = encoding
        self.errors     = errors

    @property
    def encoding(self) -> str | None:
        """same meaning as the encoding optional argument of open"""
        return self._encoding

    @encoding.setter
    def encoding(self, value:str|None) -> None:
        if value is not None and not isinstance(value,str):
            raise ValueError("ecoding must be of type str or None")
        self._encoding = value

    @property
    def errors(self) -> str | None:
        """same meaning as the errors optional argument of open"""
        return self._errors

    @errors.setter
    def errors(self, value:str|None) -> None:
        if value is not None and not isinstance(value,str):
            raise ValueError("errors must be of type str or None")
        self._errors = value

    @property
    def config(self) -> dict[str,Any]:
        """configuration of this dict"""
        config = super().config
        config.update( encoding = self.encoding, errors = self.errors )
        return config

    def __getitem__(self, key:str) -> str | bytes:
        try:
            path = self._get_path(key)
            if self.text_mode:
                return path.read_text(self.encoding, self.errors)
            else:
                return path.read_bytes()
        except (FileNotFoundError, TypeError):
            raise KeyError(key) from None

    @overload
    def __setitem__(self, key:str, value:str) -> None:...
    @overload
    def __setitem__(self, key:str, value:bytes) -> None:...

    def __setitem__(self, key:str, value:AnyStr):
        path = self._get_path(key)
        if self.text_mode:
            path.write_text(value, self.encoding, self.errors) #type: ignore
        else:
            path.write_bytes(value)                            #type: ignore

    def __delitem__(self, key:str) -> None:
        try:
            path = self._get_path(key)
            path.unlink()
        except FileNotFoundError:
            raise KeyError(key) from None

    def __contains__(self, key:Any) -> bool:
        try:
            path = self._get_path(key)
            return path.is_file()
        except (KeyError, TypeError):
            return False

    def __iter__(self) -> Iterator[str]:
        return ( f.name for f in self.path.iterdir() if f.is_file() )

    def close(self) -> None:
        pass



class FileDictExt(KeyMapDict, FileDict):
    """
    File based dictionary

    A dictionary-like object based on the file system rather than
    in-memory hash tables. It is persistent and sharable between
    proceses.

    Handle only files with a given extention if any
    """

    def __init__(self, *arg, ext:str="", **karg) -> None:
        super().__init__(*arg,**karg)
        self.ext = ext

    @property
    def ext(self) -> str:
        """File extension of the files"""
        return self._ext

    @ext.setter
    def ext(self, value:str|None) -> None:
        if value is None or isinstance(value,str):
            if value and not value.startswith("."):
                raise ValueError("the extension must began with a '.' ex: '.txt' ")
            self._ext = value or ""
        else:
            raise TypeError("value for the ext must be a str")

    @property
    def config(self) -> dict[str,Any]:
        config = super().config
        config["ext"] = self.ext
        return config

    def keymap(self, key:Any) -> str:
        return f"{key}{self.ext}"

    def __iter__(self) -> Iterator[str]:
        it = super().__iter__()
        if (ext:=self.ext):
            n = -len(ext)
            return ( key[:n] for key in it if key.endswith(ext) )
        return it



class FileSerializerDict(SerializerDictToFile, FileDictExt):
    """A file dict that support serialesing data"""

    @property
    def config(self) -> dict[str,Any]:
        config = super().config
        config["serializer"] = self.serializer
        config["read_config"] = self.read_config
        config["write_config"] = self.write_config
        return config

    def _open_key(self, key, read:bool) -> Callable[[],IO[str]|IO[bytes]] | None:
        """
        Return a function to open a new or existing file to read or write
        if self.serializer is a SerializerToFile, otherwise return None
        """
        if not isinstance(self.serializer, SerializerToFile):
            return None
        config = {}
        if self.text_mode:
            config["mode"] = "r" if read else "w"
            if (encoding:=self.encoding):
                config["encoding"] = encoding
            if (errors:=self.errors):
                config["errors"] = errors
        else:
            config["mode"] = "rb" if read else "wb"
        return partial(open, self._get_path(self.keymap(key)), **config)



class FileDictJson(FileSerializerDict):
    """
    File based dictionary

    A dictionary-like object based on the file system rather than
    in-memory hash tables and store the values as a json.

    It is persistent and sharable between
    proceses
    """
    ext = ".json"
    serializer = json
    write_config = dict(sort_keys=True, indent=4)



class FileDictPickle(FileSerializerDict):
    """
    File based dictionary

    A dictionary-like object based on the file system rather than
    in-memory hash tables and store the values as a pickle.

    It is persistent and sharable between
    proceses
    """
    ext = ".pickle"
    serializer = pickle
    text_mode = False



class SerializerDictConfig(SerializerDict, ConfigClass):
    """Cooperative class to store its values in a serialize way"""

    @property
    def config(self) -> dict[str,Any]:
        config = super().config
        config["serializer"] = self.serializer
        config["read_config"] = self.read_config
        config["write_config"] = self.write_config
        return config



class SQLJsonDict(SerializerDictConfig, SQLDict):
    serializer = json



class SQLPickleDict(SerializerDictConfig, SQLDict):
    serializer = pickle
    text_mode = False



class FolderDict(BaseFileDict):
    """
    dictionary-like object for the subfolders of the given folder
    those subfolders will have for value a FileDict instaciate in that
    subfolder
    """

    def __init__(self, *argv, filedict_factory:Callable[[Path],Any]=FileDict, **karg):
        super().__init__(*argv, **karg)
        path = self.path
        if not path.exists():
            raise ValueError("This folder doesn't exist")
        if not path.is_dir():
            raise ValueError("This path isn't a folder")
        if not callable(filedict_factory):# or not isinstance(filedict_factory, FileDict):
            raise TypeError("filedict_factory is not a callable")
        self._filedict_factory = filedict_factory

    @property
    def filedict_factory(self) -> Callable[[Path],Any]:
        return self._filedict_factory

    @property
    def config(self) -> dict[str,Any]:
        config = super().config
        config["filedict_factory"] = self.filedict_factory
        return config

    def __getitem__(self, key:str) -> Any:
        path = self._get_path(key)
        if path.is_dir():
            item = self.filedict_factory(path)
            if isinstance(item, BaseFileDict):
                item.text_mode=self.text_mode
            return item
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (p.name for p in self.path.iterdir() if p.is_dir() )

    def __contains__(self, key:Any) -> bool:
        try:
            path = self._get_path(key)
            return path.is_dir()
        except (KeyError,TypeError):
            return False

    def close(self) -> None:
        pass


#TODO add a dbm dict, maybe?
#class DBMDict(MutableMapping):



__all__ = [ x for x in dir() if not (x.startswith("_") or x in __exclude_from_all__) ]
del __exclude_from_all__


