from typing_recipes import (
    runtime_checkable,
    TracebackType,
    TYPE_CHECKING,
    SupportsRead,
    NumberLike,
    TypeAlias,
    Orderable,
    Iterator,
    Iterable,
    overload,
    Callable,
    Protocol,
    PathType,
    TypeVar,
    PathStr,
    AnyStr,
    Any,
    KT,
    VT,
    IO,
    T,
)


MOD :TypeAlias = int
DIV :TypeAlias = int


@runtime_checkable
class SerializerToString(Protocol):
    def dumps(self, value:Any, **kargv) -> str: ...
    def loads(self, value:str | bytes, **kargv) -> Any:...


@runtime_checkable
class SerializerToFile(Protocol):
    def dump(self, value:Any, file:IO[str]|IO[bytes], **kargv) -> None: ...
    def load(self, file:SupportsRead[str|bytes], **kargv) -> Any:...


@runtime_checkable
class Serializer(SerializerToString, SerializerToFile, Protocol):
    pass


