from typing_recipes import (
    runtime_checkable,
    TracebackType,
    TYPE_CHECKING,
    SupportsWrite,
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
    Generic,
    AnyStr,
    Self,
    Any,
    KT,
    VT,
    IO,
    T,
)


MOD: TypeAlias = int
DIV: TypeAlias = int



@runtime_checkable
class SerializerToFile(Protocol):
    def dump(self, obj: Any, fp: SupportsWrite[str | bytes], **kwargs: Any ) -> None: ...
    def load(self, fp: SupportsRead[str | bytes], **kwargs: Any ) -> Any: ...


@runtime_checkable
class SerializerToString(Protocol):
    def dumps(self, obj: Any, **kwargs: Any) -> str: ...
    def loads(self, s: str, **kwargs: Any) -> Any: ...


@runtime_checkable
class SerializerToBytes(Protocol):
    def dumps(self, obj: Any, **kwargs: Any) -> bytes: ...
    def loads(self, s: bytes, **kwargs: Any) -> Any: ...



@runtime_checkable
class Serializer(SerializerToString, SerializerToFile, Protocol):
    pass


