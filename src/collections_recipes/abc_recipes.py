from .cr_typing import (
    overload as _overload,
    TYPE_CHECKING as _TC,
    TypeVar as _TypeVar,
    T as _T,
)

if not _TC:
    from collections.abc import *
else:
    from collections.abc import (
        MutableSequence,
        AsyncGenerator,
        MutableMapping,
        AsyncIterable,
        AsyncIterator,
        MappingView,
        Reversible,
        Collection,
        MutableSet,
        ValuesView,
        ByteString,
        Awaitable,
        Coroutine,
        Generator,
        Container,
        ItemsView,
        Hashable,
        Iterable,
        Iterator,
        Callable,
        KeysView,
        Sequence,
        Mapping,
        Sized,
        Set,
)




class AutoSized(Sized, Iterable):
    """
    mix-in class for iterables classes that provide a default
    implmentation of __len__ and __bool__
    """

    __slots__ = ()

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __bool__(self) -> bool:
        return any(True for _ in self)


class MutableMappingExtended(AutoSized, MutableMapping):

    __slots__ = ()

    def clear(self) -> None:
        while True:
            try:
                del self[next(iter(self))]
            except (StopIteration, KeyError):
                return


_Self = _TypeVar("_Self", bound="SequenceSliceView")

class SequenceSliceView(Sequence[_T]):
    """View over an slice of a Sequence"""

    __slots__ = '_sequence', '_indices'

    def __init__(self, original:Sequence[_T], slice_index:slice) -> None:
        self._sequence = original
        self._indices = range(len(original))[slice_index]

    def __len__(self) -> int:
        return len(self._indices)

    @_overload
    def __getitem__(self:_Self, index:int) -> _T:...
    @_overload
    def __getitem__(self:_Self, index:slice) -> _Self:...

    def __getitem__(self:_Self, index:int|slice) -> _T | _Self:
        if isinstance(index, slice):
            return type(self)(self, index)
        try:
            return self._sequence[self._indices[index]]
        except IndexError:
            raise IndexError(f"{type(self).__name__} index out of range") from None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._sequence!r}, {repr(self._indices).replace('range','slice')})"


class SlicedSequence(Sequence[_T]):
    """
    mix-in class to add slice support to a Sequence class
    through a view on the original sequence
    """

    __slots__ = ()

    @_overload
    def __getitem__(self, index:int) -> _T:...
    @_overload
    def __getitem__(self, index:slice) -> SequenceSliceView[_T]:...

    def __getitem__(self, index:int|slice) -> _T | SequenceSliceView[_T]:
        if isinstance(index, slice):
            return SequenceSliceView(self, index)
        return super().__getitem__(index)




