from os import PathLike
import sys


__exclude_from_all__=set(dir())

from typing import TYPE_CHECKING

if not TYPE_CHECKING:
    from typing import *
    from typing_extensions import *
else:
    from typing import (
        no_type_check_decorator,
        AsyncContextManager,
        runtime_checkable,
        MutableSequence,
        SupportsComplex,
        ParamSpecKwargs,
        ContextManager,
        MutableMapping,
        AsyncGenerator,
        get_type_hints,
        AsyncIterator,
        AsyncIterable,
        SupportsBytes,
        SupportsFloat,
        SupportsIndex,
        SupportsRound,
        no_type_check,
        ParamSpecArgs,
        #TYPE_CHECKING,
        is_typeddict,
        Concatenate,
        AbstractSet,
        MappingView,
        SupportsAbs,
        SupportsInt,
        DefaultDict,
        OrderedDict,
        ForwardRef,
        ByteString,
        MutableSet,
        ValuesView,
        Collection,
        Reversible,
        NamedTuple,
        get_origin,
        Annotated,
        ParamSpec,
        Container,
        ItemsView,
        Awaitable,
        Coroutine,
        FrozenSet,
        TypedDict,
        Generator,
        TypeAlias,
        TypeGuard,
        Callable,
        ClassVar,
        Optional,
        Protocol,
        Hashable,
        Iterable,
        Iterator,
        KeysView,
        Sequence,
        ChainMap,
        BinaryIO,
        get_args,
        NoReturn,
        overload,
        Generic,
        Literal,
        TypeVar,
        Mapping,
        Counter,
        Pattern,
        NewType,
        TextIO,
        AnyStr,
        Final,
        Tuple,
        Union,
        Sized,
        Deque,
        Match,
        final,
        Type,
        Dict,
        List,
        cast,
        Text,
        Any,
        Set,
        IO
    )

    from typing_extensions import (
        dataclass_transform,
        clear_overloads,
        LiteralString,
        get_overloads,
        TypeVarTuple,
        assert_never,
        assert_type,
        reveal_type,
        NotRequired,
        Required,
        runtime,
        Unpack,
        IntVar,
        Never,
        Self
    )


from types import TracebackType



import numerary #work around module for numeric type checking https://github.com/python/mypy/issues/3186
from numerary import RealLike, IntegralLike
from numerary.types import RationalLike, CachingProtocolMeta, Protocol as NProtocol





# Some unconstrained type variables.  These are used by the container types.
T = TypeVar('T')  # Any type.
KT = TypeVar('KT')  # Key type.
VT = TypeVar('VT')  # Value type.
T_co = TypeVar('T_co', covariant=True)  # Any type covariant containers.
V_co = TypeVar('V_co', covariant=True)  # Any type covariant containers.
KT_co = TypeVar('KT_co', covariant=True)  # Key type covariant containers.
VT_co = TypeVar('VT_co', covariant=True)  # Value type covariant containers.
T_contra = TypeVar('T_contra', contravariant=True)  # Ditto contravariant.
# Internal type variable used for Type[].
CT_co = TypeVar('CT_co', covariant=True, bound=type)


#P = ParamSpec('P') #ParamSpec not defined in the module where it is used are buggy
                    #https://github.com/python/mypy/issues/13099 https://github.com/python/mypy/issues/12475
X = TypeVar('X')
Y = TypeVar('Y')
Z = TypeVar('Z')
S = TypeVar('S')




if sys.version_info >= (3, 11):
    @runtime_checkable
    class ComplexLike(
        numerary.types.SupportsAbs[T_co],
        SupportsComplex,
        numerary.types.SupportsConjugate,
        numerary.types.SupportsComplexOps[T_co],
        numerary.types.SupportsRealImag,
        numerary.types.SupportsComplexPow,
        NProtocol[T_co],
        metaclass=CachingProtocolMeta,
    ):
        pass
else:
    @runtime_checkable
    class ComplexLike(
        numerary.types.SupportsAbs[T_co],
        #SupportsComplex,
        numerary.types.SupportsConjugate,
        numerary.types.SupportsComplexOps[T_co],
        numerary.types.SupportsRealImag,
        numerary.types.SupportsComplexPow,
        NProtocol[T_co],
        metaclass=CachingProtocolMeta,
    ):
        pass


NumberLike = Union[IntegralLike, RealLike]


PathStr = NewType("PathStr", str)
PathType:TypeAlias = Union[PathStr, PathLike, str]




@runtime_checkable
class OrderableBasic(Protocol[T_co]):
    """Protocol for orderables objects"""

    def __eq__(self, otro:Any) -> bool:
        """self == otro"""
        ...

    def __lt__(self, otro:Any) -> bool:
        """self < otro"""
        ...


@runtime_checkable
class Orderable(OrderableBasic[T_co], Protocol[T_co]):

    def __le__(self, otro:Any) -> bool:
        """self <= otro"""
        ...

    def __gt__(self, otro:Any) -> bool:
        """self > otro"""
        ...

    def __ge__(self, otro:Any) -> bool:
        """self >= otro"""
        ...


@runtime_checkable
class Closable(Protocol):

    def close(self) -> None:
        ...


@runtime_checkable
class Logger(Protocol):

    def warning(self, msj:str) -> None:
        ...


class SentinelObject(object):
    """
    Class to use as sentinel when None is a valid value input,
    with a representation of "<...>"

    use:
    _sentinel = SentinelObject()

    def fun(x:Any=_sentinel):
       if x is _sentinel:
           ...
       else:
           ...

    >>> help(fun)
    Help on function fun in module __main__:

    fun(x: Any = <...>)

    >>>
    """

    def __repr__(self) -> str:
        return "<...>"


@runtime_checkable
class SupportsRead(Protocol[T_co]):
    def read(self, __length: int = ...) -> T_co: ...


__all__ = [ x for x in dir() if not (x.startswith("_") or x in __exclude_from_all__) ]
del __exclude_from_all__


