from typing import (
    SupportsIndex,
    TypeAlias,
    Container,
    Callable,
    Iterable,
    Iterator,
    Sequence,
    Optional,
    Hashable,
    overload,
    Tuple,
    cast,
    Any,
    Set,
)

from typing_recipes import (
    SentinelObject,
    IntegralLike,
    NumberLike,
    Orderable,
    RealLike,
    T,
    S,
    X,
    Y,
)


Function1:TypeAlias  = Callable[[Any],Any]
Function2:TypeAlias  = Callable[[Any,Any],Any]
Contenedor:TypeAlias = Callable[[Iterable[Any]],Container[Any]]

















