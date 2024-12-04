"""
Helper function and classes used in more than one Advent of Code puzzle

https://adventofcode.com

"""
from __future__ import annotations

from typing import NamedTuple, TypeVar, Iterable, Iterator, Hashable, Generic, Callable, TYPE_CHECKING
from math import copysign
from contextlib import contextmanager
import itertools
import sys
import tqdm


T = TypeVar("T")
X = TypeVar("X")
Y = TypeVar("Y")
P = TypeVar("P", bound=Hashable)
H = TypeVar("H", bound=Hashable)


__exclude_from_all__ = set(dir())

from .point_class import Point


BLACK_char = "█"
WHITE_char = '░'

progress_bar = tqdm.tqdm_gui if "idlelib" in sys.modules else tqdm.tqdm


DIRECCIONES = {
    "v": Point(1, 0),
    "^": Point(-1, 0),
    ">": Point(0, 1),
    "<": Point(0, -1),
    }
DIRECCIONES["+"] = tuple(DIRECCIONES[d] for d in "<>^v")
DIRECCIONES["x"] = tuple(DIRECCIONES[a] + DIRECCIONES[b] for a, b in ["<^", "<v", ">^", ">v"])
DIRECCIONES["*"] = DIRECCIONES["+"] + DIRECCIONES["x"]


def vecinos(point: Point, include_input: bool = False, direcciones: Iterable[Point] = DIRECCIONES["+"]) -> Iterator[Point]:
    """
    vecinos del punto dado en las 4 direcciones cardinales
    o las direcciones espesificadas
    """
    if include_input:
        yield point
    yield from (p+point for p in direcciones)


def is_valid(point: Point | complex, shape: tuple[int, int]) -> bool:
    """said if the given point fall within the given shape"""
    x, y = shape
    return 0 <= point.real < x and 0 <= point.imag < y


def make_vecinos(point: Point | complex, validator: Callable[[Point | complex], bool], include_input: bool = False, direcciones: str | tuple[Point] = "+", shape: tuple[int, int] = None) -> Iterator[Point]:
    for v in vecinos(point, include_input=include_input, direcciones=DIRECCIONES[direcciones] if isinstance(direcciones, str) else direcciones):
        if shape:
            if is_valid(v, shape) and validator(v):
                yield v
        else:
            if validator(v):
                yield v


def normalize(number: complex) -> complex:
    """
    return a complex number such that each component
    is 1 if said coordinate is non-zero
    preserving its sign
    """
    x, y = number.real, number.imag
    return complex(x and copysign(1, x), y and copysign(1, y))


def make_mirror_dict(data: Iterable[tuple[X, Y]]) -> dict[X | Y, Y | X]:
    """create a dict such that dict[k]==v and dict[v]==k"""
    return {k: v for a, b in data for k, v in [(a, b), (b, a)]}


def mirror_dict(data: dict[X, T]) -> dict[X | T, T | X]:
    return make_mirror_dict(data.items())


def mod1n(number: int, base: int) -> int:
    """A non zero modulos operator give results in [1,base]"""
    return number % base or base


def all_points2(shape: tuple[int, int]) -> Iterable[Point]:
    x, y = shape
    return itertools.starmap(Point, itertools.product(range(x), range(y)))


class PatternResult(NamedTuple, Generic[T]):
    non_periodic: T
    periodic: T
    

def find_pattern(iterable: Iterable[T], transform: Callable[[T], tuple[X, H]] = lambda x: (x, x)) -> PatternResult[list[X]]:
    non_periodic = []
    periodic = []
    memory = {}
    for i, (data, record) in enumerate(map(transform, iterable)):
        # data, record = transform(item)
        if record in memory:
            j = memory[record]
            periodic = non_periodic[j:]
            non_periodic = non_periodic[:j]
            break
        memory[record] = i
        non_periodic.append(data)
    return PatternResult(non_periodic, periodic)


def find_pattern_size(iterable: Iterable[T], key: Callable[[T], H] = None) -> PatternResult[int]:
    non_periodic = 0
    periodic = 0
    memory = {}
    for i, record in enumerate(iterable if key is None else map(key, iterable)):
        if record in memory:
            j = memory[record]
            periodic = non_periodic - j
            non_periodic = j
            break
        memory[record] = i
        non_periodic += 1
    return PatternResult(non_periodic, periodic)    
        

@contextmanager
def set_recursion_limit(limit):
    old = sys.getrecursionlimit()
    new = max(limit, old)
    try:
        sys.setrecursionlimit(new)
        yield
    finally:
        sys.setrecursionlimit(old)


def get_raw_data(path: str = "./input.txt") -> str:
    with open(path) as file:
        return file.read()


class DefaultValueDict[KT, VT](dict):

    def __init__(self, default_value: VT, /, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_value = default_value

    def __missing__(self, key) -> VT:
        return self.default_value


__all__ = [x for x in dir() if not (x.startswith("_") or x in __exclude_from_all__)]
del __exclude_from_all__

if TYPE_CHECKING:
    __all__ = [
        'BLACK_char',
        'DIRECCIONES',
        'PatternResult',
        'Point',
        'WHITE_char',
        'all_points2',
        'find_pattern',
        'find_pattern_size',
        'get_raw_data',
        'is_valid',
        'make_mirror_dict',
        'make_vecinos',
        'mirror_dict',
        'mod1n',
        'normalize',
        'progress_bar',
        'set_recursion_limit',
        'vecinos',
        'DefaultValueDict',
    ]
