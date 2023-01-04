"""
Helper function and classes used in more than one Advent of Code puzzle

https://adventofcode.com

"""
from __future__ import annotations

from typing import NamedTuple, TypeVar, Iterable, Iterator, Hashable, Any
from math import copysign
from heapq import heappush, heappop
from operator import sub, eq, add
from functools import partial
from dataclasses import dataclass, field
import itertools
import sys
import numpy
import tqdm



__exclude_from_all__=set(dir())


T = TypeVar("T")
X = TypeVar("X")
Y = TypeVar("Y")
P = TypeVar("P",bound=Hashable)
Grafo = TypeVar("Grafo")


def get_raw_data(path:str="./input.txt") -> str:
    with open(path) as file:
        return file.read()


class Point(NamedTuple):
    """2d point"""
    x: int
    y: int

    @property
    def real(self):
        return self.x

    @property
    def imag(self):
        return self.y

    def __add__(self, otro:Point) -> Point:
        "self + otro"
        x,y = self
        if isinstance(otro,(type(self),tuple)):
            ox, oy = otro
        elif isinstance(otro, (int,float)):
            ox, oy = otro,otro
        elif isinstance(otro, complex):
            ox, oy = self.from_complex(otro)
        else:
            return NotImplemented
        return type(self)(x+ox, y+oy)

    __radd__ = __add__

    def __neg__(self) -> Point:
        "-self"
        x,y = self
        return type(self)(-x, -y)

    def __sub__(self, otro:Point) -> Point:
        "self - otro"
        return self + (-otro)

    def __mul__(self, otro):
        "self * otro"
        if isinstance(otro, (int,float)):
            return type(self)(otro*self.x, otro*self.y)
        if isinstance(otro, complex):
            return self.from_complex( complex(self)*otro )
        return NotImplemented
    
    __rmul__ = __mul__

    def __mod__(self, otro:int|Point) -> Point:
        """self % otro
           if otro is a number apply the mod to each coordinate
           if otro is a Point apply the mod point-wise ( Point(self.x % otro.x, self.y % otro.y) )
        """
        if isinstance(otro, int):
            return type(self)(self.x%otro, self.y%otro)
        if isinstance(otro, (tuple,type(self))):
            ox,oy = otro
            return type(self)(self.x%ox, self.y%oy)
        return NotImplemented

    def normalize(self) -> Point:
        """
        return a Point such that each coordinate
        is 1 if said coordinate is non zero
        preserving its sign
        """
        x,y = self
        return type(self)(x and int(copysign(1,x)),y and int(copysign(1,y)))

    def distance_t(self, otro:Point) -> int:
        """
        The taxicab distance between two points
        https://en.wikipedia.org/wiki/Taxicab_geometry
        """
        if not isinstance(otro, type(self)):
            raise ValueError("No es un Point")
        x1,y1 = self
        x2,y2 = otro
        return abs(x1-x2) + abs(y1-y2)

    def __complex__(self) -> complex:
        return complex(self.x, self.y)

    @classmethod
    def from_complex(cls, number:complex) -> Point:
        return cls(int(number.real), int(number.imag))

    def __divmod__(self, otro:int|Point) -> tuple[Point,Point]:
        "divmod(self, otro)"
        if isinstance(otro, int):
            x,y = self
            dx,mx = divmod(x,otro)
            dy,my = divmod(y,otro)
            return Point(dx,dy),Point(mx,my)
        if isinstance(otro,(tuple,type(self))):
            x,y = self
            ox,oy = otro
            dx,mx = divmod(x,ox)
            dy,my = divmod(y,oy)
            return Point(dx,dy),Point(mx,my)
        return NotImplemented

    def __rdivmod__(self, otro:int|Point) -> tuple[Point,Point]:
        "divmod(otro, self)"
        if isinstance(otro, int):
            x,y = self
            dx,mx = divmod(otro,x)
            dy,my = divmod(otro,y)
            return Point(dx,dy),Point(mx,my)
        if isinstance(otro,(tuple,type(self))):
            x,y = self
            ox,oy = otro
            dx,mx = divmod(ox,x)
            dy,my = divmod(oy,y)
            return Point(dx,dy),Point(mx,my)
        return NotImplemented        


DIRECCIONES={
    "R":Point(1,0),
    "L":Point(-1,0),
    "U":Point(0,1),
    "D":Point(0,-1),
    "v":Point(1,0),
    "^":Point(-1,0),
    ">":Point(0,1),
    "<":Point(0,-1),
    "+":(Point(1,0),Point(-1,0),Point(0,1),Point(0,-1)),
    "x":(Point(1,1),Point(-1,-1),Point(-1,1),Point(1,-1)),
    }
DIRECCIONES["*"] = DIRECCIONES["+"]+DIRECCIONES["x"]


def vecinos(point:Point, include_input:bool=False, direcciones:Iterable[Point]=DIRECCIONES["+"] ) -> Iterator[Point]:
    """vecinos del punto dado en las 4 direcciones cardinales"""
    if include_input:
        yield point
    yield from (p+point for p in direcciones)


def is_valid(point:Point|complex, shape:tuple[int,int]) -> bool:
    """said if the given point fall within the wiven shape"""
    x,y = shape
    return 0<=point.real<x and 0<=point.imag<y


def make_vecinos(point:Point|complex, validator:Callable[[Point|complex],bool], include_input:bool=False, direcciones:str|tuple[Point]="+") -> Iterator[Point]:
    for v in vecinos(point,include_input=include_input, direcciones=DIRECCIONES[direcciones] if isinstance(direcciones,str) else direcciones ):
        if validator(v):
            yield v


class PriorityQueue:

    def __init__(self):
        self._pq = []                         # list of entries arranged in a heap
        self._entry_finder = {}               # mapping of tasks to entries
        self._REMOVED = '<removed-task>'      # placeholder for a removed task
        self._counter = itertools.count()     # unique sequence count

    def __bool__(self):
        return bool(self._entry_finder)

    def __repr__(self):
        return f"<{type(self).__name__}({list(self._entry_finder.keys())})>"

    def __len__(self):
        return len(self._entry_finder)

    def add_task(self, task, priority:int=0):
        'Add a new task or update the priority of an existing task'
        if task in self._entry_finder:
            self.remove_task(task)
        count = next(self._counter)
        entry = [priority, count, task]
        self._entry_finder[task] = entry
        heappush(self._pq, entry)

    def remove_task(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self._entry_finder.pop(task)
        entry[-1] = self._REMOVED

    def pop_task(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        pq = self._pq
        while pq:
            priority, count, task = heappop(pq)
            if task is not self._REMOVED:
                del self._entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')


@dataclass(order=True, eq=True, frozen=True)
class PrioritizedItem:
    priority: int
    item:Any = field(compare=False)


def where(condition:numpy.ndarray[bool,...]) -> Iterator[tuple[int,...]]:
    return zip(*numpy.where(condition))


def cost_plus_one(point1:Any, point2:Any, old_cost:int, tablero:Any) -> int:
    return old_cost+1


def shortest_path_grafo(
    inicio:P,
    meta:P|Callable[[P],bool],
    tablero:Grafo,
    neighbors:Callable[[P,Grafo],Iterable[P]],
    cost:Callable[[P,P,int,Grafo],int]=cost_plus_one,
    initial_cost:int=0) -> tuple[int,P]:
    #https://www.youtube.com/watch?v=sBe_7Mzb47Y
    visitado = set()
    if not callable(meta):
        meta = partial(eq,meta)
    queue = PriorityQueue()
    queue.add_task( (initial_cost,inicio), initial_cost )
    while queue:
        steps, p = queue.pop_task()
        if p in visitado:
            continue
        visitado.add(p)
        if meta(p):
            return steps,p
        for v in neighbors(p,tablero):
            new_steps = cost(p,v,steps,tablero)
            queue.add_task( (new_steps,v), new_steps )
    return float("inf"),None


def normalize(number:complex) -> complex:
    """
    return a complex number such that each component
    is 1 if said coordinate is non zero
    preserving its sign
    """
    x,y = number.real, number.imag
    return complex(x and copysign(1,x), y and copysign(1,y))



BLACK_char = "█"
WHITE_char = '░'


progress_bar = tqdm.tqdm_gui if "idlelib" in sys.modules else tqdm.tqdm


def make_mirror_dict(data:Iterable[tuple[X,Y]]) -> dict[X|Y,Y|X]:
    """create a dict such that dict[k]==v and dict[v]==k"""
    return {k:v for a,b in data for k,v in [(a,b),(b,a)]}


def mirror_dict(data:dict[X,T]) -> dict[X|T,T|X]:
    return make_mirror_dict(data.items())


__all__ = [ x for x in dir() if not (x.startswith("_") or x in __exclude_from_all__) ]
del __exclude_from_all__

