"""
Helper function and classes used in more than one Advent of Code puzzle

https://adventofcode.com

"""
from __future__ import annotations

from typing import NamedTuple
from math import copysign
from operator import sub
import tqdm, sys


__exclude_from_all__=set(dir())

def get_raw_data(path:str="./input.txt") -> str:
    with open(path) as file:
        return file.read()



class Point(NamedTuple):
    """2d point"""
    x: int
    y: int

    def __add__(self, otro:Point) -> Point:
        x,y = self
        if isinstance(otro,type(self)):
            ox, oy = otro
        elif isinstance(otro, (int,float)):
            ox, oy = otro,otro
        else:
            return NotImplemented
        return type(self)(x+ox, y+oy)

    def __radd__(self, otro:Point) -> Point:
        return self + otro

    def __neg__(self) -> Point:
        x,y = self
        return type(self)(-x, -y)

    def __sub__(self, otro:Point) -> Point:
        return self + (-otro)

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
        

direcciones={
    "R":Point(1,0),
    "L":Point(-1,0),
    "U":Point(0,1),
    "D":Point(0,-1),
    }
    

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




__all__ = [ x for x in dir() if not (x.startswith("_") or x in __exclude_from_all__) ]
del __exclude_from_all__

