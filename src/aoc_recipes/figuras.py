# https://adventofcode.com/2023/day/18
from __future__ import annotations
from typing import Iterable, Sequence, Self, TypeAlias
import itertools_recipes as ir
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
import operator
import abc
from .point_class import Point, dotproduct, NDVector, NDPoint, Point3


Vector: TypeAlias = Point
Vector3: TypeAlias = Point3


@dataclass
class Polygon:
    # first used on https://adventofcode.com/2023/day/18
    points: Sequence[Point]

    @property
    def area(self) -> int | Fraction:
        """ https://en.wikipedia.org/wiki/Shoelace_formula """
        points = self.points
        size = len(points)
        a = abs(sum(points[i].x * (points[(i+1) % size].y-points[i-1].y) for i in range(size)))
        if a % 2 == 0:
            return a//2
        return Fraction(a, 2)

    def __len__(self) -> int:
        """boundary points"""
        return sum(a.distance_t(b) for a, b in ir.pairwise(self.points))

    @property
    def interior_points(self) -> int | Fraction:
        """ https://en.wikipedia.org/wiki/Pick%27s_theorem """
        size = len(self)
        if size % 2 == 0:
            size //= 2
        else:
            size = Fraction(size, 2)
        return self.area + 1 - size


def make_polygon(data: Iterable[tuple[Point, int]], initial: Point = Point(0, 0)) -> Polygon:
    digger = initial
    result = [initial]
    for move, n in data:
        new = digger + move * n
        result.append(new)
        digger = new
    assert initial == result[-1], "is not a closed polygon"
    return Polygon(result)


@dataclass(frozen=True)
class NDRecta(abc.ABC):
    """
    abstract base class for a N-dimensions Recta or Line in parametric form
    """
    ini: NDPoint
    vector: NDVector

    @abc.abstractmethod
    def __contains__(self, item: NDPoint) -> bool:
        return False

    def __getitem__(self, time: int | slice) -> NDPoint | LineSegment:
        if isinstance(time, slice):
            return LineSegment(self, time)
        return self.ini + time*self.vector

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False
        if self.ini == other.ini and self.vector == other.vector:
            return True
        if are_codirectional(self.vector, other.vector):
            return self.ini in other and other.ini in self
        return False

    def __add__(self, other: NDVector) -> Self:
        if not isinstance(other, type(self.vector)):
            return NotImplemented
        new_ini = self.ini + other
        return type(self)(new_ini, self.vector)

    __radd__ = __add__

    def __sub__(self, other) -> Self:
        return self + (-other)

    __rsub__ = __sub__

    def __mul__(self, other: NDVector) -> Self:
        if not isinstance(other, type(self.vector)):
            return NotImplemented
        new_vector = self.vector * other
        return type(self)(self.ini, new_vector)

    __rmul__ = __mul__

    def find_time(self, value: NDPoint) -> int | Fraction:
        """find the value t such that: value = self.ini + t*self.vector"""
        if value not in self:
            raise ValueError("is not a point of this Recta")
        d = value - self.ini
        for a, b in zip(d, self.vector):
            if b:
                if a % b == 0:
                    return a//b
                else:
                    return Fraction(a, b)
        raise ValueError("Recta with null vector")


@dataclass(frozen=True, eq=False)
class Recta2(NDRecta):
    """2d Recta or line"""
    # first used on https://adventofcode.com/2023/day/24
    ini: Point
    vector: Vector

    @cached_property
    def coeff(self) -> tuple[int, int, int]:
        """return number a,b,c such that for any point in this Recta it satisfies that: ax + by + c = 0"""
        p1 = self.ini
        p2 = self.ini + self.vector
        d = p1-p2
        a = d.y
        b = -d.x
        c = p1.x * p2.y - p2.x * p1.y
        return a, b, c

    def __contains__(self, item) -> bool:
        if not isinstance(item, (Point, tuple)):
            return False
        x, y = item
        a, b, c = self.coeff
        return a*x + b*y + c == 0

    @classmethod
    def from_2_points(cls, ini: Point, fin: Point) -> Self:
        return cls(ini, fin-ini)


@dataclass(frozen=True, eq=False)
class Recta3(NDRecta):
    """3d Recta or line"""
    ini: Point3
    vector: Vector3

    @cached_property
    def m(self) -> Vector3:
        """ https://en.wikipedia.org/wiki/Pl%C3%BCcker_coordinates """
        return self.ini.cross_product(self.ini + self.vector)

    def __contains__(self, item: Point3) -> bool:
        if isinstance(item, Point3):
            return dotproduct(item, self.m) == 0
        return False


@dataclass
class LineSegment(Sequence):
    recta: NDRecta
    rango: range

    def __init__(self, recta: NDRecta, time: slice | int):
        self.recta = recta
        self.rango = range(time.start or 0, time.stop or 0, time.step or 1) if isinstance(time, slice) else range(time)

    def __len__(self):
        return len(self.rango)

    def __getitem__(self, item) -> NDPoint:
        if isinstance(item, int):
            return self.recta[self.rango[item]]
        raise IndexError(item)

    def __contains__(self, item: NDPoint) -> bool:
        if item in self.recta:
            return self.recta.find_time(item) in self.rango
        return False


def are_codirectional[T](a: NDVector[T], b: NDVector[T]) -> bool:
    # https://en.wikipedia.org/wiki/Dot_product
    ma = dotproduct(a, a)
    mb = dotproduct(b, b)
    ab = dotproduct(a, b)
    return ab**2 == ma * mb


def find_intersection(r1: Recta2, r2: Recta2) -> Point | Recta2 | None:
    if are_codirectional(r1.vector, r2.vector):
        if r2.ini in r1:
            return r1
        else:
            return None
    a1, b1, c1 = r1.coeff
    s1, s2, s3 = map(operator.add, r1.coeff, r2.coeff)
    x = Fraction(s3*b1 - s2*c1, s2*a1 - s1*b1)
    y = (-a1*x - c1)/b1
    if x.denominator == 1:
        x = x.numerator
    if y.denominator == 1:
        y = y.numerator
    return Point(x, y)
