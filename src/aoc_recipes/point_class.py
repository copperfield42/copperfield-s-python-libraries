"""
Helper function and classes used in more than one Advent of Code puzzle

https://adventofcode.com

"""
from __future__ import annotations

from typing import NamedTuple, Self, Sequence
from math import copysign
from numbers import Number, Complex, Real


__exclude_from_all__ = set(dir())

type NDVector[T] = Sequence[T] | Point | Point3  # point with any number of dimensions
type NDPoint[T] = Sequence[T] | Point | Point3  # vector with any number of dimensions


def taxicab_distance[T](a: NDPoint[T], b: NDPoint[T]) -> T:
    """
    The taxicab distance between two points
    https://en.wikipedia.org/wiki/Taxicab_geometry
    """
    return sum(abs(e1-e2) for e1, e2 in zip(a, b, strict=True))


def normalize_point[T](point: NDPoint[T]) -> NDPoint[T]:
    """
    return a Point such that each coordinate
    is 1 if said coordinate is non zero
    preserving its sign
    """
    return type(point)(*(e and int(copysign(1, e)) for e in point))


def dotproduct[T](v1: NDVector[T], v2: NDVector[T]) -> T:
    """ https://en.wikipedia.org/wiki/Dot_product """
    return sum(a*b for a, b in zip(v1, v2, strict=True))


class Point(NamedTuple):
    """2d point/vector"""
    x: int
    y: int

    @property
    def real(self) -> int:
        return self.x

    @property
    def imag(self) -> int:
        return self.y

    def __add__(self, other: Point | tuple[int, int] | complex | int) -> Self:
        """self + other"""
        x, y = self
        if isinstance(other, (type(self), tuple)):
            ox, oy = other
        elif isinstance(other, Real):
            ox, oy = other, 0
        elif isinstance(other, Complex):
            ox, oy = self.from_complex(other)
        elif isinstance(other, Number):
            ox, oy = other, 0
        else:
            return NotImplemented
        return type(self)(x + ox, y + oy)

    __radd__ = __add__

    def __neg__(self) -> Self:
        """-self"""
        x, y = self
        return type(self)(-x, -y)

    def __sub__(self, other: Point | complex | int) -> Self:
        """self - other"""
        return self + (-other)

    def __rsub__(self, other: Point | complex | int) -> Self:
        """other - self"""
        return (type(self)(0, 0) + other) + (-self)

    def __mul__(self, other: Point | tuple[int, int] | complex | int) -> Self:
        """
        self * other

        points multiply with one another like complex numbers
        """
        new = type(self)
        x, y = self
        if isinstance(other, (type(self), tuple)):
            ox, oy = other
        elif isinstance(other, Real):
            return new(x * other, y * other)
        elif isinstance(other, Complex):
            ox, oy = self.from_complex(other)
        elif isinstance(other, Number):
            return new(x * other, y * other)
        else:
            return NotImplemented

        return new(x*ox - y*oy, x*oy + y*ox)

    __rmul__ = __mul__

    def __mod__(self, other: int | Point | tuple[int, int]) -> Self:
        """
        self % otro
        if other is a number apply the mod to each coordinate
        if other is a Point apply the mod point-wise ( Point(self.x % otro.x, self.y % otro.y) )
        """
        if isinstance(other, int):
            return type(self)(self.x % other, self.y % other)
        if isinstance(other, (tuple, type(self))):
            ox, oy = other
            return type(self)(self.x % ox, self.y % oy)
        return NotImplemented

    def __complex__(self) -> complex:
        return complex(self.x, self.y)

    @classmethod
    def from_complex(cls, number: Complex) -> Self:
        """return an integer point from the given complex number"""
        return cls(int(number.real), int(number.imag))

    def __divmod__(self, other: int | Point | tuple[int, int]) -> tuple[Self, Self]:
        """divmod(self, other)"""
        if isinstance(other, int):
            x, y = self
            dx, mx = divmod(x, other)
            dy, my = divmod(y, other)
            return Point(dx, dy), Point(mx, my)
        if isinstance(other, (tuple, type(self))):
            x, y = self
            ox, oy = other
            dx, mx = divmod(x, ox)
            dy, my = divmod(y, oy)
            return Point(dx, dy), Point(mx, my)
        return NotImplemented

    def __rdivmod__(self, other: int | Point | tuple[int, int]) -> tuple[Self, Self]:
        """divmod(other, self)"""
        if isinstance(other, int):
            x, y = self
            dx, mx = divmod(other, x)
            dy, my = divmod(other, y)
            return Point(dx, dy), Point(mx, my)
        if isinstance(other, (tuple, type(self))):
            x, y = self
            ox, oy = other
            dx, mx = divmod(ox, x)
            dy, my = divmod(oy, y)
            return Point(dx, dy), Point(mx, my)
        return NotImplemented

    normalize = normalize_point
    distance_t = taxicab_distance
    dotproduct = dotproduct


class Point3(NamedTuple):
    """3d point/vector"""
    x: int
    y: int
    z: int

    def __add__(self, other: Point3) -> Point3:
        """self + other"""
        x, y, z = self
        if isinstance(other, (type(self), tuple)):
            ox, oy, oz = other
        elif isinstance(other, Number):
            ox, oy, oz = other, 0, 0
        else:
            return NotImplemented
        return type(self)(x + ox, y + oy, z + oz)

    __radd__ = __add__

    def __neg__(self) -> Point3:
        """-self"""
        x, y, z = self
        return type(self)(-x, -y, -z)

    def __sub__(self, other: Point3) -> Point3:
        """self - other"""
        return self + (-other)

    def __mul__(self, other):
        """self * otro"""
        if isinstance(other, Number):
            return type(self)(*(other * t for t in self))
        return NotImplemented

    __rmul__ = __mul__

    def __mod__(self, other: int | Point3) -> Point3:
        """
        self % other
        if other is a number apply the mod to each coordinate
        if other is a Point apply the mod point-wise ( Point(self.x % otro.x, self.y % otro.y) )
        """
        if isinstance(other, int):
            return type(self)(*(t % other for t in self))
        if isinstance(other, (tuple, type(self))):
            return type(self)(*(t1 % t2 for t1, t2 in zip(self, other, strict=True)))
        return NotImplemented

    def cross_product(self, other: Point3) -> Self:
        """https://en.wikipedia.org/wiki/Cross_product"""
        a1, a2, a3 = self
        b1, b2, b3 = other
        return type(self)(
            a2 * b3 - a3 * b2,
            a3 * b1 - a1 * b3,
            a1 * b2 - a2 * b1
        )

    normalize = normalize_point
    distance_t = taxicab_distance
    dotproduct = dotproduct


__all__ = [x for x in dir() if not (x.startswith("_") or x in __exclude_from_all__)]
del __exclude_from_all__
