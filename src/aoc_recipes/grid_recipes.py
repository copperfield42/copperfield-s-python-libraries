"""
Helper function and classes to read grids as numpy array

"""
from __future__ import annotations

from typing import Iterable, Iterator, Any, Final, Callable, Annotated
import itertools
import numpy
from aoc_recipes import Point, DIRECCIONES, vecinos
from itertools_recipes import interesting_lines

__exclude_from_all__ = set(dir())


BLACK_char: Final = "█"
WHITE_char: Final = '░'


type Matrix[T] = numpy.ndarray[Annotated[Any, tuple[int, int]], numpy.dtype[T]]
type DictGraph[V, C] = dict[V, dict[V, C]]


def where(condition: numpy.ndarray[Any, numpy.dtype[bool]]) -> Iterator[tuple[int, ...]]:
    return zip(*numpy.where(condition))


def where2(condition: Matrix[bool]) -> Iterator[Point]:
    return map(Point, *numpy.where(condition))


def all_points2(shape: tuple[int, int]) -> Iterable[Point]:
    x, y = shape
    return itertools.starmap(Point, itertools.product(range(x), range(y)))


def flood_fill(matrix: Matrix[bool], initial: Point = Point(0, 0), mode: str = "+") -> Matrix[bool]:
    result = numpy.zeros_like(matrix, dtype=bool)
    points = set(where2(~matrix))
    work = {initial}
    while work:
        p = work.pop()
        result[p] = True
        points.discard(p)
        work.update(nextp for nextp in vecinos(p, direcciones=DIRECCIONES[mode]) if nextp in points)
    return result


def show_bool_matrix2(matrix: Matrix[bool], *, true: str = BLACK_char, false: str = WHITE_char):
    x_axis, y_axis = matrix.shape
    for x in range(x_axis):
        for y in range(y_axis):
            print(true if matrix[x, y] else false, sep="", end="", flush=False)
        print(flush=True)
    print()


def show_str_matrix2(matrix: Matrix[str]):
    x_axis, y_axis = matrix.shape
    for x in range(x_axis):
        for y in range(y_axis):
            print(matrix[x, y], sep="", end="", flush=False)
        print(flush=True)
    print()


def show_path(matrix: Matrix[Any], path: Iterable[Point]):
    mapa = numpy.zeros_like(matrix, dtype=bool)
    for p in path:
        mapa[p] = True
    return show_bool_matrix2(mapa)


def to_bool_matrix(data: str | Iterable[str], char: str = ".") -> Matrix[bool]:
    return numpy.array([[c == char for c in line] for line in interesting_lines(data)], dtype=bool)


def to_str_matrix(data: str | Iterable[str]) -> Matrix[str]:
    return numpy.array([list(line) for line in interesting_lines(data)], dtype=str)


def to_int_matrix(data: str | Iterable[str], single_digit: bool = True) -> Matrix[int]:
    if single_digit:
        return numpy.array([list(map(int, line)) for line in interesting_lines(data)], dtype=int)
    else:
        return numpy.array([list(map(int, line.split())) for line in interesting_lines(data)], dtype=int)


def edge_contraction[T](matrix: Matrix[T], points_of_interest: list[Point], neighbors: Callable[[Matrix[T], Point], Iterable[Point]]) -> DictGraph[Point, int]:
    # used first in https://adventofcode.com/2023/day/23

    graph: dict[Point, dict[Point, int]] = {p: {} for p in points_of_interest}

    for c in points_of_interest:
        stack = [(0, c)]
        seen = {c}
        while stack:
            d, p = stack.pop()
            if d and p in points_of_interest:
                graph[c][p] = d
                continue
            for v in neighbors(matrix, p):
                if v not in seen:
                    stack.append((d+1, v))
                    seen.add(v)
    return graph


def convolve2d[T](image: Matrix[T], kernel: Matrix[T], *, fillvalue: T = 0, mode: str = "same") -> Matrix[T]:
    """
    https://www.youtube.com/watch?v=KuXjwB4LzSA
    """
    # https://medium.com/analytics-vidhya/2d-convolution-using-python-numpy-43442ff5f381
    # https://copyprogramming.com/howto/what-does-scipy-signal-convolve2d-calculate-duplicate
    match mode:
        case "same":
            padding = 1
        case "valid":
            # from scipy implementation
            if not all(si >= sk for si, sk in zip(image.shape, kernel.shape, strict=True)):
                image, kernel = kernel, image
            padding = 0
        case _:
            padding = 1

    kernel = kernel[::-1, ::-1]  # flip up

    k_shape = Point(*kernel.shape)
    i_shape = Point(*image.shape)
    pad = Point(padding, padding)
    o_shape = i_shape - k_shape + 2*pad + Point(1, 1)

    output = numpy.zeros(o_shape, dtype=kernel.dtype)

    if padding:
        image_padded = numpy.full(i_shape + 2*pad, fillvalue, dtype=image.dtype)
        image_padded[padding:-padding, padding:-padding] = image
    else:
        image_padded = image

    condition = Point(*image_padded.shape) - k_shape

    for y in range(o_shape.y):
        if y > condition.y:
            break
        for x in range(o_shape.x):
            if x > condition.x:
                break
            output[x, y] = (kernel * image_padded[x: x + k_shape.x, y: y + k_shape.y]).sum(dtype=output.dtype)
    return output


__all__ = [x for x in dir() if not (x.startswith("_") or x in __exclude_from_all__)]
del __exclude_from_all__
