"""
Recipes relates to Graph theory

Third party Libraries for working with graph: networkx
pip install --upgrade networkx


"""

from __future__ import annotations
from typing import Callable, Iterable, Any, Hashable
from dataclasses import dataclass
from aoc_recipes import set_recursion_limit, DefaultValueDict
from functools import partial
from operator import eq
from .priority_queue import PriorityQueue
from .point_class import taxicab_distance


def cost_plus_one(grafo: Any, point1: Any, point2: Any, old_cost: int) -> int:
    return old_cost + 1


def tuple_path[T](path: tuple[T, ...], item: T) -> tuple[T, ...]:
    return path + (item,)


@dataclass
class NodeStatus[T]:
    node: T
    index: int = None
    lowlink: int = None
    onStack: bool = False


@dataclass
class AStarNodeStatus[T]:
    node: T
    cost_so_far: int | float = float("inf")
    came_from: T = None


class AStarDict[T](dict[T, AStarNodeStatus[T]]):

    def __missing__(self, key: T) -> AStarNodeStatus[T]:
        value = AStarNodeStatus(key)
        self[key] = value
        return value


def reconstruct_path[T](memory: dict[T, AStarNodeStatus[T]], from_: T) -> list[T]:
    path = [from_]
    while prev := memory[from_].came_from:
        path.append(prev)
        from_ = prev
    path.reverse()
    return path


__exclude_from_all__ = set(dir())


def shortest_path_grafo[Graph, V, P, C](
        grafo: Graph,
        inicio: V,
        meta: V | Callable[[V], bool],
        neighbors: Callable[[Graph, V], Iterable[V]],
        edge_cost: Callable[[Graph, V, V, C], C] = cost_plus_one,
        build_path: Callable[[P, V], P] = tuple_path,
        initial_cost: C = 0) -> tuple[C, P]:
    """Dijkstra's shortest path"""
    # https://www.youtube.com/watch?v=sBe_7Mzb47Y
    # first used on https://adventofcode.com/2022/day/12

    visitado = set()
    if not callable(meta):
        meta = partial(eq, meta)
    queue = PriorityQueue()
    queue.add_task((initial_cost, inicio, build_path((), inicio)), initial_cost)
    while queue:
        cost, p, path = queue.pop_task()
        if p in visitado:
            continue
        visitado.add(p)
        if meta(p):
            # print("Dijkstra memory size", len(visitado), "pendientes",len(queue))
            return cost, path
        for v in neighbors(grafo, p):
            new_cost = edge_cost(grafo, p, v, cost)
            queue.add_task((new_cost, v, build_path(path, v)), new_cost)
    return float("inf"), None


def strongly_connected_components[Graph, V: Hashable](grafo: Graph, nodes: Callable[[Graph], Iterable[V]], successors: Callable[[Graph, V], Iterable[V]]) -> Iterable[list[V]]:
    """https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm"""
    # used first in https://adventofcode.com/2023/day/25
    def strongconnect(v: NodeStatus[V]) -> list[V]:
        nonlocal index, stack, vertices
        # Set the depth index for v to the smallest unused index
        v.index = index
        v.lowlink = index
        index += 1
        stack.append(v)
        v.onStack = True

        # Consider successors of v
        for nw in successors(grafo, v.node):
            w = vertices[nw]
            if w.index is None:
                # Successor w has not yet been visited; recurse on it
                strongconnect(w)
                v.lowlink = min(v.lowlink, w.lowlink)
            elif w.onStack:
                # Successor w is in stack S and hence in the current SCC
                # If w is not on stack, then (v, w) is an edge pointing to an SCC already found and must be ignored
                # The next line may look odd - but is correct.
                # It says w.index not w.lowlink; that is deliberate and from the original paper
                v.lowlink = min(v.lowlink, w.index)
        # If v is a root node, pop the stack and generate an SCC
        if v.lowlink == v.index:
            scc = []
            while True:
                w = stack.pop()
                w.onStack = False
                scc.append(w.node)
                if w is v:
                    break
            return scc
        pass

    index = 0
    stack = []
    vertices: dict[V, NodeStatus[V]] = {k: NodeStatus(k) for k in nodes(grafo)}

    with set_recursion_limit(len(vertices)):
        for ver in vertices.values():
            if ver.index is None:
                yield strongconnect(ver)


def longest_path_directed_acyclic[Graph, V](
        graph: Graph,
        inicio: V,
        goal: V,
        neighbors: Callable[[Graph, V], Iterable[V]],
        recursion_limit: int = 30_000) -> int:
    # https://www.geeksforgeeks.org/find-longest-path-directed-acyclic-graph/
    # used first in https://adventofcode.com/2023/day/23
    def topological_sort_util(node: V):
        nonlocal visited, stack
        visited.add(node)
        for v in neighbors(graph, node):
            if v in visited:
                continue
            topological_sort_util(v)
        stack.append(node)
        pass

    stack = []
    visited = set()
    dist = DefaultValueDict(float("-inf"))

    with set_recursion_limit(recursion_limit):
        topological_sort_util(inicio)

    dist[inicio] = 0

    while stack:
        u = stack.pop()
        if dist[u] != float("-inf"):
            for i in neighbors(graph, u):
                new = dist[u] + 1
                if dist[i] < new:
                    dist[i] = new
    return dist[goal]


def dfs[Graph, V: Hashable, C](
        grafo: Graph,
        start: V,
        goal: V,
        neighbors: Callable[[Graph, V], Iterable[V]],
        distance: Callable[[Graph, V, V], C],
        callback: Callable[[], None] = None,
        selector: Callable[[C, C], C] = max,
        default: C = float("-inf"),
        _seen: set[V] = None) -> C:
    # used first in https://adventofcode.com/2023/day/23
    if start == goal:
        return 0
    if _seen is None:
        _seen = set()
    if callback:
        callback()
    result = default
    _seen.add(start)
    for p in neighbors(grafo, start):
        if p in _seen:
            continue
        result = selector(result, distance(grafo, start, p) + dfs(grafo, p, goal, neighbors, distance, callback=callback, _seen=_seen, selector=selector, default=default))
    _seen.remove(start)
    return result


def a_star_shortest_path[Graph, V:Hashable, C, P, H](
        grafo: Graph,
        start: V,
        target: V | Callable[[V], bool],
        neighbors: Callable[[Graph, V], Iterable[V]],
        edge_cost: Callable[[Graph, V, V, C], C] = cost_plus_one,
        build_path: Callable[[dict[V, AStarNodeStatus[V]], V], P] | None = reconstruct_path,
        initial_cost: C = 0,
        heuristic: Callable[[V], H] = None,
        _memory_class: Callable[[], dict[V, AStarNodeStatus[V]]] = AStarDict
        ) -> tuple[C, P]:
    """A* Shortest path
    https://en.wikipedia.org/wiki/A*_search_algorithm

    By default, the heuristic used is the taxicap distance between target and the current vertice.
    If a function is provided for target but not for the heuristic function, this one default to 0
    """
    # https://www.redblobgames.com/pathfinding/a-star/introduction.html

    if heuristic is None:
        if callable(target):
            heuristic = lambda _: 0
        else:
            heuristic = partial(taxicab_distance, target)

    if not callable(target):
        target = partial(eq, target)

    frontier = PriorityQueue()
    frontier.add_task(start, 0)

    memory = _memory_class()
    memory[start].cost_so_far = initial_cost

    while frontier:
        current = frontier.pop_task()

        if target(current):
            # print("memory size", len(memory), "pendientes", len(frontier))
            return memory[current].cost_so_far, build_path(memory, current) if build_path else None

        for v in neighbors(grafo, current):
            new_cost = edge_cost(grafo, current, v, memory[current].cost_so_far)
            if v not in memory or new_cost < memory[v].cost_so_far:
                node = memory[v]
                node.cost_so_far = new_cost
                node.came_from = current
                priority = new_cost + heuristic(v)
                frontier.add_task(v, priority)
    return float("inf"), None


__all__ = [x for x in dir() if not (x.startswith("_") or x in __exclude_from_all__)]
del __exclude_from_all__
