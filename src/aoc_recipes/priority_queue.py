from __future__ import annotations
import itertools
from heapq import heappush, heappop
from dataclasses import dataclass, field

__exclude_from_all__ = set(dir())


class PriorityQueue:

    def __init__(self):
        self._pq = []                         # list of entries arranged in a heap
        self._entry_finder = {}               # mapping of tasks to entries
        self._REMOVED = '<removed-task>'      # placeholder for a removed task
        self._counter = itertools.count()     # unique sequence count

    def __bool__(self):
        return bool(self._entry_finder)

    def __repr__(self):
        return f"<{type(self).__name__}({sorted(self._entry_finder.keys())})>"

    def __len__(self):
        return len(self._entry_finder)

    def add_task(self, task, priority: int = 0):
        """Add a new task or update the priority of an existing task"""
        if task in self._entry_finder:
            self.remove_task(task)
        count = next(self._counter)
        entry = [priority, count, task]
        self._entry_finder[task] = entry
        heappush(self._pq, entry)

    def remove_task(self, task):
        """
        Mark an existing task as REMOVED.  Raise KeyError if not found.
        """
        entry = self._entry_finder.pop(task)
        entry[-1] = self._REMOVED

    def pop_task(self):
        """
        Remove and return the lowest priority task. Raise KeyError if empty.
        """
        pq = self._pq
        while pq:
            priority, count, task = heappop(pq)
            if task is not self._REMOVED:
                del self._entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')


@dataclass(order=True, eq=True, frozen=True)
class PrioritizedItem[T]:
    priority: int | float
    item: T = field(compare=False)


__all__ = [x for x in dir() if not (x.startswith("_") or x in __exclude_from_all__)]
del __exclude_from_all__
