"""Collections recipes"""

from collections import (
    OrderedDict,
    ChainMap,
    Counter,
    deque,
    abc,
)
from bisect import bisect_left, bisect_right
from operator import attrgetter, itemgetter
from itertools import chain, islice
from pathlib import Path
from time import time

from contextlib_recipes import AbstractClosableContextManager
from .abc_recipes import AutoSized, SequenceSliceView
from .cr_typing import (
    NumberLike,
    Orderable,
    PathType,
    Iterator,
    Iterable,
    Callable,
    overload,
    TypeVar,
    Any,
    DIV,
    MOD,
    KT,
    VT,
    T,
)



__exclude_from_all__=set(dir())

from .mapping_recipes import BufferDict

try: #make Counter like in py3.10
    Counter.total
except AttributeError:
    class Counter(Counter): #type: ignore
        def total(self):
            """Compute the sum of the counts."""
            return sum(self.values())




################################################################################
#----------------------------- Collections Recipes -----------------------------
################################################################################


class DeepChainMap(ChainMap[KT,VT]):
    """Variant of ChainMap that allows direct updates to inner scopes.

    The ChainMap class only makes updates (writes and deletions) to the first mapping in the chain while lookups will search the full chain. However, if deep writes and deletions are desired, it is easy to make a subclass that updates keys found deeper in the chain.

    >>> d = DeepChainMap({'zebra': 'black'}, {'elephant': 'blue'}, {'lion': 'yellow'})
    >>> d['lion'] = 'orange'         # update an existing key two levels down
    >>> d['snake'] = 'red'           # new keys get added to the topmost dict
    >>> del d['elephant']            # remove an existing key one level down
    >>> d                            # display result
    DeepChainMap({'zebra': 'black', 'snake': 'red'}, {}, {'lion': 'orange'})
    >>>
    """

    def __setitem__(self, key:KT, value:VT) -> None:
        for mapping in self.maps:
            if key in mapping:
                mapping[key] = value
                return
        self.maps[0][key] = value

    def __delitem__(self, key:KT) -> None:
        for mapping in self.maps:
            if key in mapping:
                del mapping[key]
                return
        raise KeyError(key)


def tail(filename:PathType, n:int=10) -> deque[str]:
    'Return the last n lines of a file'
    with open(filename) as f:
        return deque(f, n)


def moving_average(iterable:"Iterable[NumberLike]", n:int=3) -> "Iterator[NumberLike]":
    """
    moving_average([40, 30, 50, 46, 39, 44]) --> 40.0 42.0 45.0 43.0
    http://en.wikipedia.org/wiki/Moving_average
    """
    it = iter(iterable)
    d = deque(islice(it, n-1))
    d.appendleft(0)
    s = sum(d)
    for elem in it:
        s += elem - d.popleft()
        d.append(elem)
        yield s / n


def delete_nth(d:deque, n:int) -> None:
    "delete the n-element from the deque in-place"
    d.rotate(-n)
    d.popleft()
    d.rotate(n)


def roundrobin(*iterables:Iterable[T] ) -> Iterator[T]:
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    iterators:deque[Iterator[T]]
    iterators = deque(map(iter, iterables)) #type: ignore
    while iterators:
        try:
            while True:
                yield next(iterators[0])
                iterators.rotate(-1)
        except StopIteration:
            # Remove an exhausted iterator.
            iterators.popleft()


def constant_factory(value:T) -> Callable[[],T]:
    return lambda: value


class LastUpdatedOrderedDict(OrderedDict[KT,VT]):
    'Store items in the order the keys were last added'

    def __setitem__(self, key:KT, value:VT) -> None:
        if key in self:
            del self[key]
        super().__setitem__(key, value)


class OrderedCounter(Counter[T], OrderedDict[T,int]): #type: ignore
    'Counter that remembers the order elements are first encountered'

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class LRU(OrderedDict[KT,VT]):
    'Limit size, evicting the least recently looked-up key when full'

    def __init__(self, maxsize:int=128, /, *args, **kwds) -> None:
        self.maxsize = maxsize
        super().__init__(*args, **kwds)

    def __getitem__(self, key:KT) -> VT:
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key:KT, value:VT) -> None:
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]


class ListBasedSet(abc.Set[T]):
    '''
    Alternate set implementation favoring space over speed
    and not requiring the set elements to be hashable.
    '''

    def __init__(self, iterable:Iterable[T]) -> None:
        self.elements:list[T]
        lst:list[T]
        self.elements = lst = []
        for value in iterable:
            if value not in lst:
                lst.append(value)

    def __iter__(self) -> Iterator[T]:
        return iter(self.elements)

    def __contains__(self, value:Any) -> bool:
        return value in self.elements

    def __len__(self) -> int:
        return len(self.elements)


class TimeBoundedLRU:
    "LRU Cache that invalidates and refreshes old entries."

    def __init__(self, func:Callable[...,T], maxsize:int=128, maxage:float=30) -> None:
        self.cache:OrderedDict[tuple[Any,...],tuple[float,T]]  = OrderedDict()      # { args : (timestamp, result)}
        self.func = func
        self.maxsize = maxsize
        self.maxage = maxage

    def __call__(self, *args) -> T:
        if args in self.cache:
            self.cache.move_to_end(args)
            timestamp, result = self.cache[args]
            if time() - timestamp <= self.maxage:
                return result
        result = self.func(*args)
        self.cache[args] = time(), result
        if len(self.cache) > self.maxsize:
            self.cache.popitem(False)
        return result


class MultiHitLRUCache:
    """
    LRU cache that defers caching a result until
    it has been requested multiple times.

    To avoid flushing the LRU cache with one-time requests,
    we don't cache until a request has been made more than once.

    """

    def __init__(self, func:Callable[...,T], maxsize:int=128, maxrequests:int=4096, cache_after:int=1) -> None:
        self.requests:OrderedDict[tuple[Any,...],int] = OrderedDict()   # { uncached_key : request_count }
        self.cache:OrderedDict[tuple[Any,...],T] = OrderedDict()      # { cached_key : function_result }
        self.func = func
        self.maxrequests = maxrequests  # max number of uncached requests
        self.maxsize = maxsize          # max number of stored return values
        self.cache_after = cache_after

    def __call__(self, *args) -> T:
        if args in self.cache:
            self.cache.move_to_end(args)
            return self.cache[args]
        result = self.func(*args)
        self.requests[args] = self.requests.get(args, 0) + 1
        if self.requests[args] <= self.cache_after:
            self.requests.move_to_end(args)
            if len(self.requests) > self.maxrequests:
                self.requests.popitem(False)
        else:
            self.requests.pop(args, None)
            self.cache[args] = result
            if len(self.cache) > self.maxsize:
                self.cache.popitem(False)
        return result


################################################################################
#------------------------------ Raymond Hettinger ------------------------------
################################################################################
# Examples show by Raymond Hettinger in the video
# https://www.youtube.com/watch?v=S_ipdVNSFlo


_SelfBS = TypeVar("_SelfBS", bound="BitSet")

class BitSet(AutoSized, abc.MutableSet[int]):
    """Ordered set with compact storage for integers in a fixed range"""

    __slots__ = ("limit", "data")

    def __init__(self, limit:int, iterable:Iterable[int]=()) -> None:
        self.limit = limit
        num_bytes = (limit+7)//8
        self.data = bytearray(num_bytes)
        self |= iterable #type: ignore

    def _get_location(self, elem:int) -> "tuple[DIV,MOD]":
        if elem < 0 or elem >= self.limit:
            raise ValueError(f"{elem!r} must be in range 0 <= elem < {self.limit}")
        return divmod(elem,8)

    def __contains__(self, elem:Any) -> bool:
        try:
            bytenum, bitnum = self._get_location(elem)
            return bool( (self.data[bytenum] >> bitnum) & 1 )
        except (TypeError, ValueError):
            return False

    def add(self, elem:int) -> None:
        bytenum, bitnum = self._get_location(elem)
        self.data[bytenum] |= ( 1 << bitnum )

    def discard(self, elem:int) -> None:
        try:
            bytenum, bitnum = self._get_location(elem)
            self.data[bytenum] &= ~( 1 << bitnum )
        except (TypeError, ValueError):
            pass

    def __iter__(self) -> Iterator[int]:
        for elem in range(self.limit):
            if elem in self:
                yield elem

    def __repr__(self) -> str:
        return f"{type(self).__name__}(limit={self.limit}, iterable={list(self)})"

    def _from_iterable(self:_SelfBS, iterable:Iterable[int]) -> _SelfBS:
        #necesary because the constructor take an extra argument
        #see: Notes on using Set and MutableSet as a mixin
        #in the documentation
        return type(self)(self.limit, iterable)

    @classmethod
    def create_full(cls:type[_SelfBS], limit:int) -> _SelfBS:
        new = cls(1)
        new.limit = limit
        num_bytes = (limit+7)//8
        data = bytearray(1)
        data[0]=255
        new.data = data*num_bytes
        return new

#FileDict and SQLDict moved to its own modulo

################################################################################
#--------------------------------- Mis Recipes ---------------------------------
################################################################################

#TO DO: añadir el resto de funciones que tiene set

_SelfOS = TypeVar("_SelfOS", bound="OrderedSet")

class OrderedSet(abc.MutableSet[T]):
    """Set that remember the order of insertion"""

    __slots__ = ("_elements")

    def __init__(self, iterable:Iterable[T]=None) -> None:
        if iterable is None:
            self._elements = OrderedDict()
        else:
            self._elements = OrderedDict.fromkeys(iterable)

    def __contains__(self, value:Any) -> bool:
        return value in self._elements

    def __iter__(self) -> Iterator[T]:
        return iter(self._elements)

    def __len__(self) -> int:
        return len(self._elements)

    def __repr__(self) -> str:
        return f"{type(self).__name__}([{', '.join(map(repr,self))}])"

    def add(self, value:T) -> None:
        self._elements[value] = None

    def discard(self, value:T) -> None:
        try:
            del self._elements[value]
        except KeyError:
            pass

    def clear(self) -> None:
        """Remove all elements in-place"""
        self._elements.clear()

    def move_to_end(self, value:T, last:bool=True):
        """
        Move an existing value to either end of an ordered set.
        The item is moved to the right end if last is true (the default)
        or to the beginning if last is false.
        Raises KeyError if the value does not exist
        """
        self._elements.move_to_end(value, last)

    def copy(self:_SelfOS) -> _SelfOS:
        return self.__class__(self)

    def difference(self:_SelfOS, *other) -> _SelfOS:
        new = self.copy()
        for elem in other:
            if not new:
                break
            new -= elem
        return new

    #def difference_update(self, other):
    #    raise NotImplementedError

    def intersection(self:_SelfOS, *other) -> _SelfOS:
        new = self.copy()
        for elem in other:
            if not new:
                break
            new &= elem
        return new

    #def intersection_update(self, other):
    #    raise NotImplementedError

    def issubset(self, other) -> bool:
        return self <= set(other)

    def issuperset(self, other) -> bool:
        return self >= set(other)

    def symmetric_difference(self, other):
        return self ^ other

    #def symmetric_difference_update(self, other):
    #    raise NotImplementedError

    def union(self:_SelfOS, *other) -> _SelfOS:
        new = self.copy()
        for elem in other:
            new |= elem
        return new

    #def update(self, other):
    #    raise NotImplementedError

_Selfcr = TypeVar("_Selfcr",bound="chr_range")
_get2 = attrgetter(*("start stop".split()) )
_get3 = attrgetter(*("start stop step".split()) )
_basestring = (str,bytes)

class chr_range(abc.Sequence[str]):
    #http://stackoverflow.com/q/30362799/5644961S
    @overload
    def __init__(self, stop:str,/) -> None: ...
    @overload
    def __init__(self, start:str, stop:str, step:int=1,/) -> None: ...

    def __init__(self,*args) -> None:
        sl = slice(*args)
        argv:tuple[str,str,int]
        argv = [ x or y for x,y in zip(_get3(sl),('\x00',None,1)) ] #type: ignore
        if not all(argv):
            raise ValueError
        if not all( isinstance(x,t) for x,t in zip(argv,[str,str,int]) ):
            raise TypeError
        #print(argv)
        start, stop = map(ord,argv[:2])
        self._range = range(start,stop, argv[-1])
        self.start, self.stop, self.step = argv
        #for a,v in zip("start stop step".split(),argv):
        #    setattr(self,a,v)

    def __repr__(self) -> str:
        v = _get2(self) if self.step == 1 else _get3(self)
        return "{}({})".format(self.__class__.__name__,  ", ".join(map("{!r}".format,v)))

    def __iter__(self) -> Iterator[str]:
        return map(chr, self._range)

    def __reversed__(self) -> Iterator[str]:
        return map(chr, reversed(self._range))

    def __contains__(self,key) -> bool:
        return ord(key) in self._range

    @overload
    def __getitem__(self:_Selfcr, index:int) -> str:...
    @overload
    def __getitem__(self:_Selfcr, index:slice) -> _Selfcr:...

    def __getitem__(self:_Selfcr, index:int|slice) -> str | _Selfcr:
        if isinstance(index,slice):
            new_range = self._range[index]
            start, stop = map(chr,_get2(new_range))
            return self.__class__(start, stop, new_range.step)
        return chr( self._range[index] )

    def __len__(self) -> int:
        return len(self._range)

    def index(self, value:str, *argv, **kwarg) -> int:
        return self._range.index( ord(value), *argv, **kwarg)

    def count(self, char:str) -> int:
        return int( char in self )




class SortedSequence(abc.MutableSequence[T]):
    """Sequence that keep its elements ordered"""
    #https://code.activestate.com/recipes/577197-sortedcollection/

    def _getkey(self) -> Callable[[T], Orderable]:
        return self._key

    def _setkey(self, key:Callable[[T], Orderable]|None ) -> None:
        if key is not self._key:
            self.__init__(self._items, key=key) #type: ignore

    def _delkey(self) -> None:
        self._setkey(None)

    key = property(_getkey, _setkey, _delkey, 'key function')

    @overload
    def __init__(self, iterable:Iterable[T]=()):...
    @overload
    def __init__(self, iterable:Iterable[T]=(), *, key:Callable[[T],Orderable]):...

    def __init__(self, iterable:Iterable[T]=(), *, key:Callable[[T],Orderable]|None=None) -> None:
        self._given_key = key
        skey = (lambda x:x) if key is None else key
        sortpairs   = sorted(((skey(v),v) for v in iterable), key=itemgetter(0))
        self._keys  = list(map(itemgetter(0),sortpairs))
        self._items = list(map(itemgetter(1),sortpairs))
        self._key   = skey

    @overload
    def __getitem__(self, index:int) -> T: ...
    @overload
    def __getitem__(self, index:slice) -> list[T]: ...

    def __getitem__(self, index:int|slice) -> T | list[T]:
        return self._items[index]

    def __len__(self) -> int:
        return len(self._items)

    def __delitem__(self, index:int|slice) -> None:
        del self._items[index]
        del self._keys[index]

    def __contains__(self, item) -> bool:
        try:
            k = self._key(item)
            lo = bisect_left(self._keys,k)
            hi = bisect_right(self._keys,k)
            items = self._items
            return any( item == items[i] for i in range(lo,hi))
        except TypeError:
            return False

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def __reversed__(self) -> Iterator[T]:
        return reversed(self._items)

    def __repr__(self) -> str:
        if self._given_key:
            return f"{type(self).__name__}({self._items!r}, key={getattr(self._given_key, '__qualname__', repr(self._given_key))})"
        return f"{type(self).__name__}({self._items!r})"

    def insert(self, item:T) -> None : #type: ignore
        'Insert a new item.  If equal keys are found, add to the left'
        k = self._key(item)
        i = bisect_left(self._keys, k)
        self._keys.insert(i,k)
        self._items.insert(i,item)

    insert_left = insert

    def insert_right(self, item:T) -> None:
        'Insert a new item.  If equal keys are found, add to the right'
        k = self._key(item)
        i = bisect_right(self._keys, k)
        self._keys.insert(i, k)
        self._items.insert(i, item)

    append = insert_right #type: ignore

    def __setitem__(self, index:int|slice, item:T|Iterable[T]) -> None:
        """this does not support item assignment"""
        raise TypeError(f"{type(self).__name__!r} object does not support item assignment")
        k = self._key(item)
        i = bisect_left(self._keys, k)
        j = bisect_right(self._keys, k)
        print(f"{item=} {index=} {i=} {j=}")
        if (i==j ) and i == index:
            self._keys[i] =  k
            self._items[i] = item
        else:
            raise ValueError("Can't insert item in position {index} and preserve the order")




class LineSeekableFile(abc.Sequence[str], AbstractClosableContextManager):
    """
    Alternative for file.readlines() that doesn't store the
    whole content of the file in memory
    """
    #https://stackoverflow.com/a/59185917/5644961

    def __init__(self, filepath:PathType, *, encoding:str="utf8", errors:str=None, linesoffset:abc.Sequence[int]=None) -> None:
        self._filepath = Path(filepath).resolve()
        self._encoding = encoding
        self._errors = errors
        self._file = file = open(filepath, encoding=encoding, errors=errors)
        if linesoffset:
            self._linesoffset = linesoffset
        else:
            self._linesoffset = lst =[0,*( file.tell() for _ in iter(file.readline,"")) ]
            lst.pop()

    def close(self) -> None:
        self._file.close()

    def __repr__(self) -> str:
        args = [f"{self._filepath!r}"]
        if self._encoding:# != "utf8":
            args.append( f"encoding={self._encoding!r}" )
        if self._errors:
            args.append( f"errors={self._errors!r}" )
        return f"{type(self).__name__}({', '.join(args)})"

    def __len__(self) -> int:
        """Number of lines of this file"""
        return len(self._linesoffset)

    @overload
    def __getitem__(self, index:int) -> str :...
    @overload
    def __getitem__(self, index:slice) -> SequenceSliceView[str]:...

    def __getitem__(self, index:int|slice) -> str | SequenceSliceView[str]:
        if isinstance(index,slice):
            return SequenceSliceView(self, index)
        self._file.seek(self._linesoffset[index])
        return self._file.readline()

    def reload(self, recalculate_lines:int=True) -> None:
        self.close()
        self.__init__( #type: ignore
            self._filepath,
            encoding=self._encoding,
            errors = self._errors,
            linesoffset = None if recalculate_lines else self._linesoffset
            )



class ChainSet(abc.MutableSet[T]):
    """Set-like class for creating a single view of multiple sets"""

    def __init__(self, *sets:abc.MutableSet[T]) -> None:
        self.sets = list(sets) or [set()] # always at least one set

    @classmethod
    def _from_iterable(cls, iterable:Iterable[T]) -> "ChainSet[T]":
        return cls(set(iterable))

    def __contains__(self, value:Any) -> bool:
        return any( value in s for s in self.sets)

    def __iter__(self) -> Iterator[T]:
        return iter(set(chain(*self.sets)))

    def __len__(self) -> int:
        return len(set().union(*self.sets))

    def add(self, value:T) -> None:
        self.sets[0].add(value)

    def discard(self, value:T) -> None:
        self.sets[0].discard(value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(map(repr,filter(None,self.sets))) })"

    def __bool__(self) -> bool:
        return any(self.sets)



_SelfRS = TypeVar("_SelfRS", bound="RangedSet")

class RangedSet(AutoSized, abc.MutableSet[int], abc.Sequence, abc.Reversible):
    """
    Ordered set with compact storage for integers in a fixed range
    represented by range build-in object which element can be removed
    or added back in so long they can be found in the set of values
    from a range-object of the same characteristics.

    Similar to BitSet but for arbitrary range(...)

    Additionally it support indexing
    """

    __slots__ = ("_range", "_data")

    @overload
    def __init__(self, stop:int, /, *, iterable:Iterable[int]=() ) -> None:...
    @overload
    def __init__(self, start:int, stop:int, step:int=1, /, *, iterable:Iterable[int]=() ) -> None:...

    def __init__(self, *rangedata:int, iterable:Iterable[int]=() ) -> None:
        self._range = r = range(*rangedata)
        self._data = BitSet(len(r))
        self |= iterable #type: ignore

    @property
    def range(self) -> range:
        """range of valid values of this set"""
        return self._range

    def __contains__(self, elem:Any) -> bool:
        try:
            return self._range.index(elem) in self._data
        except (ValueError,TypeError):
            return False

    def __iter__(self) -> Iterator[int]:
        for i,n in enumerate(self._range):
            if i in self._data:
                yield n

    def __reversed__(self) -> Iterator[int]:
        r = self._range
        for i,n in zip(reversed(range(len(r))),reversed(r)) :
            if i in self._data:
                yield n

    def add(self, elem:int) -> None:
        self._data.add(self._range.index(elem))

    def discard(self, elem:int) -> None:
        try:
            self._data.discard(self._range.index(elem))
        except ValueError:
            pass

    def _from_iterable(self:_SelfRS, iterable:Iterable[int]) -> _SelfRS :
        #necesary because the constructor take an extra argument
        #see: Notes on using Set and MutableSet as a mixin
        #in the documentation
        r = self._range
        return type(self)(r.start, r.stop, r.step, iterable=iterable)

    def __repr__(self) -> str:
        r=str(self._range).replace("range(","")
        return f"{type(self).__name__}({r[:-1]}, iterable={list(self)})"

    @overload
    def __getitem__(self, index:int) -> int:...
    @overload
    def __getitem__(self, index:slice) -> SequenceSliceView[int]:...

    def __getitem__(self, index:int|slice) -> int | SequenceSliceView[int]:
        if isinstance(index, slice):
            return SequenceSliceView(self, index)
        if index>=0:
            for i,val in enumerate(self):
                if i==index:
                    return val
        else:
            for i,val in zip(range(-1,-len(self)-1,-1),reversed(self)):
                if i==index:
                    return val
        raise IndexError(index)

    def count(self, value:int) -> int:
        return int( value in self )

    @overload
    @classmethod
    def create_full(cls:type[_SelfRS], stop:int, /) -> _SelfRS:...

    @overload
    @classmethod
    def create_full(cls:type[_SelfRS], start:int, stop:int, step:int=1, /) -> _SelfRS:...

    @classmethod
    def create_full(cls:type[_SelfRS], *rangedata) -> _SelfRS:
        new = cls(0)
        new._range = r = range(*rangedata)
        new._data  = BitSet.create_full(len(r))
        return new












__all__ = [ x for x in dir() if not (x.startswith("_") or x in __exclude_from_all__) ]
del __exclude_from_all__






