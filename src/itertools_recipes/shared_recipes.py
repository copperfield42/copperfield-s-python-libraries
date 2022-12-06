"""shared recipes with more_itertools"""


from itertools import (
    filterfalse,
    zip_longest,
    groupby,
    starmap,
    islice,
    repeat,
    chain,
    count,
    cycle,
    tee,
)
from collections import deque
from random import choice, sample, randrange
import operator, random

from .it_typing import (
    SentinelObject,
    Orderable,
    Iterable,
    Optional,
    Callable,
    Hashable,
    Tuple,
    Set,
    Any,
    T
)

__exclude_from_all__=set(dir())

_sentinela = SentinelObject()


############################################################################################
##-------------------------------- Itertools recipes ---------------------------------------
############################################################################################





def consume(iterator:Iterable, n:int|None=None) -> None: # type: ignore
    """Advance the iterator n-steps ahead. If n is none, consume entirely."""
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)


def nth(iterable:Iterable[T], n:int, default:Optional[T]|None=None) -> T:
    """Returns the nth item or a default value"""
    return next(islice(iterable, n, None), default) #type: ignore


def pad_none(iterable:Iterable[T]) -> Iterable[Optional[T]]:
    """Returns the sequence elements and then returns None indefinitely.

        Useful for emulating the behavior of the built-in map() function."""
    return chain(iterable, repeat(None))


def tabulate(function:Callable[[int],T], start:int=0) -> Iterable[T]:
    """Return function(0), function(1), ..."""
    return map(function, count(start))


def ncycles(iterable:Iterable[T], n:int) -> Iterable[T]:
    """Returns the sequence elements n times

        ncycles("XYZ",3) --> X Y Z X Y Z X Y Z"""
    return chain.from_iterable(repeat(tuple(iterable), n))


try:
    from itertools import pairwise #py 3.10+
except ImportError:
    def pairwise(iterable:Iterable[T]) -> Iterable[Tuple[T,T]]: #type: ignore
        """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)


def flatten(listOfLists:Iterable[Iterable[Any]]) -> Iterable[Any]:
    """Flatten one level of nesting"""
    return chain.from_iterable(listOfLists)


def quantify(iterable:Iterable[T], pred:Callable[[T],bool]=bool) -> int:
    """Count how many times the predicate is true"""
    return sum(map(pred, iterable))


def roundrobin(*iterables:Iterable[Any]) -> Iterable[Any]:
    """roundrobin('ABC', 'D', 'EF') --> A D E B F C"""
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            pending -= 1
            nexts = cycle(islice(nexts, pending))


#the more_itertools version is better
def partition(pred:Callable[[T],bool], iterable:Iterable[T]) -> Tuple[Iterable[T],Iterable[T]]:
    """Use a predicate to partition entries into false entries and true entries
        partition(is_odd, range(10)) --> 0 2 4 6 8  and  1 3 5 7 9"""
    t1, t2 = tee(iterable)
    return filterfalse(pred, t1), filter(pred, t2)


#the more_itertools version is better
def unique_everseen(iterable:Iterable[Any], key:Callable[[Any],Hashable]|None=None) -> Iterable[Any]:
    """List unique elements, preserving order. Remember all elements ever seen.
        unique_everseen('AAAABBBCCDAABBB') --> A B C D
        unique_everseen('ABBCcAD', str.lower) --> A B C D"""
    seen:Set[Any] = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element


def unique_justseen(iterable:Iterable[Any], key:Callable[[Any],Hashable]|None=None) -> Iterable[Any]:
    """List unique elements, preserving order. Remember only the element just seen.
        unique_justseen('AAAABBBCCDAABBB') --> A B C D A B
        unique_justseen('ABBCcAD', str.lower) --> A B C A D"""
    return map(next, map(operator.itemgetter(1), groupby(iterable, key)))


def iter_except(func:Callable[[],Any], exception:Exception|Tuple[Exception,...], first:Callable[[],Any]|None=None) -> Iterable[Any]:
    """Call a function repeatedly until an exception is raised.

        Converts a call-until-exception interface to an iterator interface.
        Like __builtin__.iter(func, sentinel) but uses an exception instead
        of a sentinel to end the loop.

        Examples:
            iter_except(functools.partial(heappop, h), IndexError)   # priority queue iterator
            iter_except(d.popitem, KeyError)                         # non-blocking dict iterator
            iter_except(d.popleft, IndexError)                       # non-blocking deque iterator
            iter_except(q.get_nowait, Queue.Empty)                   # loop over a producer Queue
            iter_except(s.pop, KeyError)                             # non-blocking set iterator"""
    try:
        if first is not None:
            yield first()  # For database APIs needing an initial cast to db.first()
        while True:
            yield func()
    except exception: #type: ignore
        pass


def first_true(iterable:Iterable[Any], default:Any=False, pred:Callable[[Any],bool]|None=None) -> Any:
    """Returns the first true value in the iterable.

        If no true value is found, returns *default*

        If *pred* is not None, returns the first item
        for which pred(item) is true.


        first_true([a,b,c], x) --> a or b or c or x
        first_true([a,b], x, f) --> a if f(a) else b if f(b) else x """
    return next(filter(pred, iterable), default)


def random_product(*args:Iterable[Any], repeat:int=1) -> Tuple[Any,...]:
    """Random selection from itertools.product(*args, **kwds)"""
    #es asi para hacerlo compatible con python 2
    pools = list(map(tuple, args)) * repeat
    return tuple(choice(pool) for pool in pools)


def random_permutation(iterable:Iterable[Any], r:int|None=None) -> Tuple[Any,...]:
    """Random selection from itertools.permutations(iterable, r)"""
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(sample(pool, r))


def random_combination(iterable:Iterable[Any], r:int) -> Tuple[Any,...]:
    """Random selection from itertools.combinations(iterable, r)"""
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(sample(range(n), r))
    return tuple(pool[i] for i in indices)


def random_combination_with_replacement(iterable:Iterable[Any], r:int) -> Tuple[Any,...]:
    """Random selection from itertools.combinations_with_replacement(iterable, r)"""
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.randrange(n) for i in range(r))
    return tuple(pool[i] for i in indices)


def tail(n:int, iterable:Iterable[T]) -> Iterable[T]:
    """Return an iterator over the last n items,
        if n is none return a iterator over all
        elemens in iterable save the first

        tail(3,'ABCDEFG') --> E F G
        tail(None,'ABCDEFG')   --> B C D E F G """
    if n is None:
        return islice(iterable,1,None)
    return iter(deque(iterable, maxlen=n))


#the more_itertools version is better
def grouper(iterable:Iterable[Any], n:int=2, fillvalue:Any=_sentinela) -> Iterable[Tuple[Any,...]]:
    """Collect data into fixed-length chunks or blocks.
       grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
       grouper('ABCDEFG', 3)      --> ABC DEF """
    # pendiente con https://stackoverflow.com/a/8998040/5644961 para optimizaciones
    args = [iter(iterable)] * n
    if fillvalue is not _sentinela:
        return zip_longest(*args, fillvalue=fillvalue)
    else:
        return zip(*args)


#the more_itertools version is better
def sliding_window(iterable:Iterable[T], n:int) -> Iterable[Tuple[T,...]]:
    """s -> (s0,s1,...,sn-1), (s1,s2,...,sn), (s2, s3,...,sn+1), ..."""
    grupo = tee(iterable,n)
    for i,e in enumerate(grupo):
        consume(e,i)
    return zip( *grupo )


def chunked(iterable:Iterable[Any], n:int, *, container:Callable[[Iterable[Any]],T]=tuple) -> Iterable[T]: #type: ignore
    """Regresa un iterador con los datos agrupados en bloques de a lo
        sumo n elementos del iterable.
        Ejemplo:
        >>> list(chunker(2,range(9)))
        [(0, 1), (2, 3), (4, 5), (6, 7), (8,)]
        >>>

        Se puede espesificar el contenedor con el parametro key-only
        'container' en cuyo caso, los elementos son guardados en
        una instancia del mismo."""
    #http://stackoverflow.com/a/41245197/5644961
    it = iter(iterable)
    while True:
        yield container(islice(it,n))
        try:
            it = chain((next(it),),it)
        except StopIteration:
            return 
    #while (elem := container( islice(it,n) )):
    #    yield elem


def _len_range(ran:range) -> int:
    #http://stackoverflow.com/questions/14754877/pre-compute-lenrangestart-stop-step
    #http://stackoverflow.com/a/19440091/5644961
    if isinstance(ran,range):
        try:
            return len(ran)
        except OverflowError:
            pass
        start,stop,step = 0, 0, 1
        if all( hasattr(ran,x) for x in ("start","stop","step") ):
            start,stop,step =  ran.start, ran.stop, ran.step
        else:
            elem = list(map(int,repr(ran).replace("xrange(","").replace(")","").split(",")))
            if len(elem) == 1:
                stop = elem[0]
            elif len(elem) == 2:
                start,stop = elem
            else:
                start,stop,step = ran
        return max(0, (stop - start) // step + bool((stop - start) % step))
    else:
        raise ValueError("Se esperaba una instancia de {}".format(range))


#the more_itertools version is better for iterables without len
def ilen(iterable:Iterable) -> int:
    """Dice la cantidad de elementos del iterable iterando sobre el mismo si es necesario."""
    try:
        return len(iterable)  #type: ignore
    except TypeError:
        return sum( 1 for _ in iterable )
    except OverflowError as oe:
        #print("over")
        if isinstance(iterable,range):
            #print("range")
            return _len_range(iterable)
        else:
            raise oe


def is_sorted(seq:Iterable[Any], key:Callable[[Any],Orderable]|None=None, reverse:bool=False, strict:bool=False) -> bool:
    """Dice si los elementos del itereble estan ordenados
       Si strict es True los elementos deben ser estrictamente mayores/menores
       que el elemento anterior, de lo contrario deben ser mayores/menores o iguales

       >>> is_sorted(range(10))
       True
       >>> is_sorted(range(10),reverse=True)
       False
       >>> is_sorted(reversed(range(10)),reverse=True)
       True
       >>> is_sorted([1,1,1])
       True
       >>> is_sorted([1,1,1],reverse=True)
       True
       >>> is_sorted([1,1,1],strict=True)
       False
       >>>
       """
    if key is not None:
        seq = map(key,seq)
    if strict:
        comp = operator.gt if reverse else operator.lt
    else:
        comp = operator.ge if reverse else operator.le
    return all( starmap(comp,pairwise(seq)) )





try:
    from more_itertools import (#type: ignore
        random_combination_with_replacement,
        random_permutation,
        random_combination,
        unique_everseen,
        unique_justseen,
        random_product,
        sliding_window,
        iter_except,
        roundrobin,
        first_true,
        partition,
        is_sorted,
        pad_none,
        tabulate,
        pairwise,
        quantify,
        consume,
        ncycles,
        flatten,
        grouper,
        chunked,
        tail,
        ilen,
        nth,
    )
except ImportError:
    pass


groupwise = sliding_window



__all__ = [ x for x in dir() if not (x.startswith("_") or x in __exclude_from_all__) ]
del __exclude_from_all__
