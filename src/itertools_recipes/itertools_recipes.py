#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Implementación de los recipes de Itertools y otros adicionales
"""

from functools import partial, reduce
from collections import deque
from collections.abc import (
    Iterable as ABC_Iterable,
    Sequence as ABC_Sequence
)

import operator

from .it_typing import (
    SentinelObject,
    IntegralLike,
    Contenedor,
    NumberLike,
    Container,
    Orderable,
    Function1,
    Function2,
    Callable,
    Iterable,
    Iterator,
    Sequence,
    RealLike,
    overload,
    Tuple,
    cast,
    Any,
    T,
    S,
    X,
    Y,
)

try:
    from functools_recipes import identity
except ImportError:
    identity = lambda x:x


version="3.1.1"

from itertools import (
    combinations,
    zip_longest,
    accumulate,
    dropwhile,
    takewhile,
    compress,
    groupby,
    starmap,
    repeat,
    islice,
    chain,
    count,
    cycle,
    tee,
)

from .shared_recipes import consume, pad_none, pairwise

__exclude_from_all__=set(dir())

_sentinela = SentinelObject()
_basestring = (str,bytes)

#comunes a py2.7 y py3.5
##from itertools import chain, combinations, combinations_with_replacement, \
##                      compress, count, cycle, dropwhile, groupby, islice, \
##                      permutations, product, repeat, starmap, takewhile, tee \
##

#from itertools import chain, count, cycle, dropwhile, repeat, takewhile
#from itertools import filterfalse #, zip_longest




###posible fallos de importacion en python anteriores son:
##accumulate new in py3.2 agregado func in py3.3
##combinations_with_replacement new in py3.1 and py2.7
##compress new in py3.1 and py2.7
##count añadido el step in py3.1 and py2.7
##chain.from_iterable new in py2.6
##combinations new in py2.6
#groupby new in py2.4
##islice Changed in version py2.5: accept None values for default start and step.
#izip Changed in version py2.4 When no iterables are specified,
#     returns a zero length iterator instead of raising a TypeError exception.
##izip_longest new in py2.6
##permutations new in py2.6
##product new in py2.6
##starmap Changed in version py2.6: Previously, starmap() required the function arguments to be tuples.
#        Now, any iterable is allowed
##tee new in py2.4
#new 2.5 dropped support for old version of python, now only py3.9+
#new 2.6 5/6/2022 signatures and tested with mypy. grouper change a little, removed dual_reduce and moved to functools_recipes
#new 3.0.0 15/06/22 now is a warper around more_itertools to add some extra function, work even is more_itertools isn't available and some signature changes
#new 3.1.0 29/06/2022 added stubfile for main module, and explicity import everyting in init so mypy can see it
#new 3.1.1 08/07/2022 removed stubfile for main module, the explicit import everyting is now condicional on typing.TYPE_CHECKING in init so mypy can see it




################################################################################
##-------------------------- Itertools recipes modificados --------------------#
################################################################################
#also in more_itertools, but I like mine more

@overload
def take(n:int, iterable:Iterable[T]) -> Tuple[T,...]: ...
@overload
def take(n:int, iterable:Iterable[T], container:Callable[[Iterable[T]],S]) -> S: ...

def take(n:int, iterable:Iterable[T], container:Callable[[Iterable[T]],Any]=tuple) -> Any:
    """
    Regresa los primeros n elementos del iterable en el contenedor espesificado.

    >>> take(3,"ABCDEFGHI")
    ('A', 'B', 'C')
    >>> take(3,"ABCDEFGHI",list)
    ['A', 'B', 'C']
    >>> take(3,"ABCDEFGHI","".join)
    'ABC'
    >>>
    >>> take(10,range(1000))
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    >>> take(10,range(1000),sum)
    45
    >>>
    >>> take(10**10,itertools.count(),sum)
    49999999995000000000
    >>>


    """
    return container(islice(iterable, n))


@overload
def dotproduct(vec1:Iterable[int], vec2:Iterable[int]) -> int:...
@overload
def dotproduct(
    vec1:Iterable[IntegralLike], 
    vec2:Iterable[IntegralLike] 
    ) -> IntegralLike:...
@overload
def dotproduct(vec1:Iterable[RealLike], vec2:Iterable[RealLike]) -> RealLike:...
@overload
def dotproduct(
    vec1:Iterable[X],
    vec2:Iterable[X],
    *,
    sum:Callable[[Iterable[Y]],T]=...,
    map:Callable[[Callable[[X,X],Y],Iterable[X],Iterable[X]],Iterable[Y]]=...,
    mul:Callable[[X,X],Y]=...
    ) -> T:...

def dotproduct(
    vec1:Iterable[X],
    vec2:Iterable[X],
    *,
    sum:Callable[[Iterable[Y]],T]=sum,                            #type: ignore
    map:Callable[[Callable[[X,X],Y],Iterable[X],Iterable[X]],Iterable[Y]]=map,
    mul:Callable[[X,X],Y]=operator.mul
    ) -> T:
    """
    Return the dot product of two iterables

    >>> dotproduct([10, 10], [20, 20])
    400
    >>>

    basically: sum(map(mul, vec1, vec2))

    """
    return sum(map(mul, vec1, vec2))


def powerset(iterable:Iterable[T], ini:int=0, fin:int|None=None) -> Iterator[Tuple[T,...]]:
    """
    Da todas las posibles combinaciones de entre ini y fin elementos del iterable.
    Si fin no es otorgado default a la cantidad de elementos del iterable

    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    powerset([1,2,3],2) -->  (1,2) (1,3) (2,3) (1,2,3)
    powerset([1,2,3],0,2) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3)
    """
    if ini<0 or (fin is not None and fin<0):
        raise ValueError("El rango de combinaciones debe ser no negativo")
    elem = tuple(iterable)
    if fin is None:
        fin = len(elem)
    assert 0<=ini<=fin<=len(elem)
    return chain.from_iterable( combinations(elem, r) for r in range(ini,fin+1) )


def all_equal(iterable:Iterable[Any], key:Callable[[Any],Any]|None=None) -> bool:
    """
    Returns True if all the elements are equal to each other
    """
    #new recipe in 3.6
    g:Iterator[Any] = groupby(iterable,key)
    return next(g, True) and not next(g, False)


def iteratefunc(
    func:Callable[[T],T],
    start:T,
    *,
    times:int|None=None,
    iter_while:Callable[[T],bool]|None=None
    ) -> Iterator[T]:
    """
    Generador de relaciones de recurrencia de primer order:
    F0 = start
    Fn = func( F(n-1) )

    iteratefunc(func,start)   -> F0 F1 F2 ...
    iteratefunc(func,start,times=n) -> F0 F1 F2 ... Fn-1

    Sirve por ejemplo para mapas logisticos: Xn+1=rXn(1-Xn) que seria
    iteratefunc(lambda x: r*x*(1-x),x0)

    El conjunto de Mandelbrot podria ser:
    iteratefunc(lambda z: c + z**2 , 0, iter_while=lambda x: abs(x)<=2 )
    para algun número complejo c

    func: función de 1 argumento cuyo resultado es usado para
      producir el siguiente elemento de secuencia en la
      siguiente llamada.
    start: elemento inicial de la secuencia.
    times: cantidad de elementos que se desea generar de la
       secuencia, si es None se crea una secuencia infinita.
    iter_while: si es otorgado se producen elementos de la secuencia
            mientras estos cumplan con esta condición.

    """
    def iterate(f,s):
        while True:
            yield s
            s = f(s)
    result = iterate(func, start)
    if times is not None:
        result = islice(result, times)
    if iter_while:
        return takewhile(iter_while,result)
    return result


iterate = iteratefunc

def repeatfunc(
    func:Callable[...,T],
    times:int|None=None,
    *args:Any,
    **kwarg:Any
    ) -> Iterator[T]:
    """
    Call *func* with *args* and *kwarg* repeatedly, returning an iterable over the
    results.

    If *times* is specified, the iterable will terminate after that many
    repetitions:

        >>> from operator import add
        >>> times = 4
        >>> args = 3, 5
        >>> list(repeatfunc(add, times, *args))
        [8, 8, 8, 8]

    If *times* is ``None`` the iterable will not terminate:

        >>> from random import randrange
        >>> times = None
        >>> args = 1, 11
        >>> take(6, repeatfunc(randrange, times, *args))  # doctest:+SKIP
        [2, 4, 8, 1, 8, 4]



        >>> def fun(*a,**k):
        ...     print(a,k)
        ...
        ...
        >>> for _ in repeatfunc(fun,5,1,2,3,a=4,b=5,c=6):
        ...     pass
        ...
        (1, 2, 3) {'a': 4, 'b': 5, 'c': 6}
        (1, 2, 3) {'a': 4, 'b': 5, 'c': 6}
        (1, 2, 3) {'a': 4, 'b': 5, 'c': 6}
        (1, 2, 3) {'a': 4, 'b': 5, 'c': 6}
        (1, 2, 3) {'a': 4, 'b': 5, 'c': 6}
        >>>



        Example:  repeatfunc(random.random)"""
    if kwarg:
        func = partial(func, **kwarg)
    if times is None:
        return starmap(func, repeat(args))
    return starmap(func, repeat(args, times))


################################################################################
##----------------------------------- Mis recipes -----------------------------#
################################################################################



def flatten_total(
    iterable:Iterable[Any],
    flattype:type|Tuple[type,...]=ABC_Iterable,
    ignoretype:type|Tuple[type,...]=_basestring
    ) -> Iterator[Any]:
    """
    Flatten all level of nesting of a arbitrary iterable
    """
    #http://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists-in-python
    #unutbu version
    #collapse(iterable,flattype) in more_itertools
    remanente = iter(iterable)
    while True:
        try: #para futura compativilidad con python 3.7
            elem = next(remanente)
        except StopIteration:
            return
        if isinstance(elem,flattype) and not isinstance(elem,ignoretype):
            remanente = chain( elem, remanente )
        else:
            yield elem


def flatten_level(
    iterable:Iterable[Any], 
    nivel:int=1, 
    flattype:type|Tuple[type,...]=ABC_Iterable, 
    ignoretype:type|Tuple[type,...]=_basestring
    ) -> Iterator[Any]:
    """
    Flatten N levels of nesting of a arbitrary iterable

    >>> list(flatten_level([[[1, 2, 3,[23]], [4, 5]], 6]))
    [[1, 2, 3, [23]], [4, 5], 6]
    >>> list(flatten_level([[[1, 2, 3,[23]], [4, 5]], 6],2))
    [1, 2, 3, [23], 4, 5, 6]
    >>> list(flatten_level([[[1, 2, 3,[23]], [4, 5]], 6],3))
    [1, 2, 3, 23, 4, 5, 6]
    >>> 


    """
    #http://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists-in-python
    #Cristian version modificada
    #collapse(iterable,flattype,nivel) in more_itertools
    if nivel < 0:
        yield iterable
        return
    for elem in iterable:
        if isinstance(elem,flattype) and not isinstance(elem,ignoretype):
            yield from flatten_level(elem, nivel-1, flattype, ignoretype)
            #for sub in flatten_level(elem,nivel-1,flattype,ignoretype):
            #    yield sub
        else:
            yield elem


@overload
def irange(start:int, stop:int, step:int=1) -> Iterator[int]:...
@overload
def irange(start:IntegralLike, stop:IntegralLike, step:IntegralLike=1) -> Iterator[IntegralLike]:...
@overload
def irange(start:RealLike, stop:RealLike, step:RealLike=1) -> Iterator[RealLike]:...

def irange(start:"NumberLike", stop:"NumberLike", step:"NumberLike"=1) -> "Iterator[NumberLike]":
    """Simple iterador para producir valores en [start,stop)"""
    it_count = cast(Iterable[NumberLike], count(start,step)) #to please mypy
    return takewhile(lambda x: x<stop, it_count)


def ida_y_vuelta(iterable:Iterable[T]) -> Iterator[T]:
    """s-> s0,s1,s2,...,sn-2,sn-1,sn,sn-1,sn-2,...,s2,s1,s0"""
    try:
        ida = iter(iterable)
        vue = reversed(iterable)  #type: ignore
        next(vue,None)
        yield from chain(ida,vue)
    except TypeError:
        vue = []
        for x in iterable:
            yield x
            vue.append(x)
        if vue:
            vue.pop()
        yield from reversed(vue)


@overload
def iteratefunc_with_index(
    func:Callable[[int,int],int],
    start:int,
    times:int|None=None,
    iter_while:Callable[[int],bool]|None=None
    ) -> Iterator[int] :...
@overload
def iteratefunc_with_index(
    func:Callable[[IntegralLike,int],IntegralLike],
    start:IntegralLike,
    times:int|None=None,
    iter_while:Callable[[IntegralLike],bool]|None=None
    ) -> Iterator[IntegralLike] :...
@overload
def iteratefunc_with_index(
    func:Callable[[RealLike,int],RealLike],
    start:RealLike,
    times:int|None=None,
    iter_while:Callable[[RealLike],bool]|None=None
    ) -> Iterator[RealLike] :...
@overload
def iteratefunc_with_index(
    func:Callable[[T,int],T],
    start:T,
    times:int|None=None,
    iter_while:Callable[[T],bool]|None=None
    ) -> Iterator[T] :...
@overload
def iteratefunc_with_index(
    func:Callable[[Any,int],Any],
    start:Any,
    times:int|None=None,
    iter_while:Callable[[Any],bool]|None=None
    ) -> Iterator[Any] :...

def iteratefunc_with_index(
    func:Callable[[Any,int],Any],
    start:Any,
    times:int|None=None,
    iter_while:Callable[[Any],bool]|None=None
    ) -> Iterator[Any] :
    """
    Generador de relaciones de recurrencia de primer order 
    que incluyen el indice del elemento:
    F0 = start
    Fn+1 = func(Fn,n)

    iteratefunc_with_index(func,start)   -> F0 F1 F2 ...
    iteratefunc_with_index(func,start,n) -> F0 F1 F2 ... Fn-1

    por ejemplo el factorial seria: iteratefunc_with_index(lambda x,n:x*(n+1),1)

    func: función de 2 argumentos, el primero siendo el elemento anterior en
          la secuencia y el segundo el indice de ese elemento y cuyo resultado
          es usado para producir el siguiente elemento de secuencia en la
          siguiente llamada.
    start: elemento inicial de la secuencia.
    times: cantidad de elementos que se desea generar de la
           secuencia, si es None se crea una secuencia infinita.
    iter_while: si es otorgado se producen elementos de la secuencia
                mientras estos cumplan con esta condición. 
    """
    seq:Iterator[Any|int] = chain([start],count(0))
    if times is not None:
        seq = islice(seq,times)
    result = accumulate(seq,func)
    if iter_while:
        return takewhile(iter_while,result)
    return result


def iteratefunc2ord(
    a:T,
    b:T,
    binop:Callable[[T,T],T]=operator.add,
    func1:Callable[[T],T]=identity,
    func2:Callable[[T],T]=identity,
    times:int|None=None
    ) -> Iterator[T]:
    '''
    Generador de relaciones de recurrencia de 2do order:
    F0 = a
    F1 = b
    Fn = binop( func1( F(n-1) ),  func2( F(n-2)) )

    por ejemplo los numeros de Fibonacci
    F0=0
    F1=1
    Fn = F(n-1) + F(n-2)
    es  iteratefunc2ord(0,1,add) -> 0 1 1 2 3 5 8 ....

    F0=10
    F1=1
    Fn = 8*F(n-1) - F(n-2)**2
    es iteratefunc2ord(10,1,sub,lambda x: x*8,lambda y:y**2)
    -> 10, 1, -92, -737, -14360, ...


    a,b: casos bases
    func1, func2: funciones de 1 argumento que reciben el elemento
                  anterior, y el anterior a este en la secuencia
                  respectivamente. Por defecto son la funcion identidad.
    binop: funcion de 2 argumentos que toma el resultado de las funciones
           anteriormente nombradas y produce el siguiente elemento de
           secuencia.
    times: cantidad de elementos que se desea generar de la
           secuencia, si es None se crea una secuencia infinita.

        '''
    seq = pad_none([a,b])
    if times is not None:
        seq = islice(seq,times)
    try:
        yield next(seq) # type: ignore # a
        yield next(seq) # type: ignore # b
    except StopIteration:
        return
    for _ in seq:
        a,b = b, binop(func1(b),func2(a))
        yield b


@overload
def dual_accumulate(
    iterable:Iterable[T], 
    func1:Callable[[T,T],T]=..., 
    func2:Callable[[T,T],T]=...
    ) -> Iterator[Tuple[T,T]]: ...
@overload
def dual_accumulate(
    iterable:Iterable[T], 
    func1:Callable[[T|X,T],X]=..., 
    func2:Callable[[T|Y,T],Y]=...
    ) -> Iterator[Tuple[X,Y]]: ...

def dual_accumulate(
    iterable:Iterable[T], 
    func1:Callable[[T|X,T],X]=min,   #type: ignore
    func2:Callable[[T|Y,T],Y]=max    #type: ignore
    ) -> Iterator[Tuple[X,Y]]: 
    """
    Regresa una serie de valores correspondientes a la aplicación
    de las funciones dadas a cada elemento del iterable en una
    sola iteración.
    Equivalente a zip(accumulate(iterable,func1),accumulate(iterable,func2))
    """
    def dual_fun(previous:tuple[X|T,Y|T],new:T) -> tuple[X,Y]:
        x,y = previous
        return func1(x,new), func2(y,new)
    iterable = iter(iterable)
    try:
        ini = next(iterable)
    except StopIteration:
        return iter( () )
    return accumulate( chain([(ini,ini)],iterable), dual_fun ) #type: ignore


def strip_last(iterable:Iterable[T], n:int) -> Iterator[T]:
    """
    Entrega todos menos los ultimos n elementos del iterable
    strip_last("123",1) -> 1 2 
    """
    try:
        stop = len(iterable) - n #type: ignore
        if stop > 0:
            yield from islice(iterable,stop)
    except TypeError:
        n    = n+1
        it   = iter(iterable)
        cola = deque(islice(it,n), maxlen=n)
        while len(cola)==n:
            yield cola.popleft()
            try: #para futura compativilidad con python 3.7
                cola.append( next(it) )
            except StopIteration:
                return

@overload
def range_of(iterable:Iterable[Orderable[T]], stop:Orderable[T], /, *, key:Callable[[T],Orderable[T]]=... ) -> Iterator[T]: ...
@overload
def range_of(iterable:Iterable[T], stop:Orderable[X], /, *, key:Callable[[T],Orderable[X]]=... ) -> Iterator[T]: ...
@overload
def range_of(iterable:Iterable[Orderable[T]], start:Orderable[T], stop:Orderable[T], step:int|None=..., /, *, key:Callable[[T],Orderable[T]]=... ) -> Iterator[T]: ...
@overload
def range_of(iterable:Iterable[T], start:Orderable[X], stop:Orderable[X], step:int|None=..., /, *, key:Callable[[T],Orderable[X]]=... ) -> Iterator[T]: ...

def range_of( #type: ignore
    iterable:Iterable[T], 
    *argv,
    key:Callable[[T],Orderable[T]]=identity
    ) -> Iterator[T]:
    """
    range_of(iterable,stop[,*,key])
    range_of(iterable,start,stop[,step][,*,key])
    Dado un iterable ordenado de elementos entrega el equivalente a

    it = iter(iterable)
    x = next(it)
    while x <= stop:
       if star <= x:
           yield x
       x = next(it)
    """
    lim = slice(*argv)
    #result:Iterable[T]
    if lim.stop is None:
        result = iter(iterable)
    else:
        result = takewhile(lambda x: key(x)<=lim.stop, iterable )
    if lim.start is not None:
        result = dropwhile(lambda x: lim.start>key(x), result )
    if lim.step is None:
        return result
    else:
        return islice(result,None,None,lim.step)


def splitAt(n:int,iterable:Iterable[T]) -> Tuple[Iterator[T],Iterator[T]]:
    """Regresa 2 iterables, el primero contiene los primeros n elementos
       del iterable dado y el segundo contiene el resto"""
    a,b = tee(iterable,2)
    return islice(a,n),islice(b,n,None)


def recursive_map(iterable:Iterable[Any], func:Callable[[Any],T], recurtype:type|Tuple[type,...]=ABC_Sequence, ignoretype:type|Tuple[type,...]=_basestring ) -> Iterator[T]:
    """
    >>> numbers = (1, 2, (3, (4, 5)), 7)
    >>> mapped = recursive_map(numbers, str)
    >>> tuple(mapped)
    ('1', '2', ('3', ('4', '5')), '7')
    >>> complex_list = (1, 2, [3, (complex('4+2j'), 5)], map(str, (range(7, 10))))
    >>> tuple(recursive_map(complex_list, lambda x: x.__class__.__name__))
    ('int', 'int', ['int', ('complex', 'int')], 'map')

    """
    #http://stackoverflow.com/a/42095505/5644961
    for item in iterable:
        if isinstance(item, recurtype) and not isinstance(item,ignoretype):
            yield type(item)( recursive_map(item, func, recurtype, ignoretype ) )
        else:
            yield func(item)


@overload
def imean(iterable:Iterable[int]) -> float:...
@overload
def imean(iterable:Iterable[IntegralLike]) -> RealLike:...
@overload
def imean(iterable:Iterable[RealLike]) -> RealLike:...

def imean(iterable:"Iterable[NumberLike]") -> "NumberLike":
    """promedio de los valores del iterable"""
    n=0
    total=0
    for n,x in enumerate(iterable,1):
        total += x
    if n:
        return total/n
    else:
        raise ValueError("Iterable vacio")


def ijoin(conector:X, iterable:Iterable[T]) -> Iterator[T|X]:
    """ijoin(x,s) -> s0 x s1 x s2 x ... sn-1 x sn
       Similar a conector.join(iterable) de strings"""
    it = iter(iterable)
    try:
        a = next(it)
    except StopIteration:
        return
    try:
        b = next(it)
    except StopIteration:
        yield a
        return
    for a,b in pairwise(chain([a,b],it)):
        yield a
        yield conector
    yield b


def rindex(seq:Sequence[T], x:T) -> int:
    """Return seq.index(x) but look from the right,
       aka the last position of x, or -1 if it is not present"""
    #https://twitter.com/raymondh/status/987735142016864256
    return next(compress(count(len(seq)-1,-1), map(partial(operator.eq,x),reversed(seq))), -1)


def vectorsum(iterable:Iterable,
              container:Contenedor=tuple,
              vsum:Callable[[Function2,Iterable],Iterable]=starmap,
              add:Function2=operator.add,
              fillvalue:Any=0) -> Any:
    """Suma los "vectores" contenidos en el iterable
    >>> a=1,2,3
    >>> b=4,5,6
    >>> [a,b]
    [(1, 2, 3), (4, 5, 6)]
    >>> vectorsum([a,b])
    (5, 7, 9)
    >>>
    >>> c=0,0,0,23
    >>> vectorsum([a,b,c])
    (5, 7, 9, 23)
    >>>
    reduce(lambda x,y: container(vsum(add, zip_longest(x,y,fillvalue=fillvalue))),iterable)"""
    return reduce(lambda x,y: container( vsum(add, zip_longest(x,y,fillvalue=fillvalue))),iterable)


def alternatesign(iterable: Iterable[Any], pos:Function1=operator.pos, neg:Function1=operator.neg, posfirst:bool=True ) -> Iterator[Any]:
    """
    alternatesign(xs) -> pos(x0) neg(x1) pos(x2) neg(x3) ...
    >>> list(alternatesign(range(1,11)))
    [1, -2, 3, -4, 5, -6, 7, -8, 9, -10]
    >>> list(alternatesign(range(1,11),posfirst=False))
    [-1, 2, -3, 4, -5, 6, -7, 8, -9, 10]
    >>>
    """
    signs=(pos,neg) if posfirst else (neg,pos)
    return (sign(x) for sign,x in zip(cycle(signs),iterable))


def alternate(iterable:Iterable[Any], *alternatefuncs:Callable[[Any],T]) -> Iterator[T]:
    """
    alternate(xs,f1,f2,f3) -> f1(x0) f2(x1) f3(x2) f1(x3) f2(x4) f3(x5) ...
    """
    if alternatefuncs:
        if len(alternatefuncs)==1:
            return map(alternatefuncs[0],iterable)
        return ( func(x) for x,func in zip(iterable, cycle(alternatefuncs)))
    else:
        return iter(iterable)


def unzip(iterable:Iterable[Iterable[T]]) -> Iterator[Tuple[T,...]]:
    return zip(*iterable)


@overload
def rotations(iterable:Iterable[T]) -> Iterator[Tuple[T]]:...
@overload
def rotations(iterable:Iterable[T], container:Contenedor ) -> Iterator[Container[T]]:...

def rotations(iterable:Iterable[T], container:Contenedor=tuple ) -> Iterator[Container[T]]:
    """
    >>> list(rotations([1, 2, 3, 4]))
    [(1, 2, 3, 4), (2, 3, 4, 1), (3, 4, 1, 2), (4, 1, 2, 3)]
    >>> 
    """
    data = tuple(iterable)
    it = cycle(data)
    n = len(data)
    for _ in range(n):
        yield container(islice(it,n))
        next(it)


def lookahead(iterable:Iterable[T],looksteps:int=1, fillvalue:X|None=None) -> Iterator[Tuple[T|X,...]]:
    """
    s -> (s0,s1), (s1,s2), (s2, s3), ..., (sn-1,sn), (sn,None)

    >>> for x in ir.lookahead(range(10),3):
            x

    (0, 1, 2, 3)
    (1, 2, 3, 4)
    (2, 3, 4, 5)
    (3, 4, 5, 6)
    (4, 5, 6, 7)
    (5, 6, 7, 8)
    (6, 7, 8, 9)
    (7, 8, 9, None)
    (8, 9, None, None)
    (9, None, None, None)
    >>>
    """
    grupo = tee(iterable,looksteps+1)
    for i,e in enumerate(grupo):
        consume(e,i)
    return zip_longest(*grupo,fillvalue=fillvalue)


def isplit(iterable:Iterable[T], separator:Callable[[T],bool]|T=operator.not_, container:Contenedor=tuple) -> Iterator[Any]:
    """
    split the iterable into tuples (or whatever container)
    where the cut off are the given element
    or where the given predicate is true

    >>> list(isplit(range(50),lambda x:x%5==0))
    [(1, 2, 3, 4), (6, 7, 8, 9), (11, 12, 13, 14), (16, 17, 18, 19), (21, 22, 23, 24), (26, 27, 28, 29), (31, 32, 33, 34), (36, 37, 38, 39), (41, 42, 43, 44), (46, 47, 48, 49)]
    >>>
    >>> list(isplit([1,2,3,4,5,6,7,8,9,0,5,3,6,8,2,5,6],5))
    [(1, 2, 3, 4), (6, 7, 8, 9, 0), (3, 6, 8, 2), (6,)]
    >>>


    """
    if separator is operator.not_:
        key = bool
    else:
        if not callable(separator):
            key:Callable[[Any],bool] = partial(operator.ne,separator)
        else:
            key = lambda x: not separator(x) #type: ignore
    for k,v in groupby(iterable, key=key):
        if k:
            yield container(v)


def interesting_lines(iterable:Iterable[str], enum:bool|int|None=None, *, striper:Callable[[str],str]=str.strip, splitlines:Callable[[str],str]|None=str.splitlines) -> Iterator[str|Tuple[int,str]]:
    """
    strip and filter out the empty string from the given iterable.
    If enum is True enumerate the iterable first and then firter it out.
    If enum is an int enumerate the iterable first starting at enum and then firter it out.
    
    if splitlines is given and iterable is a string it apply it to the 
    iterable first before doing anything else


    >>> lines = ["a","","b"," ","c","   d", "e   ", " g "]
    >>>
    >>> list(interesting_lines(lines))
    ['a', 'b', 'c', 'd', 'e', 'g']
    >>> list(ir.interesting_lines(lines,True))
    [(0, 'a'), (2, 'b'), (4, 'c'), (5, 'd'), (6, 'e'), (7, 'g')]
    >>>
    >>> list(ir.interesting_lines(lines,17))
    [(17, 'a'), (19, 'b'), (21, 'c'), (22, 'd'), (23, 'e'), (24, 'g')]
    >>> list(ir.interesting_lines(lines,0))
    [(0, 'a'), (2, 'b'), (4, 'c'), (5, 'd'), (6, 'e'), (7, 'g')]


    """
    if splitlines and isinstance(iterable,str):
        iterable = splitlines(iterable)
    it = map(striper,iterable)
    if enum is not None:
        if isinstance(enum, bool):
            if enum:
                return filter(operator.itemgetter(1),enumerate(it))
        elif isinstance(enum, int):
            return filter(operator.itemgetter(1),enumerate(it, enum))
    return filter(None,it)


def dict_zip(*dicts, fillvalue:Any=_sentinela) -> Iterator[Tuple[Any,...]]:
    """

    https://www.youtube.com/watch?v=k2ZWyHdahEk
    
    """
    if not dicts:
        return
    first_dict,*others_dicts = dicts
    if fillvalue is not _sentinela:
        for key, first_val in first_dict.items():
            yield (key, first_val, *(other.get(key,fillvalue) for other in others_dicts))
        return
    n = len(first_dict)
    if any(len(d)!=n for d in others_dicts):
        raise ValueError("Arguments must have same length")
    for key, first_val in first_dict.items():
        yield (key, first_val, *(other[key] for other in others_dicts))


def skip(iterable:Iterable[T], to_skip:int) -> Iterator[T]:
    """
    skip every to_skip element from the iterable

    >>> list(skip(range(51),5)) #no multiple of 5
    [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 36, 37, 38, 39, 41, 42, 43, 44, 46, 47, 48, 49]
    >>>
    >>> list(skip("aAbBcC",2))
    ['A', 'B', 'C']
    >>>

    """
    if to_skip < 1:
        yield from iterable
    else:
        yield from (x for i,x in enumerate(iterable) if i%to_skip)


def insert_marker(iterable:Iterable[T], marker:T, predicate:Callable[[T],bool], after:bool=True) -> Iterator[T]:
    """
    insert the given element before or after the given predicate is true
    
    >>> list(ir.insert_item(range(1,21),"-", lambda x:x%5==0,after=True))
    [1, 2, 3, 4, 5, '-', 6, 7, 8, 9, 10, '-', 11, 12, 13, 14, 15, '-', 16, 17, 18, 19, 20, '-']
    >>> 
    >>> list(ir.insert_item(range(1,21),"-", lambda x:x%5==0,after=False))
    [1, 2, 3, 4, '-', 5, 6, 7, 8, 9, '-', 10, 11, 12, 13, 14, '-', 15, 16, 17, 18, 19, '-', 20]
    >>> 
        
    
    
    """
    for elem in iterable:
        if predicate(elem):
            pack = (elem, marker) if after else (marker, elem)
            yield from pack
        else:
            yield elem




__all__ = [ x for x in dir() if not (x.startswith("_") or x in __exclude_from_all__) ]
del __exclude_from_all__
