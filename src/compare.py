# -*- coding: utf-8 -*-
'''
https://gist.github.com/bnlucas/dcc77c788e8823916ce3
def test1(stop=100):
    for i in range(stop):
        pass
    
def test2(stop=100):
    for i in xrange(stop):
        pass
        
compare = Compare('Testing the use of range and xrange.')
compare.add(test1, 10**5)
compare.add(test2, 10**5)
compare.run(1000, 100)


Outputs:

--------------------------------------------------------------------------------
 Testing the use of range and xrange.

 comparing best 100 of 1000 loops.
--------------------------------------------------------------------------------
 test2                                                    @        0.002969923s 
 test1                                                    @        0.004339960s 
--------------------------------------------------------------------------------
'''
from __future__ import print_function, division
from functools import wraps, partial
from contextlib import contextmanager
from time import time
import profile, timeit

__all__ = ['Compare', 'FuncCompare', 'benchmark', 'timeblock' ]

try:
    from itertools_recipes import consume
except ImportError:
    from collections import deque
    def consume(iterable):
        deque(iterable,maxlen=0)
try:
    from itertools import imap as map
except ImportError:
    pass

    

class Compare(object):

    def __init__(self, title=None):
        self.title = title
        self.methods = []

    def benchmark(self, loops, method, *args, **kwargs):
        call = method.__name__
        t1 = time()
        for i in range(loops):
            method(*args, **kwargs)
        t2 = time()
        runs =  t2 - t1
        t3 = int(1 / ((t2 - t1) / loops))
        return (call, runs, t3)


    def add(self, method, *args, **kwargs):
        self.methods.append( (method, (args, kwargs)) )

    def run(self, loops=100):

        runs = []
        for method in self.methods:
            benchmark = self.benchmark( loops, method[0], *method[1][0], **method[1][1] )
            runs.append(benchmark)

        runs.sort(key=lambda x:x[1]) 

        print( '-' * 80 )

        if self.title:
            print( ' ' + self.title )
            print( '' )

        print( ' Comparando {} loops.'.format( loops ) )
        print( '-' * 80 )

        for fun,tiempo,des in runs:
            print( ' {: <40} @ {: >18.9f}s {} call/s'.format(fun, tiempo,des) )

        print( '-' * 80 )
        
    def perfil(self,loops=None):
        if self.title:
            print( ' ' + self.title )
            print( '' )
        for method in self.methods:
            fun = method[0].__name__ + "("
            if method[1][0]: fun += ",".join( map(str,method[1][0]) )
            if method[1][1]: fun += ",".join( map(lambda x: str(x[0])+"="+str(x[1]), method[1][1].items() ) )
            fun +=")"
            print("*"*80)
            print(fun)
            profile.run(fun)
            if loops:
                fun,tiempo,des = self.benchmark( loops, method[0], *method[1][0], **method[1][1] )
                print( ' Comparando {} loops: {: >18.9f}s {} call/s'.format(loops, tiempo,des) )
            print("*"*80)

    def timeit(self,cantidad=1):
        if self.title:
            print( ' ' + self.title )
            print( '' )
        resul = []
        for method in self.methods:
            fun = lambda: method[0](*method[1][0], **method[1][1])
            resul.append( (method[0].__name__, timeit.timeit(fun,number=cantidad) ) )
        resul.sort(key=lambda x:x[1])
        for name,time in resul:
            print("{: <40} @ {: >18.9f}s con {} llamadas".format(name,time,cantidad) )
            
        
            
@contextmanager    
def timeblock(label="Este bloque se tardo"):
    """Contex manager para medir el tiempo de ejecucion de un bloque de codigo.
       Uso:
       with timethis("Este bloque se tardo"):
           #bloque de codigo
       """
    #https://www.youtube.com/watch?v=5-qadlG7tWo
    ini = time()
    try:
        yield
    finally:
        fin = time()
        print("%s: %0.8f seg"%(label,fin-ini))
        
def benchmark(iterations=10000):
    """Decorador para medir el tiempo de ejecucion de una función"""
    def wrapper(function):
        @wraps(function)
        def func(*args, **kwargs):
            t1 = time()
            for i in range(iterations):
                call = function(*args, **kwargs)
            t2 = time()
            t3 = int(1 / ((t2 - t1) / iterations))
            print( func.__name__, 'at', iterations, 'iterations:', t2 - t1)
            print( 'Can perform', t3, 'calls per second.')
            return call
        return func
    return wrapper

class FuncCompare(object):
    """Clase para la comparación de funciones"""

    def __init__(self, *funciones, **kwarg):
        self.titulo = kwarg.pop("titulo","")
        self.setup  = kwarg.pop("setup","pass")
        self.verbose = kwarg.pop("verbose",False)
        self._funciones = [(_get_name(f),f) for f in funciones]

    def add(self,func):
        """añade una función"""
        self._funciones.append( (_get_name(func),func ) )

    
    def _runer(self,fun_list):
        result = []
        for name, func in fun_list:
            if self.verbose:
                print("timing:", name,flush=True)
            timer = timeit.Timer(func, self.setup)
            n,_ = _autorange(timer)
            r = timer.repeat(3,n)
            best=min(r)
            result.append( (n,best,name) )
        print("",self.titulo,"\n" if self.titulo else "" )
        return _show_result( result )        

    def run(self):
        """timeit cada función"""
        return self._runer(self._funciones)

    def runwith(self,*argv, **kwarg):
        """timeit cada función con argumentos dados"""
        return self._runer( [(n, partial(f,*argv,**kwarg)) for n,f in self._funciones ] )

    def runfor(self, iterable):
        """para cada funcion realisa el timeit de
           for arg in iterable:
               func(arg)
            """
        def inner(fun):
            for x in iterable:
                fun(x)
        return self._runer( [(n, partial(inner,f)) for n,f in self._funciones ] )

    def runforwith(self, iterable,*argv, **kwarg):
        """para cada funcion realisa el timeit de
           for a in iterable:
               func(x,*argv, **kwarg)
            """
        def inner(fun):
            for x in iterable:
                fun(x,*argv, **kwarg)
        return self._runer( [(n, partial(inner,f)) for n,f in self._funciones ] )    

    
def _show_result(resul_list):
    units = {"usec": 1, "msec": 1e3, "sec": 1e6}
    scales = sorted( ((s,u) for u,s in units.items()), reverse=True )
    for loops, best, name in sorted(resul_list, reverse=True, key=lambda x:(x[0],-x[1],x[2])):
        print(name)
        print("%d loops"%loops, end=" ")
        usec = best*1e6 / loops
        for scale, time_unit in scales:
            if usec >= scale:
                break
        print("best of %d: %.*g %s per loop" % (3, 3, usec/scale, time_unit))
        print()
    
        

def _get_name(func):
    try:
        return func.__name__
    except AttributeError:
        return str(func)

def _autorange(timer):
    try:
        return timer.autorange()
    except AttributeError:
        pass
    for i in range(1,10):
        number = 10**i
        time_taken = timer.timeit(number)
        if time_taken >=0.2:
            break
    return number, time_taken    

    























