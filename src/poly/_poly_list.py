"""
Modulo para simples polinomios canonicos finitos
"""
import itertools
import operator
from itertools import starmap, zip_longest, repeat, chain
from numbers import Number, Integral
from operator import itemgetter
from collections import Iterable, Iterator

from naturales.combinatoria import triangulo_pascal
from constante import Constante
from powclass import PowClass

__all__ =[
        'Polinomio',
        'Constante',
        'X',
        'a',
        'binomio',
        'poly_x_n'
    ]

class NotANumberError(Exception):
    pass

class NotAPolynomialError(Exception):
    pass

def sumatoria(data):
    if isinstance(data,Number):
        return data
    return sum(data)

class Polinomio(PowClass,list):
    ''' m0 + m1X + m2X**2 + ... + mnX**n '''

    @property
    def grado(self) -> 'n':
        '''grado de este polinomio'''
        self._clean()
        return (len(self)-1) if self else 0

    @property
    def dom_term(self) -> '(mn,n)':
        '''Coeficiente del termino dominante del polinomio'''
        n=self.grado
        mn = self[n] if self else 0
        return mn, n

    @property
    def size(self):
        '''cantidad de terminos del polinomio'''
        self._clean()
        return sum( 1 for _ in self.terminos() )
      
    def terminos(self) -> '[(i,mi)]':
        self._clean()
        return filter(itemgetter(1), enumerate(self))

    def __call__(self,valor):
        '''Evalua este polinomio en el valor dado'''
        return sum( a*pow(valor,e) for e,a in self.terminos() ) 
          
    def _clean(self):
        '''remueve los ceros del final'''
        while self and not self[-1]:
            self.pop()
                
    def __repr__(self):
        self._clean()
        return '{}({})'.format(self.__class__.__qualname__, super().__repr__())

    def __str__(self):
        def _x_(e,a):
            a = "" if a==1 else str(a)
            x = "x"
            if a :
                if not e:
                    return a
                else:
                    if e==1:
                        return a+"x"
                    return a+"x**%d"%e
            else:
                if not e:
                    return "1"
                else:
                    if e==1:
                        return "x"
                    return "x**%d"%e
                
        return " + ".join( starmap(_x_, self.terminos() ) ) or "0"

    def map(self, func:'F(x[,*xs])->y',*iterable):
        '''Aplica la funcion a cada elemento del polinomio'''
        if iterable:
            for i,ms in enumerate(zip(self,*iterable)):
                self[i] = func(*ms)
        else:
            for i,m in enumerate(self):
                self[i] = func(m)

    def __irshift__(self,n):
        '''P>>=n multiplica este polinomio por X**n'''
        self[0:0] = repeat(0,n)
        return self

    def __rshift__(self,n):
        '''P>>n multiplica este polinomio por X**n y lo regresa como un nuevo polinomio'''
        return self.__class__( chain(repeat(0,n),self) )


    def __neg__(self):
        '''-X'''
        return self.__class__( map(operator.neg,self) )

    def __pos__(self):
        '''+X'''
        return self.__class__( map(operator.pos,self) )    

    def __iadd__(self,otro):
        '''X+=Y suma este polinomio con otro'''
        if isinstance(otro,Iterable):
            poly=iter(otro)
            self.map(lambda x,m: x+sumatoria(m),poly)
            self.extend( map(sumatoria,poly) ) 
        elif isinstance(otro,Number):
            if len(self)>0:
                self[0] += otro
            else:
                self.append(otro)
        else:
            return NotImplemented
        self._clean()
        return self

    def suma(self,*poly):
        '''X+=Y1+Y2+...+Yn'''
        self+= zip_longest(*poly,fillvalue=0)

    def suma_monomio(self,m,e):
        '''X += mx**e'''
        g=self.grado
        if m:
            if g < e:
                if len(self) == 0:
                    g-=1
                self.extend( repeat(0,e-g) )
            self[e]+=m

    def __add__(self,otro):
        '''X+Y'''
        if isinstance(otro,Iterable):
            new=self.__class__()
            new.suma(self,otro)
            return new
        new=self.__class__(self)
        new+=otro
        return new
        

    def __radd__(self,otro):
        '''Y+X'''
        return self + otro

    def resta(self,*poly):
        '''X-=Y1+Y2+...+Yn'''
        self+=map(operator.neg,map(sumatoria, zip_longest(*poly,fillvalue=0)))

    def __isub__(self,otro):
        '''X-=Y'''
        if isinstance(otro, Iterable):
            self += map(operator.neg,otro)
        elif isinstance(otro,Number):
            self += (-otro)
        else:
            return NotImplemented
        return self

    def __sub__(self,otro):
        '''X-Y'''
        new = self.__class__(self)
        new -=otro
        return new

    def __rsub__(self,otro):
        '''Y-X'''
        return (-self)+otro

    def __mul__(self,otro):
        '''X*Y'''
        poly = self.__class__
        self._clean()
        if not self:
            return poly()
        if isinstance(otro,poly):
            if otro:
                new = poly( repeat(0,1 + self.grado + otro.grado) )
                A = self
                B = otro
                if len(B) > len(A):
                    A,B = B,A
                for e1,m1 in A.terminos():
                    for e2,m2 in B.terminos():
                        new[e1+e2] += m1*m2
                return new
            return poly()   
        elif isinstance(otro,Iterable):
            new = poly()
            for e,m in enumerate(otro):
                new += chain( repeat(0,e), map(lambda y:y*m,self) )
            return new
        elif isinstance(otro,Number):
            if otro:
                return poly( map(lambda x:x*otro,self) )
            return poly()
        return NotImplemented

    def __rmul__(self,otro):
        '''Y*X'''
        return self * otro

    def __imul__(self,otro):
        '''X *= c'''
        self._clean()
        if not self:
            return self
        if isinstance(otro,Number):
            if otro:
                self.map(lambda x:x*otro)
            else:
                self.clear()
            return self
        return NotImplemented 

    def __pow__ (self,n,m=None):
        ''' X**n [mod m] '''
        return super().__pow__(n,m)  

    def _division_escalar(self, n, div=operator.truediv, in_place=False):
        '''X/n con n un numero'''
        if isinstance(n,Number):
            if not n:
                raise ZeroDivisionError
            if in_place:
                self.map(div,repeat(n))
                return self
            return self.__class__( map(div,self,repeat(n)) )
        raise NotANumberError

    def _division_poly(self, otro, div=operator.truediv, in_place=False,mod=False):
        '''X / Y -> (Q,R) <==> X = YQ + R
           si in_place & mod entonces al final X=R
           si mod entoces Q no es calculado'''
        poly = self.__class__
        if isinstance(otro,poly):
            self._clean()
            md,ed= otro.dom_term
            if md:
                q = poly()
                r = self if in_place and mod else poly(self)
                while r and r.grado > ed:
                    mr,er = r.dom_term
                    t = div(mr,md)
                    if not mod:
                        q.suma_monomio(t,er-ed)
                    r -= chain(repeat(0,er-ed),map(operator.mul,otro,repeat(t)))
                return q, r
            else:
                raise ZeroDivisionError
        raise NotAPolynomialError

    def _division(self, otro, div=operator.truediv, in_place=False,mod=False):
        '''X / Y o X / n'''
        try:
            return self._division_escalar(otro, div, in_place), 0
        except NotANumberError:
            pass
        try:
            return self._division_poly(otro, div, in_place,mod)
        except NotAPolynomialError:
            pass
        return NotImplemented, NotImplemented

    def __mod__ (self,m):
        '''X%m --> X mod m'''
        if isinstance(m,Number):
            new = self.__class__(map(operator.mod,self,repeat(m)))
            new._clean()
            return new
        return self._division(m, operator.floordiv, False,True)[1]

    def __rmod__ (self,otro):
        return NotImplemented

    def __imod__ (self,m):
        '''X %= m'''
        if isinstance(m,Number):
            self.map(lambda x: x%m)
            self._clean()
            return self
        return self._division(m, operator.floordiv, True,True)[1]            
  
    def __floordiv__ (self,otro):
        '''X // Y'''
        return self._division(otro, operator.floordiv, False)[0]
    
    def __truediv__ (self,otro):
        '''X / Y'''
        return self._division(otro, operator.truediv, False)[0]
    
    def __rtruediv__ (self,otro):
        return pow(self,-1) * otro
    
    def __rfloordiv__ (self,otro):
        return (self.__class__([1]) * otro) // self 

    def __itruediv__ (self,otro):
        return self._division(otro, operator.truediv, True)[0]
    
    def __ifloordiv__ (self,otro):
        return self._division(otro, operator.floordiv, True)[0]

    def __divmod__(self,otro):
        q,r = self._division(otro, operator.floordiv, False)
        if q is NotImplemented:
            return NotImplemented
        return q,r

######################################################################################
# ------------------------------------- Funciones adicionales ----------------------------------------------------------------    
######################################################################################
Constante._poly_ = Polinomio

X = Polinomio([0,1])
a = Constante('a')

def binomio(n,a=Constante('a'),m=None) -> Polinomio:
    '''regresa el polinomio: (X+a)**n [mod m]'''
    return Polinomio( ((c%m) if m else c)*pow(a,n-i,m) for i,c in enumerate(triangulo_pascal(n)) )

def poly_x_n(n) -> Polinomio:
    '''Regresa el polinomio: x**n'''
    return Polinomio( chain( repeat(0,n),[1]) )
