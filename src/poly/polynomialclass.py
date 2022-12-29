"""
Modulo para simples polinomios canonicos finitos
"""
from collections.abc import Iterable
from collections import namedtuple
from itertools import starmap
from operator import itemgetter, neg, pos, truediv 
from operator import add, sub, mul, mod, floordiv
from numbers import Number, Integral
from abc import abstractmethod, ABC

from .powclass import PowClass

__all__=['PolinomioBase']

Dom_term = namedtuple("Dom_term","coef grado")

try:
    from naturales.lib_combinatoria import triangulo_pascal, factorialDescendente
except (ImportError, SyntaxError):
    from functools import reduce
    from math import factorial

    def factorialDescendente(n,k):
        return reduce(mul,range(n+1-k,n+1),1)

    def combinatorio(n:int, k:int) -> int:
        """Número combinatorio N en K.
           Cuenta el número de sub-conjuntos de tamaño K de un N-conjunto
           O la cantidad de formas de elegir K objetos de entre N de ellos"""
        if k < 0:
            return 0
        if k == 0 or k == n:
            return 1
        k = min(k,n-k) if 0<=k<=n else k # take advantage of symmetry
        if k == 1:
            return n
        return factorialDescendente(n,k)//factorial(k)
        
    def triangulo_pascal(n):
        return (combinatorio(n,k) for k in range(n+1))


class NotANumberError(Exception):
    pass

class NotAPolynomialError(Exception):
    pass

def coeffunc(func:'F(coef,[*xs])->coef',*prev)-> 'G((exp,coef),*xs)->(exp,newcoef)':
    def fun(term:'(exp,coef)',*argv)->'(exp,newcoef)':
        i,e=term
        return i, func(e,*prev,*argv)
    return fun

def expfunc(func:'F(exp,[*xs])->exp',*prev)-> 'G((exp,coef),*xs)->(newexp,coef)':
    def fun(term:'(exp,coef)',*argv) -> '(newexp,coef)':
        i,e=term
        return func(i,*prev,*argv),e
    return fun



class PolinomioBase(PowClass,ABC):
    ''' m0 + m1X + m2X**2 + ... + mnX**n '''

    @property
    def grado(self) -> 'n':
        '''Grado de este polinomio'''
        return max( map(itemgetter(0), self.terminos()), default=0 )

    @property
    def dom_term(self) -> '(m,n)':
        '''Tupla (m,n) del termino mX**n de mayor grado'''
        n  = self.grado
        mn = self[n] #if self else 0
        return Dom_term(mn, n)

    @property
    def size(self):
        '''Cantidad de terminos del polinomio'''
        #return len(self)
        return sum( 1 for _ in self.terminos() )
        
    @abstractmethod 
    def terminos(self) -> '[(i,mi)]':
        '''Generador de tuplas (i,m) que representa el termino mX**i con m distinto de cero'''
        raise NotImplementedError

    @abstractmethod
    def __bool__(self):
        '''X != 0'''
        return False

    def __call__(self,valor):
        '''Evalua este polinomio en el valor dado'''
        return sum( a*pow(valor,e) for e,a in self.terminos() ) 

    def __repr__(self):
        return '{}({})'.format(self.__class__.__qualname__, super().__repr__())
        
    @staticmethod
    def format_x(e,a=1,x='x'):
        '''Regresa el string: ax**e'''
        if not a:
            return '0'
        a = '' if a==1 and e>0 else ("-" if a==-1 and e>0 else str(a) )
        if a :
            if not e:
                return a
            else:
                if e==1:
                    return a+x
                return a+'%s**%s'%(x,e)
        else:
            if not e:
                return '1'
            else:
                if e==1:
                    return x
                return '%s**%s'%(x,e)
            
    def __str__(self):
        result=[]
        for i,(e,a) in enumerate(sorted(self.terminos(),key=itemgetter(0),reverse=True)):
            try:
                if i and a>0:
                    result.append("+")
            except TypeError:
                if i: result.append("+")
            result.append(self.format_x(e,a))
        return " ".join(result) or "0"
        #return " + ".join( starmap(self.format_x, sorted(self.terminos(),key=itemgetter(0)) ) ) or "0"

    @abstractmethod
    def __getitem__(self, key):
        '''Optiene el coeficiente del termino de grado key'''
        return 0

    def __missing__(self, key):
        '''El coeficiente de un termino que no esta presente en este polinomo'''
        return 0

    @abstractmethod
    def __setitem__(self, key, value):
        '''Asigna este valor al coeficiente de grado key'''
        raise NotImplementedError

    @abstractmethod
    def __delitem__(self, key):
        '''Elimina el termino de grado key'''
        raise NotImplementedError

    @abstractmethod        
    def __contains__(self,key):
        '''Dice si el coeficiente del termino de grado key es no nulo'''
        return False       

    def aplicar(self, func:'F(x[,*xs])->y',*argv):
        '''Aplica la funcion a cada elemento de este polinomio in-place'''
        for i,m in self.terminos():
            self[i] = func(m,*argv)

    @classmethod
    def from_terminos(cls,iterable):
        '''Crea un polinomio a partir de una lista de tuplas (exp,coef) que representan el termino coef*X**exp'''
        new=cls()
        for i,m in filter(itemgetter(1),iterable):
            if i>=0:
                new[i] = m
        return new

    @classmethod
    def from_coeficientes(cls, iterable):
        '''Crea un polinomio a partir de una lista de coeficientes.
           Por ejemplo la lista [0,1] crea el polinomio x, [0,0,10] crea 10x**2, etc'''
        return cls.from_terminos( enumerate(iterable) )

    @classmethod
    def from_grados(cls,iterable):
        '''Crea un polnomio con todos sus coeficiente iguales a 1
           para los terminos de los grados espesificados.
           Por ejemplo la lista [0,1,4,7] crea el polinomio: 1+ x +x**4 +x**7'''
        return cls.from_terminos( (e,1) for e in iterable ) 

    @abstractmethod
    def __irshift__(self,n):
        '''P>>=n multiplica este polinomio por X**n'''
        return NotImplemented

    def __rshift__(self,n):
        '''P>>n multiplica este polinomio por X**n y lo regresa como un nuevo polinomio'''
        if isinstance(n,Integral):
            if n>=0:
                return self.from_terminos( starmap(lambda i,m: (i+n,m), self.terminos()) )
            raise ValueError('El valor del shift debe ser no negativo')
        raise TypeError('El valor del shift debe ser un numero entero')

    @abstractmethod
    def __ilshift__(self,n):
        '''P<<=n reduce el grado de cada termino de este polinomio por n y descarta grados negativos'''
        return NotImplemented        
        
    def __lshift__(self,n):
        '''P<<n reduce el grado de cada termino de este polinomio por n y lo regresa como un nuevo polinomio
           descartando grados negativos'''
        if isinstance(n,Integral):
            if n>=0:
                return self.from_terminos( starmap(lambda i,m: (i-n,m), self.terminos()) )
            raise ValueError('El valor del shift debe ser no negativo')
        raise TypeError('El valor del shift debe ser un numero entero')        
        
    def __neg__(self):
        '''-X'''
        return self.from_terminos( map(coeffunc(neg),self.terminos()) )

    def __pos__(self):
        '''+X'''
        return self.from_terminos( map(coeffunc(pos),self.terminos()) )

    def suma(self,otro,func=add):
        '''Suma este polinomio con otro, o con un iterable de terminos (exp,coef)
           in-place '''
        if isinstance(otro,PolinomioBase):
            poly = otro.terminos()
        else:
            poly = iter(otro)
        for i,m in poly:
            self[i] = func(self[i],m)
        self.clean()

    def __iadd__(self,otro):
        '''X+=Y suma este polinomio con otro'''
        if isinstance(otro,Number):
            self[0] += otro
            if not self[0]:
                del self[0]
        else:
            self.suma(otro,add)
        return self

    def suma_monomio(self,coef,exp=0):
        '''X += mx**e'''
        self[exp] += coef
        if not self[exp]:
            del self[exp]

    def __add__(self,otro):
        '''X+Y'''
        new = self.from_terminos(self.terminos())
        new+= otro
        return new
        
    def __radd__(self,otro):
        '''Y+X'''
        return self + otro

    def __isub__(self,otro):
        '''X-=Y'''
        if isinstance(otro,Number):
            self[0] -= otro
            if not self[0]:
                del self[0]
        else:
            self.suma(otro,sub)
        return self
        
    def __sub__(self,otro):
        '''X-Y'''
        new = self.from_terminos(self.terminos())
        new-= otro
        return new

    def __rsub__(self,otro):
        '''Y-X'''
        new = -self
        new += otro
        return new 

    def __mul__(self,otro):
        '''X*Y'''
        poly = self.__class__
        new = poly()
        if not self:
            return new
        if isinstance(otro,PolinomioBase):
            if otro:
                A = self
                B = otro
                if B.size > A.size:
                    A,B = B,A
                for e1,m1 in A.terminos():
                    for e2,m2 in B.terminos():
                        new[e1+e2] += m1*m2
        elif isinstance(otro,Iterable):
            for e1,m1 in otro:
                for e2,m2 in self.terminos():
                    new[e1+e2] += m1*m2
        elif isinstance(otro,Number):
            if otro:
                return poly.from_terminos( map(coeffunc(mul,otro),self.terminos()) )
        else:
            return NotImplemented
        new.clean()
        return new

    def __rmul__(self,otro):
        '''Y*X'''
        return self * otro

    @abstractmethod
    def clear(self):
        '''Borra todos los terminos de este polinomio'''
        raise NotImplementedError
        
    @abstractmethod
    def clean(self):
        '''Elimina todos los coeficientes nulos de este polinomio'''
        raise NotImplementedError

    def __imul__(self,otro):
        '''X *= c'''
        if not self:
            return self
        if isinstance(otro,Number):
            if otro:
                self.aplicar(mul,otro)
            else:
                self.clear()
            return self
        return NotImplemented 

    def __pow__ (self,n,m=None):
        ''' X**n [mod m] '''
        return super().__pow__(n,m)  

    def _division_escalar(self, n, div=truediv, in_place=False):
        '''X/n con n un numero'''
        if isinstance(n,Number):
            if not n:
                raise ZeroDivisionError
            if in_place:
                self.aplicar(div,n)
                self.clean()
                return self
            return self.from_terminos( map(coeffunc(div,n),self.terminos()) )
            #return  self.from_terminos( starmap(lambda i,m: (i,div(m,n)),self.terminos()) )
        raise NotANumberError

    def _division_poly(self, otro, div=truediv, in_place=False,mod=False):
        '''X / Y -> (Q,R) <==> X = YQ + R
           si in_place & mod entonces al final X=R
           si mod entoces Q no es calculado'''
        poly = self.__class__
        if isinstance(otro,PolinomioBase):
            md,ed= otro.dom_term
            if ed == 1 and md == 1 and not self[0] and not otro[0] and not in_place and not mod:
                # caso especial divicion entre x 
                return self<<1,0
            if md:
                q = poly()
                r = self if in_place and mod else poly.from_terminos(self.terminos())
                while r and r.grado >= ed:
                    mr,er = r.dom_term
                    t = div(mr,md)
                    if not mod:
                        q.suma_monomio(t,er-ed)
                    r.suma( starmap(lambda e,m:(e+(er-ed), t*m),otro.terminos()), sub )
                return q, r
            else:
                raise ZeroDivisionError
        raise NotAPolynomialError

    def _division(self, otro, div=truediv, in_place=False,mod=False):
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

    def division(self,otro,div=truediv):
        '''X/Y = (Q,R) <==> X = Q*Y + R
           Se emplea la division Euclidiana'''
        if isinstance(otro,(Number,PolinomioBase)):
            if callable(div):
                return self._division(otro,div,False,False)
            raise TypeError('Se debe proporcionar una funcion de divicion adecuada')
        raise TypeError('Solo se puede dividir por un numero u otro polinomio')

    def __mod__ (self,m):
        '''X%m --> X mod m'''
        if isinstance(m,Number):
            return self.from_terminos( map(coeffunc(mod,m),self.terminos()) )
        return self._division(m, floordiv, False,True)[1]

##    def __rmod__ (self,otro):
##        return NotImplemented

    def __imod__ (self,m):
        '''X %= m'''
        if isinstance(m,Number):
            self.aplicar(mod,m)
            self.clean()
            return self
        return self._division(m, floordiv, True,True)[1]            
  
    def __floordiv__ (self,otro):
        '''X // Y'''
        return self._division(otro, floordiv, False)[0]
    
    def __truediv__ (self,otro):
        '''X / Y'''
        return self._division(otro, truediv, False)[0]
    
    def __rtruediv__ (self,otro):
        return pow(self,-1) * otro
    
    def __rfloordiv__ (self,otro):
        return (self.from_coeficientes([1]) * otro) // self 

    def __itruediv__ (self,otro):
        return self._division(otro, truediv, True)[0]
    
    def __ifloordiv__ (self,otro):
        return self._division(otro, floordiv, True)[0]

    def __divmod__(self,otro):
        q,r = self._division(otro, floordiv, False)
        if q is NotImplemented:
            return NotImplemented
        return q,r


    @classmethod
    def binomio(cls,n,a=1,m=None) -> 'Polinomio':
        '''Regresa el polinomio: (X+a)**n [mod m]'''
        return cls.from_terminos( (i, ((c%m) if m else c)*pow(a,n-i,m) ) for i,c in enumerate(triangulo_pascal(n)) )

    @classmethod
    def monomio(cls,coef=1,exp=1) -> 'Polinomio':
        '''Regresa el polinomio: mX**n'''
        return cls.from_terminos( [(exp,coef)] )
        
    def derivada(self,n=1):
        '''Regresa el polinomio derivada de este polinomio
           Si n es suministrado calcula la n-esima derivada  '''
        if n==1:
            return self.from_terminos(  (i-1,m*i) for i,m in self.terminos() if i )
        elif n<0:
            raise ValueError('No se puede calcular derivadas negativas')
        ter = ( (i-n, m*factorialDescendente(i,n) ) for i,m in self.terminos() )
        return self.from_terminos( ter )

    @classmethod
    def _especialmod(cls,poly,r,n=None):
        '''Calcula X mod (x**r -1,n)'''
        new = cls()
        new.suma( filter(itemgetter(1), ( (i%r,(m%n) if n is not None else m) for i,m in poly) ) )
        return new

    def especialmod(self,r,n=None):
        '''Calcula X mod (x**r -1,n)'''
        return self._especialmod(self.terminos(),r,n)

    @classmethod
    def binomio_especialmod(cls,m,a,r,n=None):
        '''Calcula (X+a)**m mod (x**r -1,n)
           se asume (pero no se verifica) que r es menor que el grado de este polinomio'''        
        bino = ( (i%r, ((c%n) if n else c)*pow(a,m-i,n) ) for i,c in enumerate(triangulo_pascal(m)) )
        new = cls()
        new.suma( filter(itemgetter(1),bino) )
        return new

    
