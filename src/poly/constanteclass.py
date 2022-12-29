"""
Modulo para constantes arbitraria, para uso en calculos simbolicos
"""
import math, cmath, typing, operator, functools
from abc import abstractmethod,ABC
from fractions import Fraction
from numbers import Number, Real, Complex, Rational, Integral
from itertools import chain
#from collections import namedtuple
from .powclass import PowClass
from typing import Callable, Tuple

try:
    from abc_recipes import ConfigClass
except ImportError:
    
    class ConfigClass(ABC):
        """Cooperative class that offer a default __repr__ method
           based on the abstract property .config"""

        @property
        @abstractmethod
        def config(self) -> dict:
            """configuration of this class"""
            return {}

        def __repr__(self) -> str:
            return f"{type(self).__name__}({', '.join( f'{k}={v!r}' for k,v in self.config.items() )})"

        def make_new(self,**new_config):
            """Create a new instance of this class with the same configuration, plus whatever change are requested"""
            c = self.config
            c.update(new_config)
            return type(self)(**c)

#Partes_de_Constante = namedtuple("Partes_de_Constante","a m e value name")


__all__ =['ConstanteABC','ConstanteBase','sqrt','evaluar_constante','evaluar_formula']

class ConstanteABC(Number, PowClass, ConfigClass):
    '''Constante numerica arbitraria de forma: (a+mC**e) con a,m,e conocidos'''

    __slots__ = ()
        
    @property
    @abstractmethod
    def a(self) -> Number:
        """(a+mC**e) -> a"""
        return 0

    @property
    @abstractmethod
    def m(self) -> Number:
        """(a+mC**e) -> m"""
        return 1

    @property
    @abstractmethod
    def e(self) -> Number:
        """(a+mC**e) -> e"""
        return 1

    @property
    def config(self) -> dict:
        return dict(a=self.a, m=self.m, e=self.e)

    def __call__(self, valor:Number) -> Number:
        return evaluar_constante(self, valor, self.name)
        
    @abstractmethod
    def __add__(self,otro:Number) -> Number:
        """self + otro"""
        return NotImplemented

    @abstractmethod
    def __mul__(self,otro:Number) -> Number:
        """self * otro"""
        return NotImplemented

    @abstractmethod
    def __mod__(self,otro:Number) -> Number:
        """self % otro"""
        return NotImplemented

    @abstractmethod
    def _division(self,otro:Number,div:Callable[[Number,Number],Number]) -> Number:
        """self / otro y/o self // otro """
        return NotImplemented

    @abstractmethod    
    def _rdivision(self,otro:Number,div:Callable[[Number,Number],Number]) -> Number:
        """otro/self y/o otro//self """
        return NotImplemented    
    
    def __eq__(self,otro:Number) -> bool:
        """self == otro"""
        if isinstance(otro,ConstanteABC):
            return self.config == otro.config
        return False

    def __hash__(self):
        """hash(self)"""
        #https://docs.python.org/3/reference/datamodel.html#object.__hash__
        #if a child overwrite __eq__ it should also overwrite this even if just to call this one
        return hash(frozenset(self.config.items()))  

    def __bool__(self) -> bool:
        """bool(self)"""
        return bool(self.m)

    def __neg__(self) -> Number:
        """- self"""
        return self * (-1)

    def __pos__(self) -> Number:
        """+ self"""
        return self * 1
        
    def __sub__(self,otro) -> Number:
        """self - otro"""
        return self +(-otro)  
        
    def __rsub__(self,otro:Number) -> Number:
        """otro - self"""
        return -self + otro

    def __radd__(self,otro:Number) -> Number:
        """otro + self"""
        return self + otro

    def __rmul__(self,otro:Number) -> Number:
        """otro * self"""
        return self * otro

    def __rfloordiv__(self,otro:Number) -> Number:
        """self // otro"""
        return self._rdivision(otro, rational_div_maker(operator.floordiv))

    def __rtruediv__(self,otro:Number) -> Number:
        """self / otro"""
        return self._rdivision(otro, rational_div_maker(operator.truediv))

    def __floordiv__(self,otro:Number) -> Number:
        """self // otro"""
        return self._division(otro, rational_div_maker(operator.floordiv))

    def __truediv__(self,otro:Number) -> Number:
        """self / otro"""
        return self._division(otro, rational_div_maker(operator.truediv))

    def __rmod__(self,otro):
        """otro % self"""
        return NotImplemented
    
    def __rpow__(self, otro):
        """otro ** self"""
        return NotImplemented
    
    

    def _casting(self, caster):
        val = self()
        if isinstance(val,type(self)):#just in case to not fall into an infinete recursion loop
            raise TypeError(f"can't optain concrete value in order to cast it to: {caster}")
        return caster(val)

    def __abs__(self) -> Number:
        """abs(self)"""
        return self._casting(abs)
        
    def __float__(self) -> float:
        """float(self)"""
        return self._casting(float)
        
    def __int__(self) -> int:
        """int(self)"""
        return self._casting(int)

    # def __trunc__(self) -> Integral:
        # """math.trunc(self)"""
        # return self._casting(math.trunc)

    # def __floor__(self) -> Integral:
        # """math.floor(self)"""
        # return self._casting(math.floor)

    # def __ceil__(self) -> Integral:
        # """math.ceil(self)"""
        # return self._casting(math.ceil)
        
    # def __round__(self, ndigits:int=None) -> Number:
        # """round(self,ndigits)"""
        # return self._casting(functools.partial(round,ndigits=ndigits))
    



class ConstanteBase(ConstanteABC):
    '''Constante numerica arbitraria de forma: (a+mC**e) con a,m,e conocidos'''

    __slots__= ("_a", "_m", "_e", "_value", "_name")

    def __init__(self,*, a:Number=0, m:Number=1, e:Number=1, value:Number=None, name:str=None):
        if not m:
            raise ValueError("El parametro m no puede ser zero")
        if not e:
            raise ValueError("El parametro e no puede ser zero")
        if value is None and not name:
            raise ValueError("Constante sin valor o nombre")
        self._a = a or 0
        self._m = m
        self._e = e
        self._value = value
        if value is not None and not name:
            name = f"({value})"
        self._name = name

    @property
    def a(self) -> Number:
        """(a+mC**e) -> a"""
        return self._a

    @property
    def m(self) -> Number:
        """(a+mC**e) -> m"""
        return self._m

    @property
    def e(self) -> Number:
        """(a+mC**e) -> e"""
        return self._e
    
    @property
    def value(self) -> Number:
        return self._value
    
    @property
    def name(self) -> str:
        return self._name
        
    @property
    def config(self) -> dict:
        c = super().config
        c.update(value=self.value, name=self.name)
        return c

    def __str__(self) -> str:
        a = str(self.a or "")
        e = str(self.e) if self.e != 1 else ""
        m = str(self.m if self.m !=-1 else "-") if self.m != 1 else ""
        C = self.name
        try:
            sig = '+' if self.m>=0 else ''
        except TypeError:
            sig = '+'
        if e.startswith("-"):
            e = f'({e})'
        resul=""
        if a:
            resul += a + ' ' + sig
        resul += m+C
        if e:
            resul += '**' + e
        return f'({resul})'

    def _partes(self) -> Tuple[Number,str]:
        return self.a, self.m, self.e, self.value, self.name
        
    def __add__(self,otro:Number) -> Number:
        """C+X"""
        #print(f"add: \n{self=} \n{otro=}\n")
        if not isinstance(otro,Number):
            return NotImplemented
        a,m,e,V,n = self._partes()
        if V is not None and ( e == 1 or V in (0,1)):
            return m*V + a + otro
        if not m:
            return a+otro
        if isinstance(otro,type(self)):
            a2,m2,e2,V2,n2 = otro._partes()
            if e==e2 and ((V is not None and V==V2 ) or (V is None and n==n2 )):
                m = m+m2
                n = n or n2
                if m:
                    return type(self)(value=V, e=e, m=m, a=a+a2, name=n)
                else:
                    return a+a2
        return self.make_new(a=self.a+otro)

    def __mul__(self,otro:Number) -> Number:
        """C * X"""
        #print(f"mul: \n{self=} \n{otro=}\n")
        if not isinstance(otro,Number):
            return NotImplemented
        if not otro:
            return 0
        a1,m1,e1,V,n = self._partes()
        if V is not None and ( e1 == 1 or V in (0,1)):
            return (m1*V + a1) * otro
        if not m1:
            return a1*otro
        if isinstance(otro, type(self)):# and ( (V is not None and V==otro.value) or (V is None and n==otro.name)):
            new = type(self)
            a2,m2,e2,V2,n2 = otro._partes()
            if (V is not None and V==V2) or (V is None and n==n2):
                e = e1 + e2
                m = m1*m2
                a = a1*a2
                n = n or n2
                if not e or e==1:
                    result = new(value=V, name=n, e=e1, m=m1)*a2 + new(value=V, name=n, e=e2, m=m2)*a1 + a
                    if not e:
                        return result + m
                    #e==1
                    if V is None:
                        return new(value=V, name=n, m=m) + result
                    else:
                        return result + m*V
                if e1==e2:
                    m3 = m1*a2 + m2*a1
                    return new(value=V, name=n, e=e, m=m) + new(value=V, name=n, e=e1)*m3 + a
                return new(value=V, name=n, e=e, m=m) + new(value=V, name=n, e=e1, m=m1)*a2 + new(value=V, name=n, e=e2, m=m2)*a1 + a
            else:
                res = new(value=V,name=n,e=e1, m=m1)*a2 + new(value=V2,name=n2,e=e2, m=m2)*a1 + a1*a2
                m = m1*m2
                if V is not None and V2 is not None:
                    if isinstance(e1, Rational) and isinstance(e2, Rational):
                        if e1.denominator == e2.denominator:
                            X = V**e1.numerator
                            Y = V2**e2.numerator
                            V = X*Y
                            if isinstance(V, ConstanteABC):
                                return (V**Fraction(1,e1.denominator))*m + res
                            else:
                                return new(value=V, e=Fraction(1,e1.denominator), name=n)*m + res
                return new(value=V, name=n, e=e1, m=new(value=V2, name=n2, e=e2)*m) + res
        return self.make_new(m=self.m*otro, a=self.a*otro)

    def __mod__(self,otro:Number) -> Number:
        """C%X"""
        if not isinstance(otro,Number):
            return NotImplemented
        if not otro:
            raise ZeroDivisionError
        a,m,e,V,n = self._partes()
        a = a%otro
        m = m%otro
        if isinstance(e,Rational) and e.denominator==1:
            e = int(e)
        if isinstance(V, Integral) and isinstance(e,Integral):
            V = pow(V,e,otro)
            return (V*m+a)%otro
        return type(self)(value=V, name=n, e=e)*m + a

    def _division(self,otro:Number,div:Callable[[Number,Number],Number]) -> Number:
        """C/X y/o C//X """
        if not isinstance(otro,Number):
            return NotImplemented
        if not otro:
            raise ZeroDivisionError
        if self==otro:
            return 1
        a,m,e,V,n = self._partes()
        return type(self)(value=V, name=n, e=e)*div(m,otro) + div(a,otro)

    def _rdivision(self,otro:Number,div:Callable[[Number,Number],Number]) -> Number:
        """X/C y/o X//C """
        if not isinstance(otro,Number):
            return NotImplemented
        if not self:
            raise ZeroDivisionError
        if not otro:
            return 0
        if self==otro:
            return 1
        if (hasattr(self.e,"denominator") and self.e.denominator==2) or self.e==0.5:
            #Rationalizing the Denominator
            a,m,e,V,n = self._partes()
            con = type(self)(value=V, name=n,e=e, m=m, a=-a)
            dem = self*con
            return otro*con*div(1,dem)
        if not self.a:
            return type(self)(value=self.value, name=self.name, e=-self.e)*div(1,self.m)*otro
        return type(self)(value=self, e=-1) * otro

    def __pow__(self, otro:Number, mod:Number=None) -> Number:
        if not isinstance(otro, Number):
            return NotImplemented
        if mod is not None and not isinstance(otro, Integral):
            raise ValueError("pow() 3rd argument cannot be provided if the 2nd argument is not and Integral number")
        if not otro:
            return 1
        if not self:
            return 0
        r = super().__pow__(otro, mod)
        if r is NotImplemented:
            if not self.a:
                _,m,e,V,n = self._partes()
                e = self.e*otro
                if isinstance(otro, Integral):
                    m = pow(m,otro,mod)
                else:
                    if isinstance(m, ConstanteABC):
                        m = m**otro
                    else:
                        m = type(self)(value=self.m, e=otro, name=n)
                if not e:
                    return m
                if e==1:
                    if self.value is not None:
                        return V * m
                    else:
                        return type(self)(name=n) * m
                return type(self)(value=V, name=n, e=e) * m
            return type(self)(value=self, e=otro, name=None)
        return r
        

    def __call__(self, valor:Number=None) -> Number:
        if self.value is None:
            if valor is None:
                raise ValueError("Se debe proveer el valor para esta constante")
            return super().__call__(valor)
        else:
            if valor is not None:
                raise ValueError("Esta constante ya posee un valor")
            return super().__call__(self.value)
    
    

def rational_div_maker(div):
    def rational_div(a,b):
        """Si ambos argumentos son Racionales, entonces regresa Fraction(a,b)
            sino regresa div(a,b)"""
        if isinstance(a,Rational) and isinstance(b,Rational):
            return Fraction(a,b)
        return div(a,b)
    return rational_div

def sqrt(x:typing.Union[Real,Complex,ConstanteABC], _fraction_resul=False) -> typing.Union[Real,Complex,ConstanteABC]:
    """Calcula la raiz cuadrada de x, segun el valor de x
       retornando un número complejo o una Constante de ser necesario

       _fraction_resul si es True y x es una fraccion no negativa, entoces el
       resultado sera una fraccion"""
    if isinstance(x,ConstanteABC):
        return x**Fraction(1,2)
    try:
        if _fraction_resul and x>=0 and isinstance(x,Fraction):
            n = Fraction( *(sqrt(x.numerator).as_integer_ratio()   ) )
            d = Fraction( *(sqrt(x.denominator).as_integer_ratio() ) )
            return n/d
        return math.sqrt(x) if x>=0 else cmath.sqrt(x)
    except TypeError:
        return cmath.sqrt(x)

def evaluar_constante(cons:ConstanteABC, valor:Number, name:str=None) -> typing.Union[Number, ConstanteABC]:
    """Evalua el valor de la formula de la constante dada con el valor
       otorgado para la constante del nombre dado, que en caso de ser
       omitido sera cons.name"""
    if not isinstance(cons, ConstanteABC):
        return cons
    if name is None:
        return evaluar_constante(cons,valor,cons.name)
    if cons.value is not None:
        c = evaluar_constante(cons.value, valor, name)
    else:
        c = valor if cons.name == name else type(cons)(name=cons.name)
    a = evaluar_constante(cons.a,valor,name)
    m = evaluar_constante(cons.m,valor,name)
    e = evaluar_constante(cons.e,valor,name)
    if e == 0.5:
        return sqrt(c)*m + a
    return m*c**e + a        

def evaluar_formula(formula:ConstanteABC,*valores:[("nombre","valor")]) -> typing.Union[Number, ConstanteABC]:
    """Evalua la formula contenida en la constante dada."""
    resul = formula
    for c,v in valores:
        resul = evaluar_constante(resul,v,c)
    return resul

