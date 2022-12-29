"""
Modulo para constantes arbitraria, para uso en calculos simbolicos
"""
import math, cmath, typing, operator
from abc import abstractmethod
from fractions import Fraction
from numbers import Number, Real, Complex, Rational
from itertools import chain
from collections import namedtuple

Partes_de_Constante = namedtuple("Partes_de_Constante","a m e C name")


__all__ =['ConstanteABC','ConstanteBase','sqrt','evaluar_constante','evaluar_formula']

class ConstanteABC(Number):
    '''Constante numerica arbitraria de forma: (a+mC**e) con a,m,e conocidos'''
        
    @property
    @abstractmethod
    def a(self):
        """(a+mC**e) -> a"""
        raise NotImplementedError

    @property
    @abstractmethod
    def m(self):
        """(a+mC**e) -> m"""
        raise NotImplementedError

    @property
    @abstractmethod
    def e(self):
        """(a+mC**e) -> e"""
        raise NotImplementedError

    @property
    @abstractmethod
    def C(self):
        """(a+mC**e) -> C"""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self):
        """Nombre de esta constante"""
        raise NotImplementedError

    def partes(self):
        """Regresa una tupla con las partes constitullentes de esta constante"""
        return Partes_de_Constante(a=self.a, m=self.m, e=self.e, C=self.C, name=self.name)

    def __call__(self,valor):
        return evaluar_constante(self,valor,self.name)

    def __repr__(self):
        a = ('a='+repr(self.a) ) if self.a    else ''
        m = ('m='+repr(self.m) ) if self.m!=1 else ''
        e = ('e='+repr(self.e) ) if self.e!=1 else ''
        C = ('C='+repr(self.C) ) if self.C    else ''
        return '{}({})'.format(self.__class__.__qualname__, ', '.join(chain([repr(self.name)],filter(None,(a,m,e,C)))) )

    def __str__(self):
        a,m,e,C,name = self.partes()
        add = str(a) if a else ''
        try:
            sig = '+' if m>=0 else ''
        except TypeError:
            sig = '+'
        con = ( str(m if m!=-1 else '-') if m!=1 else '' ) + (name if not C else str(C))
        exp = str(e) if e != 1 else ''
        if exp and exp.startswith("-"):
            exp = '({exp})'.format(exp=exp)
        resul=''
        if add:
            resul += add + ' ' + sig
        resul += con
        if exp:
            resul += '**'+exp
        return '({resul})'.format(resul=resul)
    
    def __lt__(self,otro):
        """C<X"""
        return NotImplemented

    def __eq__(self,otro):
        """C == X"""
        if isinstance(otro,ConstanteABC):
            return self.partes() == otro.partes()
        return False

    def __le__(self,otro):
        """C<=X"""
        return self==otro or self<otro

    def __bool__(self):
        """bool(C)"""
        return bool(self.m)

    def __neg__(self):
        """-C"""
        return self * (-1)

    def __pos__(self):
        """+C"""
        return self * 1
        
    def __sub__(self,otro):
        """C - X"""
        return self +(-otro)  
        
    def __rsub__(self,otro):
        """X - C"""
        return -self + otro

    def __radd__(self,otro):
        """X + C"""
        return self + otro

    def __rmul__(self,otro):
        """X * C"""
        return self * otro

    def __rfloordiv__(self,otro):
        """C//X"""
        return self._rdivision(otro, rational_div_maker(operator.floordiv))

    def __rtruediv__(self,otro):
        """C/X"""
        return self._rdivision(otro, rational_div_maker(operator.truediv))

    def __mod__(self,otro):
        """C%X"""
        if isinstance(otro, ConstanteABC) or not isinstance(otro,Number):
            return NotImplemented
        if otro == 0:
            raise ZeroDivisionError
        if self.C:
            C = self.__class__(self.C,e=self.e)
        else:
            C = self.__class__(self.name,e=self.e)
        m = self.m % otro
        a = self.a % otro
        return C*m + a

    def __floordiv__(self,otro):
        """C//X"""
        return self._division(otro, rational_div_maker(operator.floordiv))

    def __truediv__(self,otro):
        """C/X"""
        return self._division(otro, rational_div_maker(operator.truediv))


    @abstractmethod
    def __add__(self,otro):
        """C+X"""
        return NotImplemented

    @abstractmethod
    def __mul__(self,otro):
        """C * X"""
        return NotImplemented

    @abstractmethod
    def __pow__(self,otro,modulo=None):
        """C**X
           pow(C,X,m)"""
        return NotImplemented

    @abstractmethod
    def _division(self,otro,div):
        """C/X y/o C//X """
        return NotImplemented

    @abstractmethod    
    def _rdivision(self,otro,div):
        """X/C y/o X//C """
        return NotImplemented        


class ConstanteBase(ConstanteABC):
    '''Constante numerica arbitraria de forma: (a+mC**e) con a,m,e conocidos'''
    #implementación parcial, solo el __init__ y las propiedades

    def __init__(self,nombre,*,a=0,m=1,e=1,C=None):
        if C is not None and not isinstance(C,ConstanteABC):
            raise TypeError("El parametro C debe ser una instancia de {A}, no {B}".format(A=ConstanteABC,B=type(C)))
        if not m:
            raise ValueError("El parametro m no puede ser zero")
        if not e:
            raise ValueError("El parametro e no puede ser zero")
        if isinstance(nombre,ConstanteABC) and C is not None:
            raise ValueError("Si el parametro nombre es de tipo {A}, el parametro C no debe ser otorgado".format(A=ConstanteABC))
        if isinstance(nombre,ConstanteABC):
            C = nombre
            nombre = C.name
        if isinstance(nombre,str):
            name = nombre
            if C:
                if C.name != name:
                    raise ValueError("El parametro C debe tener el mismo nombre de esta constante que se desea crear")
                if not C.a:
                    if e==0.5:
                        m *= sqrt(C.m)
                    else:
                        m *= (C.m)**e
                    e *= C.e
                    C = None
                else:
                    if e == 1:
                        e = C.e
                        a = a + m*C.a
                        m = m*C.m
                        C = None
        else:
            raise TypeError("el parametro nombre debe ser de tipo string o {A} , no {B}".format(A=ConstanteABC,B=type(nombre)))
        self._c = C
        self._a = a
        self._m = m
        self._e = e
        self._name = name

    @property
    def a(self):
        """(a+mC**e) -> a"""
        return self._a

    @property
    def m(self):
        """(a+mC**e) -> m"""
        return self._m

    @property
    def e(self):
        """(a+mC**e) -> e"""
        return self._e

    @property
    def C(self):
        """(a+mC**e) -> C"""
        return self._c

    @property
    def name(self):
        """Nombre de esta constante"""
        return self._name

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

def evaluar_constante(cons:ConstanteABC,valor:Number,name:str=None) -> typing.Union[Number, ConstanteABC]:
    """Evalua el valor de la formula de la constante dada con el valor
       otorgado para la constante del nombre dado, que en caso de ser
       omitido sera cons.name"""
    if not isinstance(cons, ConstanteABC):
        return cons
    if name is None:
        return evaluar_constante(cons,valor,cons.name)
    if cons.C:
        c = evaluar_constante(cons.C,valor,name)
    else:
        c = valor if cons.name == name else cons.__class__(cons.name)
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

