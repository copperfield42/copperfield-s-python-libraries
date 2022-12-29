"""
Modulo para constantes arbitraria, para uso en calculos simbolicos
"""
if __name__ == "__main__" and not __package__:
    from relative_import_helper import relative_import_helper
    __package__ = relative_import_helper(__file__,1)
    del relative_import_helper
    print("idle trick")

from .constanteclass import ConstanteABC, ConstanteBase, sqrt, evaluar_constante, evaluar_formula
from .powclass import PowClass


from fractions import Fraction
from numbers import Number, Integral
from itertools import count as _count
import operator as _operator


__all__ = ['Constante', 'ConstanteABC', 'ConstanteBase', 'sqrt', 'evaluar_constante', 'evaluar_formula']

class Constante(PowClass, ConstanteBase):
    '''Constante numerica arbitraria de forma: (a+mC**e) con a,m,e conocidos'''

    def __add__(self,otro):
        """C+X"""
        #print("add",self,otro)
        if isinstance(otro,ConstanteABC):
            X,Y = sorted([self,otro],key=lambda x:x.name)
            a1,m1,e1,C1,name1 = X.partes()
            a2,m2,e2,C2,name2 = Y.partes()
            C = C1
            e = e1
            name = name1
            if (C1 and C2 and C1==C2 and e1==e2) or (not C1 and not C2 and name1==name2 and e1==e2):
                m = m1 + m2
                a = a1 + a2
            else:
                m = m1
                a = a1 + Y
            if not m:
                return a
            return self.__class__(name, a=a, m=m, e=e, C=C)
        else:
            return self.__class__(self.name, e=self.e, m=self.m, C=self.C, a=self.a+otro)
        raise NotImplementedError

    def __mul__(self,otro):
        """C * X"""
        #print("mul",self,otro)
        if otro:
            if isinstance(otro,ConstanteABC):
                X,Y = sorted([self,otro],key=lambda x:x.name)
                #print("mul",self,otro,"->",X,Y)
                a1,m1,e1,C1,name1 = X.partes()
                a2,m2,e2,C2,name2 = Y.partes()
                cons = self.__class__
                XX = cons(name1,C=C1,e=e1,m=m1)*a2
                YY = cons(name2,C=C2,e=e2,m=m2)*a1
                AA = a1*a2
                M  = m1*m2
                if (C1 and C2 and C1==C2) or (not C1 and not C2 and name1==name2):
                    e = e1+e2
                    if e:
                        return cons(name1,C=C1,e=e,m=M) + XX + YY + AA
                    else:
                        return XX + YY + M + AA
                XYA = XX + YY + AA
                if e1 == e2 and C1 and C2:
                    e = e1
                    return ((C1*C2)**e)*M + XYA
                return cons(name1,C=C1,e=e1,m=m1*cons(name2,C=C2,e=e2,m=m2)) + XYA
            else:
                return self.__class__(self.name, C=self.C, e=self.e, m=self.m*otro, a=self.a*otro)
        else:
            return 0

    def __pow__(self,otro,modulo=None):
        """C**X
           pow(C,X,m)"""
        if isinstance(otro, ConstanteABC) or not isinstance(otro,Number):
            return NotImplemented
        if modulo is not None:
            if not (isinstance(otro,Integral) and isinstance(modulo,Integral) ):
                raise TypeError("pow() 3rd argument not allowed unless all arguments are integers")
        if not otro:
            return (1%modulo) if modulo is not None else (1)
        elif otro == 1:
            return (self%modulo) if modulo is not None else (+self)
        else:
            # (a+m(C)**e) ** x
            if int(otro) == otro:
                otro = int(otro)
            if otro == 0.5:
                otro = Fraction(1,2)
            a,m,e,C,name = self.partes()
            if a:
                # (a+m(C)**e) ** x
                if otro>0:
                    new = super().__pow__(otro,modulo)
                    if new is not NotImplemented:
                        return new
                else:
                    if modulo is not None:
                        raise ValueError("pow() 2nd argument cannot be negative when 3rd argument specified")
                return self.__class__(self,e=otro)
            else:
                # (m(C)**e) ** x
                if m != 1:
                    m = sqrt(m) if otro == 0.5 else m**otro                            
                if C:
                    if modulo is None:
                        return (C**(e*otro))*m
                    else:
                        return pow(C,e*otro,modulo)*(m%modulo)
                else:
                    return self.__class__(name,e=e*otro,m=m)
                raise NotImplementedError("(m(C)**e) ** x")
        raise NotImplementedError

    def _division(self,otro,div):
        """C/X y/o C//X """
        if not isinstance(otro,Number):
            return NotImplemented
        if otro:
            if self == otro:
                return 1
            if isinstance(otro,ConstanteABC):
                a1,m1,e1,C1,name1 = self.partes()
                a2,m2,e2,C2,name2 = otro.partes()
                cons = self.__class__
                if a2: # (a1+m1(C1)**e1) / (a2+m2(C2)**e2)
                    return self * (otro)**(-1)
                else:  # (a1+m1(C1)**e1) / (m2(C2)**e2)
                    if C2:
                        C3 = C2**(-e2)
                    else:
                        C3 = cons(name2,e=-e2)
                    A = C3*div(a1,m2)
                    if C1:
                        return cons(C1,e=e1,m=div(m1,m2))*C3 + A
                    else:
                        return cons(name1,e=e1,m=div(m1,m2))*C3 + A
            else:
                return self.__class__(self.name, C=self.C, e=self.e, m=div(self.m,otro), a=div(self.a,otro) )            
        else:
            raise ZeroDivisionError

    def _rdivision(self, otro, div):
        return NotImplemented

class RadicalConstante(PowClass, ConstanteBase):

    def __init__(self, value:Number, raiz:int=None,**kwarg):
        self._valor = value
        if not raiz:
            raiz = kwarg.pop("e", Fraction(1,2))
        else:
            raiz = Fraction(1,raiz)
        if raiz==1 or raiz==0:
            raise ValueError("the exponent can't be 1 or 0")
        super().__init__(f"({value})", e=raiz,**kwarg)

    @property
    def value(self):
        return self._valor

    def __eq__(self,otro):
        if isinstance(otro, type(self)):
            return self.value==otro.value and super().__eq__(otro)
        return super().__eq__(otro)

    def __add__(self, otro):
        if not isinstance(otro,Number):
            return NotImplemented
        if self.e == 1 or self.value in (0,1):
            return self.value*self.m + self.a + otro
        if isinstance(otro, type(self)):
            a1,m1,e1,_,_ = self.partes()
            a2,m2,e2,_,_ = otro.partes()
            if self.value == otro.value and e1 == e2:
                m = m1+m2
                if m:
                    return type(self)(self.value, e=e1, m=m, a=a1+a2)
                else:
                    return a1+a2
        return type(self)(self.value, e=self.e, m=self.m, a=self.a + otro)

    def __mul__(self,otro):
        if not isinstance(otro,Number):
            return NotImplemented
        if not otro:
            return 0
        if self.e == 1 or self.value in (0,1):
            return (self.value*self.m + self.a) * otro
        if isinstance(otro, type(self)) and self.value == otro.value:
            new = type(self)
            R = self.value
            a1,m1,e1,_,_ = self.partes()
            a2,m2,e2,_,_ = otro.partes()
            e = e1+e2
            m = m1*m2
            a = a1*a2
            if not e:
                return new(R,e=e1,m=m1)*a2 + new(R,e=e2,m=m2)*a1 + (a + m)
            if e == 1:
                return new(R,e=e1,m=m1)*a2 + new(R,e=e2,m=m2)*a1 + (a + m*R)
            if e1==e2:
                m3 = m1*a2 + m2*a1
                return new(R, e=e, m=m) + new(R, e=e1)*m3 + a               
            return new(R,e=e,m=m) + new(R,e=e1,m=m1)*a2 + new(R,e=e2,m=m2)*a1 + a
        return type(self)(self.value, e=self.e, m=self.m*otro, a=self.a*otro)
        
    def _division(self, otro, div):
        """self/otro"""
        if not isinstance(otro,Number):
            return NotImplemented
        if not otro:
            raise ZeroDivisionError
        if self==otro:
            return 1
        return type(self)(self.value, e=self.e)*div(self.m,otro) + div(self.a,otro)

    def _rdivision(self, otro, div):
        """otro/self"""
        if not isinstance(otro,Number):
            return NotImplemented
        if not self:
            raise ZeroDivisionError
        if not otro:
            return 0
        if self==otro:
            return 1
        if not self.a:
            return type(self)(self.value,e=-self.e)*div(1,self.m) * otro
        return type(self)(self, e=-1) * otro        


    def __call__(self):
        return super().__call__(self.value)

    def __mod__(self, mod):
        if not isinstance(mod,Number):
            return NotImplemented
        if not mod:
            raise ZeroDivisionError
        a,m,e,_,_ = self.partes()
        m = m%mod
        a = a%mod
        return type(self)(self.value,e=e)*m + a

    def __abs__(self):
        return abs(self())
        


a,b,c = map(Constante,"abc")

golden = (RadicalConstante(5)+1)//2 #golden ratio
img = RadicalConstante(-1) #imaginary constant

def fib(n):
    """calculate the nth numer in the fibbonacci sequence"""
    r5 = RadicalConstante(5)
    g  = (r5+1)//2
    return (g**n - (1-g)**n)//r5

def exp(x, prec=1e-20, keep_rational=False, verbose=False, _iteraciones=20):
    """Return e raised to the power of x."""
    div = Fraction if keep_rational else _operator.truediv
    fac    = 1
    result = 0
    power  = 1
    prev   = 0
    for i in _count(1):
        prev = result
        result += power*fac
        fac *= div(1,i)
        power *= x
        try:
            if abs(result-prev)<prec:
                break
        except TypeError:
            if i > _iteraciones:
                break
    if verbose:
        print("numero de iteraciones:",i)
    return result
    
    
