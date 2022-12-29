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
from functools import total_ordering
import operator as _operator


__all__ = ['Constante', 'ConstanteABC', 'ConstanteBase', 'sqrt', 'evaluar_constante', 'evaluar_formula']

class Constante(ConstanteBase):
    '''Constante numerica arbitraria de forma: (a+mC**e) con a,m,e conocidos'''

    def __init__(self, name:str, **kwarg):
        super().__init__(name=name,**kwarg)
    
 
@total_ordering
class RadicalConstante(ConstanteBase):
    '''Constante numerica arbitraria de forma: (a+mC**e) con a,m,e,C conocidos
       para representar raizes sin calcular su valor numerico '''

    def __init__(self, value:Number, raiz:int=None,**kwarg):
        if raiz is None:
            raiz = kwarg.pop("e", Fraction(1,2))
        else:
            raiz = Fraction(1,raiz)
        if not raiz:
            raise ValueError("the exponent can't be 0")
        
        super().__init__(value=value, e=raiz,**kwarg)
    
    def __lt__(self,otro:Number) -> bool:
        """self < otro"""
        if not isinstance(otro, Number):
            return NotImplemented
        return self() < otro
        



a,b,c = map(Constante,"abc")

golden = (RadicalConstante(5)+1)//2 #golden ratio
img = RadicalConstante(-1) #imaginary constant
_r69 = RadicalConstante(69)
plastic = RadicalConstante( (9+_r69)//18,3) + RadicalConstante( (9-_r69)//18,3) #plstic ratio

def fib(n):
    """calculate the nth number in the fibbonacci sequence"""
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
    
    
