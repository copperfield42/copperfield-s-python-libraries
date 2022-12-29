# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 13:50:51 2016

@author: david
"""

from .polynomialclass import PolinomioBase 
from .constante import Constante, sqrt, evaluar_formula, RadicalConstante
from operator import itemgetter, floordiv, truediv
from numbers import Number, Integral, Real, Rational, Complex
from fractions import Fraction
from collections import namedtuple
from decimal import Decimal
import math, cmath, numbers, typing

try:
    from naturales import factores as _factores
except (ImportError, SyntaxError):
    def _factores(n):
        return ( f for f in range(1,n+1) if n%f==0)

Termino = namedtuple("Termino","grado coef")


class Polinomio(dict,PolinomioBase):   

    def __setitem__(self,exp,coef):
        if isinstance(exp,Integral):
            if exp>=0:
                return super().__setitem__(exp,coef)
            raise ValueError('los exponentes del polinomio deben ser numeros no negativos')
        raise TypeError('Los exponentes del polinomio solo puede ser numeros enteros no negativos')
    
    def __repr__(self):
        return '{}({})'.format(self.__class__.__qualname__, super().__repr__())

    def __bool__(self):
        return any( self.values() )
   
    def clean(self):
        to_remove = tuple( e for e,m in self.items() if not m or not isinstance(e,Integral) or e<0 ) 
        for k in to_remove:
            del self[k]
            
    def terminos(self):
        return map(Termino._make,filter(itemgetter(1), self.items()))

    @classmethod
    def from_terminos(cls,terminos:'[(exp,coef)]'):
        return cls( (e,c) for e,c in terminos if c and e>=0 )
        
    def __round__(self,*a,**k):
        #print(f"{a=} {k=}")
        return self

    def __floordiv__ (self,otro):
        '''X // Y'''
        return self._division(otro, fdiv, False)[0]        


            
 

X=Polinomio.monomio(1,1)
x=Polinomio.monomio(1,1)
a,b,c=map(Constante,"abc")

def monomio_div_binomio(m,n,a,r,div=truediv) -> Polinomio:
    '''Calcula (mx**n)/(ax**r-1)'''
    if all( isinstance(d,Number) for d in [m,n,a,r] ):
        if not all( isinstance(d,Integral) for d in [n,r] ):
            raise TypeError('Los exponenttes deben ser numeros enteros')
        if a and m:
            q,f = divmod(n,r)
            Q = Polinomio()
            if a ==1:
                R = Polinomio.monomio(m,f)
                Q.suma( (n-k*r, m) for k in range(1,q+1) )
            else:
                R = Polinomio.monomio(div(m,pow(a,q)),f)
                Q.suma( (n-k*r ,div(m,pow(a,k))) for k in range(1,q+1) )
            return Q,R
        if not m:
            return 0,0
        if not a:
            Q=Polinomio.monomio(-m,n)
            return Q,0
    raise TypeError('Todos los datos deben ser numeros')
            
        
def polyGCD(A,B,div=floordiv) -> Polinomio:
    if all( isinstance(x , PolinomioBase) for x in [A,B]):
        A = Polinomio.from_terminos(A.terminos())
        B = Polinomio.from_terminos(B.terminos())
        while B:
            A._division(B,div,True,True)
            A,B = B,A
        return A
    while B:
        A,B = B, A%B
    return A

def polyIntegral(Px,ini=None,fin=None,*,coeffunc = truediv, consName='Constante de integracion'):
    '''Calcula la integral del polinomio Px
       Si ini y fin son proporcionados, calcula la integral definida entre esos puntos,
       sino regresa el polinomio correspondiente.

       coeffunc: es la funcion para calcular la fraccion c/(i+1) de la regla de la potencia para
       el termino cx^i
       consName: nombre de la constante de integracion para el caso de la integral indefinida'''
    if isinstance(Px, Number):
        Px = Polinomio.monomio(Px,0)
    if isinstance(Px,PolinomioBase):
        Px.clean()
        C = Constante( consName )
        I = None
        if Px:
            I = Polinomio.from_terminos(  ( i+1, coeffunc(c,i+1) ) for i,c in Px.terminos() )
            I += C
        else:
            I = Polinomio.monomio(C,0)
        if ini is None and fin is None:
            return I
        elif ini is not None and fin is not None:
            return I(fin) - I(ini)
        else:
            raise ValueError('Si proporciona uno de los limites de integracion, se debe proporcionar  el otro')
    else:
        raise ValueError('Se debe proporcionar un polinomio')
        

def rational_roots(Px) -> [Fraction]:
    '''Busca si el polinomio dado tiene raices racionales segun el teorema
       de raices racionales y regresa una lista con las misma.
       
       Sea el polinomio
       a0 + a1X**1 + a2X**2 + ... + a(n-1)X**(n-1) + anX**n 
       con ai numeros enteros y con a0 y an =/=0
       las raices racionales x = ±P/Q con P y Q coprimos satisfacen que:
       >P es un factor de a0
       >Q es un factor de an

       https://en.wikipedia.org/wiki/Rational_root_theorem'''
    if Px[0]==0 and Px.grado>0:
        return [0, *filter(None,rational_roots(Px<<1))]
    try:
        cadidatos={Fraction(s*p,q) for p in _factores(abs(Px[0])) for q in _factores(abs(Px.dom_term.coef)) for s in (1,-1)}
        return [c for c in cadidatos if Px(c)==0]
    except TypeError:
        return []

def fdiv(a,b,div=None):
    """Trata de regresar Fraction(a,b) y si falla regresa a/b
       o regresa div(a,b) si otorgado
       (para evitar en lo posible el uso de flotantes)"""
    try:
        return Fraction(a,b)
    except TypeError:
        if div:
            return div(a,b)
        return a/b

def linear_root(Px:Polinomio) -> Number:
    """Raices de un polinomio de grado 1"""
    if not Px.grado == 1:
        raise ValueError("El polinomio no es de grado 1")
    a,b = Px[1],Px[0]
    if b==0:
        return 0
    if a==1:
        return -b
    return fdiv(-b,a)

def quadratic_roots(Px:Polinomio,show_discriminante:bool=False,_fr=False) -> typing.Tuple[Number,Number]:
    """Raices de un polinomio de grado 2
       https://en.wikipedia.org/wiki/Quadratic_function"""
    if not Px.grado==2:
        raise ValueError("El polinomio no es de grado 2")
    if Px(0)==0:
        return (0,linear_root(Px<<1) )
    a,b,c = Px[2],Px[1],Px[0]
    D = b**2 -4*a*c
    if show_discriminante:
        print(Px)
        print("Discriminate:",D)
    if _fr:
        try:
            D = Fraction(D)
        except TypeError:
            pass
    D2 = sqrt(D,_fr)
    return fdiv(-b+D2,2*a),fdiv(-b-D2,2*a)
    
def cubic_roots(Px:Polinomio,show_discriminante:bool=False,_fr=False,_rr=True) -> typing.Tuple[Number,Number,Number]:
    """Raices de un polinomio de grado 3
       https://en.wikipedia.org/wiki/Cubic_equation#General_cubic_formula"""
    if not Px.grado == 3:
        raise ValueError("El polinomio no es de grado 4")
    if Px(0)==0:
        return (0,) + quadratic_roots(Px<<1,show_discriminante,_fr)
    d,c,b,a = (Px[n] for n in range(4))
    if _rr: #check if there are a rational root
        try:
            rr = rational_roots(Px)
            if len(rr) == 3:
                if show_discriminante:
                    print("Encontradas 3 raiz racionales: {rr}".format(rr=rr))
                return tuple(rr)
            elif len(rr) == 2:
                r1,r2 = rr
                x = Polinomio.monomio(1,1)
                if show_discriminante:
                    print("Encontradas 2 raiz racionales: {rr}".format(rr=rr))
                return r1,r2,linear_root(Px/(x-r1)/(x-r2))
            elif len(rr)== 1:
                r = rr[0]
                Δ = b**2 -4*a*c -2*a*b*r -3*(a**2)*(r**2)
                if show_discriminante:
                    print("Encontrada una raiz racional: {r}\nDiscriminante: {Δ}".format(r=r,Δ=Δ))
                Dq = sqrt(Δ,_fr)
                bra = -b -r*a
                return r, fdiv(bra+Dq,2*a), fdiv(bra-Dq,2*a)
        except Exception:
            pass
    #Δ = chr(916)
    Δ = 18*a*b*c*d - 4*(b**3)*d + (b*c)**2 - 4*a*(c**3) - 27*(a*d)**2
    Δ0 = b**2 -3*a*c    
    Δ1 = 2*(b**3)-9*a*b*c +27*(a**2)*d
    C = fdiv(Δ1 + sqrt(-27*Δ*a**2,_fr),2)**Fraction(1,3)    
    if not Δ0:
        C = ( Δ1 )**Fraction(1,3)
    if show_discriminante:
        print(Px)
        print("Discriminante Δ:", Δ)
        try:
            if Δ > 0:
                print("the equation has three distinct real roots.")
            elif Δ < 0:
                print("the equation has one real root and two non-real complex conjugate roots.")
            else:
                print("the equation has a multiple root and all of its roots are real.")
        except TypeError:
            pass
        print("")
        print("Δ0 =",Δ0)
        print("Δ1 =",Δ1)
        print("C  =",C)
    if not Δ:
        if not Δ0:
            r = fdiv(-b,3*a)
            return r,r,r
        else:
            dr = fdiv(9*a*d - b*c, 2*Δ0)
            sr = fdiv(4*a*b*c -9*(a**2)*d - b**3, a*Δ0)
            return sr,dr,dr
    root3_1 = 1,complex(-0.5,0.5*sqrt(3)),complex(-0.5,-0.5*sqrt(3))
    # ζ = chr(950)
    result = [fdiv(-1,3*a)*(b +ζk*C + fdiv(Δ0,ζk*C) ) for ζk in root3_1]
    try:
        if Δ>0:
            for i,r in enumerate(result):
                if isinstance(r,Complex):
                    result[i] = r.real
    except TypeError:
        pass
    return tuple(result)
        
    raise NotImplementedError

def quartic_roots(Px:Polinomio,show_discriminante:bool=False,_fr=False,_biquadratic=True) -> typing.Tuple[Number,Number,Number,Number]:
    """Raices de un polinomio de grado 4
       https://en.wikipedia.org/wiki/Quartic_function"""
    if not Px.grado == 4:
        raise ValueError("El polinomio no es de grado 4")
    if Px(0)==0:
        return (0,) + cubic_roots(Px<<1,show_discriminante,_fr)    
    e,d,c,b,a = (Px[n] for n in range(5))
    if not b and not d and _biquadratic:
        #biquadratic case
        if show_discriminante:
            print("Biquadratic case")
        Bcx = Polinomio.from_coeficientes([e,c,a])
        r1,r2 = (sqrt(r,_fr) for r in quadratic_roots(Bcx,show_discriminante,_fr))
        return +r1,-r1,+r2,-r2
    #Δ = chr(916)
    Δ = 256*(a*e)**3 - 192*b*d*(a*e)**2 -128*(a*c*e)**2 + 144*c*e*(a*d)**2 -27*(a**2)*(d**4)  \
        + 144*a*c*(b*e)**2 - 6*a*e*(b*d)**2 - 80*a*b*d*e*c**2 + 18*a*b*c*d**3 + 16*a*e*c**4   \
        - 4*a*(c**3)*d**2 - 27*(b**4)*e**2 + 18*c*d*e*b**3 - 4*(b*d)**3 - 4*e*(b**2)*c**3 + (b*c*d)**2
    P = 8*a*c - 3*b**2
    R = b**3 + 8*d*a**2 -4*a*b*c
    Δ0= c**2 -3*b*d + 12*a*e
    D = 64*e*a**3 - 16*(a*c)**2 +16*a*c*b**2 - 16*d*d*a**2 - 3*b**4
    if show_discriminante:
        print(Px)
        print("Discriminate Δ:",Δ)
        print("P =",P)
        print("R =",R)
        print("Δ0=",Δ0)
        print("D =",D)
        try:
            if Δ<0:
                print("the equation has two distinct real roots and two complex conjugate non-real roots.")
            elif Δ>0:
                if P<0 and D<0:
                    print("all four roots are real and distinct.")
                elif P>0 and D>0:
                    print("there are two pairs of non-real complex conjugate roots.")
                else:
                    print("either the equation's four roots are all real or none is.")
            else:#Δ==0
                if P<0 and D<0 and Δ0!=0:
                    print("there are a real double root and two real simple roots.")
                elif D>0 or (P>0 and (D!=0 or R!=0)):
                    print("there are a real double root and two complex conjugate roots.")
                elif Δ0==0 and D!=0:
                    print("there are a triple root and a simple root, all real.")
                elif Δ==0:
                    if P<0:
                        print("there are two real double roots.")
                    elif P>0 and R==0:
                        print("there are two complex conjugate double roots.")
                    elif Δ0==0:
                        print("all four roots are equal to -{b}/(4*{a})".format(b=b,a=a))
                else:
                    print("###unexpected option###")
        except TypeError:
            pass
    if Δ==0 and Δ0==0:
        r = fdiv(-b,4*a)
        return r,r,r,r
    p  = fdiv(P,8*a**2)
    q  = fdiv(R,8*a**3)
    Δ1 = 2*c**3 -9*b*c*d + 27*e*b**2 + 27*a*d**2 -72*a*c*e
    if Δ>0 and P<0 and D<0:
        # ϕ = chr(981)
        ϕ = math.acos( Δ1/(2*math.sqrt(Δ0**3) ) )
        S = sqrt( fdiv(2*sqrt(Δ0)*math.cos(ϕ/3),3*a) + Fraction(-2,3)*p )*Fraction(1,2)
        Q = None
    else:
        Q = fdiv( sqrt(-27*Δ) + Δ1, 2)**Fraction(1,3)
        S = sqrt( fdiv(Q+fdiv(Δ0,Q),3*a) + Fraction(-2,3)*p )*Fraction(1,2)
        ϕ = None
    if Q==0 or S==0:
        if show_discriminante:
            print("caso S=0 o Q=0")
        try:
            if Δ>0:
                t = math.acos( fdiv(Δ1,2*sqrt(Δ0**3) ) )
                S = sqrt( p*Fraction(-2,3)+fdiv(2,3*a)*sqrt(Δ0)*math.cos(t/3) )
            elif Δ != 0 and Δ0 == 0:
                Q = Δ1**Fraction(1,3)
                S = sqrt( fdiv(Q,3*a) + Fraction(-2,3)*p )*Fraction(1,2)
        except TypeError:
            pass    
    if show_discriminante:
        print("")
        print("S =",S)
        print("Q =",Q)
        print("Δ1=",Δ1)
        print("p =",p)
        print("q =",q)
        print("a =",a)
        print("b =",b)        
        print("ϕ =",ϕ)        
    b_4a = fdiv(-b,4*a)
    r12 = sqrt( -4*S**2 - 2*p + fdiv(q,S) )*Fraction(1,2)
    r34 = sqrt( -4*S**2 - 2*p - fdiv(q,S) )*Fraction(1,2)
    return b_4a - S + r12, b_4a - S - r12, \
           b_4a + S + r34, b_4a + S - r34
    raise NotImplementedError

def newton_root(Fx,initial,derivative=None,*,tolerance:float=1e-9,iterations:int=100, verbose:bool=False):
    """Metodo de Newton para encontrar una raiz de la funcion dada
       https://en.wikipedia.org/wiki/Newton%27s_method"""
    Dx = derivative if derivative is not None else Fx.derivada()
    r0 = initial
    for i in range(iterations):
        r = r0 - Fx(r0)/Dx(r0)
        if abs(Fx(r)) < tolerance or r==r0:
            if verbose: print("solo tomo",i,"iteraciones")
            break
        r0 = r
    return r
    pass


##e=Constante("e")    
##Px4 = x**4 + x**2 + e
##Cx3 = 4*x**3 -3*x - fdiv(35, 13*sqrt(13))
##Cx3f = 4*x**3 -3*x - fdiv(35, 13*sqrt(Fraction(13),1))
##d=Constante("d")
##Cx3d = 4*x**3 -3*x - d
##Px6 = x**6 - 2*x**5 - 26*x**4 + 28*x**3+ 145*x**2 - 26*x - 80

##r1=Constante('b', a=BinomioConstante(Constante('b', a=Constante('a', m=Constante('c', m=-4)), e=2), m=Constante('a', m=Fraction(1, 2), e=-1), e=Fraction(1, 2)), m=Constante('a', m=Fraction(-1, 2), e=-1))
##r2=Constante('b', a=BinomioConstante(Constante('b', a=Constante('a', m=Constante('c', m=-4)), e=2), m=Constante('a', m=Fraction(-1, 2), e=-1), e=Fraction(1, 2)), m=Constante('a', m=Fraction(-1, 2), e=-1))
## 
    
