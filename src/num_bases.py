"""Modulo para manipulación de números en distintas bases"""

from typing import Iterator, Iterable, Tuple, Mapping, Union, List, Final
import math, numbers
from fractions import Fraction
from typing_recipes import RealLike

__exclude_from_all = set(dir())

BASE:Final[str] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ" 
#quito 'I' y 'O' para evitar confucion con 1 y 0
#cojunto de caracteres de las distintas base
#FROM_BASE = {k:v for v,k in enumerate(BASE) }
#TO_BASE =dict(enumerate(BASE))



def num_dig(n:int, base:int=10 ) -> Iterator[int]:
    """Generador de los digitos del número en la base dada.
       123 -> 3 2 1
       42 base=16 -> 10 2
       42 base=2  -> 0 1 0 1 0 1"""
    if base < 2:
        raise ValueError("La base debe ser mayor o igual que 2")
    if n < 0:
        n *= -1
    if n == 0:
        yield 0
        return
    while n:
        n,d = divmod(n,base)
        yield d


def num_digits(n:int, base:int=10) -> Tuple[int,...]:
    """Regresa una tuple con los digitos del número dado en la base dada
       123 -> 3 2 1
       42 base=16 -> 10 2
       42 base=2  -> 0 1 0 1 0 1"""
    return tuple(num_dig(n,base))


def num_from_digits(digitos:Iterable[int], base:int=10) -> int:
    """Dados los digitos del número en la base dada, reconstrulle el número que representan.
       Los digitos deben estar ordenados desde el primero al último
       [3,2,1] -> 123
       [10,2] base=16 -> 42
       [0,1,0,1,0,1], base=2 -> 42 """
    if base < 2:
        raise ValueError("La base debe ser mayor o igual que 2")
    n,b = 0,1
    for d in digitos:
        if d:
            n += d*b
        b *= base
    return n


def num_len(n:int, base:int=10) -> int:
    """Dice cuantos digitos tiene el número dado en la base espesificada.
       Equivalente a len(str(n)) pero sin la transformación a string"""
    if n<0:
        n*=-1
    if 0 <= n <base:
        return 1
    if base==2:
        return n.bit_length()
    log = math.log10 if base==10 else (lambda x: math.log(x,base))
    estimado:int = math.floor( log(n) ) +1
    borde = base**(estimado-1)
    #print(estimado)
    if borde <= n < base*borde:
        #print("correcto")
        return estimado
    elif n<borde:
        #print("me pase")
        return estimado -1
    else:
        #print("no le llegue")
        return estimado +1
    #return ilen(num_dig(n,base))


def num_concat(a:int, b:int, *, base:int=10) -> int:
    """Concatena los numeros dados en la base espesificada
       num_concat(32,1) -> 321
       num_concat(32,115) -> 32115
       num_concat(42,10,base=2) -> 682 (42=0b101010, 10=0b1010, 682=0b1010101010)"""
    #re:int = a*base**num_len(b,base) + b #to please mypy --strict
    return a*base**num_len(b,base)


def num_reverse(n:int, base:int=10) -> int:
    """Regresa el valor del numero con los digitos revertidos en la base dada.
       123 -> 321
       300 -> 3
       42 base=16 -> 162 (42=0x2a 162=0xa2)
       56 base=2  -> 7   (52=0b111000 7=0b111)"""
    return num_from_digits(reversed(num_digits(n,base)),base)


def toBase(n:RealLike, base:int, *, symbols:Mapping[int,str]|str|None=None, prec:int=16, sep:str=".", neg:str="-" ) -> str:
    """Transforma el número a un string que representa ese numero en la base dada"""
    if base == 10 and not symbols:
        return str(n)
    if n < 0:
        return neg + toBase(-n, base=base, symbols=symbols, prec=prec, sep=sep)
    if symbols is None:
        symbols = BASE
    if len(symbols) < base:
        raise ValueError("No hay suficientes simbolos para representar esta base")
    if isinstance(n,numbers.Real):
        entero = int(n)
        er = "".join( symbols[d] for d in reversed(num_digits(entero,base)) )
        if entero == n :
            return er
        decimal = n - entero
        dr = ""
        for _ in range(prec):
            if not decimal:
                break
            decimal *= base
            t = int(decimal)
            dr += symbols[t]
            decimal -= t
        dr = dr.rstrip(symbols[0])
        return sep.join(filter(None,[er,dr]))
    else:
        raise ValueError("Not a Real number")


def fromBase(n:str|RealLike, base:int, *, symbols:Mapping[str,int]|None=None, sep:str=".", neg:str="-") -> Union[int,float] :
    """Dado un string de un número en la base dada, regresa el valor numerico del mismo"""
    if isinstance(n,str):
        n = n.strip()
    else:
        n = str(n)
    if symbols is None:
        if 2 <= base <= 36 and sep not in n:
            return int(n,base)
        symbols = {k:v for v,k in enumerate(BASE[:base]) }
        if base <= 36:
            n = n.lower()
    if n.startswith(neg):
        return -fromBase(n[1:],base, symbols=symbols, sep=sep)
    if sep not in n:
        return num_from_digits( (symbols[d] for d in reversed(n)), base)
    ent,dec = n.split(sep,maxsplit=1)
    return num_from_digits( (symbols[d] for d in reversed(ent)), base) + \
           sum( symbols[d]*base**(-i) for i,d in enumerate(dec,1) )


def change_base(n:Union[str,RealLike],old_base:int,new_base:int,*, old_sym:Mapping[str,int]|None=None, new_sym:Mapping[int,str]|None=None) -> str:
    """Cambia la representación del numero dado de la base vieja a la nueva"""
    return toBase(fromBase(n,old_base,symbols=old_sym), new_base, symbols=new_sym )


def all_bases(n:Union[str,int], base:int=10) -> None:
    """Dado un numero en la base dada, imprime la representación
       del mismos en todas las bases disponibles"""
    for new_base in range(2,1+len(BASE)):
        print(f"base={new_base:02}:", change_base(n,base,new_base), "*original" if new_base == base else "") 


def repeating_decimal(a:int, b:int|None=None, *, prec:int|None=None, base:int=10) -> Tuple[int,List[int],List[int]]:
    """Calcula el periodo de a/b y regresa una tupla (entero,digitos,periodo)
       donde
       entero es la parte entera de a/b
       digitos es una lista con los digitos NO periodicos de a/b
       periodo es una lista con los digitos periodicos de a/b        """
    #https://en.wikipedia.org/wiki/Repeating_decimal
    if b is None:
        a,b = a.as_integer_ratio()
    if not prec:
        prec = abs(a)+abs(b)
    entero, remainder = divmod(a,b)
    if not remainder:
        return entero,[],[]
    mod:dict[tuple[int,int],int] = {}
    digits:list[int] = []
    period:list[int] = []
    for i in range(prec):
        d,remainder = divmod(remainder*base,b)
        #if show: print(f"{i=} digit={d}, {remainder=}")
        if remainder==0:
            if d:
                digits.append(d)
            break
        elif (d,remainder) in mod:  #repeat = True
            #if show: print(f"repeat from i={mod[(d,remainder)]}")
            i = mod[(d,remainder)]
            period = digits[i:]
            digits = digits[:i]
            break
        else:
            mod[(d,remainder)] = i
            digits.append(d)
    return entero, digits, period


def repeating_decimal_str(a:int, b:int|None=None, *, prec:int=100, **karg) -> str:
    n,d,p = repeating_decimal(a,b,prec=prec,**karg)
    if karg.get("base",10)!=10:
        if d or p:
            return toBase(n,karg["base"]) + "." + "".join(map(BASE.__getitem__,d)) + (("(" + "".join(map(BASE.__getitem__,p)) + ")") if p else "")
        else:
            return toBase(n,karg["base"])
    if d or p:
        #return "".join([str(n), ".", "".join(map(str,d)), (("(" + "".join(map(str,p)) + ")") if p else "")
        return str(n) + "." + "".join(map(str,d)) + (("(" + "".join(map(str,p)) + ")") if p else "")
    else:
        return str(n)


def make_fraction(n:int=0, decimals:Union[int,List[int]]|None=None, periodo:Union[int,List[int]]|None=None) -> Fraction:
    """retorna la fraccion corespondiente al numero n.decimals(periodo)
       make_fraction(0,None,69) is 0.(69) -> 23/33
       make_fraction(1,0,5)  is 1.0(5) -> 19/18
       make_fraction(0,[0,0,0,0,0],3) is 0.00000(3) -> 1/300000 
       """
    if decimals is None:
        d = 1
        decimals = 0
    elif isinstance(decimals,Iterable):
        dl = list(decimals)
        d = 10**len(dl)
        decimals = num_from_digits(reversed(dl))
    else:
        d = 10**num_len(decimals)
    if periodo is None:
        p = 1
        periodo = 0
    elif isinstance(periodo,Iterable):
        pl = list(periodo)
        p = 10**len(pl) - 1
        periodo = num_from_digits(reversed(pl))
    else:
        p = 10**num_len(periodo) - 1
    return n + Fraction(decimals,d) + Fraction(periodo,d*p)


__all__ = [x for x in dir() if x not in __exclude_from_all and not x.startswith("_") ]
del __exclude_from_all

