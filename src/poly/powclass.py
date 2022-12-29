from numbers import Integral

try:
    from abc_recipes import PowClass
except ImportError:

    class PowClass(object):
        '''Clase que implementa pow(x,n,m) para n Integral'''

        def __pow__(self,n,m=None):
            '''self**n [mod m]'''
            if isinstance(n,Integral):
                if m is not None and not m:
                    raise ValueError('pow() 3rd argument cannot be 0')
                if n:
                    if n==1:
                        return self if m is None else (self%m)
                    if n<0:
                        if m is not None:
                            raise ValueError('pow() 2nd argument cannot be negative when 3rd argument specified')
                        return 1/pow(self,-n)
                    y=1
                    x=self
                    while n>1:
                        if n&1: #es impar
                            y = y*x
                            n-=1
                        x=x*x
                        n//=2
                        if m is not None:
                            y%=m
                            x%=m
                    return ( x*y ) if m is None else ( (x*y)%m )
                return 1 if m is None else (1%m)
            try:
                return super().__pow__(n,m)
            except AttributeError:
                return NotImplemented
