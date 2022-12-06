from typing_recipes import Any, TypeVar, Literal, cast
from numbers import Integral 

__exclude_from_all__=set(dir())

from abc import (
    update_abstractmethods,
    abstractstaticmethod,
    abstractclassmethod,
    abstractproperty,
    get_cache_token,
    abstractmethod,
    ABCMeta,
    ABC,
)



class _PropertyConfigInit:

    _new_default:dict[str,Any]

    def __init__(self,*arg,**kwargs) -> None:
        for k,v in self._new_default.items():
            if k not in kwargs:
                kwargs[k]=v
        super().__init__(*arg,**kwargs)


class PropertyConfigMeta(ABCMeta):
    """
    This metaclass extract the atributes defined for the main class
    that would overwrite any inherited attribute that was defined with
    the @property decorator somewhere in the inheritance chain an put 
    them in the class attribute "_new_default"

    And insert an extra class in the mro containing a __init__ 
    in order to add the content of "_new_default" into the call
    to the parents __init__
    """

    def __new__(mcls, name, bases, namespace, /, **kwargs):
        #arrive at this by looking at ABCMeta implementation on _py_abc
        #source code
        
        #list the properties that the new class would inherit
        properties = {p for bcls in bases
                        for cls in bcls.__mro__
                        for p,v in vars(cls).items()
                      if isinstance(v,property)
                      }
        #procede to extract the atributes that would
        #overwrite the properties inherited by non-property
        new_default = {}
        new_namespace = {}
        for k,v in namespace.items():
            if k in properties:
                if isinstance(v,property):
                    new_namespace[k] = v
                else:
                    new_default[k] = v
            else:
                new_namespace[k] = v
        cls = super().__new__(mcls, name, bases, new_namespace, **kwargs)
        if hasattr(cls,"_new_default"):
            cls._new_default = {**cls._new_default, **new_default}
        else:
            cls._new_default = new_default
        return cls

    def mro(cls) -> list[type]:
        old = super().mro()
        if _PropertyConfigInit in old:
            old.remove(_PropertyConfigInit)
        if len(old)>1:
            return [ old[0], _PropertyConfigInit, *old[1:] ]
        else:
            return [ _PropertyConfigInit, *old ]


class PropertyConfig(metaclass=PropertyConfigMeta):
    """
    Cooperative class that transform

    class A(SomeClass):
       a = 1
       b = 2

    into

    class A(SomeClass):
       def __init__(self, *arg, a = 1, b = 2, **karg):
           super().__init__(*arg, a = a, b = b, **karg)

    so long as a and b are defined as properties in SomeClass 
    (or somewhere in the inheritance chain)

    class SomeClass:

       @property
       def a(self):
           ...

       @property
       def b(self):
           ...

    Use as

    class A(PropertyConfig, SomeClass):
       a = 1
       b = 2
       
    It also make itself the imediate parents of any subclass
    of a class that inhering from this class

    class B(A):
        pass 
        
    also become

    class B(PropertyConfig, A):
        pass
    """

    _new_default:dict[str,Any]
    
    #def __new__(cls,*arg,**karg):
        #print(f"PropertyConfig.__new__({cls},{arg=},{karg=})")
        #result = super().__new__(cls,*arg,**karg)
        #result.__init__.__kwdefaults__.update(cls._new_default)
        #print("__mro__",result.__class__.__mro__)
        #result.__class__.__mro__ = (PropertyConfig,(c for c in result.__class__.__mro__ if c!=PropertyConfig))
        #return result

    # def __init__(self,*arg,**kwargs):
        # print("PropertyConfig.__init__",arg,kwargs)
        # print(self._new_default)
        # for k,v in self._new_default.items():
            # if k not in kwargs:
                # kwargs[k]=v
        # print(kwargs)
        # super().__init__(*arg,**kwargs)
    pass


_CSelf = TypeVar("_CSelf",bound="ConfigClass")

class ConfigClass(ABC):
    """
    Cooperative class that offer a default __repr__ method
    based on the abstract property .config
    """

    @property
    @abstractmethod
    def config(self) -> dict[str,Any]:
        """configuration of this class"""
        return {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join( f'{k}={v!r}' for k,v in self.config.items() )})"

    def make_new(self:_CSelf,**new_config) -> _CSelf:
        """Create a new instance of this class with the same configuration, plus whatever change are requested"""
        c = self.config
        c.update(new_config)
        return type(self)(**c)




class PowClass(ABC):
    '''
    Mix-in class that implements self**n for some n instance of numbers.Integral.
    
    In order to use with pow of 3 arguments you also need to implement __mod__ 
    and if you want negative exponet support you also need __rtruediv__
    '''
    #not type hinted because mypy is a diva with numeric-like types
    __slots__ = ()    

    @abstractmethod
    def __mul__(self, otro):
        return NotImplemented
        

    @property
    def unity(self):
        """the number 1 or equivalent of this class 
           (overwrite this if the unity for your class is differente)"""
        return 1

    def __pow__(self, n:int, m:int=None):
        '''
        self**n
        pow(self,n,m)
        '''
        if not isinstance(n,Integral):
            try:
                return super().__pow__(n,m) #type: ignore
            except AttributeError:
                pass
            return NotImplemented
        if m is not None and not m:
            raise ValueError('pow() 3rd argument cannot be 0')
        one = self.unity 
        if not n:
            return one if m is None else (one%m)
        if n==1:
            return self if m is None else (self%m)
        if n<0:
            if m is not None:
                raise ValueError('pow() 2nd argument cannot be negative when 3rd argument specified')
            return 1/pow(self,-n) 
        y = one
        x = self
        while n>1:
            if n&1: #is odd
                y  = y*x
                n -= 1
            x = x*x
            n //= 2
            if m is not None:
                y %= m
                x %= m
        return ( x*y ) if m is None else ( (x*y)%m )





class NonDataDescriptor(ABC):

    @abstractmethod
    def __get__(self, instance, owner=None):
         raise NotImplementedError

class DataDescriptor(NonDataDescriptor):

    @abstractmethod
    def __set__(self, instance, value):
        raise AttributeError("can't set attribute")

    @abstractmethod
    def __delete__(self, instance):
        raise AttributeError("can't delete attribute")

class DataDescriptorNamed(DataDescriptor):

    @abstractmethod
    def __set_name__(self, owner, name):
        raise NotImplementedError



 
__all__ = [ x for x in dir() if not (x.startswith("_") or x in __exclude_from_all__) ]
del __exclude_from_all__


        