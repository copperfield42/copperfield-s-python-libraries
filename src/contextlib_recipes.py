import os
from functools import wraps
from abc import abstractmethod
from typing_recipes import (
    ParamSpec, 
    PathType, 
    Iterator, 
    Callable, 
    TypeVar,
    Any,
    T, 
)

P = ParamSpec('P')



__exclude_from_all__=set(dir())

from contextlib import *

class AbstractClosableContextManager(AbstractContextManager):
    """Abtract Context Manager for a class with a close method with a default __enter__ that return self and default __exit__ that call close on exit"""

    __slots__ = ()

    @abstractmethod
    def close(self) -> None:
        pass

    def __del__(self) -> None:
        self.close()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


try:
    from contextlib import chdir #type: ignore #py 3.11+
except ImportError:

    @contextmanager
    def chdir(folder:PathType) -> Iterator[None]:
        r"""
        Cambia el directorio de trabajo del bloque de este contexto
        a la carpetada y regresa al anterior al terminar el bloque

        >>> import os
        >>> os.getcwd()
        'C:\\Users\\Bob'
        >>> with chdir(r"C:\test"):
                print(os.getcwd())

        C:\test
        >>> os.getcwd()
        'C:\\Users\\Bob'
        >>>
        """
        if folder:
            original = os.getcwd()
            os.chdir(folder)
            try:
                yield
            finally:
                os.chdir(original)
        else:
            yield


def chdir_decorator(folder:PathType) -> Callable[[Callable[P, T]], Callable[P, T]]: 
    """
    Decorator to allow a function to always excecute in a given folder

    >>> import os
    >>> os.getcwd()
    'C:\\Users\\Bob'
    >>> @chdir_decorator(r"C:\test")
    def fun():
        print(os.getcwd())

    >>> fun()
    C:\test
    >>> os.getcwd()
    'C:\\Users\\Bob'
    >>>
    """

    def decorator(func:Callable[P,T]) -> Callable[P,T]:
        @wraps(func)
        def wrapper(*args:P.args,**kwarg:P.kwargs) -> T:
            with chdir(folder):
                return func(*args,**kwarg)
        return wrapper
    return decorator




def __getattr__(name:str) -> Any:
    if name == "redirect_folder":
        from warnings import warn
        import sys
        warn(f"'{name}' is deprecated, use 'chdir' instead", DeprecationWarning, stacklevel=2)
        setattr(sys.modules[__name__], name, chdir)
        return chdir
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")













__all__ = [ x for x in dir() if not (x.startswith("_") or x in __exclude_from_all__) ]
del __exclude_from_all__



