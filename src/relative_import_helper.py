# -*- coding: utf-8 -*-
import sys, os, importlib

def relative_import_helper(path,nivel=1,verbose=False):
    u"""Dada una ruta de un archivo python dentro de un paquete y el nivel del paquete padre
       importa todos los paquetes intermedios desde el paquete padre hasta el paquete
       donde se encuentra el archivo dado y regresa un string tal que sustitullendo
       __package__ por este string se pueden realizar import relativos si el archivo
       es ejecutado como script. Además agrega el directorio del paquete padre a sys.path 

       ejemplo:
       
       sea la sigueinte carpeta
       /test_mod
            __init__.py
            mod_a.py
            /sub_mod
                __init__.py
                mod_b.py
                test.py
       
       donde test.py por ejemplo contiene
           from . import mod_b
           from .. import mod_a
       
       esos impor fallaran si test.py es ejecutado como script, para resolver esto con esta
       funcion basta con agregar __package__ = relative_import_helper(__file__,2) antes de los 
       import relativos quedando como 
       
           __package__ = relative_import_helper(__file__,2)
           from . import mod_b
           from .. import mod_a       
       

      Nota, la forma recomendada de resolver este problema es
      python -m test_mod.sub_mod.test
       """
    namepack = os.path.dirname(path)
    packs = []
    for _ in range(nivel):
        temp = os.path.basename(namepack)
        if temp:
            packs.append( temp )
            namepack = os.path.dirname(namepack)
        else:
            break
    pack = ".".join(reversed(packs))
    if verbose:
        print("package:",pack)
        print("path:",namepack)
    sys.path.append(namepack)
    importlib.import_module(pack)
    return pack