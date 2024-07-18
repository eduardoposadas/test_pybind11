* [Jugando con Pybind11](#jugando-con-pybind11)
   * [Búsqueda de números primos usando la criba de Eratóstenes en Python.](#búsqueda-de-números-primos-usando-la-criba-de-eratóstenes-en-python)
   * [Pybind11](#pybind11)
   * [Búsqueda de números primos usando la criba de Eratóstenes en C++.](#búsqueda-de-números-primos-usando-la-criba-de-eratóstenes-en-c)
      * [Implementaciones de la criba de Eratóstenes en C++ usando hilos.](#implementaciones-de-la-criba-de-eratóstenes-en-c-usando-hilos)
   * [Cálculo del valor de pi usando la fórmula de Leibniz.](#cálculo-del-valor-de-pi-usando-la-fórmula-de-leibniz)
      * [Cálculo del valor de pi usando la fórmula de Leibniz y una implementación multiproceso en Python.](#cálculo-del-valor-de-pi-usando-la-fórmula-de-leibniz-y-una-implementación-multiproceso-en-python)
      * [Cálculo del valor de pi usando la fórmula de Leibniz y una implementación multihilo en C++.](#cálculo-del-valor-de-pi-usando-la-fórmula-de-leibniz-y-una-implementación-multihilo-en-c)
      * [Cálculo del valor de pi usando la fórmula de Leibniz y una implementación C++ en GPU con CUDA.](#cálculo-del-valor-de-pi-usando-la-fórmula-de-leibniz-y-una-implementación-c-en-gpu-con-cuda)
   * [Cálculo del valor de pi usando integración numérica.](#cálculo-del-valor-de-pi-usando-integración-numérica)
   * [Conclusiones](#conclusiones)
   * [Compilación del código fuente](#compilación-del-código-fuente)

# Jugando con Pybind11
Python es un lenguaje excelente para prototipado rápido. Gracias a la amplia variedad de su ecosistema de librerías permite la creación de pequeñas aplicaciones en muy poco tiempo, con poco esfuerzo y excelentes resultados. Sin embargo, si el problema a resolver es computacionalmente intensivo, Python rápidamente deja ver que no es el lenguaje apropiado para este tipo de problemas. Al ser un lenguaje interpretado es intrínsecamente lento y además el interprete por defecto de Python  ([CPython](https://github.com/python/cpython)) tiene un [bloqueo](https://realpython.com/python-gil/) global, conocido como [GIL](https://wiki.python.org/moin/GlobalInterpreterLock), que impide que varios hilos se ejecuten concurrentemente, lo que provoca que no se pueda aprovechar de forma eficiente la capacidad de multiproceso que tienen la mayoría de los procesadores actuales.

En el momento de escribir esto ya hay un [intento](https://peps.python.org/pep-0703/) [serio](https://www.blog.pythonlibrary.org/2024/03/14/python-3-13-allows-disabling-of-the-gil-subinterpreters/) de eliminar el GIL de CPython, pero aún está en fase experimental.
Mientras tanto, como indica este [artículo](https://realpython.com/python-parallel-processing/), las únicas maneras de evitar el GIL son:

 - Usar paralelismo basado en procesos en vez de hilos.
 - Usar un interprete de Python alternativo.
 - Usar una librería GIL-Immune como [NumPy](https://numpy.org/)
 - [Escribir](https://docs.python.org/3/extending/extending.html) un módulo de extensión en C o C++ para CPython que no bloquee el GIL.
 - Dejar que [Cython](https://cython.org/) genere un módulo de extensión en C por ti.
 - Hacer llamadas a funciones C externas usando [ctypes](https://docs.python.org/3/library/ctypes.html)

Cada una de estas opciones tiene sus ventajas y sus inconvenientes como indica el [artículo](https://realpython.com/python-parallel-processing/) mencionado.
De estas posibles opciones me pareció interesante hacer un módulo de extensión en C para CPython y explorar las posibilidades que ofrecía esta solución al problema del GIL. Dio la casualidad de que acababa de conocer [pybind11](https://pybind11.readthedocs.io/en/stable/index.html) y de que también estaba repasando las últimas [actualizaciones](https://github.com/AnthonyCalandra/modern-cpp-features) que C++ ha recibido, por lo que decidí hacer un pequeño módulo para Python usando C++ y pybind11.

Para evaluar la mejora de rendimiento que se consigue usando C++ en vez de Python elegí tres problemas cuya resolución es computacionalmente costosa:

 - Búsqueda de números primos usando la [criba de Eratóstenes](https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes)
 - Cálculo del valor de $\pi$ usando la [fórmula de Leibniz](https://en.wikipedia.org/wiki/Leibniz_formula_for_%CF%80)
 - Cálculo del valor de $\pi$ usando [integración numérica](https://www.stolaf.edu/people/rab/os/pub0/modules/PiUsingNumericalIntegration/index.html)

Cada problema se ha implementado de diferentes maneras para comprobar cual es la más rápida.

## Búsqueda de números primos usando la criba de Eratóstenes en Python.
La criba de Eratóstenes es un algoritmo muy sencillo que puede ser implementado en Python en unas pocas líneas:
```Python
# Do not use this function. It is very slow, but is readable
def sieve_of_Eratosthenes_naive(n: int) -> list:  
    sieve = [True] * (n + 1)  
  
    for p in range(2, int(math.sqrt(n)) + 1):  
        if sieve[p]:  
            for i in range(p * p, n + 1, p):  
                sieve[i] = False  
    
    out = [] 
    for p in range(2, n + 1):  
        if sieve[p]:  
            out.append(p)  
  
    return out
```
Aunque esta es la forma más intuitiva de implementar el algoritmo, es extremadamente lenta. Haciendo un perfilado de rendimiento de la función con [line_profiler](https://github.com/pyutils/line_profiler) y `n=1_000_000` se puede ver cuales son las líneas problemáticas:
```
$ kernprof -lv main.py 
09:06:27.749 Inside start: sieve_of_Eratosthenes_naive
09:06:31.521 Inside end: sieve_of_Eratosthenes_naive Duration: 3771.614 ms
Wrote profile results to main.py.lprof
Timer unit: 1e-06 s

Total time: 1.75165 s
File: main.py
Function: sieve_of_Eratosthenes_naive at line 90

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    90                                           @profile
    91                                           def sieve_of_Eratosthenes_naive(name: str, n: int) -> list:
    92         1         68.9     68.9      0.0      clock = StopWatch(message=f' Inside start: {name}')
    93                                           
    94         1       3131.4   3131.4      0.2      sieve = [True] * (n + 1)
    95                                           
    96      1000        261.0      0.3      0.0      for p in range(2, int(math.sqrt(n)) + 1):
    97       999        329.3      0.3      0.0          if sieve[p]:
    98   2122216     530544.0      0.2     30.3              for i in range(p * p, n + 1, p):
    99   2122048     661989.4      0.3     37.8                  sieve[i] = False
   100                                           
   101         1          0.9      0.9      0.0      out = []
   102   1000000     243830.2      0.2     13.9      for p in range(2, n + 1):
   103    999999     279405.1      0.3     16.0          if sieve[p]:
   104     78498      32030.7      0.4      1.8              out.append(p)
   105                                           
   106         1         61.8     61.8      0.0      clock.elapsed(f' Inside end: {name} Duration: ')
   107         1          0.4      0.4      0.0      return out
```
Entre las líneas 98 y 99 consumen más del 67% del tiempo de ejecución. Ese bucle se ejecuta tantas veces como números primos se encuentren, por lo que está claro que hay que cambiarlo por algo más óptimo. En este caso, el código más rápido que he conseguido ha sido asignando a una lista [segmentadada](https://docs.python.org/es/3/reference/expressions.html#slicings) otra lista con el mismo número de elementos `False`.
El otro 30% del tiempo de ejecución corresponde a las líneas 102 y 103. ¿Un bucle `for`  para recorrer una lista y ejecutar un `append`?  ¡Eso es muy poco *pythonico*! Mejor usar [comprensión de listas](https://docs.python.org/es/3/tutorial/datastructures.html#list-comprehensions).

Después de realizar los cambios en el código, el perfilado de la función resultante es:
```
$ kernprof -lv main.py 
09:19:00.735 Inside start: sieve_of_Eratosthenes
09:19:00.976 Inside end: sieve_of_Eratosthenes Duration: 240.885 ms
Wrote profile results to main.py.lprof
Timer unit: 1e-06 s

Total time: 0.240088 s
File: main.py
Function: sieve_of_Eratosthenes at line 110

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   110                                           @profile
   111                                           def sieve_of_Eratosthenes(name: str, n: int) -> list:
   112         1         40.7     40.7      0.0      clock = StopWatch(message=f' Inside start: {name}')
   113                                           
   114         1       3338.9   3338.9      1.4      sieve = [True] * (n + 1)
   115                                           
   116      1000        301.1      0.3      0.1      for p in range(2, int(math.sqrt(n)) + 1):
   117       999        348.5      0.3      0.1          if sieve[p]:
   118       168      27456.9    163.4     11.4              sieve[p * p: n + 1: p] = [False] * (((n + 1) - (p * p) + p - 1) // p)
   119                                           
   120         1     208542.9 208542.9     86.9      out = [i for i in range(2, n + 1) if sieve[i]]
   121                                           
   122         1         59.2     59.2      0.0      clock.elapsed(f' Inside end: {name} Duration: ')
   123         1          0.2      0.2      0.0      return out
```
Con estos cambios el 86% del tiempo de ejecución se consume en la línea 120, generando la lista de números primos. Se podría haber usado un [generador](https://wiki.python.org/moin/Generators) para evitar este consumo de tiempo sustituyendo la linea 120 por `out = (i for i in range(2, n + 1) if sieve[i])`, pero haciendo esto simplemente se estaría pasando el consumo de tiempo al código donde se usase el generador.

Al hacer optimizaciones, el código ha dejado de ser legible. Este es uno de los problemas que presentan todos los lenguajes interpretados y que pretenden ser "fáciles de aprender". Si escribes código de forma que sea lo más natural y legible posible, la mayoría de las veces no será un código óptimo. Para escribir código optimo necesitas aprender cómo el interprete va a ejecutar el código, y por tanto eliminas la ventaja de utilizar un lenguaje "fácil de aprender".

El siguiente paso para hacer un código más rápido es dejar de usar Python puro y usar algún módulo, seguramente escrito en C o en algún otro lenguaje con fama de ser difícil de aprender, que sea apropiado para la tarea que se va a implementar. Es aquí donde Python muestra su verdadero potencial. Existe una variedad enorme de módulos que ayudan a no reinventar la rueda y aumentan drásticamente la eficacia del programador.
En este caso, el modulo [NumPy](https://numpy.org/) permite realizar operaciones sobre listas de forma mucho más rápida.
El perfilado de la función reescrita usando NumPy es:
```
$ kernprof -lv main.py 
09:47:25.792 Inside start: sieve_of_Eratosthenes_numpy
09:47:25.797 Before filling in the list of prime numbers. Duration: 4.386 ms
09:47:25.798 Inside end: sieve_of_Eratosthenes_numpy Duration: 6.142 ms
Wrote profile results to main.py.lprof
Timer unit: 1e-06 s

Total time: 0.00537282 s
File: main.py
Function: sieve_of_Eratosthenes_numpy at line 126

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   126                                           @profile
   127                                           def sieve_of_Eratosthenes_numpy(name: str, n: int) -> np.ndarray:
   128         1         36.5     36.5      0.7      clock = StopWatch(message=f' Inside start: {name}')
   129                                           
   130         1        906.7    906.7     16.9      sieve = np.full(n + 1, True)
   131         1          1.7      1.7      0.0      sieve[0] = False
   132         1          0.4      0.4      0.0      sieve[1] = False
   133                                           
   134      1000        306.8      0.3      5.7      for p in range(2, int(math.sqrt(n)) + 1):
   135       999        421.6      0.4      7.8          if sieve[p]:
   136       168       1917.1     11.4     35.7              sieve[p * p: n + 1: p] = False
   137                                           
   138         1         35.1     35.1      0.7      clock.elapsed(f' Before filling in the list of prime numbers. Duration: ')
   139         1       1720.7   1720.7     32.0      out = np.nonzero(sieve)[0]
   140                                           
   141         1         26.1     26.1      0.5      clock.elapsed(f' Inside end: {name} Duration: ')
   142         1          0.2      0.2      0.0      return out
```
El primer dato llamativo es que el tiempo de ejecución ha pasado de 240 ms a 6 ms. Además, si se conoce el significado de las llamadas `np.full` y `np.nonzero`, el código es intuitivo y legible de un solo vistazo. El lado negativo es que estas funciones de NumPy no se ejecutan de forma paralela, es decir no usan todos los núcleos de los procesadores disponibles.
Como nota al margen, hay que destacar que algunas funciones de NumPy, como las funciones para [álgebra lineal](https://numpy.org/doc/stable/reference/routines.linalg.html), sí se pueden ejecutar de forma paralela con muy pocos cambios en el código Python.

Llegados a este punto, la única forma de disminuir el tiempo de ejecución es crear un módulo para Python en un lenguaje no interpretado que implemente la criba de Eratóstenes y se ejecute usando varios hilos.

## Pybind11
[Pybind11](https://pybind11.readthedocs.io/en/stable/index.html) es una librería C++ que permite, entre otras muchas cosas, crear módulos en C++ para Python de una forma muy sencilla.
Para crear un módulo se crea un fichero en el que se define la API del modulo, en este caso `test_pybind11.cpp`. Como se explica en la [documentación](https://pybind11.readthedocs.io/en/stable/basics.html#header-and-namespace-conventions) de pybind11, este fichero al menos debe contener:
```C++
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "my_func.h"

PYBIND11_MODULE(test_pybind11, m) {
	m.def("python_name_for_my_func", &my_func, "Documentation for my_func");
}
```
Una vez creado este fichero hay que implementar la función C++ en sí misma, es decir se crea un fichero `my_func.cpp` con una función llamada `my_func`. Después se compila con:
```
$ c++ -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) test_pybind11.cpp -o test_pybind11$(python3-config --extension-suffix)
```
esto genera un fichero con un nombre parecido a `test_pybind11.cpython-310-x86_64-linux-gnu.so`. Si aparece algún error por que no encuentra la orden `c++` o algún fichero terminado en .h instalar:
```bash
# sudo apt install g++ python3-dev python3-pybind11
```
Una vez generado el módulo Python se puede usar importándolo y llamando a la función definida en la API:
```Python
$ python3  
Python 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] on linux  
Type "help", "copyright", "credits" or "license" for more information.  
>>> import test_pybind11  
>>> test_pybind11.python_name_for_my_func()
```
Para automatizar la compilación del módulo se puede usar `cmake` como se explica en la [documentación](https://pybind11.readthedocs.io/en/stable/compiling.html#modules-with-cmake). 

## Búsqueda de números primos usando la criba de Eratóstenes en C++.
La implementación en C++ de la criba de Eratóstenes se ha hecho de ocho formas distintas. Las cuatro primeras son implementaciones con un único hilo de ejecución, es decir sin paralelismo. Las cuatro últimas están implementadas usando varios hilos de ejecución, es decir con procesamiento en paralelo.
El código que permite que estas funciones C++ puedan ser utilizadas en Python ha quedado reflejado en el fichero `test_pybind11.cpp` en las líneas:
```C++
m.def("sieve_std_list", &SieveOfEratosthenes_std_list, "Sieve of Eratosthenes. Returns std::list.",  
      py::arg("name") = "C++", py::arg("n") = 10);  
m.def("sieve_python_list", &SieveOfEratosthenes_python_list, "Sieve of Eratosthenes. Returns py::list.",  
      py::arg("name") = "C++", py::arg("n") = 10);  
m.def("sieve_std_vector", &SieveOfEratosthenes_std_vector, "Sieve of Eratosthenes. Returns opaque type VectorULongLongInt.",  
      py::arg("name") = "C++", py::arg("n") = 10);  
m.def("sieve_as_array_nocopy", &SieveOfEratosthenes_as_array_nocopy, "Sieve of Eratosthenes. Returns numpy.ndarray without copy.",  
      py::arg("name") = "C++", py::arg("n") = 10);  
m.def("sieve_as_array_nocopy_omp", &SieveOfEratosthenes_as_array_nocopy_omp, "Sieve of Eratosthenes. Returns numpy.ndarray without copy. Uses OpenMP for parallelization.",  
      py::arg("name") = "C++", py::arg("n") = 10);  
m.def("sieve_as_array_nocopy_thread", &SieveOfEratosthenes_as_array_nocopy_thread, "Sieve of Eratosthenes. Returns numpy.ndarray without copy. Launch one thread for each piece of sieve. Sieve is shared between all the threads.",  
      py::arg("name") = "C++", py::arg("n") = 10);  
m.def("sieve_as_array_nocopy_thread_pool", &SieveOfEratosthenes_as_array_nocopy_thread_pool, "Sieve of Eratosthenes. Returns numpy.ndarray without copy. Launch a fixed pool of threads and use a queue to dispatch the jobs.",  
      py::arg("name") = "C++", py::arg("n") = 10);  
m.def("sieve_as_array_nocopy_generic_thread_pool", &SieveOfEratosthenes_as_array_nocopy_generic_thread_pool, "Sieve of Eratosthenes. Returns numpy.ndarray without copy. Use a generic pool of threads.",  
      py::arg("name") = "C++", py::arg("n") = 10);
```
Estas funciones están implementadas en el fichero `SieveOfEratosthenes.cpp`.  Las declaraciones de las funciones se pueden ver en el fichero `SieveOfEratosthenes.h`:
```C++
std::list<unsigned long long> SieveOfEratosthenes_std_list(const std::string& name, unsigned long long n);
py::list SieveOfEratosthenes_python_list(const std::string& name, unsigned long long n);
std::vector<unsigned long long> SieveOfEratosthenes_std_vector(const std::string& name, unsigned long long n);
py::array_t<unsigned long long> SieveOfEratosthenes_as_array_nocopy(const std::string& name, unsigned long long n);
py::array_t<unsigned long long> SieveOfEratosthenes_as_array_nocopy_omp(const std::string& name, unsigned long long n);
py::array_t<unsigned long long> SieveOfEratosthenes_as_array_nocopy_thread(const std::string& name, unsigned long long n);
py::array_t<unsigned long long> SieveOfEratosthenes_as_array_nocopy_thread_pool(const std::string& name, unsigned long long n);
py::array_t<unsigned long long> SieveOfEratosthenes_as_array_nocopy_generic_thread_pool(const std::string& name, unsigned long long n);
```
Pybind11 [permite](https://pybind11.readthedocs.io/en/stable/advanced/cast/index.html) utilizar tipos de Python en C++, aunque esto implica hacer una copia de los datos cuando se realiza la conversión de tipos de C++ a Python. Para evitar esta copia se pueden utilizar [tipos opacos](https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html#making-opaque-types). En el caso de los contenedores de la STL pybind11 dispone de [funciones](https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html#binding-stl-containers) para [crear](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/test_pybind11.cpp#L18) los tipos opacos de forma directa.
Para los  `numpy.array` pybind11 dispone de `py::array_t`. Se puede crear una variable del tipo `py::array_t` a partir de un `std::vector` sin copia de datos usando un pequeño template que realiza la [conversión](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/SieveOfEratosthenes.cpp#L117) de tipos.

De las ocho implementaciones del algoritmo en C++, las cuatro primeras son monohilo y son esencialmente iguales. Son la traducción directa de la implementación en Python a C++:
```C++
std::vector<unsigned long long> SieveOfEratosthenes_std_vector(unsigned long long n)
{
    std::vector<bool> sieve(n + 1, true);

    for (unsigned long long p = 2; p * p <= n; p++) {
        if (sieve[p]) {
            for (unsigned long long i = p * p; i <= n; i += p)
                sieve[i] = false;
        }
    }

    std::vector<unsigned long long> primes;
    for (unsigned long long p = 2; p <= n; p++)
        if (sieve[p])
            primes.push_back(p);

    return primes;
}
```

La única diferencia es el tipo de datos donde se almacena la lista de números primos:

 - `SieveOfEratosthenes_std_list` devuelve un `std::list`, es decir usa un tipo de datos C++ y pybind11 hace una copia de los datos del `std::list` de C++ a un tipo de datos `list` de Python.
 - `SieveOfEratosthenes_python_list` usa y devuelve una variable del tipo `py::list`. Este tipo de datos es la forma en la que pybind11 permite usar un tipo de datos `list` de Python en código C++.
 - `SieveOfEratosthenes_std_vector` devuelve un `std::vector` tratado como un tipo de datos pybind11 opaco, es decir sin copia de datos en la conversión de tipos de C++ a Python.
 - `SieveOfEratosthenes_as_array_nocopy` devuelve un `py::array_t` que en Python pasa a ser un `numpy.array`. Este array se genera a partir de un `std::vector` sin realizar copia de datos (FIXME ref linea 129 SieveOfEratosthenes.cpp).

Para n=100_000_000 los tiempos de ejecución son:
```
Sieve Of Eratosthenes. Prime numbers less than 100_000_000:
Implementation                             Time (ms)   Primes found      Returned type
_________________________________________________________________________________________
Sieve Python                               8309.202       5761455         <class 'list'>
Sieve Python NumPy                          809.954       5761455         <class 'numpy.ndarray'>
Sieve C++ Serial std::list                 1396.550       5761455         <class 'list'>
Sieve C++ Serial py::list                  1044.407       5761455         <class 'list'>
Sieve C++ Serial opaque type                884.972       5761455         <class 'test_pybind11.VectorULongLongInt'>
Sieve C++ Serial np.ndarray_nocopy          864.377       5761455         <class 'numpy.ndarray'>
```
Como se ve, ninguna de estas implementaciones es más rápida que el uso de NumPy en Python. No tengo una respuesta clara para esto. Posiblemente esté relacionado con el uso de instrucciones [SIMD](https://numpy.org/doc/stable/reference/simd/index.html) por parte de NumPy, o con la [implementación](https://en.cppreference.com/w/cpp/container/vector_bool) de `std::vector<bool>` en la STL de C++.

Con esta tabla de tiempos de ejecución hay que volver a repetir que Python ofrece la posibilidad de programar de forma muy eficiente sin realizar un gran esfuerzo gracias al uso de los módulos disponibles.

### Implementaciones de la criba de Eratóstenes en C++ usando hilos.
Las cuatro implementaciones en C++ del algoritmo de la criba de Eratóstenes que usan hilos lo hacen con estrategias diferentes. Las declaraciones, como ya se ha dicho, están en el fichero `SieveOfEratosthenes.h`:
```C++
py::array_t<unsigned long long> SieveOfEratosthenes_as_array_nocopy_omp(const std::string& name, unsigned long long n);
py::array_t<unsigned long long> SieveOfEratosthenes_as_array_nocopy_thread(const std::string& name, unsigned long long n);
py::array_t<unsigned long long> SieveOfEratosthenes_as_array_nocopy_thread_pool(const std::string& name, unsigned long long n);
py::array_t<unsigned long long> SieveOfEratosthenes_as_array_nocopy_generic_thread_pool(const std::string& name, unsigned long long n);
```
Como reflejan sus nombres:

 - `SieveOfEratosthenes_as_array_nocopy_omp` usa el algoritmo usado en la versión en serie con directivas `#pragma` [OpenMP](https://www.openmp.org/) para la implementación del paralelismo. La ventaja de usar OpenMP es que apenas hay que modificar el código. La desventaja es que no es todo lo eficiente que debería.
El único cambio hecho al código de la versión en serie, aparte de añadir las directivas `#pragma` de OpenMP, ha sido cambiar `std::vector<bool>` por `std::shared_ptr<bool[]>` ya que no se pueden hacer accesos concurrentes a elementos de un `std::vector<bool>`:
```C++
py::array_t<unsigned long long> SieveOfEratosthenes_as_array_nocopy_omp(unsigned long long n)
{
    std::shared_ptr<bool[]> sieve(new bool[n + 1]);
    sieve[0] = sieve[1] = false;  // 0 and 1 are not primes
    std::memset(sieve.get() + 2, true, n - 1); // -1 = +1 - 2

    unsigned long long n_sqrt = sqrt(n);  // OpenMP doesn't like p * p <= n as cond-expression in for loops
#pragma omp parallel for schedule(dynamic)
    for (unsigned long long p = 2; p <= n_sqrt; p++) {
        if (sieve[p]) {
            for (unsigned long long i = p * p; i <= n; i += p)
                sieve[i] = false;
        }
    }

    std::vector<unsigned long long> primes;
// #pragma omp parallel for shared(sieve, primes)  // push_back it is no thread safe
    for (unsigned long long p = 2; p <= n; p++)
        if (sieve[p])
            primes.push_back(p);

    return as_pyarray(std::move(primes));
}
```

 - `SieveOfEratosthenes_as_array_nocopy_thread` divide la criba en trozos iguales del tamaño de la caché L1 para [datos](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/utils.cpp#L11) y lanza un hilo de ejecución para cada trozo. Es un ejemplo de como NO HAY QUE HACERLO.
 - `SieveOfEratosthenes_as_array_nocopy_thread_pool` [lanza](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/SieveOfEratosthenes.cpp#L335) tantos hilos de ejecución como núcleos tenga el ordenador donde se ejecuta:
```C++
    // Number of workers
    auto n_workers = std::thread::hardware_concurrency();
    if (n_workers == 0)
        n_workers = 1;

    // Queue of jobs
    jobs_fifo_queue<std::shared_ptr<job_type>> jobs_queue(n_workers * 100);

    // Workers pool
    std::vector<std::thread> workers_pool;
    for(decltype(n_workers) i = 0; i < n_workers; i++)
        workers_pool.emplace_back(SieveOfEratosthenes_pool_worker, std::ref(jobs_queue));
``` 
Estos hilos [recogen](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/SieveOfEratosthenes.cpp#L298) trabajos de una [cola](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/jobs_fifo_queue.h#L26) FIFO. En cada trabajo se indica el trozo de criba que el hilo tiene que calcular y hay una entrada para que el hilo guarde el resultado. Guardar el resultado en la propia cola de trabajos evita copias de datos innecesarias:
```C++
// Job type for the queue of jobs
struct job_type{
    unsigned long long input_start;
    unsigned long long input_end;
    std::vector<unsigned long long> output_primes;
};
```
Los trozos de la criba de cada trabajo son del tamaño de la caché para datos L1 de los procesadores. Esto evita que los diferentes núcleos del procesador compitan por los mismos datos en la caché del procesador:
```C++
    const unsigned long long chunk_size = get_CPU_L1_data_cache_size() / sizeof(bool);
    
    // Produce jobs for workers
    unsigned long long start = 0;
    while (start <= n){
        auto job = std::make_unique<job_type>();
        job->input_start = start;
        job->input_end = start + chunk_size > n ? n : start + chunk_size;
        job->output_primes = {};
        jobs_queue.enqueue(std::move(job));

        start += chunk_size;
    }
```
Cuando ya no hay más trabajos en la cola FIFO se recogen los resultados de todos los trabajos para crear la lista de números primos resultante.
 - `SieveOfEratosthenes_as_array_nocopy_generic_thread_pool` es una [implementación](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/SieveOfEratosthenes.cpp#L414) más genérica que la anterior. Se ha usado una [cola](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/thread_pool.h#L15) de tareas en la que se puede encolar cualquier función C++. En este caso se encolan funciones que comparten una criba común y que calculan los números primos de un trozo de criba que tiene el tamaño de la caché L1 de los procesadores para datos. Los resultados de estas tareas se conservan en la propia cola de tareas y son recogidos una vez se ha indicado a la cola de tareas que no se van a mandar más tareas. Las listas se ordenan y se devuelven unidas en una sola lista de números primos como resultado final.
```C++
    std::shared_ptr<bool[]> sieve(new bool[n + 1]);
    ThreadPool<std::vector<unsigned long long>> workers_pool;
    const unsigned long long chunk_size = get_CPU_L1_data_cache_size() / sizeof(bool);

    // Produce jobs for workers
    unsigned long long start = 0;
    while (start <= n){
        auto end = start + chunk_size > n ? n : start + chunk_size;
        workers_pool.enqueueAndCollect(SieveOfEratosthenes_worker_2, sieve, start, end);

        start += chunk_size;
    }

    // No more jobs. Waits until all tasks have finished and ends the threads
    workers_pool.finish();
```
Los tiempos de ejecución para n=1_000_000_000:
```
Sieve Of Eratosthenes. Prime numbers less than 1_000_000_000:
Implementation                             Time (ms)   Primes found      Returned type
_________________________________________________________________________________________
Sieve Python                             181266.518       50847534        <class 'list'>
Sieve Python NumPy                         9242.615       50847534        <class 'numpy.ndarray'>
Sieve C++ Serial std::list                15064.946       50847534        <class 'list'>
Sieve C++ Serial Python list              12073.701       50847534        <class 'list'>
Sieve C++ Serial std::vector              10938.307       50847534        <class 'test_pybind11.VectorULongLongInt'>
Sieve C++ Serial np.ndarray_nocopy        10463.328       50847534        <class 'numpy.ndarray'>
Sieve C++ OpenMP np.ndarray_nocopy         7306.659       50847534        <class 'numpy.ndarray'>
Sieve C++ Multi thread np.ndarray_nocopy   4547.855       50847534        <class 'numpy.ndarray'>
Sieve C++ Thread pool np.ndarray_nocopy    2834.655       50847534        <class 'numpy.ndarray'>
Sieve C++ Gen thr pool np.ndarray_nocopy   3911.589       50847534        <class 'numpy.ndarray'>
```
Como se puede ver, la implementación más rápida es `SieveOfEratosthenes_as_array_nocopy_thread_pool`, llamada "Sieve C++ Thread pool np.ndarray_nocopy" en la tabla anterior. Sin duda, el hacer una implementación especifica para el problema a tratar ayuda a mejorar el tiempo de ejecución.
Las pruebas se han hecho en un ordenador con cuatro núcleos y ocho núcleos virtuales (Intel Hyperthreading), por lo que era de esperar que el tiempo de ejecución de la implementación monohilo más rápida y el tiempo de ejecución de la implementación multihilo más rápida fuera un múltiplo de cuatro u ocho. Sin embargo la proporción es cercana a tres: 9242.615 / 2834.655 = 3.26
Esto es debido a que:
 - La tecnología multihilo a nivel de núcleo (Intel Hyperthreading) no es útil para aumentar el paralelismo en este caso en concreto.
 - La propia naturaleza del algoritmo de la criba de Eratóstenes. Cada hilo que se usa para calcular los números primos en un trozo de la criba necesita calcular los números primos anteriores a la posición del inicio del trozo de la criba, cada hilo está haciendo cálculos que ya ha realizado otro hilo.

## Cálculo del valor de $\pi$ usando la fórmula de Leibniz.
La [fórmula](https://en.wikipedia.org/wiki/Leibniz_formula_for_%CF%80) de Leibniz para el cálculo de $\pi$ es muy sencilla de [implementar](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/main.py#L187) en Python:
```Python
def pi_leibniz(n: int) -> float:
    s = 1
    k = 3
    for i in range(1, n + 1):
        # s += (-1)**(i) / (2 * i + 1)
        if i % 2 == 0:  # much faster
            s += 1 / k
        else:
            s -= 1 / k
        k += 2
    s *= 4

    return s
```
Como se puede ver en el comentario, la primera optimización que se ha hecho es no usar la exponenciación para decidir el signo del sumando. La exponenciación es una operación muy lenta, incluso en C++, mientras que realizar la operación de modulo dos es inherentemente rápida en un ordenador, ya que internamente está utilizando una representación binaria de los números.
La otra pequeña optimización que se ha realizado es utilizar una variable llamada `k` a la que en cada iteración se le suma 2 para evitar tener que calcular `2 * i + 1`.

La [implementación](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/CalculatingPi.cpp#L85) en C++, que está en el fichero `CalculatingPi.cpp`(FIXME ref linea 89 a fichero), es la traducción directa del código en Python:
```C++
long double pi_leibniz(unsigned long long n)
    long double s = 1;
    long double k = 3;
    for (unsigned long long i = 1; i <= n; i++){
        // s += pow(-1, i) / (2 * i + 1);
        if (i % 2 == 0)  // much faster
            s += 1.0 / k;
        else
            s -= 1.0 / k;
        k += 2;
    }
    s *= 4;

    return s;
}
```
Los tiempos para una ejecución con n=100_000_000:
```
Implementation                             Time (ms)  Calculated value    Returned type
______________________________________________________________________________________
Pi Leibniz Python                         11711.682  3.141592663589326    <class 'float'>
Pi Leibniz C++ one thread                   124.887  3.141592663589794    <class 'float'>
```
Simplemente por traducir el algoritmo a C++ se consigue que el tiempo de ejecución sea casi un 99% menor.

Parece evidente que si el algoritmo a codificar es sencillo, es un punto crítico en la ejecución del programa y no existe un módulo en Python que ya tenga implementada la tarea a realizar, la implementación en C++ siempre debería ser considerada una opción a tener en cuenta.

### Cálculo del valor de $\pi$ usando la fórmula de Leibniz y una implementación multiproceso en Python.
Python ofrece la posibilidad de implementar procesos [multihilo](https://docs.python.org/3/library/threading.html) pero debido al GIL solamente un hilo se ejecutará a la vez lo que proporciona rendimientos ridículos. Para evitar esto se puede utilizar el módulo de paralelismo basado en [procesos](https://docs.python.org/3/library/multiprocessing.html). Con este módulo en vez de hilos se lanzan procesos para cada tarea, lo que implica que:

 - La coste computacional de lanzar un proceso es mucho mayor que el de lanzar un hilo. Para aminorar este coste se lanza un grupo inicial de procesos que se encargan de realizar una lista de tareas.
 - El paso de datos entre diferentes procesos es mucho más costoso que el paso de datos entre hilos ya que se hace usando memoria compartida. En el caso que nos ocupa no es un problema puesto que los procesos hijos únicamente devuelven al padre un número en coma flotante.

Como en este problema el trasiego de datos entre los diferentes procesos es mínimo es posible que la implementación usando el módulo de paralelismo basado en procesos de Python dé buenos resultados.
La [implementación](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/main.py#L220) en Python ha sido, como siempre, muy sencilla:
```Python
def pi_leibniz_concurrent(n: int) -> float:
    n_cpu = os.cpu_count()
    chunk_size = (n + n_cpu - 1) // n_cpu
    chunks = [[(i * chunk_size) + 1, (i * chunk_size) + chunk_size] for i in range(n_cpu)]
    chunks[-1:][0][1] = n  # end of last chunk = n

    with cf.ProcessPoolExecutor(max_workers=n_cpu) as executor:
        results = [executor.submit(pi_leibniz_concurrent_worker, a, b) for (a, b) in chunks]
        sum_ = sum(r.result() for r in cf.as_completed(results))

    return sum_ * 4.0
```
Simplemente se han repartido el número de iteraciones entre el número de procesadores con la [fórmula](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/main.py#L224) `chunk_size = (n + n_cpu - 1) // n_cpu`. De esta manera se lanzan tantos procesos como procesadores haya disponibles y cada proceso se encarga de `chunk_size` operaciones. Si el número de iteraciones no es múltiplo del número de procesadores, el último proceso se encargará de menos iteraciones.
El [código](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/main.py#L205) de cada hilo es muy similar al código de la función monoproceso:
```Python
def pi_leibniz_concurrent_worker(start: int, end: int) -> float:
    s = 1 if start == 1 else 0
    k = 2 * start + 1

    for i in range(start, end + 1):
        # s += (-1)**(i) / (2 * i + 1)
        if i % 2 == 0:  # much faster
            s += 1 / k
        else:
            s -= 1 / k
        k += 2

    return s
```
Sin embargo los tiempos de ejecución no son los esperados. La implementación multiproceso en Python no es más rápida que la implementación monohilo en C++:
```
Pi calculation. 100_000_000 iterations:
Implementation                             Time (ms)  Calculated value    Returned type
______________________________________________________________________________________
Pi Leibniz Python                         11711.682  3.141592663589326    <class 'float'>
Pi Leibniz Python Concurrent               3608.394  3.1415926635898788   <class 'float'>
Pi Leibniz C++ one thread                   124.887  3.141592663589794    <class 'float'>
```
Además la proporción entre el tiempo de ejecución de la implementación con un único proceso y la implementación con varios procesos concurrentes es de 3,2 (11711.682 / 3608.394 = 3,24) lo que indica que la implementación multiproceso no es capaz de aprovechar los cuatro núcleos del ordenador en el que se ejecutaron las pruebas.

### Cálculo del valor de $\pi$ usando la fórmula de Leibniz y una implementación multihilo en C++.
El siguiente paso para intentar reducir el tiempo de ejecución es hacer una implementación multihilo en C++ del algoritmo anterior. Esto es bastante sencillo, ya que no hay datos compartidos entre los distintos hilos como en la criba de Eratóstenes.
La [traducción](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/CalculatingPi.cpp#L124) a C++ del proceso padre:
```C++
long double pi_leibniz_threads(unsigned long long n)
{
    long double sum = 0;
    std::vector<std::thread> workers;
    std::vector<std::future<decltype(sum)>> futures;
    auto n_cpu = std::thread::hardware_concurrency();

    // Launch n_cpu threads
    unsigned long long start, end;
    unsigned long long chunk_size = (n + n_cpu - 1) / n_cpu;
    for (uint proc = 0; proc < n_cpu; proc++){
        std::promise<decltype(sum)> p;
        futures.push_back(p.get_future());
        start = (chunk_size * proc) + 1;
        end = start + chunk_size - 1 > n ? n : start + chunk_size - 1;
        workers.emplace_back(pi_leibniz_worker, std::move(p), start ,end);
    }

    for (auto & w : workers)
        w.join();

    for(auto & f : futures)
        sum += f.get();
    sum *= 4;

    return sum;
}
``` 
La [traducción](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/CalculatingPi.cpp#L105) a C++ del código del hilo:
``` C++
void pi_leibniz_worker(std::promise<long double> && p,
                       unsigned long long start,
                       unsigned long long end)
{
    long double s = start == 1 ? 1: 0;
    long double k = 2 * start + 1;

    for (unsigned long long i = start; i <= end; i++){
        // s += pow(-1, i) / (2 * i + 1);
        if (i % 2 == 0)  // much faster
            s += 1.0 / k;
        else
            s -= 1.0 / k;
        k += 2;
    }

    p.set_value(s);
}
```
El hilo padre se encarga de [recoger](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/CalculatingPi.cpp#L147) todos los resultados, sumarlos y multiplicarlos por cuatro para obtener el valor de $\pi$.

Los tiempos de ejecución para n=100_000_000:
```
Pi calculation. 100_000_000 iterations:
Implementation                             Time (ms)  Calculated value    Returned type
______________________________________________________________________________________
Pi Leibniz Python                         11711.682  3.141592663589326    <class 'float'>
Pi Leibniz Python Concurrent               3608.394  3.1415926635898788   <class 'float'>
Pi Leibniz C++ one thread                   124.887  3.141592663589794    <class 'float'>
Pi Leibniz C++ multi thread                  33.666  3.141592663589793    <class 'float'>
```
Al igual que en la implementación multihilo de la criba de Eratóstenes, las pruebas se han hecho en un ordenador con cuatro núcleos y ocho hilos (Intel Hyperthreading), sin embargo el tiempo de ejecución de la implementación multihilo no es la octava parte del tiempo de ejecución de la implementación con un solo hilo. No obstante, la proporción se acerca a un cuarto (124.887 / 33.666 = 3,7). Esto seguramente es debido a que los dos procesadores lógicos del mismo núcleo comparten una misma [FPU](https://en.wikipedia.org/wiki/Floating-point_unit) que actúa como cuello de botella.
En la tabla de tiempo también se puede apreciar que la reducción de tiempo de ejecución entre la implementación más sencilla en Python y la implementación multihilo en C++ es de un 99.7%.

### Cálculo del valor de $\pi$ usando la fórmula de Leibniz y una implementación C++ en GPU con CUDA.
El cálculo del valor de $\pi$ usando la fórmula de Leibniz es un algoritmo que es fácilmente paralelizable y que reduce su tiempo de ejecución con una progresión geométrica en relación con los hilos usados, como se ha visto en el apartado anterior. Es por esto que me pareció interesante implementar este algoritmo usando la GPU.
La [implementación](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/CalculatingPi_gpu.cu#L100) realizada es sumamente sencilla. La función de entrada se encarga de calcular el número de hilos óptimo que se lanzarán en la GPU usando la [función](https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/) `cudaOccupancyMaxPotentialBlockSize`, reserva espacio en memoria para el resultado de cada hilo y, cuando los hilos han concluido, recoge los resultados y devuelve el valor de $\pi$ calculado:
```C++
long double pi_leibniz_gpu(const unsigned long long iterations) {
    // check for GPU
    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
        return 0;
    }

    // https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
    int blockSize; // The launch configurator returned block size
    int gridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    HANDLE_ERROR(cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, pi_leibniz, 0, 0));

    const int n_threads = gridSize * blockSize;
    float_type result[n_threads];
    float_type *dev_result;

    HANDLE_ERROR(cudaMalloc((void **) &dev_result, n_threads * sizeof(float_type)));
    pi_leibniz<<<gridSize, blockSize>>>(dev_result, iterations);
    HANDLE_ERROR(cudaMemcpy(result, dev_result, n_threads * sizeof(float_type), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(dev_result));

    // result array has only a few thousand items. It's not necessary to use:
    // https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    // https://github.com/mark-poscablo/gpu-sum-reduction
    long double pi = std::reduce(result, result + n_threads, static_cast<float_type>(0));
    pi *= 4;

    return pi;
}
```
El [código](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/CalculatingPi_gpu.cu#L77) de los hilos que se lanzan en la GPU es muy parecido al de los hilos que se lanzan en la CPU  del apartado anterior:
```C++
__global__ void pi_leibniz(float_type *result, const unsigned long int iterations) {
    const unsigned int n_threads = gridDim.x * blockDim.x;
    const unsigned long int chunk_size = (iterations + n_threads - 1) / n_threads;

    const unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned long int start = (chunk_size * index) + 1;
    const unsigned long int end = start + chunk_size - 1 > iterations ? iterations : start + chunk_size - 1;

    float_type s = start == 1 ? 1: 0;
    float_type k = 2.0 * start + 1;

    for (unsigned long long i = start; i <= end; i++){
        // s += pow(-1, i) / (2 * i + 1);
        if (i % 2 == 0)  // much faster
            s += 1.0 / k;
        else
            s -= 1.0 / k;
        k += 2;
    }

    result[index] = s;
}
```
La diferencia fundamental con el código que se ejecuta en CPU es que el inicio y el fin del número de iteraciones que va a calcular cada hilo, las variables `start` y `end`, se tienen que calcular dentro del hilo ya que todos los hilos se lanzan a la vez con la [llamada](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/CalculatingPi_gpu.cu#L121)`
pi_leibniz<<<gridSize, blockSize>>>(dev_result, iterations);`.

De igual manera, como no existe la posibilidad de que cada hilo devuelva un valor, el resultado de cada hilo se guarda en un array llamado `result` en la posición indicada por la variable `index`. Esta variable indica el número de hilo y se calcula en base a unas variables propias de CUDA [llamadas](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#built-in-variables) `blockIdx` y `blockDim`.

Otra [diferencia](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/CalculatingPi_gpu.cu#L17) con el código que se ejecuta en CPU es el tipo del valor devuelto por cada hilo. Mientras que en el código ejecutado en la CPU el valor es del tipo `long double`, en el código ejecutado en la GPU es del tipo `double` ya que el tipo `long double` no es [soportado](https://docs.nvidia.com/cuda/archive/9.2/cuda-c-programming-guide/#long-double) por CUDA en el código que se ejecuta en GPU.

El ordenador en el que estuve haciendo estas pruebas tenía una tarjeta Nvidia GTX 1050, una tarjeta bastante modesta pero que en este caso cumple su función perfectamente. Los tiempos de ejecución para n=100_000_000:
```
Pi calculation. 100_000_000 iterations:
Implementation                             Time (ms)  Calculated value    Returned type
______________________________________________________________________________________
Pi Leibniz C++ one thread                   124.887  3.141592663589794    <class 'float'>
Pi Leibniz C++ multi thread                  34.243  3.141592663589793    <class 'float'>
Pi Leibniz C++ GPU                           93.967  3.1415926635897824   <class 'float'>
```
Sorprendentemente el tiempo de ejecución de la implementación que usa la GPU es casi el triple del tiempo de ejecución de la implementación multihilo en CPU. Sin embargo, si se aumentan el número de iteraciones:
```
Pi calculation. 100_000_000 iterations:
Implementation                             Time (ms)  Calculated value    Returned type
______________________________________________________________________________________
Pi Leibniz C++ multi thread                  34.243  3.141592663589793    <class 'float'>
Pi Leibniz C++ GPU                           93.967  3.1415926635897824   <class 'float'>

Pi calculation. 1_000_000_000 iterations:
Implementation                             Time (ms)  Calculated value    Returned type
______________________________________________________________________________________
Pi Leibniz C++ multi thread                 339.635  3.141592654589794    <class 'float'>
Pi Leibniz C++ GPU                          233.614  3.141592654589722    <class 'float'>

Pi calculation. 10_000_000_000 iterations:
Implementation                             Time (ms)  Calculated value    Returned type
______________________________________________________________________________________
Pi Leibniz C++ multi thread                3392.365  3.1415926536897945   <class 'float'>
Pi Leibniz C++ GPU                         2387.844  3.141592653689783    <class 'float'>

Pi calculation. 100_000_000_000 iterations:
Implementation                             Time (ms)  Calculated value    Returned type
______________________________________________________________________________________
Pi Leibniz C++ multi thread               33889.705  3.141592653599798    <class 'float'>
Pi Leibniz C++ GPU                        23338.685  3.1415926535997496   <class 'float'>
```
A partir de 1.000.000.000 iteraciones el tiempo de ejecución aumenta en la misma proporción que el número de iteraciones y la proporción entre ambas implementaciones se mantiene constante, siendo el tiempo de ejecución en GPU aproximadamente 2/3 del tiempo de ejecución en CPU.

## Cálculo del valor de $\pi$ usando integración numérica.
Para el cálculo del valor de $\pi$ usando [integración numérica](https://www.stolaf.edu/people/rab/os/pub0/modules/PiUsingNumericalIntegration/index.html) se ha seguido la misma estrategia que en el apartado anterior para el cálculo del valor de $\pi$ usando la [fórmula de Leibniz](https://en.wikipedia.org/wiki/Leibniz_formula_for_%CF%80). Se han hecho las siguientes implementaciones:
 - En Python sin [concurrencia](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/main.py#L147).
 - En Python con [concurrencia](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/main.py#L171) multiproceso.
 - En C++ con un único [hilo](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/CalculatingPi.cpp#L10).
 - En C++ con múltiples [hilos](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/CalculatingPi.cpp#L50).
 - En C++ con CUDA, es decir ejecutando el algoritmo con múltiples [hilos](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/CalculatingPi_gpu.cu#L40) en GPU.

La implementación en Python es:
```Python
def pi_num_integration(n: int) -> float:
    sum_ = 0
    width = 2.0 / n
    for i in range(n):
        x = -1 + (i + 0.5) * width
        sum_ += math.sqrt(1 - x * x) * width

    return sum_ * 2.0
```
Y su traducción en C++:
```C++
long double pi_num_integration_cpp(unsigned long long numRect)
{
    long double width;       /* width of a rectangle subinterval */
    long double x;           /* an x_i value for determining height of rectangle */
    long double sum;         /* accumulates areas all rectangles so far */

    sum = 0;
    width = 2.0 / numRect;
    for (unsigned long long i = 0; i < numRect; i++) {
        x = -1 + (i + 0.5) * width;
        sum += std::sqrt(1 - x * x) * width;
    }

    return sum * 2;
}
```
Lo único destacable es que el algoritmo incluye una raíz cuadrada, que va a influir en las ganancias que se obtienen al implementar el algoritmo de diferentes formas. Las variaciones de las diferentes implementaciones son similares a las que se han hecho en el apartado anterior para la fórmula de Leibniz, por lo que no se van a comentar.

Los tiempos para n=100_000_000:
```
Pi calculation. 100_000_000 iterations:
Implementation                             Time (ms)  Calculated value    Returned type
______________________________________________________________________________________
Pi Area Int. Python                       22656.576  3.1415926535910885   <class 'float'>
Pi Area Int. Python Concurrent             6604.540  3.141592653590649    <class 'float'>
Pi Area Int. C++ one thread                 275.097  3.141592653590767    <class 'float'>
Pi Area Int. C++ multi thread               110.970  3.1415926535907674   <class 'float'>
Pi Area Int. C++ GPU                         61.598  3.141592653590776    <class 'float'>

```
Como ocurría con la fórmula de Leibniz la proporción del tiempo de ejecución entre la implementación monoproceso en Python y la implementación multiproceso en Python es cercana a 3,4 (22656.576 / 6604.540 = 3,43) lo que indica que la implementación multiproceso no aprovecha de una forma óptima los cuatro núcleos del ordenador donde se ejecutaron las pruebas.
Mucho más llamativa es la proporción del tiempo de ejecución entre la implementación en C++ monohilo y la implementación en C++ multihilo: 2,47 (275.097 / 110.970 =2,47). La operación de raiz cuadrada, `std::sqrt` en C++, parece influir notablemente en la capacidad de los núcleos de paralelizar los hilos. Al igual que en el cálculo de $\pi$ con la fórmula de Leibniz, las operaciones en coma flotante inciden notablemente en el rendimiento en paralelo de los núcleos del procesador.

## Conclusiones
Lo que empezó como un juego para ponerme al día con las nuevas mejoras que está recibiendo C++ ha terminado siendo un sumidero de tiempo libre, aunque he de confesar que lo ha sido con gusto. Al menos, después de todas estas pruebas he podido llegar a las siguientes conclusiones:
 - Python es intrínsecamente lento en cuanto a tiempo de ejecución, pero es muy rápido en cuanto a tiempo de desarrollo.
 - Es conveniente usar los módulos disponibles para Python, puesto que mejoran el tiempo de desarrollo y el tiempo de ejecución.
 - Si el tiempo de ejecución es crucial, una vez que se haya concluido el primer prototipo de la aplicación conviene localizar los puntos del código en Python que más inciden en el tiempo de ejecución usando un perfilador de rendimiento. Una vez que se tengan localizado los puntos críticos se pueden seguir dos estrategias:
	 - Intentar buscar algún módulo que aborde el mismo problema y haya sido programado en un lenguaje no interpretado.
	 - Programarse uno mismo, en un lenguaje no interpretado, un módulo que ejecute las partes computacionalmente más costosas. Esto se puede hacer usando directamente la API que el interprete de Python facilita, pero es mucho más sencillo utilizar alguna librería. En el caso de C++ la librería pybind11 simplifica mucho este paso.
 - Si la aplicación que se desarrolla en Python gasta la mayor parte del tiempo de ejecución en esperas de entrada y salida de datos, se puede crear una aplicación multiproceso o multihilo con un buen rendimiento. Para esto se puede usar alguno de los módulos de Python como [concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html).
 - Si la aplicación que se desarrolla en Python gasta la mayor parte del tiempo de ejecución usando la CPU, o lo que es lo mismo la aplicación es computacionalmente intensiva, y el problema a resolver es fácilmente paralelizable es conveniente evaluar la posibilidad de crear un módulo en C o C++ que haga uso concurrente de todos los núcleos displonibles de la CPU o de la GPU.
 De forma general se obtendrán grandes ganancias en el tiempo de ejecución, pero dependerá de la propia naturaleza del algoritmo a implementar, por ejemplo los cálculos con números en coma flotante o el acceso a zonas de memoria de forma concurrente pueden disminuir mucho la ganancia en tiempo de ejecución.

## Compilación del código fuente
Para compilar el código fuente en Ubuntu 24.04 y Ubuntu 20.04 los pasos son:
 - Instala los paquetes necesarios.
   ```bash
   $ sudo apt install cmake g++ python3-dev python3-pybind11
   ```
   Si tienes una tarjeta gráfica Nvidia instala además `nvidia-cuda-toolkit`.
   ```bash
   $ sudo apt install nvidia-cuda-toolkit
   ```
 - Clona el repositorio git.
   ```bash
   git clone https://github.com/eduardoposadas/test_pybind11.git
   ```
 - Cambia al directorio creado y ejecuta `cmake` para configurar el proyecto y generar un sistema de compilación.
   ```bash
   $ cd test_pybind11
   $ cmake -DCMAKE_BUILD_TYPE=Release -S . -B build_dir_Release
   ```
 - Lanza de nuevo `cmake` para compilar y enlazar
   ```bash
   $ cmake --build build_dir_Release -- -j $( nproc )
   ```
 - Estos pasos habrán generado el módulo con un nombre parecido a `test_pybind11.cpython-310-x86_64-linux-gnu.so` en el propio directorio del código fuente.
 - Lanza el script `main.py`
   ```bash
   $ ./main.py
   ```

