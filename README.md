* [Playing with Pybind11](#playing-with-pybind11)
   * [Prime number search using the sieve of Eratosthenes in Python.](#prime-number-search-using-the-sieve-of-eratosthenes-in-python)
   * [Pybind11](#pybind11)
   * [Prime number search using the sieve of Eratosthenes in C++.](#prime-number-search-using-the-sieve-of-eratosthenes-in-c)
      * [Implementations of the sieve of Eratosthenes in C++ using threads.](#implementations-of-the-sieve-of-eratosthenes-in-c-using-threads)
   * [Calculation of the value of pi using Leibniz's formula.](#calculation-of-the-value-of-pi-using-leibnizs-formula)
      * [Calculation of the value of pi using Leibniz's formula and a multiprocess implementation in Python.](#calculation-of-the-value-of-pi-using-leibnizs-formula-and-a-multiprocess-implementation-in-python)
      * [Calculation of the value of pi using Leibniz's formula and a multithreaded implementation in C++.](#calculation-of-the-value-of-pi-using-leibnizs-formula-and-a-multithreaded-implementation-in-c)
      * [Calculation of the value of pi using Leibniz's formula and a C++ implementation on GPU with CUDA.](#calculation-of-the-value-of-pi-using-leibnizs-formula-and-a-c-implementation-on-gpu-with-cuda)
   * [Calculation of the value of pi using numerical integration.](#calculation-of-the-value-of-pi-using-numerical-integration)
   * [Conclusions](#conclusions)
   * [Compiling the source code](#compiling-the-source-code)

# Playing with Pybind11
Python is an excellent language for rapid prototyping. Thanks to the wide variety of its ecosystem of libraries, it allows the creation of small applications in a very short time, with little effort and excellent results. However, if the problem to be solved is computationally intensive, Python quickly reveals that it is not the appropriate language for this type of problem. Being an interpreted language it is inherently slow and furthermore the default Python interpreter ([CPython](https://github.com/python/cpython)) has a global [lock](https://realpython.com/python-gil/), known as [GIL](https://wiki.python.org/moin/GlobalInterpreterLock), which prevents multiple threads from executing concurrently, resulting in not being able to efficiently take advantage of the multithreading capability that most current processors have.

At the time of this writing there is already an [attempt](https://peps.python.org/pep-0703/) [serious](https://www.blog.pythonlibrary.org/2024/03/14/python-3-13-allows-disabling-of-the-gil-subinterpreters/) to remove GIL from CPython, but it is still in the experimental phase.
In the meantime, as this [article](https://realpython.com/python-parallel-processing/) indicates, the only ways to avoid GIL are:

 - Use process-based parallelism instead of multithreading.
 - Use an alternative runtime environment for python.
 - Install a GIL-immune library like [NumPy](https://numpy.org/)
 - [Write](https://docs.python.org/3/extending/extending.html) a C or C++ extension module with the GIL released.
 - Have [Cython](https://cython.org/) generate a C extension module for you.
 - Call a foreign C function using [ctypes](https://docs.python.org/3/library/ctypes.html)

Each of these options has its advantages and disadvantages as indicated in the [article](https://realpython.com/python-parallel-processing/) mentioned above.
From these possible options I found it interesting to make a C extension module for CPython and explore the possibilities offered by this solution to the GIL problem. It so happened that I had just learned about [pybind11](https://pybind11.readthedocs.io/en/stable/index.html) and was also reviewing the latest [updates](https://github.com/AnthonyCalandra/modern-cpp-features) that C++ has received, so I decided to make a small module for Python using C++ and pybind11.

To evaluate the performance improvement achieved by using C++ instead of Python I chose three problems that are computationally expensive to solve:

 - Prime number search using [sieve of Eratosthenes](https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes)
 - Calculating the value of $\pi$ using [Leibniz's formula](https://en.wikipedia.org/wiki/Leibniz_formula_for_%CF%80)
 - Calculating the value of $\pi$ using [numerical integration](https://www.stolaf.edu/people/rab/os/pub0/modules/PiUsingNumericalIntegration/index.html)

Each problem has been implemented in different ways to test which is the fastest.

## Prime number search using the sieve of Eratosthenes in Python.
The sieve of Eratosthenes is a very simple algorithm that can be implemented in Python in a few lines:
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
Although this is the most intuitive way to implement the algorithm, it is extremely slow. Doing a performance profiling of the function with [line_profiler](https://github.com/pyutils/line_profiler) and `n=1_000_000` you can see which are the problematic lines:
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
Between lines 98 and 99 they consume more than 67% of the execution time. That loop is executed as many times as prime numbers are found, so it is clear that it must be changed for something more optimal. In this case, the fastest code I have achieved was to assign to a [segmented](https://docs.python.org/es/3/reference/expressions.html#slicings) list another list with the same number of `False` elements.
The other 30% of the execution time corresponds to lines 102 and 103. A `for` loop to traverse a list and execute an `append`? That's very *unpythonic*! Better to use [list comprehension](https://docs.python.org/es/3/tutorial/datastructures.html#list-comprehensions).

After making the changes to the code, the resulting function profiling is:
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
With these changes 86% of the execution time is consumed in line 120, generating the list of prime numbers. A [generator](https://wiki.python.org/moin/Generators) could have been used to avoid this time consumption by replacing line 120 with `out = (i for i in range(2, n + 1) if sieve[i])`, but doing this would simply pass the time consumption to the code where the generator was used.

By doing optimizations, the code is no longer readable. This is one of the problems with all interpreted languages that claim to be “easy to learn”. If you write code in a way that is as natural and readable as possible, most of the time it will not be optimal code. To write optimal code you need to learn how the interpreter is going to execute the code, and therefore you eliminate the advantage of using an “easy to learn” language.

The next step to make faster code is to stop using pure Python and use some module, most likely written in C or some other language with a reputation for being difficult to learn, that is appropriate for the task to be implemented. This is where Python shows its true potential. There is a huge variety of modules that help not to reinvent the wheel and drastically increase the programmer's efficiency.
In this case, the [NumPy](https://numpy.org/) module allows you to perform operations on lists much faster.
The profiling of the function rewritten using NumPy is:
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
The first striking fact is that the execution time has been reduced from 240 ms to 6 ms. Also, if you know the meaning of the `np.full` and `np.nonzero` calls, the code is intuitive and readable at a glance. On the downside, these NumPy functions do not run in parallel, i.e. they do not use all available processor cores.
As a side note, it should be noted that some NumPy functions, such as the functions for [linear algebra](https://numpy.org/doc/stable/reference/routines.linalg.html), can indeed be run in parallel with very few changes to the Python code.

At this point, the only way to decrease execution time is to create a module for Python in a non-interpreted language that implements Eratosthenes' sieve and runs using multiple threads.

## Pybind11
[Pybind11](https://pybind11.readthedocs.io/en/stable/index.html) is a C++ library that allows, among many other things, to create C++ modules for Python in a very simple way.
To create a module, a file is created in which the API of the module is defined, in this case `test_pybind11.cpp`. As explained in the pybind11 [documentation](https://pybind11.readthedocs.io/en/stable/basics.html#header-and-namespace-conventions), this file must at least contain:
```C++
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "my_func.h"

PYBIND11_MODULE(test_pybind11, m) {
	m.def("python_name_for_my_func", &my_func, "Documentation for my_func");
}
```
Once this file is created, the C++ function itself must be implemented, that is, a `my_func.cpp` file is created with a function called `my_func`. Then you compile it with:
```
$ c++ -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) test_pybind11.cpp -o test_pybind11$(python3-config --extension-suffix)
```
this generates a file with a name similar to `test_pybind11.cpython-310-x86_64-linux-gnu.so`. If you get an error because it does not find the `c++` command or a file ending in .h install:
```bash
# sudo apt install g++ python3-dev python3-pybind11
```
Once the Python module is generated, it can be used by importing it and calling the function defined in the API:
```Python
$ python3
Python 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import test_pybind11
>>> test_pybind11.python_name_for_my_func()
```
To automate the compilation of the module you can use `cmake` as explained in the [documentation](https://pybind11.readthedocs.io/en/stable/compiling.html#modules-with-cmake).

## Prime number search using the sieve of Eratosthenes in C++.
The C++ implementation of the sieve of Eratosthenes has been done in eight different ways. The first four are single-threaded implementations, i.e. without parallelism. The last four are implemented using multiple threads, i.e. with parallel processing.
The code that allows these C++ functions to be used in Python has been reflected in the file `test_pybind11.cpp` in the lines:
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
These functions are implemented in the `SieveOfEratosthenes.cpp` file.  The declarations of the functions can be seen in the `SieveOfEratosthenes.h` file:
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
Pybind11 [allows](https://pybind11.readthedocs.io/en/stable/advanced/cast/index.html) to use Python types in C++, although this implies making a copy of the data when performing type conversion from C++ to Python. To avoid this copying, [opaque types](https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html#making-opaque-types) can be used. For STL containers pybind11 has [functions](https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html#binding-stl-containers) to [create](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/test_pybind11.cpp#L18) the opaque types directly.
For `numpy.array` pybind11 has `py::array_t`. A variable of type `py::array_t` can be created from a `std::vector` without copying data using a small template that performs type [conversion](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/SieveOfEratosthenes.cpp#L117).

Of the eight C++ implementations of the algorithm, the first four are single-threaded and are essentially the same. They are the direct translation of the Python implementation into C++:
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

The only difference is the data type where the list of prime numbers is stored:

 - `SieveOfEratosthenes_std_list` returns a `std::list`, i.e. it uses a C++ data type and pybind11 makes a copy of the data from the C++ `std::list` to a Python `list` data type.
 - `SieveOfEratosthenes_python_list` uses and returns a variable of type `py::list`. This data type is the way pybind11 allows to use a Python `list` data type in C++ code.
 - `SieveOfEratosthenes_std_vector` returns a `std::vector` treated as an opaque pybind11 data type, i.e. without data copying in the C++ to Python type conversion.
 - `SieveOfEratosthenes_as_array_nocopy` returns a `py::array_t` which in Python becomes a `numpy.array`. This array is generated from a `std::vector` without [copying](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/SieveOfEratosthenes.cpp#L157) data.

For n=100_000_000 the execution times are:
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
As you can see, none of these implementations is faster than using NumPy in Python. I don't have a clear answer for this. Possibly it is related to NumPy's use of [SIMD](https://numpy.org/doc/stable/reference/simd/index.html) instructions, or the [implementation](https://en.cppreference.com/w/cpp/container/vector_bool) of `std::vector<bool>` in the C++ STL.

With this runtime table, it bears repeating once again that Python offers the possibility of programming very efficiently without a great deal of effort thanks to the use of the available modules.

### Implementations of the sieve of Eratosthenes in C++ using threads.
The four C++ implementations of the sieve of Eratosthenes algorithm that use threads do so with different strategies. The declarations, as mentioned above, are in the `SieveOfEratosthenes.h` file:
```C++
py::array_t<unsigned long long> SieveOfEratosthenes_as_array_nocopy_omp(const std::string& name, unsigned long long n);
py::array_t<unsigned long long> SieveOfEratosthenes_as_array_nocopy_thread(const std::string& name, unsigned long long n);
py::array_t<unsigned long long> SieveOfEratosthenes_as_array_nocopy_thread_pool(const std::string& name, unsigned long long n);
py::array_t<unsigned long long> SieveOfEratosthenes_as_array_nocopy_generic_thread_pool(const std::string& name, unsigned long long n);
```
As their names reflect:

 - `SieveOfEratosthenes_as_array_nocopy_omp` uses the algorithm used in the serial version with `#pragma` [OpenMP](https://www.openmp.org/) directives for parallelism implementation. The advantage of using OpenMP is that you hardly have to modify the code. The disadvantage is that it is not as efficient as it should be.
The only change made to the serial version code, other than adding the OpenMP `#pragma` directives, was to change `std::vector<bool>` to `std::shared_ptr<bool[]>` since you cannot concurrently access elements of a `std::vector<bool>`:
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

 - `SieveOfEratosthenes_as_array_nocopy_thread` splits the sieve into equal chunks of the size of the L1 cache for [data](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/utils.cpp#L11) and launches a thread for each chunk. This is an example of how NOT TO DO IT.
 - `SieveOfEratosthenes_as_array_nocopy_thread_pool` [launches](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/SieveOfEratosthenes.cpp#L335) as many threads as there are cores in the computer where it is executed:
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
These threads [collect](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/SieveOfEratosthenes.cpp#L298) jobs from a FIFO [queue](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/jobs_fifo_queue.h#L26). In each job the sieve chunk that the thread has to compute is specified and there is an entry for the thread to save the result. Saving the result in the job queue itself avoids unnecessary data copies:
```C++
// Job type for the queue of jobs
struct job_type{
    unsigned long long input_start;
    unsigned long long input_end;
    std::vector<unsigned long long> output_primes;
};
```
The sieve chunks of each job are the size of the processors' L1 data cache. This prevents different processor cores from competing for the same data in the processor cache:
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
When there are no more jobs in the FIFO queue, the results of all jobs are collected to create the resulting list of prime numbers.
 - `SieveOfEratosthenes_as_array_nocopy_generic_thread_pool` is a more generic [implementation](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/SieveOfEratosthenes.cpp#L414) than the previous one. A task [queue](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/thread_pool.h#L15) has been used in which any C++ function can be enqueued. In this case functions are queued that share a common sieve and compute the prime numbers from a sieve chunk that has the size of the processors L1 cache for data. The results of these tasks are kept in the task queue itself and are collected once the task queue has been told that no more tasks are to be sent. The lists are sorted and returned together as a single list of prime numbers as the final result.
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
The execution times for n=1_000_000_000_000:
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
As you can see, the fastest implementation is `SieveOfEratosthenes_as_array_nocopy_thread_pool`, called “Sieve C++ Thread pool np.ndarray_nocopy” in the table above. Undoubtedly, making an implementation specific to the problem to be addressed helps to improve runtime.
The tests have been done on a computer with four cores and eight virtual cores (Intel Hyperthreading), so it was expected that the execution time of the fastest single-threaded implementation and the execution time of the fastest multithreaded implementation would be a multiple of four or eight. However the ratio is close to three: 9242,615 / 2834,655 = 3.26.
This is because:
 - Core-level multithreading technology (Intel Hyperthreading) is not useful for increasing parallelism in this particular case.
 - The very nature of the sieve of Eratosthenes algorithm. Each thread that is used to calculate the prime numbers in a chunk of the sieve needs to calculate the prime numbers prior to the start position of the sieve chunk, each thread is doing calculations that another thread has already performed.

## Calculation of the value of $\pi$ using Leibniz's formula.
Leibniz's [formula](https://en.wikipedia.org/wiki/Leibniz_formula_for_%CF%80) for the calculation of $\pi$ is very easy to [implement](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/main.py#L187) in Python:
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
As you can see from the comment, the first optimization that has been done is not to use exponentiation to decide the sign of the adder. Exponentiation is a very slow operation, even in C++, whereas performing the modulo two operation is inherently fast on a computer, since internally it is using a binary representation of the numbers.
The other small optimization that has been done is to use a variable called `k` to which at each iteration 2 is added to avoid having to calculate `2 * i + 1`.

The C++ [implementation](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/CalculatingPi.cpp#L85), which is in the file `CalculatingPi.cpp`, is the direct translation of the Python code:
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
The times for a run with n=100_000_000:
```
Implementation                             Time (ms)  Calculated value    Returned type
______________________________________________________________________________________
Pi Leibniz Python                         11711.682  3.141592663589326    <class 'float'>
Pi Leibniz C++ one thread                   124.887  3.141592663589794    <class 'float'>
```
Simply by translating the algorithm to C++ the execution time is almost 99% less.

It seems obvious that if the algorithm to be coded is simple, it is a critical point in the execution of the program and there is no Python module that already has the task to be performed implemented, the C++ implementation should always be considered as an option to be taken into account.

### Calculation of the value of $\pi$ using Leibniz's formula and a multiprocess implementation in Python.
Python offers the possibility to implement [multithreaded](https://docs.python.org/3/library/threading.html) processes but due to the GIL only one thread will run at a time which gives ridiculous performances. To avoid this you can use the parallelism module based on [processes](https://docs.python.org/3/library/multiprocessing.html). With this module instead of threads processes are launched for each task, which implies that:

 - The computational cost of launching a process is much higher than that of launching a thread. To reduce this cost, an initial group of processes is launched to perform a list of tasks.
 - Passing data between different processes is much more expensive than passing data between threads since it is done using shared memory. In this case it is not a problem since the child processes only return a floating point number to the parent.

Since in this problem the data transfer between the different processes is minimal, it is possible that the implementation using the Python process-based parallelism module will give good results.
The Python [implementation](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/main.py#L220) has been, as usual, very simple:
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
We have simply divided the number of iterations between the number of processors with the [formula](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/main.py#L224) `chunk_size = (n + n_cpu - 1) // n_cpu`. In this way as many processes are launched as processors are available and each process takes care of `chunk_size` operations. If the number of iterations is not a multiple of the number of processors, the last process will handle fewer iterations.
The [code](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/main.py#L205) of each thread is very similar to the code of the single-process function:
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
However the execution times are not as expected. The multiprocess implementation in Python is not faster than the single-threaded implementation in C++:
```
Pi calculation. 100_000_000 iterations:
Implementation                             Time (ms)  Calculated value    Returned type
______________________________________________________________________________________
Pi Leibniz Python                         11711.682  3.141592663589326    <class 'float'>
Pi Leibniz Python Concurrent               3608.394  3.1415926635898788   <class 'float'>
Pi Leibniz C++ one thread                   124.887  3.141592663589794    <class 'float'>
```
Furthermore, the ratio between the execution time of the single-process implementation and the multi-process implementation is 3.2 (11711,682 / 3608,394 = 3.24), which indicates that the multi-process implementation is not able to take advantage of the four cores of the computer on which the tests were run.

### Calculation of the value of $\pi$ using Leibniz's formula and a multithreaded implementation in C++.
The next step in trying to reduce the execution time is to do a multithreaded C++ implementation of the above algorithm. This is fairly straightforward, since there is no data sharing between threads as in the sieve of Eratosthenes.
The C++ [translation](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/CalculatingPi.cpp#L124) of the parent process:
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
The C++ [translation](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/CalculatingPi.cpp#L105) of the thread code:
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
The parent thread is responsible for [collecting](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/CalculatingPi.cpp#L147) all the results, adding them up and multiplying by four to get the value of $\pi$.

The execution times for n=100_000_000:
```
Pi calculation. 100_000_000 iterations:
Implementation                             Time (ms)  Calculated value    Returned type
______________________________________________________________________________________
Pi Leibniz Python                         11711.682  3.141592663589326    <class 'float'>
Pi Leibniz Python Concurrent               3608.394  3.1415926635898788   <class 'float'>
Pi Leibniz C++ one thread                   124.887  3.141592663589794    <class 'float'>
Pi Leibniz C++ multi thread                  33.666  3.141592663589793    <class 'float'>
```
As in the multithreaded implementation of the sieve of Eratosthenes, the tests have been done on a computer with four cores and eight threads (Intel Hyperthreading), however the execution time of the multithreaded implementation is not one eighth of the execution time of the single-threaded implementation. Nevertheless, the ratio is close to a quarter (124,887 / 33,666 = 3.7). This is most likely due to the fact that the two logical processors of the same core share the same [FPU](https://en.wikipedia.org/wiki/Floating-point_unit) which acts as a bottleneck.
From the timing table it can also be seen that the reduction in execution time between the simpler Python implementation and the multithreaded C++ implementation is 99.7%.

### Calculation of the value of $\pi$ using Leibniz's formula and a C++ implementation on GPU with CUDA.
The calculation of the value of $\pi$ using Leibniz's formula is an algorithm that is easily parallelizable and that reduces its execution time with a geometric progression in relation to the threads used, as seen in the previous section. This is why I found it interesting to implement this algorithm using the GPU.
The [implementation](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/CalculatingPi_gpu.cu#L100) is extremely simple. The entry function is in charge of calculating the optimal number of threads to be launched on the GPU using the [function](https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/) `cudaOccupancyMaxPotentialBlockSize`, reserves space in memory for the result of each thread and, when the threads have finished, collects the results and returns the calculated $\pi$ value:

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
The [code](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/CalculatingPi_gpu.cu#L77) for threads launched on the GPU is very similar to the code for threads launched on the CPU in the previous section:
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
The fundamental difference with the code running on CPU is that the start and end of the number of iterations to be calculated by each thread, the `start` and `end` variables, have to be calculated within the thread since all threads are launched at the same time with the [call](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/CalculatingPi_gpu.cu#L121) `
pi_leibniz<<<gridSize, blockSize><(dev_result, iterations);`.

Similarly, since there is no possibility for each thread to return a value, the result of each thread is stored in an array called `result` in the position indicated by the variable `index`. This variable indicates the thread number and is calculated based on CUDA's own variables [calls](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#built-in-variables) `blockIdx` and `blockDim`.

Another [difference](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/CalculatingPi_gpu.cu#L17) with the code running on CPU is the type of the value returned by each thread. While in the code running on CPU the value is of type `long double`, in the code running on GPU it is of type `double` since the `long double` type is not [supported](https://docs.nvidia.com/cuda/archive/9.2/cuda-c-programming-guide/#long-double) by CUDA in the code running on GPU.

The computer I was running these tests on had an Nvidia GTX 1050 card, a rather modest card but in this case it does its job perfectly. The execution times for n=100_000_000:
```
Pi calculation. 100_000_000 iterations:
Implementation                             Time (ms)  Calculated value    Returned type
______________________________________________________________________________________
Pi Leibniz C++ one thread                   124.887  3.141592663589794    <class 'float'>
Pi Leibniz C++ multi thread                  34.243  3.141592663589793    <class 'float'>
Pi Leibniz C++ GPU                           93.967  3.1415926635897824   <class 'float'>
```
Remarkably, the execution time of the GPU implementation is almost three times the execution time of the multithreaded CPU implementation. However, if the number of iterations is increased:
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
From 1,000,000,000,000 iterations onwards the execution time increases in the same proportion as the number of iterations and the ratio between the two implementations remains constant, with the GPU execution time being approximately 2/3 of the CPU execution time.

## Calculation of the value of $\pi$ using numerical integration.
For the calculation of the value of $\pi$ using [numerical integration](https://www.stolaf.edu/people/rab/os/pub0/modules/PiUsingNumericalIntegration/index.html) the same strategy has been followed as in the previous section for the calculation of the value of $\pi$ using the [Leibniz formula](https://en.wikipedia.org/wiki/Leibniz_formula_for_%CF%80). The following implementations have been made:
 - In Python without [concurrency](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/main.py#L147).
 - In Python with multiprocess [concurrency](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/main.py#L171).
 - In C++ with a single [thread](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/CalculatingPi.cpp#L10).
 - In C++ with multiple [threads](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/CalculatingPi.cpp#L50).
 - In C++ with CUDA, i.e. executing the algorithm with multiple [threads](https://github.com/eduardoposadas/test_pybind11/blob/64592f1bae25b268000c96c123261bb66d8bfcb0/CalculatingPi_gpu.cu#L40) on GPU.

The Python implementation is:
```Python
def pi_num_integration(n: int) -> float:
    sum_ = 0
    width = 2.0 / n
    for i in range(n):
        x = -1 + (i + 0.5) * width
        sum_ += math.sqrt(1 - x * x) * width

    return sum_ * 2.0
```
And its translation in C++:
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
The only remarkable thing is that the algorithm includes a square root, which will influence the gains obtained by implementing the algorithm in different ways. The variations of the different implementations are similar to those made in the previous section for the Leibniz formula, so they are not going to be commented.

The times for n=100_000_000:
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
As with the Leibniz formula, the execution time ratio between the single-process Python implementation and the multiprocess Python implementation is close to 3.4 (22656,576 / 6604,540 = 3.43), which indicates that the multiprocess implementation does not make optimal use of the four cores of the computer where the tests were run.
Much more striking is the ratio of execution time between the single-threaded C++ implementation and the multithreaded C++ implementation: 2.47 (275,097 / 110,970 =2.47). The square root operation, `std::sqrt` in C++, seems to noticeably influence the ability of the kernels to parallelize threads. As in the calculation of $\pi$ with the Leibniz formula, floating point operations have a significant impact on the parallel performance of the processor cores.

## Conclusions
What started as a game to catch me up with the new improvements that C++ is receiving has ended up being a free time sink, although I must confess that it has been with pleasure. If nothing else, after all this testing I have been able to come to the following conclusions:
 - Python is inherently slow in terms of runtime, but it is very fast in terms of development time.
 - It is convenient to use the modules available for Python, since they improve both development time and runtime.
 - If run time is crucial, once the first prototype of the application has been completed, it is a good idea to locate the points in the Python code that have the greatest impact on run time using a performance profiler. Once the critical points have been located, two strategies can be followed:
	 - Try to find a module that addresses the same problem and has been programmed in a non-interpreted language.
	 - Program yourself, in a non-interpreted language, a module that executes the most computationally expensive parts. This can be done by directly using the API provided by the Python interpreter, but it is much easier to use a library. In the case of C++, the pybind11 library greatly simplifies this step.
 - If the application being developed in Python spends most of its execution time waiting for data input and output, a multiprocess or multithreaded application can be created with good performance. For this you can use one of the Python modules such as [concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html).
 - If the application to be developed in Python spends most of the execution time using the CPU, or in other words the application is computationally intensive, and the problem to be solved is easily parallelizable, it is convenient to evaluate the possibility of creating a module in C or C++ that makes concurrent use of all the displaceable cores of the CPU or GPU.
 In general, large gains in execution time will be obtained, but it will depend on the nature of the algorithm to be implemented, for example, calculations with floating point numbers or concurrent access to memory areas can greatly reduce the gain in execution time.

## Compiling the source code
To compile the source code in Ubuntu 24.04 and Ubuntu 20.04 the steps are:
 - Install the necessary packages.
   ```bash
   $ sudo apt install cmake g++ python3-dev python3-pybind11
   ```
   If you have an Nvidia graphics card install ``nvidia-cuda-toolkit`` in addition.
   ```bash
   sudo apt install nvidia-cuda-toolkit
   ```
 - Clone the git repository.
   ```bash
   git clone https://github.com/eduardoposadas/test_pybind11.git
   ```
 - Change to the created directory and run `cmake` to configure the project and generate a build system.
   ```bash
   cd test_pybind11
   $ cmake -DCMAKE_BUILD_TYPE=Release -S . -B build_dir_Release
   ```
 - Launch ```cmake``` again to compile and bind
   ```bash
   $ cmake --build build_dir_Release -- -j $( nproc )
   ```
 - These steps will have generated the module with a name similar to `test_pybind11.cpython-310-x86_64-linux-gnu.so` in the source code directory itself.
 - Launch the `main.py` script
   ```bash
   ./main.py
   ```





