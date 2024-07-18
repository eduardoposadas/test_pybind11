#!/usr/bin/env python3

import os
import time
import math
import numpy as np
import concurrent.futures as cf
import test_pybind11


def launch_functions(functions: tuple, partial_results: list, total_results: dict) -> None:
    partial_results.clear()
    for f, name, iterations in functions:
        clock = StopWatch(message=f' Outside start: {name}')

        result = f(n=iterations, name=name)

        duration = clock.elapsed(f' Outside end: {name} Duration: ')
        print()

        type_ = str(type(result))
        if (isinstance(result, list)
                or isinstance(result, test_pybind11.VectorULongLongInt)
                or isinstance(result, np.ndarray)):
            result = len(result)

        partial_results.append([name, duration, result, type_])

        if name not in total_results:
            total_results[name] = {}
        if iterations not in total_results[name]:
            total_results[name][iterations] = {}
        total_results[name][iterations]['result'] = result
        total_results[name][iterations]['duration'] = duration


def print_results(r: list) -> None:
    for (name, duration, result, type_) in r:
        duration = round(duration / 1e6, 3)  # from nano to milliseconds with 3 decimals
        print(f'{name:40} {duration:10.3f} {result:^20}  {type_}')


def results_to_csv(r: dict, file_name: str = 'results.csv') -> None:
    sep = ","
    with open(file_name, 'w') as f:
        # Write the first line with the fields
        fields = f'Implementation{sep}'
        k1 = next(iter(r))  # first key of r
        for n_iter in r[k1]:
            fields += f'{n_iter} iter. result{sep}'
        for n_iter in r[k1]:
            fields += f'{n_iter} iter. duration{sep}'
        f.write(f'{fields[:-1]}\n')

        # Write data lines
        for implementation in r:
            f.write(f'{implementation}{sep}')
            line = ''
            for n_iter in r[implementation]:
                data = r[implementation][n_iter]['result']
                line += f'{data}{sep}'
            for n_iter in r[implementation]:
                data = r[implementation][n_iter]['duration']
                line += f'{data}{sep}'
            f.write(f'{line[:-1]}\n')


class StopWatch:
    def __init__(self, message: str = None):
        self._start = time.time_ns()
        self._elapsed = 0
        if message is not None:
            print(f'{self._str_time(self._start)}{message}')

    def elapsed(self, message: str = None) -> int:
        self._elapsed = time.time_ns() - self._start
        if message is not None:
            duration = round(self._elapsed / 1e6, 3)  # from nano to milliseconds with 3 decimals
            print(self._str_time() + message + str(duration) + ' ms')
        return self._elapsed

    @staticmethod
    def _str_time(ns: int = None) -> str:
        if ns is None:
            ns = time.time_ns()
        s, ns = divmod(int(ns), int(1e9))
        st = time.localtime(s)
        return f'{st.tm_hour:02d}:{st.tm_min:02d}:{st.tm_sec:02d}.{math.floor(ns / 1e6):03}'


# Do not use this function. It is very slow, but is readable
# @profile
def sieve_of_Eratosthenes_naive(name: str, n: int) -> list:
    clock = StopWatch(message=f' Inside start: {name}')

    sieve = [True] * (n + 1)

    for p in range(2, int(math.sqrt(n)) + 1):
        if sieve[p]:
            for i in range(p * p, n + 1, p):
                sieve[i] = False

    out = []
    for p in range(2, n + 1):
        if sieve[p]:
            out.append(p)

    clock.elapsed(f' Inside end: {name} Duration: ')
    return out


# @profile
def sieve_of_Eratosthenes(name: str, n: int) -> list:
    clock = StopWatch(message=f' Inside start: {name}')

    sieve = [True] * (n + 1)

    for p in range(2, int(math.sqrt(n)) + 1):
        if sieve[p]:
            sieve[p * p: n + 1: p] = [False] * (((n + 1) - (p * p) + p - 1) // p)

    out = [i for i in range(2, n + 1) if sieve[i]]

    clock.elapsed(f' Inside end: {name} Duration: ')
    return out


# @profile
def sieve_of_Eratosthenes_numpy(name: str, n: int) -> np.ndarray:
    clock = StopWatch(message=f' Inside start: {name}')

    sieve = np.full(n + 1, True)
    sieve[0] = False
    sieve[1] = False

    for p in range(2, int(math.sqrt(n)) + 1):
        if sieve[p]:
            sieve[p * p: n + 1: p] = False

    clock.elapsed(f' Before filling in the list of prime numbers. Duration: ')
    out = np.nonzero(sieve)[0]

    clock.elapsed(f' Inside end: {name} Duration: ')
    return out


def pi_num_integration(name: str, n: int) -> float:
    """https://www.stolaf.edu/people/rab/os/pub0/modules/PiUsingNumericalIntegration/index.html"""
    clock = StopWatch(message=f' Inside start: {name}')

    sum_ = 0
    width = 2.0 / n
    for i in range(n):
        x = -1 + (i + 0.5) * width
        sum_ += math.sqrt(1 - x * x) * width

    clock.elapsed(f' Inside end: {name} Duration: ')
    return sum_ * 2.0


def pi_num_integration_concurrent_worker(n: int, start: int, end: int) -> float:
    sum_ = 0
    width = 2.0 / n
    for i in range(start, end):
        x = -1 + (i + 0.5) * width
        sum_ += math.sqrt(1 - x * x) * width

    return sum_


def pi_num_integration_concurrent(name: str, n: int) -> float:
    clock = StopWatch(message=f' Inside start: {name}')

    n_cpu = os.cpu_count()
    chunk_size = (n + n_cpu - 1) // n_cpu
    chunks = [[(i * chunk_size), (i * chunk_size) + chunk_size] for i in range(n_cpu)]
    chunks[-1:][0][1] = n  # end of last chunk = n

    with cf.ProcessPoolExecutor(max_workers=n_cpu) as executor:
        results = [executor.submit(pi_num_integration_concurrent_worker, n, a, b) for (a, b) in chunks]
        sum_ = sum(r.result() for r in cf.as_completed(results))

    clock.elapsed(f' Inside end: {name} Duration: ')
    return sum_ * 2.0


def pi_leibniz(name: str, n: int) -> float:
    clock = StopWatch(message=f' Inside start: {name}')

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

    clock.elapsed(f' Inside end: {name} Duration: ')
    return s


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


def pi_leibniz_concurrent(name: str, n: int) -> float:
    clock = StopWatch(message=f' Inside start: {name}')

    n_cpu = os.cpu_count()
    chunk_size = (n + n_cpu - 1) // n_cpu
    chunks = [[(i * chunk_size) + 1, (i * chunk_size) + chunk_size] for i in range(n_cpu)]
    chunks[-1:][0][1] = n  # end of last chunk = n

    with cf.ProcessPoolExecutor(max_workers=n_cpu) as executor:
        results = [executor.submit(pi_leibniz_concurrent_worker, a, b) for (a, b) in chunks]
        sum_ = sum(r.result() for r in cf.as_completed(results))

    clock.elapsed(f' Inside end: {name} Duration: ')
    return sum_ * 4.0


if __name__ == "__main__":
    total_results = {}
    partial_results = []
    for iterations in [1_000_000, 10_000_000, 100_000_000, 1_000_000_000]:
        print(f'\nLaunching Sieve Of Eratosthenes {iterations:_} iterations:')
        functions = (
            (sieve_of_Eratosthenes, 'Sieve Python', iterations),
            (sieve_of_Eratosthenes_numpy, 'Sieve Python NumPy', iterations),
            (test_pybind11.sieve_std_list, 'Sieve C++ Serial std::list', iterations),
            (test_pybind11.sieve_python_list, 'Sieve C++ Serial py::list', iterations),
            (test_pybind11.sieve_std_vector, 'Sieve C++ Serial opaque type', iterations),
            (test_pybind11.sieve_as_array_nocopy, 'Sieve C++ Serial np.ndarray_nocopy', iterations),
            (test_pybind11.sieve_as_array_nocopy_omp, 'Sieve C++ OpenMP np.ndarray_nocopy', iterations),
            (test_pybind11.sieve_as_array_nocopy_thread, 'Sieve C++ Multi thread np.ndarray_nocopy', iterations),
            (test_pybind11.sieve_as_array_nocopy_thread_pool, 'Sieve C++ Thread pool np.ndarray_nocopy', iterations),
            (test_pybind11.sieve_as_array_nocopy_generic_thread_pool, 'Sieve C++ Gen thr pool np.ndarray_nocopy', iterations),
        )
        launch_functions(functions, partial_results, total_results)
        print(f'Sieve Of Eratosthenes. Prime numbers less than {iterations:_}:\n'
              'Implementation                             Time (ms)   Primes found      Returned type\n'
              '_________________________________________________________________________________________')
        print_results(partial_results)

        print(f'\nLaunching Pi calculation {iterations:_} iterations:')
        functions = (
            (pi_leibniz, 'Pi Leibniz Python', iterations),
            (pi_leibniz_concurrent, 'Pi Leibniz Python Concurrent', iterations),
            (test_pybind11.pi_leibniz_cpp, 'Pi Leibniz C++ one thread', iterations),
            (test_pybind11.pi_leibniz_cpp_threads, 'Pi Leibniz C++ multi thread', iterations),
        )
        if hasattr(test_pybind11, 'pi_leibniz_gpu'):  # if the module has GPU support
            functions += ((test_pybind11.pi_leibniz_gpu, 'Pi Leibniz C++ GPU', iterations), )

        launch_functions(functions, partial_results, total_results)
        print(f'Pi calculation. {iterations:_} iterations:\n'
              'Implementation                             Time (ms)  Calculated value    Returned type\n'
              '______________________________________________________________________________________')
        print_results(partial_results)

        print(f'\nLaunching Pi calculation {iterations:_} iterations:')
        functions = (
            (pi_num_integration, 'Pi Area Int. Python', iterations),
            (pi_num_integration_concurrent, 'Pi Area Int. Python Concurrent', iterations),
            (test_pybind11.pi_num_integration_cpp, 'Pi Area Int. C++ one thread', iterations),
            (test_pybind11.pi_num_integration_cpp_threads, 'Pi Area Int. C++ multi thread', iterations),
        )
        if hasattr(test_pybind11, 'pi_num_integration_gpu'):  # if the module has GPU support
            functions += ((test_pybind11.pi_num_integration_gpu, 'Pi Area Int. C++ GPU', iterations), )

        launch_functions(functions, partial_results, total_results)
        print(f'Pi calculation. {iterations:_} iterations:\n'
              'Implementation                             Time (ms)  Calculated value    Returned type\n'
              '______________________________________________________________________________________')
        print_results(partial_results)

    results_to_csv(total_results)
