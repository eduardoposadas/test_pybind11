#pragma once

#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

std::list<unsigned long long> SieveOfEratosthenes_std_list(const std::string& name, unsigned long long n);
py::list SieveOfEratosthenes_python_list(const std::string& name, unsigned long long n);
std::vector<unsigned long long> SieveOfEratosthenes_std_vector(const std::string& name, unsigned long long n);
py::array_t<unsigned long long> SieveOfEratosthenes_as_array_nocopy(const std::string& name, unsigned long long n);
py::array_t<unsigned long long> SieveOfEratosthenes_as_array_nocopy_omp(const std::string& name, unsigned long long n);
py::array_t<unsigned long long> SieveOfEratosthenes_as_array_nocopy_thread(const std::string& name, unsigned long long n);
py::array_t<unsigned long long> SieveOfEratosthenes_as_array_nocopy_thread_pool(const std::string& name, unsigned long long n);
py::array_t<unsigned long long> SieveOfEratosthenes_as_array_nocopy_generic_thread_pool(const std::string& name, unsigned long long n);
