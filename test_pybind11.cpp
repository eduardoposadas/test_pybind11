#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<unsigned long long>);

#include "SieveOfEratosthenes.h"
#include "CalculatingPi.h"
#ifdef HAS_CUDA
#include "CalculatingPi_gpu.cuh"
#endif


PYBIND11_MODULE(test_pybind11, m) {
    m.doc() = "pybind11 testing plugin.";

    py::bind_vector<std::vector<unsigned long long>>(m, "VectorULongLongInt");

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

    m.def("pi_leibniz_cpp", &pi_leibniz, "Calculate pi with one thread.",
          py::arg("name") = "pi_leibniz_cpp", py::arg("n") = 10000000);
    m.def("pi_leibniz_cpp_threads", &pi_leibniz_threads, "Calculate pi with multiple threads.",
          py::arg("name") = "pi_leibniz_cpp_threads", py::arg("n") = 10000000);

    m.def("pi_num_integration_cpp", &pi_num_integration_cpp, "Calculate pi with one thread.",
          py::arg("name") = "pi_area_serial_cpp", py::arg("n") = 10000000);
    m.def("pi_num_integration_cpp_threads", &pi_num_integration_cpp_threads, "Calculate pi with multiple threads.",
          py::arg("name") = "pi_area_cpp_threads", py::arg("n") = 10000000);

#ifdef HAS_CUDA
    m.def("pi_leibniz_gpu", &pi_leibniz_gpu, "Calculate pi with GPU.",
          py::arg("name") = "pi_leibniz_gpu", py::arg("n") = 10000000);
    m.def("pi_num_integration_gpu", &pi_num_integration_gpu, "Calculate pi with GPU.",
          py::arg("name") = "pi_area_serial_gpu", py::arg("n") = 10000000);
#endif
}
