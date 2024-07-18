#include "utils.h"

#include <cmath>
#include <unistd.h>

// For py::print
#include <pybind11/pybind11.h>
namespace py = pybind11;


long int get_CPU_L1_data_cache_size()
{
    return sysconf(_SC_LEVEL1_DCACHE_SIZE);
}

StopWatch::StopWatch(const std::optional<std::string>& message):
    start(std::chrono::high_resolution_clock::now()),
    elapsed_(std::chrono::milliseconds(0))
{
    if(message)
        py::print(str_time(), message.value());
}

void StopWatch::elapsed(const std::optional<std::string>& message)
{
    elapsed_ = std::chrono::high_resolution_clock::now() - start;
    if(message){
        char buf[16];
        std::snprintf(buf, sizeof buf, "%.3f", elapsed_.count());
        py::print(str_time(), message.value(), std::string(buf), "ms");
    }
}

std::string StopWatch::str_time()
{
    std::timespec ts;
    std::timespec_get(&ts, TIME_UTC);
    char buf[10];
    std::strftime(buf, sizeof buf, "%H:%M:%S", std::localtime(&ts.tv_sec));
    auto ns = std::to_string(static_cast<int>(floor(ts.tv_nsec / 1e6))).substr(0,3);
    int zero_pad_len = 3 - ns.length();
    return std::string(buf) + '.' + std::string(zero_pad_len, '0') + ns;

}
