#pragma once

#include <optional>
#include <string>
#include <chrono>

long int get_CPU_L1_data_cache_size();

class StopWatch
{
public:
    explicit StopWatch(const std::optional<std::string>& message = std::nullopt);
    void elapsed(const std::optional<std::string>& message = std::nullopt);
    static std::string str_time();
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::duration<float,std::milli> elapsed_;
};
