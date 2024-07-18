#pragma once

#include <string>

long double pi_num_integration_gpu(const std::string& name, unsigned long long iterations);
long double pi_leibniz_gpu(const std::string& name, unsigned long long iterations);
