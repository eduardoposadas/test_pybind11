#pragma once

#include <string>

long double pi_num_integration_cpp(const std::string& name, unsigned long long numRect);
long double pi_num_integration_cpp_threads(const std::string& name, unsigned long long numRect);

long double pi_leibniz(const std::string& name, unsigned long long n);
long double pi_leibniz_threads(const std::string& name, unsigned long long n);
