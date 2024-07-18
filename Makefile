# Install:
# sudo apt install make g++ python3-dev python3-pybind11 nvidia-cuda-toolkit

PYTHON = python3
PYTHON-CONFIG = python3-config
# For a specific Python version:
# PYTHON = python3.11
# PYTHON-CONFIG = python3.11-config

TARGET = test_pybind11$(shell $(PYTHON-CONFIG) --extension-suffix)
BUILD_DIR_RELEASE = build-Release
BUILD_DIR_DEBUG   = build-Debug

RM         = rm -rf
MKDIR      = mkdir -p
MAKEFLAGS := --jobs=$(shell nproc)

CPPFLAGS = -MP -MD
CXXFLAGS = -std=c++17 -Wall -Wextra -pedantic -Werror -fPIC -fopenmp $(shell $(PYTHON) -m pybind11 --includes)
LDFLAGS  = -shared -fopenmp

NVCC := $(shell command -v nvcc 2> /dev/null)
ifdef NVCC
  CXXFLAGS     += -D HAS_CUDA
  CUDA_CPPFLAGS = -MP -MD
  CUDA_CXXFLAGS = -Werror all-warnings --use_fast_math --compiler-options '-fPIC' $(shell $(PYTHON) -m pybind11 --includes)
  CUDA_LDFLAGS  = -shared
else
  $(info CUDA not enabled. Command nvcc not found)
endif

# Use "make DEBUG=1" to generate a target with debug info
DEBUG ?= 0
ifeq ($(DEBUG), 0)
  CXXFLAGS      += -D NDEBUG -O3
  CUDA_CXXFLAGS += -D NDEBUG -O3
  LDFLAGS       += -s
  BUILD_DIR      = $(BUILD_DIR_RELEASE)
else
  $(info DEBUG Options enabled)
  CXXFLAGS      += -ggdb3 -Og
  CUDA_CXXFLAGS += -g -G
  BUILD_DIR      = $(BUILD_DIR_DEBUG)
endif

# TARGET in BUILD_DIR directory. As the target is a Python module, I prefer in the Makefile directory with main.py
#TARGET_FILES = $(TARGET:%=$(BUILD_DIR)/%)
TARGET_FILES      = $(TARGET)
SOURCE_FILES      = $(wildcard *.cpp)
OBJECT_FILES      = $(SOURCE_FILES:%.cpp=$(BUILD_DIR)/%.o)

ifdef NVCC
  CUDA_SOURCE_FILES = $(wildcard *.cu)
  CUDA_OBJECT_FILES = $(CUDA_SOURCE_FILES:%.cu=$(BUILD_DIR)/%.o)
  LINKER            = $(NVCC) --linker-options "$(LDFLAGS)" $(CUDA_LDFLAGS)
else
#  CUDA_SOURCE_FILES =
#  CUDA_OBJECT_FILES =
  LINKER            = $(CXX) $(LDFLAGS)
endif

DEPS_FILES        = $(OBJECT_FILES:.o=.d) $(CUDA_OBJECT_FILES:.o=.d)



.PHONY: all clean make_dirs
all: make_dirs $(TARGET_FILES)

clean:
	@echo Cleaning $(BUILD_DIR_RELEASE) $(BUILD_DIR_DEBUG) $(TARGET_FILES)
	@$(RM) $(BUILD_DIR_RELEASE) $(BUILD_DIR_DEBUG) $(TARGET_FILES)

make_dirs: $(BUILD_DIR)
$(BUILD_DIR):
	@echo Making directory $@
	@$(MKDIR) $@

$(OBJECT_FILES): $(BUILD_DIR)/%.o: %.cpp
	@echo Compiling $<
	@$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

$(CUDA_OBJECT_FILES): $(BUILD_DIR)/%.o: %.cu
	@echo Compiling $<
	@$(NVCC) $(CUDA_CXXFLAGS) $(CUDA_CPPFLAGS) -c -o $@ $<

$(TARGET_FILES): $(OBJECT_FILES) $(CUDA_OBJECT_FILES)
	@echo Linking $@
	@$(LINKER) -o $@ $^
	@echo "Build successful!"

-include ${DEPS_FILES}
