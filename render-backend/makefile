SRC_DIR := csrc
CUSRC_DIR := cuda
OBJ_DIR := obj
CPP_FILES := $(wildcard $(SRC_DIR)/*.cpp)
CU_FILES := $(wildcard $(CUSRC_DIR)/*.cu)
CPPOBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_FILES))
CUOBJ_FILES := $(patsubst $(CUSRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_FILES))
OBJ_FILES := $(CPPOBJ_FILES) $(CUOBJ_FILES)
TARGET := bin/main
CPPFLAGS := -I./include -I/usr/local/cuda/include -I./third $(shell pkg-config --cflags opencv4) -DTINYOBJLOADER_IMPLEMENTATION #-DVRT_DEBUG
CXXFLAGS := -std=c++17 -Wall -Wextra -Werror -pedantic -O3 -march=native -mtune=native -fopenmp
NVCCFLAGS := -std=c++17 -O3 -arch=native -Xcompiler -fopenmp -dopt on
LDFLAGS := $(shell pkg-config --libs opencv4)

.PHONY: all clean

all: $(TARGET)

clean:
	rm -f $(OBJ_FILES) $(TARGET)

$(TARGET): $(OBJ_FILES)
	@mkdir -p $(@D)
	nvcc $(CPPFLAGS) $(LDFLAGS) $(NVCCFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	g++ $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

$(OBJ_DIR)/%.o: $(CUSRC_DIR)/%.cu
	@mkdir -p $(@D)
	nvcc $(CPPFLAGS) $(NVCCFLAGS) -c -o $@ $<
