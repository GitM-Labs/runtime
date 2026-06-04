cmake_minimum_required(VERSION 3.16)
project(hft_generator)

set(CMAKE_CXX_STANDARD 20)

# Arrow/Parquet from pyarrow
execute_process(
    COMMAND python3 -c "import pyarrow; print(pyarrow.get_include())"
    OUTPUT_VARIABLE ARROW_INCLUDE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND python3 -c "import pyarrow; print(pyarrow.get_library_dirs()[0])"
    OUTPUT_VARIABLE ARROW_LIB
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

include_directories(${ARROW_INCLUDE})
link_directories(${ARROW_LIB})

add_executable(hft_gen main.cpp)
target_compile_options(hft_gen PRIVATE -Wno-unused-result)
target_link_libraries(hft_gen arrow parquet)
