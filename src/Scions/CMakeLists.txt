include(GenerateExportHeader)


add_library(CPU ep/common/runtime_sequential_graph.cpp
        ep/cpu/cpu_ep.cpp
        ep/cpu/op/tensor_ops.cpp)



add_library(Scions::CPU ALIAS CPU)

find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})
message(STATUS "Imported oneMKL targets: ${MKL_IMPORTED_TARGETS}")

target_link_libraries(CPU PRIVATE Scions_options Scions_warnings)
target_link_libraries(CPU PUBLIC fmt::fmt-header-only MKL::MKL)

target_include_directories(CPU ${WARNING_GUARD} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

target_compile_features(CPU PUBLIC cxx_std_23)

set_target_properties(
        CPU
        PROPERTIES VERSION ${PROJECT_VERSION}
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES)

generate_export_header(CPU EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/Scions/cpu_export.hpp)

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(CPU PUBLIC SAMPLE_LIBRARY_STATIC_DEFINE)
    message(STATUS "Sample library is built for static linking")
endif()
