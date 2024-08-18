#include(GenerateExportHeader)


add_library(CPU_EP INTERFACE ${PROJECT_SOURCE_DIR}/include/scions/ep/cpu)


add_library(Scions::CPU ALIAS CPU_EP)


target_link_libraries(CPU_EP INTERFACE Scions_options Scions_warnings)
#target_link_libraries(CPU_EP INTERFACE fmt::fmt-header-only)

target_include_directories(CPU_EP ${WARNING_GUARD}
        INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

target_precompile_headers(CPU_EP INTERFACE ${PROJECT_SOURCE_DIR}/include/manifold/common.hpp)

target_compile_features(CPU_EP INTERFACE cxx_std_23)

set_target_properties(
        CPU_EP
        PROPERTIES VERSION ${PROJECT_VERSION})

#generate_export_header(Manifold EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/Scions/cpu_export.hpp)

#if (NOT BUILD_SHARED_LIBS)
#    target_compile_definitions(Manifold PUBLIC SCIONS_LIBRARY_STATIC_DEFINE)
#    message(STATUS "Manifold library is built for static linking")
#endif ()
