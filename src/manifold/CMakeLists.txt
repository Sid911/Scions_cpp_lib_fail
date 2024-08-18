#include(GenerateExportHeader)

add_library(Manifold INTERFACE ${PROJECT_SOURCE_DIR}/include/manifold)

add_library(Manifold::Manifold ALIAS Manifold)

target_link_libraries(Manifold INTERFACE Scions_options Scions_warnings)
#target_link_libraries(Manifold INTERFACE fmt::fmt-header-only)

target_include_directories(Manifold ${WARNING_GUARD} INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/>
                                                               $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

target_precompile_headers(Manifold INTERFACE ${PROJECT_SOURCE_DIR}/include/manifold/common.hpp)

target_compile_features(Manifold INTERFACE cxx_std_23)

set_target_properties(Manifold PROPERTIES VERSION ${PROJECT_VERSION})

#generate_export_header(Manifold EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/Scions/cpu_export.hpp)

#if (NOT BUILD_SHARED_LIBS)
#    target_compile_definitions(Manifold PUBLIC SCIONS_LIBRARY_STATIC_DEFINE)
#    message(STATUS "Manifold library is built for static linking")
#endif ()
