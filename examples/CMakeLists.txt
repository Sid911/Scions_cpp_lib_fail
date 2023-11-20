
add_executable(Gates gates.cpp)
add_executable(Gates2 gates_v2.cpp)

target_compile_features(Gates PUBLIC cxx_std_23)
#target_link_libraries(Gates PRIVATE Scions_options Scions_warnings)
target_link_libraries(Gates PUBLIC Scions::CPU MKL::MKL fmt::fmt-header-only)

target_compile_features(Gates2 PUBLIC cxx_std_23)
#target_link_libraries(Gates2 PRIVATE Scions_options Scions_warnings)
target_link_libraries(Gates2 PUBLIC Scions::CPU MKL::MKL fmt::fmt-header-only)
