
add_executable(Gates gates.cpp)
target_compile_features(Gates PUBLIC cxx_std_23)
#target_link_libraries(Gates PRIVATE Scions_options Scions_warnings)
target_link_libraries(Gates PUBLIC Scions::CPU MKL::MKL fmt::fmt-header-only)