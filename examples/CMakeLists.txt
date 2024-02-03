add_executable(Gates2 gates_v2.cpp)

target_compile_features(Gates2 PUBLIC cxx_std_23)
target_link_libraries(Gates2 PRIVATE Scions_options Scions_warnings)
#target_compile_options(Gates2 PUBLIC "-E")
target_link_libraries(Gates2 PUBLIC Manifold::Manifold)
target_precompile_headers(Gates2 INTERFACE <algorithm> <array> <vector> <string_view>)
