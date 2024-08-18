add_executable(Gates3 gates_v3.cpp)

target_compile_features(Gates3 PUBLIC cxx_std_23)
target_link_libraries(Gates3 PRIVATE Scions_options Scions_warnings)
#target_compile_options(Gates3 PUBLIC "-E")
target_link_libraries(Gates3 PUBLIC Manifold::Manifold flux::flux)
target_precompile_headers(
    Gates3
    INTERFACE
    <algorithm>
    <array>
    <vector>
    <string_view>
    <vector>)
