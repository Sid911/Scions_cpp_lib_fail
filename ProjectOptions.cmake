include(cmake/SystemLink.cmake)
include(cmake/LibFuzzer.cmake)
include(CMakeDependentOption)
include(CheckCXXCompilerFlag)


macro(Scions_supports_sanitizers)
  if((CMAKE_CXX_COMPILER_ID MATCHES ".*Clang.*" OR CMAKE_CXX_COMPILER_ID MATCHES ".*GNU.*") AND NOT WIN32)
    set(SUPPORTS_UBSAN OFF) # Turn it off for now
  else()
    set(SUPPORTS_UBSAN OFF)
  endif()

  if((CMAKE_CXX_COMPILER_ID MATCHES ".*Clang.*" OR CMAKE_CXX_COMPILER_ID MATCHES ".*GNU.*") AND WIN32)
    set(SUPPORTS_ASAN OFF)
  else()
    set(SUPPORTS_ASAN OFF) # Turn it off
  endif()
endmacro()


macro(Scions_setup_options)
  option(Scions_ENABLE_HARDENING "Enable hardening" OFF)
  option(Scions_ENABLE_COVERAGE "Enable coverage reporting" OFF)
  option(Scions_BUILD_EXAMPLE "Enable building of examples" ON)
  option(Scions_TRACE_TIME_CLANG "Enable Clang -ftime-trace feature" OFF)
  cmake_dependent_option(
    Scions_ENABLE_GLOBAL_HARDENING
    "Attempt to push hardening options to built dependencies"
    ON
    Scions_ENABLE_HARDENING
    OFF)

  Scions_supports_sanitizers()

  if(NOT PROJECT_IS_TOP_LEVEL OR Scions_PACKAGING_MAINTAINER_MODE)
    option(Scions_ENABLE_IPO "Enable IPO/LTO" OFF)
    option(Scions_WARNINGS_AS_ERRORS "Treat Warnings As Errors" OFF)
    option(Scions_ENABLE_USER_LINKER "Enable user-selected linker" ON)
    option(Scions_ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" OFF)
    option(Scions_ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
    option(Scions_ENABLE_SANITIZER_UNDEFINED "Enable undefined sanitizer" OFF)
    option(Scions_ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
    option(Scions_ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)
    option(Scions_ENABLE_UNITY_BUILD "Enable unity builds" OFF)
    option(Scions_ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
    option(Scions_ENABLE_CPPCHECK "Enable cpp-check analysis" OFF)
    option(Scions_ENABLE_PCH "Enable precompiled headers" OFF)
    option(Scions_ENABLE_CACHE "Enable ccache" ON)
  else()
    option(Scions_ENABLE_IPO "Enable IPO/LTO" ON)
    option(Scions_WARNINGS_AS_ERRORS "Treat Warnings As Errors" OFF)
    option(Scions_ENABLE_USER_LINKER "Enable user-selected linker" ON)
    option(Scions_ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" ${SUPPORTS_ASAN})
    option(Scions_ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
    option(Scions_ENABLE_SANITIZER_UNDEFINED "Enable undefined sanitizer" ${SUPPORTS_UBSAN})
    option(Scions_ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
    option(Scions_ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)
    option(Scions_ENABLE_UNITY_BUILD "Enable unity builds" OFF)
    option(Scions_ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
    option(Scions_ENABLE_CPPCHECK "Enable cpp-check analysis" OFF)
    option(Scions_ENABLE_PCH "Enable precompiled headers" ON)
    option(Scions_ENABLE_CACHE "Enable ccache" ON)
  endif()

  if(NOT PROJECT_IS_TOP_LEVEL)
    mark_as_advanced(
      Scions_ENABLE_IPO
      Scions_WARNINGS_AS_ERRORS
      Scions_ENABLE_USER_LINKER
      Scions_ENABLE_SANITIZER_ADDRESS
      Scions_ENABLE_SANITIZER_LEAK
      Scions_ENABLE_SANITIZER_UNDEFINED
      Scions_ENABLE_SANITIZER_THREAD
      Scions_ENABLE_SANITIZER_MEMORY
      Scions_ENABLE_UNITY_BUILD
      Scions_ENABLE_CLANG_TIDY
      Scions_ENABLE_CPPCHECK
      Scions_ENABLE_COVERAGE
      Scions_ENABLE_PCH
      Scions_ENABLE_CACHE)
  endif()

  Scions_check_libfuzzer_support(LIBFUZZER_SUPPORTED)
  if(LIBFUZZER_SUPPORTED AND (Scions_ENABLE_SANITIZER_ADDRESS OR Scions_ENABLE_SANITIZER_THREAD OR Scions_ENABLE_SANITIZER_UNDEFINED))
    set(DEFAULT_FUZZER ON)
  else()
    set(DEFAULT_FUZZER OFF)
  endif()

  option(Scions_BUILD_FUZZ_TESTS "Enable fuzz testing executable" ${DEFAULT_FUZZER})

endmacro()

macro(Scions_global_options)
  if(Scions_ENABLE_IPO)
    include(cmake/InterproceduralOptimization.cmake)
    Scions_enable_ipo()
  endif()

  if (Scions_TRACE_TIME_CLANG)
    if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftime-trace")
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftime-trace")
    endif ()
  endif ()
  Scions_supports_sanitizers()

  if(Scions_ENABLE_HARDENING AND Scions_ENABLE_GLOBAL_HARDENING)
    include(cmake/Hardening.cmake)
    if(NOT SUPPORTS_UBSAN 
       OR Scions_ENABLE_SANITIZER_UNDEFINED
       OR Scions_ENABLE_SANITIZER_ADDRESS
       OR Scions_ENABLE_SANITIZER_THREAD
       OR Scions_ENABLE_SANITIZER_LEAK)
      set(ENABLE_UBSAN_MINIMAL_RUNTIME FALSE)
    else()
      set(ENABLE_UBSAN_MINIMAL_RUNTIME TRUE)
    endif()
    message("${Scions_ENABLE_HARDENING} ${ENABLE_UBSAN_MINIMAL_RUNTIME} ${Scions_ENABLE_SANITIZER_UNDEFINED}")
    Scions_enable_hardening(Scions_options ON ${ENABLE_UBSAN_MINIMAL_RUNTIME})
  endif()
endmacro()

macro(Scions_local_options)
  if(PROJECT_IS_TOP_LEVEL)
    include(cmake/StandardProjectSettings.cmake)
  endif()

  add_library(Scions_warnings INTERFACE)
  add_library(Scions_options INTERFACE)

  include(cmake/CompilerWarnings.cmake)
  Scions_set_project_warnings(
    Scions_warnings
    ${Scions_WARNINGS_AS_ERRORS}
    ""
    ""
    ""
    "")

  if(Scions_ENABLE_USER_LINKER)
    include(cmake/Linker.cmake)
    Scions_configure_linker(Scions_options)
  endif()

  include(cmake/Sanitizers.cmake)
  Scions_enable_sanitizers(
    Scions_options
    ${Scions_ENABLE_SANITIZER_ADDRESS}
    ${Scions_ENABLE_SANITIZER_LEAK}
    ${Scions_ENABLE_SANITIZER_UNDEFINED}
    ${Scions_ENABLE_SANITIZER_THREAD}
    ${Scions_ENABLE_SANITIZER_MEMORY})

  set_target_properties(Scions_options PROPERTIES UNITY_BUILD ${Scions_ENABLE_UNITY_BUILD})

  if(Scions_ENABLE_PCH)
    target_precompile_headers(
      Scions_options
      INTERFACE <vector> <string> <string_view> <array> <utility> <algorithm>
    )
  endif()

  if(Scions_ENABLE_CACHE)
    include(cmake/Cache.cmake)
    Scions_enable_cache()
  endif()

  include(cmake/StaticAnalyzers.cmake)
  if(Scions_ENABLE_CLANG_TIDY)
    Scions_enable_clang_tidy(Scions_options ${Scions_WARNINGS_AS_ERRORS})
  endif()

  if(Scions_ENABLE_CPPCHECK)
    Scions_enable_cppcheck(${Scions_WARNINGS_AS_ERRORS} "" # override cppcheck options
    )
  endif()

  if(Scions_ENABLE_COVERAGE)
    include(cmake/Tests.cmake)
    Scions_enable_coverage(Scions_options)
  endif()

  if(Scions_WARNINGS_AS_ERRORS)
    check_cxx_compiler_flag("-Wl,--fatal-warnings" LINKER_FATAL_WARNINGS)
    if(LINKER_FATAL_WARNINGS)
      # This is not working consistently, so disabling for now
      # target_link_options(Scions_options INTERFACE -Wl,--fatal-warnings)
    endif()
  endif()

  if(Scions_ENABLE_HARDENING AND NOT Scions_ENABLE_GLOBAL_HARDENING)
    include(cmake/Hardening.cmake)
    if(NOT SUPPORTS_UBSAN 
       OR Scions_ENABLE_SANITIZER_UNDEFINED
       OR Scions_ENABLE_SANITIZER_ADDRESS
       OR Scions_ENABLE_SANITIZER_THREAD
       OR Scions_ENABLE_SANITIZER_LEAK)
      set(ENABLE_UBSAN_MINIMAL_RUNTIME FALSE)
    else()
      set(ENABLE_UBSAN_MINIMAL_RUNTIME TRUE)
    endif()
    Scions_enable_hardening(Scions_options OFF ${ENABLE_UBSAN_MINIMAL_RUNTIME})
  endif()

endmacro()
