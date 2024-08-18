macro(Scions_configure_linker project_name)
  include(CheckCXXCompilerFlag)

  set(USER_LINKER_OPTION
      "mold"
      CACHE STRING "Linker to be used")
  set(USER_LINKER_OPTION_VALUES "lld" "gold" "bfd" "mold")
  set_property(CACHE USER_LINKER_OPTION PROPERTY STRINGS ${USER_LINKER_OPTION_VALUES})
  list(
    FIND
    USER_LINKER_OPTION_VALUES
    ${USER_LINKER_OPTION}
    USER_LINKER_OPTION_INDEX)

  if(${USER_LINKER_OPTION_INDEX} EQUAL -1)
    message(
      STATUS
        "Using custom linker: '${USER_LINKER_OPTION}', explicitly supported entries are ${USER_LINKER_OPTION_VALUES}")
  else()
    message(STATUS "Using ${USER_LINKER_OPTION} Linker")
  endif()

  if(NOT Scions_ENABLE_USER_LINKER)
    return()
  endif()
  if (USER_LINKER_OPTION MATCHES "mold")
    if (Scions_ENABLE_IPO AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      add_link_options("-Wl,--icf=safe,--color-diagnostics=always,--thinlto-jobs=all")
    else ()
      add_link_options("-Wl,--icf=safe,--color-diagnostics=always")
    endif ()
  endif()
  set(LINKER_FLAG "-fuse-ld=${USER_LINKER_OPTION}")

  check_cxx_compiler_flag(${LINKER_FLAG} CXX_SUPPORTS_USER_LINKER)
  if(CXX_SUPPORTS_USER_LINKER)
    add_link_options(${LINKER_FLAG})
  endif()
endmacro()
