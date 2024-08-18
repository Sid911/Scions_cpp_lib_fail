include(cmake/CPM.cmake)

# Done as a function so that updates to variables like
# CMAKE_CXX_FLAGS don't propagate out to other
# targets
function(Scions_setup_dependencies)

  # For each dependency, see if it's
  # already been provided to us by a parent project

  # if(NOT TARGET fmtlib::fmtlib)
  #   cpmaddpackage("gh:fmtlib/fmt#10.1.1")
  # endif()


  if(NOT TARGET Catch2::Catch2WithMain)
    cpmaddpackage("gh:catchorg/Catch2@3.3.2")
  endif()

#  if(NOT TARGET tools::tools)
#    cpmaddpackage("gh:lefticus/tools#update_build_system")
#  endif()

  include(FetchContent)

  FetchContent_Declare(
      flux
      GIT_REPOSITORY https://github.com/tcbrindle/flux.git
      GIT_TAG main # Replace with a git commit id to fix a particular revision
  )

  FetchContent_MakeAvailable(flux)

endfunction()
