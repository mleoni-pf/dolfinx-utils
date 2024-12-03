set(FFCFLAGS CACHE INTERNAL "")

function(preprocess_ufl out_var)
  set(result)
  foreach(in_f ${ARGN})
      #    file(RELATIVE_PATH out_f ${CMAKE_CURRENT_SOURCE_DIR} ${in_f})
    string(REGEX REPLACE "\\.[^.]*$" "" in_f_noext ${in_f})
    set(out_f "${in_f_noext}.c")
    get_filename_component(in_f_name ${in_f} NAME)
    add_custom_command(OUTPUT ${out_f} ${in_f_noext}.h
      COMMAND ffcx ${FFCFLAGS} ${in_f_name} > /dev/null
      DEPENDS ${in_f}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Compiling py file to ${out_f}"
      VERBATIM
      )
    list(APPEND result ${out_f})
  endforeach()
  set(${out_var} "${result}" PARENT_SCOPE)
endfunction()

preprocess_ufl(ufl2cpp_files
    ${CMAKE_CURRENT_SOURCE_DIR}/area.py
    )

set(DOLFINX-UTILSSRC
           ${CMAKE_CURRENT_SOURCE_DIR}/DirichletBCDefinitions.h
           ${CMAKE_CURRENT_SOURCE_DIR}/TimeDependentFunction.h
           ${CMAKE_CURRENT_SOURCE_DIR}/TimeIntervals.cpp
           ${CMAKE_CURRENT_SOURCE_DIR}/TimeIntervals.h
           ${CMAKE_CURRENT_SOURCE_DIR}/assembleSolve.cpp
           ${CMAKE_CURRENT_SOURCE_DIR}/assembleSolve.h
           ${CMAKE_CURRENT_SOURCE_DIR}/computeArea.cpp
           ${CMAKE_CURRENT_SOURCE_DIR}/computeArea.h
           ${CMAKE_CURRENT_SOURCE_DIR}/computeHmin.cpp
           ${CMAKE_CURRENT_SOURCE_DIR}/computeHmin.h
           )

add_library(dolfinx-utils SHARED ${DOLFINX-UTILSSRC} ${ufl2cpp_files})

# Target libraries
target_link_libraries(dolfinx-utils dolfinx)



include(GNUInstallDirs)

install(FILES 
           ${CMAKE_CURRENT_SOURCE_DIR}/DirichletBCDefinitions.h
           ${CMAKE_CURRENT_SOURCE_DIR}/TimeDependentFunction.h
           ${CMAKE_CURRENT_SOURCE_DIR}/TimeIntervals.h
           ${CMAKE_CURRENT_SOURCE_DIR}/assembleSolve.h
           ${CMAKE_CURRENT_SOURCE_DIR}/computeArea.h
           ${CMAKE_CURRENT_SOURCE_DIR}/computeHmin.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dolfinx-utils/ COMPONENT Development)

# Set target include location (for build and installed)
target_include_directories(dolfinx-utils PUBLIC
                           $<INSTALL_INTERFACE:include>
                           "$<BUILD_INTERFACE:${DOLFINX-UTILS_SOURCE_DIR};${DOLFINX-UTILS_SOURCE_DIR}/dolfinx-utils>")

        
# Install library and header files
install(TARGETS dolfinx-utils
  EXPORT DOLFINX-UTILSTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development
  )

# Generate DOLFINX-UTILSTargets.cmake
install(EXPORT DOLFINX-UTILSTargets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dolfinx-utils)

# Generate CMake config files (DOLFINX-UTILSConfig{,Version}.cmake)
include(CMakePackageConfigHelpers)
#write_basic_package_version_file(${CMAKE_BINARY_DIR}/dolfinx-utils/DOLFINX-UTILSConfigVersion.cmake
  #VERSION ${DOLFINX_VERSION}
  #COMPATIBILITY ExactVersion)

configure_package_config_file(${DOLFINX-UTILS_SOURCE_DIR}/cmake/templates/DOLFINX-UTILSConfig.cmake.in
${CMAKE_BINARY_DIR}/dolfinx-utils/DOLFINX-UTILSConfig.cmake
INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dolfinx-utils)

# Install CMake helper files
install(
  FILES
  ${CMAKE_BINARY_DIR}/dolfinx-utils/DOLFINX-UTILSConfig.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dolfinx-utils
  COMPONENT Development)
