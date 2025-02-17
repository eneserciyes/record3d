cmake_minimum_required(VERSION 3.13)
project(record3d)
set(CMAKE_CXX_STANDARD 14)

if (APPLE)
    set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE INTERNAL "" FORCE)
endif()

if(UNIX AND NOT APPLE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# Add libusbmuxd
if(WIN32)
    set(libusbmuxd_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs/win/libusbmuxd)
else()
    set(libusbmuxd_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs/unix/libusbmuxd)
endif()
add_subdirectory(${libusbmuxd_DIR})
include_directories(${libusbmuxd_DIR}/include)



########################################################################
### Create C++ library
########################################################################
include(GNUInstallDirs)
add_library(record3d_cpp STATIC src/Record3DStream.cpp )
include_directories(include)
target_link_libraries(record3d_cpp PUBLIC usbmuxd)
install(TARGETS record3d_cpp usbmuxd plist EXPORT Record3DConfig
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT Record3DConfig DESTINATION share/Record3D/cmake)


########################################################################
### Create Python module (using pybind11)
########################################################################
if(BUILD_PYTHON_BINDINGS)
    include(python-bindings/pybind11/tools/FindPythonLibsNew.cmake)
    include_directories(python-bindings/pybind11/include)

    include_directories(${PYTHON_INCLUDE_DIRS})
    add_library(record3d_py STATIC src/Record3DStream.cpp)
    target_compile_definitions(record3d_py PRIVATE PYTHON_BINDINGS_BUILD)
    target_link_libraries(record3d_py usbmuxd)

    add_subdirectory(python-bindings/pybind11)
    pybind11_add_module(record3d python-bindings/src/PythonBindings.cpp)
    target_compile_definitions(record3d PRIVATE PYTHON_BINDINGS_BUILD)
    target_link_libraries(record3d PRIVATE record3d_py)
endif()
