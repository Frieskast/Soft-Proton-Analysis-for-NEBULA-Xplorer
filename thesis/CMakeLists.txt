cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(Simulation)

set(CMAKE_BUILD_TYPE Debug)

find_package(Geant4 REQUIRED ui_all vis_all)
find_package(ROOT REQUIRED)

include(${Geant4_USE_FILE})
include_directories(${ROOT_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
link_directories(${ROOT_LIBRARY_DIRS})
add_definitions(${ROOT_CXX_FLAGS})

# Gather all source and header files from src/ and include/
file(GLOB sources
    ${PROJECT_SOURCE_DIR}/src/*.cc
)
file(GLOB headers
    ${PROJECT_SOURCE_DIR}/include/*.hh
)

# Copy macro and data files if needed
file(GLOB MACRO_FILES "*.mac")
file(COPY ${MACRO_FILES} DESTINATION ${PROJECT_BINARY_DIR})

# file(GLOB DATA_FILES "*.dat")
# file(COPY ${DATA_FILES} DESTINATION ${PROJECT_BINARY_DIR})

add_executable(thesis
    ${sources}
)

target_link_libraries(thesis ${Geant4_LIBRARIES} ${ROOT_LIBRARIES})

add_custom_target(Simulation DEPENDS thesis)
