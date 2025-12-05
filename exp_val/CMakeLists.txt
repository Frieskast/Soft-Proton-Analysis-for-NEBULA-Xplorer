cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(exp_val)

# --- Find Dependencies ---
find_package(Geant4 REQUIRED ui_all vis_all)
find_package(ROOT REQUIRED)

# --- Configure Project with Geant4 ---
include(${Geant4_USE_FILE})

# --- Add Include Directories ---
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${ROOT_INCLUDE_DIRS})

# --- Define Source Files ---
file(GLOB SOURCES "src/*.cc")

# --- Create the Executable ---
add_executable(exp_val ${SOURCES})

# --- Link Libraries ---
target_link_libraries(exp_val ${Geant4_LIBRARIES} ${ROOT_LIBRARIES})

# --- Copy Macro Files (Optional) ---
file(GLOB MACRO_FILES "${PROJECT_SOURCE_DIR}/*.mac")
if(MACRO_FILES)
    file(COPY ${MACRO_FILES} DESTINATION ${PROJECT_BINARY_DIR})
endif()
