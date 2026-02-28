# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.10)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

# Honor <PackageName>_ROOT variables in find_package().
if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

project ("DriveLens" LANGUAGES CXX)

# ── C++20 standard (global) ──────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Find third-party packages (installed via vcpkg) ──────────────────
find_package(OpenCV REQUIRED)
find_package(cpr REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# Include sub-projects.
add_subdirectory ("DriveLens")
