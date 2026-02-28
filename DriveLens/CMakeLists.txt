# CMakeList.txt : CMake project for DriveLens, include source and define
# project specific logic here.
#

# Add source to this project's executable.
add_executable (DriveLens "DriveLens.cpp" "DriveLens.h")

# ── Link third-party libraries ───────────────────────────────────────
# OpenCV  – image capture & processing on the edge device
target_link_libraries(DriveLens PRIVATE ${OpenCV_LIBS})
target_include_directories(DriveLens PRIVATE ${OpenCV_INCLUDE_DIRS})

# cpr     – modern C++ HTTP client (wraps libcurl) for cloud upload
target_link_libraries(DriveLens PRIVATE cpr::cpr)

# nlohmann/json – parse server JSON responses
target_link_libraries(DriveLens PRIVATE nlohmann_json::nlohmann_json)


