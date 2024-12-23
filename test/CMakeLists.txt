include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/f3c355f9dd382bc2c323be2713e351a578b68c61.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

include_directories(${PROJECT_SOURCE_DIR}/shinx)

add_executable(
  test_hello
  test_hello.cc
)

target_link_libraries(
  test_hello
  GTest::gtest_main
  shinx
)

include(GoogleTest)
gtest_discover_tests(test_hello)