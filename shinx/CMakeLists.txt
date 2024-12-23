set(SHINX_SRC
  HNSWIndex.cc
)

add_library(shinx SHARED ${SHINX_SRC})

add_executable(runner runner.cc)
target_link_libraries(runner shinx)

