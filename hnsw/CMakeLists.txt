set(HNSW_SRC
  HNSWIndex.cc
)

add_library(hnsw SHARED ${HNSW_SRC})

add_executable(runner runner.cc)
target_link_libraries(runner hnsw)

