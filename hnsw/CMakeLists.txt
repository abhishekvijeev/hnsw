set(HNSW_SRC
  HNSWIndex.cc
)

add_library(hnsw SHARED ${HNSW_SRC})

