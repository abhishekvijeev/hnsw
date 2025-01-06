include_directories(${PROJECT_SOURCE_DIR}/hnsw)

add_executable(example example.cc)
target_link_libraries(example hnsw)

add_executable(sift10k sift10k.cc)
target_link_libraries(sift10k hnsw)