set(HNSW_SRC
  HNSWIndex.cc
)

set(HNSW_HEADERS
  HNSWIndex.h
)

add_library(hnsw SHARED ${HNSW_SRC})

# foreach(header ${HNSW_HEADERS})
#   get_filename_component(dir ${header} DIRECTORY )
#   message(dir="${dir}")
#   install(FILES ${header}
#     DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hnsw/${dir}
#   )
# endforeach()

# install(TARGETS hnsw
#   EXPORT hnsw-targets
#   RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
#   ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#   LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#   INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
# )

# set_target_properties(hnsw PROPERTIES PUBLIC_HEADER "HNSWIndex.h")

# install(TARGETS hnsw
#   LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#   PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
# )

