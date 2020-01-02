set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/dht)

set(FILE_LIST
   dht.cc
   pybind.cc
   trim_dht.cc
)

pybind11_add_module(dhtcxx ${FILE_LIST})

target_link_libraries(dhtcxx
   PRIVATE
   ${Boost_LIBRARIES}
   fmt-header-only
   ${FFTW_LIBRARIES}
)

target_include_directories(dhtcxx
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
