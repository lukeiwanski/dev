include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(allocator allocator.cpp)
add_test(TestAllocator allocator)

add_executable(list list.cpp)
add_test(TestList list)

add_executable(queue queue.cpp)
add_test(TestQueue queue)
