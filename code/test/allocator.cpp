#include "allocator.hpp"

int main() {
  // constructors
  ltl::allocator<float> a;
  ltl::allocator<float> c();
  ltl::allocator<float> b(a);

  // allocate
  auto ptr = a.allocate(128);
  a.deallocate(ptr, 128);
  return 0;
}
