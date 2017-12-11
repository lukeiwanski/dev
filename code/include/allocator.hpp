#include "limits.hpp"
#include "util.hpp"
#ifndef LTL_ALLOCATOR
#define LTL_ALLOCATOR
namespace ltl {
// Forward declaration
template <class Scalar> class allocator;
template <> class allocator<void> {
public:
  typedef void *pointer;
  typedef const void *const_pointer;
  typedef void value_type;
  template <class U> struct rebind { typedef allocator<U> other; };
};

// Allocator implementation that follows
// http://www.cplusplus.com/reference/memory/allocator/
template <class Scalar> class allocator {
public:
  using value_type = Scalar;
  using pointer = Scalar *;
  using reference = Scalar &;
  using const_pointer = const Scalar *;
  using const_reference = const Scalar &;
  using size_type = unsigned long long;
  using difference_type = long long;

  template <class Type> struct rebund { typedef allocator<Type> other; };

  allocator() noexcept {}
  allocator(const allocator &alloc) noexcept {}
  template <class U> allocator(const allocator<U> &alloc) noexcept {}

  ~allocator() {} // throw {}

  pointer address(reference x) { return &x; }
  const_pointer address(const_reference x) const { return &x; }

  pointer allocate(size_type n, allocator<void>::const_pointer hint = 0) {
    return static_cast<pointer>(operator new(n * sizeof(value_type)));
  }

  void deallocate(pointer p, size_type n) { delete (p); }

  size_type max_size() const noexcept {
    numeric_limits<size_type>::max() / sizeof(value_type);
  }

  template <class U, class... Args> void construct(U *p, Args &&... args) {
    new ((void *)p) U(forward<Args>(args)...);
  }

  template <class U> void destroy(U *p) { p->~U(); }
};

template <class T, class U>
constexpr bool operator==(const allocator<T> &, const allocator<U> &) noexcept;

template <class T, class U>
constexpr bool operator!=(const allocator<T> &, const allocator<U> &) noexcept;

} // ltl
#endif // LTL_ALLOCATOR
