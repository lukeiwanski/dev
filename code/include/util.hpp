#ifndef LTL_UTILS
#define LTL_UTILS
namespace ltl {
template <typename T> struct identity { typedef T type; };

template <typename T> T &&forward(typename identity<T>::type &&param) {
  return static_cast<typename identity<T>::type &&>(param);
}
} // ltl
#endif // LTL_UTILS
