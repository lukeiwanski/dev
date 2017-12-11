#include "allocator.hpp"
#include "list.hpp"
#ifndef LTL_QUEUE
#define LTL_QUEUE
namespace ltl {

// Queue implementation that follows
// http://www.cplusplus.com/reference/queue/queue/
template <class Scalar, class Container = list<Scalar>> class queue {
public:
  using value_type = Scalar;
  using container_type = Container;
  using size_type = unsigned int;

  container_type data;

  explicit queue(const container_type &ctnr = container_type()) { data = ctnr; }

  // Returns: true if the container size is 0, false otherwise
  bool empty() const { return data.empty(); }

  // Returns: The number of elements in the container
  size_type size() const { return data.size(); }

  // Returns: A reference to the next element
  value_type &front() { return data.front(); }
  const value_type &front() const { return data.front(); }

  // Returns: A reference to the last element in the queue
  value_type &back() { return data.back(); }
  const value_type &back() const { return data.back(); }

  // Add element at the end
  // in: value to which the inserted element is initialized
  void push(const value_type &element) { data.push_back(element); }

  //
  void pop() { data.pop_front(); }
};

} // ltl
#endif // LTL_QUEUE
