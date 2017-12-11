#include "allocator.hpp"
#include <stdio.h>
#ifndef LTL_LIST
#define LTL_LIST
namespace ltl {
// Deque implementation that follows
// http://www.cplusplus.com/reference/list/list/
template <class Scalar, class Alloc = allocator<Scalar>> class list {
public:
  using value_type = Scalar;
  using allocator_type = Alloc;
  using reference = typename allocator_type::reference;
  using const_reference = typename allocator_type::const_reference;
  using pointer = typename allocator_type::pointer;
  using const_pointer = typename allocator_type::const_pointer;
  // using const_iterator = const_pointer;
  // reverse_iterator
  // const_reverse_iterator
  // difference_type
  using size_type = unsigned int;

  // http://www.cplusplus.com/reference/list/list/list/
  explicit list(const allocator_type &alloc = allocator_type()) {}
  explicit list(size_type n, const value_type &val = value_type(),
                const allocator_type &alloc = allocator_type()) {
    for (size_type it = 0; it < n; it++) {
      push_back(val);
    }
  }

/// TODO: why is that overriding above signature?
#if 0
template <class InputIterator> list(InputIterator first, InputIterator last, const allocator_type& alloc = allocator_type()){
	}
#endif

  list(const list &x) {}

  ~list() {
    node *tmp;
    while (head) {
      tmp = head;
      head = head->next;
      delete tmp;
    }
  }

  // http://www.cplusplus.com/reference/list/list/operator=/
  list &operator=(const list &x) {}

  bool empty() const { return head == nullptr; }

  size_type size() const { return elements; }

  reference front() { return head->data; }

  const_reference front() const { return head->data(); }

  reference back() { return tail->data; }

  const_reference back() const { return tail->data; }

  void push_back(const value_type &val) {
    node *newNode = new node(val, nullptr, tail);
    if (head == nullptr)
      head = newNode;
    if (tail != nullptr)
      tail->next = newNode;
    tail = newNode;
    ++elements;
  }

  void pop_front() {
    node *tmp = head;
    head = head->next;
    if (head != nullptr)
      head->prev = nullptr;
    --elements;
    delete tmp;
  }

private:
  struct node {
    value_type data;
    node *next, *prev;
    node(const_reference data, node *next, node *prev)
        : data(data), next(next), prev(prev) {}
  };

  class iterator {
    friend class list;
    iterator(node *n) : m_node(n) {}

  public:
    iterator() : m_node(0) {}

    iterator &operator++() {
      m_node = m_node->next;
      return *this;
    }

    reference operator*() { return static_cast<node *>(m_node)->data; }

    bool operator!=(const iterator &rhs) { return m_node != rhs.m_node; }

  private:
    node *m_node;
  };

public:
  iterator begin() { return iterator(head); }
  const iterator begin() const { return const_iterator(head); }

  iterator end() { return iterator(tail); }
  const iterator end() const { return iterator(tail); }

private:
  size_type elements = 0;
  node *head = nullptr;
  node *tail = nullptr;
};
} // ltl
#endif // LTL_LIST
