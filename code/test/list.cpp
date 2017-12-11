#include "list.hpp"
int main() {
  // constructors
  ltl::list<float> a;

  float x = 5;
  a.push_back(x);

  if (a.size() < 1)
    return 1; // problem

  if (a.front() != a.back())
    return 2; // problem

  if (a.front() != x)
    return 3; // problem

  if (a.empty() != false)
    return 4; // problem

  a.pop_front();
  if (a.size() != 0)
    return 5;

  ltl::list<int> b(100, 20);

  for (auto element : b) {
    if (element != 20)
      return 6;
  }

  return 0;
}
