#include "queue.hpp"

int main() {
  // constructors
  ltl::queue<float> a;

  float x = 5;
  a.push(x);

  if (a.size() < 1)
    return 1; // problem

  if (a.front() != a.back())
    return 2; // problem

  if (a.front() != x)
    return 3; // problem

  if (a.empty() != false)
    return 4; // problem

  a.pop();
  return a.size(); // should be empty
}
