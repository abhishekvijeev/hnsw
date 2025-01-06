#include "HNSWIndex.h"

#include <iostream>
#include <iterator>
#include <vector>

int main()
{
  hnsw::HNSWIndex h;
  std::vector<float> p0{1.0, 1.0};
  std::vector<float> p1{2.0, 2.0};
  std::vector<float> p2{4.0, 0.0};
  std::vector<float> p3{0.0, 10.0};
  std::vector<float> p4{3.0, 6.0};
  h.Insert(p0);
  h.Insert(p1);
  h.Insert(p2);
  h.Insert(p3);
  h.Insert(p4);

  std::vector<float> q{3, 3};
  std::vector<std::pair<int64_t, std::vector<float>>> knn = h.KNNSearch(q, 2);
  std::cout << "Nearest neighbours:" << std::endl;
  for (auto& point : knn) {
    std::copy(point.second.begin(), point.second.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;
  }
  std::cout << std::endl;
}