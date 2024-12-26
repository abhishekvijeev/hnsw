#include "HNSWIndex.h"

#include <cmath>
#include <iostream>
#include <iterator>
#include <queue>
#include <random>
#include <unordered_set>
#include <vector>

using namespace hnsw;

void HNSWIndex::PrintParameters() {
  std::cout << std::endl;
  std::cout << "HNSW Parameters:" << std::endl << std::endl;
  std::cout << "entry_point: " << entry_point << std::endl;
  std::cout << "L: " << L << std::endl;
  std::cout << "ef_construction: " << ef_construction << std::endl;
  std::cout << "M: " << M << std::endl;
  std::cout << "MMax: " << MMax << std::endl;
  std::cout << "MMax0: " << MMax0 << std::endl;
  std::cout << "mL: " << mL << std::endl;
  std::cout << std::endl << std::endl;
}

uint64_t HNSWIndex::GenerateLevel() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return std::floor(-1 * log(dist(gen)) * mL);
}

void HNSWIndex::Insert(const std::vector<float>& embedding) {
  std::cout << "HNSWIndex::Insert(";
  std::copy(embedding.begin(), embedding.end(), std::ostream_iterator<float>(std::cout, ","));
  std::cout << ")" << std::endl;

  uint64_t point_id = points.size();
  uint64_t point_level = GenerateLevel();
  Point p{point_id, point_level, embedding};
  points.push_back(p);

  /// If necessary, add more levels to the graph
  while (graph.size() <= point_level) {
    graph.push_back(std::vector<std::vector<uint64_t>>());
  }

  /// Phase 1
  ///
  /// Traverse the graph starting from the top layer to find the
  /// closest neighbor (at that layer) to the point being inserted
  ///
  /// When Phase 1 completes, 'ep' contains the entry point to the
  /// highest layer that will hold this new point
  uint64_t ep = entry_point;
  for (uint64_t lc = L; lc > point_level; lc--) {
    std::priority_queue<HNSWIndex::PointDistFarther> W = SearchLayer(p, ep, 1, lc);
    ep = W.top().id;
  }

  std::cout << "ep: " << ep << std::endl;
}

std::priority_queue<HNSWIndex::PointDistFarther> HNSWIndex::SearchLayer(
  const HNSWIndex::Point& q,
  uint64_t ep,
  uint64_t ef,
  uint64_t lc) {

  std::cout << "SearchLayer\n";
  std::unordered_set<uint64_t> v;
  std::priority_queue<HNSWIndex::PointDistCloser> C;
  std::priority_queue<HNSWIndex::PointDistFarther> W;

  v.insert(ep);
  C.push({ep, points[ep].distance(q)});
  W.push({ep, points[ep].distance(q)});

  while (C.size() > 0) {
    uint64_t c = C.top().id; C.pop();
    uint64_t f = W.top().id;
    if (points[c].distance(q) > points[f].distance(q))
      break;
    for (uint64_t e : graph[lc][c]) {
      if (v.find(e) == v.end()) {
        v.insert(e);
        f = W.top().id;
        if ((points[e].distance(q) < points[f].distance(q)) || (W.size() < ef))  {
          C.push({e, points[e].distance(q)});
          W.push({e, points[e].distance(q)});
          if (W.size() > ef) {
            W.pop();
          }
        }
      }
    }
  }
  return W;
}