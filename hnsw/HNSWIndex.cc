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

int64_t HNSWIndex::GenerateLevel() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return std::floor(-1 * log(dist(gen)) * mL);
}

void HNSWIndex::Insert(const std::vector<float>& embedding) {
  std::cout << "HNSWIndex::Insert(";
  std::copy(embedding.begin(), embedding.end(), std::ostream_iterator<float>(std::cout, ","));
  std::cout << ")" << std::endl;

  int64_t point_id = points.size();
  int64_t point_level = GenerateLevel();
  Point q{point_id, point_level, embedding};
  points.push_back(q);

  /// If necessary, add more levels to the graph
  while (graph.size() <= point_level) {
    graph.push_back(std::unordered_map<int64_t, std::vector<int64_t>>());
  }

  // First point
  if (points.size() == 1) {
    for (int64_t lc = point_level; lc >= 0; lc--) {
      graph[lc][point_id] = std::vector<int64_t>();
    }
    entry_point = point_id;
    L = point_level;

    std::cout << "First point\n";
    std::cout << "Point ID: " << point_id << "\n";
    std::cout << "Point level: " << point_level << "\n";
    std::cout << "entry_point: " << entry_point << "\n";
    std::cout << "L: " << L << "\n";
    std::cout << std::endl;
    return;
  }

  /// Phase 1
  ///
  /// Traverse the graph starting from the top layer to find the
  /// closest neighbor (at that layer) to the point being inserted
  ///
  /// When Phase 1 completes, 'ep' contains the entry point to the
  /// highest layer that will hold this new point
  int64_t ep = entry_point;
  for (int64_t lc = L; lc > point_level; lc--) {
    std::vector<int64_t> W = SearchLayer(q, ep, 1, lc);
    ep = W[0];
  }

  std::cout << "Phase 1 complete\n";
  std::cout << "ep: " << ep << std::endl;

  for (int64_t lc = std::min(L, point_level); lc >= 0; lc--) {
    std::vector<int64_t> W = SearchLayer(q, ep, ef_construction, lc);
    std::vector<int64_t> neighbours = SelectNeighbours(q, W, M, lc);
  }

  std::cout << std::endl;
}

std::vector<int64_t> HNSWIndex::SearchLayer(
  const HNSWIndex::Point& q,
  int64_t ep,
  int64_t ef,
  int64_t lc) {

  std::cout << "SearchLayer\n";
  std::unordered_set<int64_t> v;
  std::priority_queue<HNSWIndex::PointDistCloser> C;
  std::priority_queue<HNSWIndex::PointDistFarther> W;

  v.insert(ep);
  C.push({ep, points[ep].distance(q)});
  W.push({ep, points[ep].distance(q)});

  while (C.size() > 0) {
    int64_t c = C.top().id; C.pop();
    int64_t f = W.top().id;
    if (points[c].distance(q) > points[f].distance(q))
      break;
    for (int64_t e : graph[lc][c]) {
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

  std::vector<int64_t> W_vec;
  while (!W.empty()) {
      W_vec.push_back(W.top().id);
      W.pop();
  }
  return W_vec;
}

std::vector<int64_t> HNSWIndex::SelectNeighbours(
  const HNSWIndex::Point& q,
  const std::vector<int64_t>& C,
  int64_t M,
  int64_t lc) {

  std::vector<int64_t> R;
  std::priority_queue<HNSWIndex::PointDistCloser> W;
  for (int64_t c : C) {
    W.push({c, points[c].distance(q)});
  }

  while ((W.size() > 0) && (R.size() < M)) {
    int64_t e = W.top().id; W.pop();
    if (R.size() == 0) {
      R.push_back(e);
    }
    else {
      // If 'e' is closer to any 'r' than it is to 'q', prune 'e'
      bool prune_e = false;
      for (int64_t r : R) {
        if (points[e].distance(points[r]) < points[e].distance(q)) {
          prune_e = true;
          break;
        }
      }

      if (!prune_e) {
        R.push_back(e);
      }
    }
  }
  return R;
}