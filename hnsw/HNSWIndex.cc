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

void HNSWIndex::Insert(const std::vector<float>& q_embedding) {
  std::cout << "HNSWIndex::Insert(";
  std::copy(q_embedding.begin(), q_embedding.end(), std::ostream_iterator<float>(std::cout, ","));
  std::cout << ")" << std::endl;

  int64_t point_id = points.size();
  // int64_t point_level = GenerateLevel();

  int64_t point_level;

  switch(point_id) {
    case 0: point_level = 0; break;
    case 1: point_level = 0; break;
    case 2: point_level = 0; break;
    case 3: point_level = 1; break;
    case 4: point_level = 1; break;
  }

  Point q{point_id, point_level, q_embedding};
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

  std::cout << "Point ID: " << point_id << "\n";
  std::cout << "Point level: " << point_level << "\n";
  std::cout << "L: " << L << "\n";

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

    std::cout << "Node " << point_id << "'s neighbour candidates at level " << lc << ": ";
    std::copy(W.begin(), W.end(), std::ostream_iterator<float>(std::cout, ","));
    std::cout << std::endl;

    std::vector<int64_t> neighbours = SelectNeighbours(q, W, M, lc);

    std::cout << "Node " << point_id << "'s neighbours at level " << lc << ": ";
    std::copy(neighbours.begin(), neighbours.end(), std::ostream_iterator<float>(std::cout, ","));
    std::cout << std::endl;

    /// Add bidirectional connections between 'q' and all nodes in
    /// 'neighbours' at layer 'lc'
    graph[lc][point_id] = neighbours;
    for (int64_t neighbour : neighbours) {
      graph[lc][neighbour].push_back(point_id);
    }

    /// Now that additional edges have been added to the nodes in
    /// 'neighbours', we need to ensure that the total number of edges
    /// for each node in 'neighbours' doesn't exceed 'MMax'
    for (int64_t e : neighbours) {
      const std::vector<int64_t>& e_conn = graph[lc][e];
      int64_t max_edges = (lc == 0 ? MMax0 : MMax);
      if (e_conn.size() > max_edges) {
        std::vector<int64_t> e_new_conn = SelectNeighbours(points[e], e_conn, max_edges, lc);
        graph[lc][e] = e_new_conn;
      }
    }
    ep = W[W.size() - 1]; // W[0] contains the farthest node, W[W.size() - 1] contains the nearest node
  }

  if (point_level > L) {
    L = point_level;
    entry_point = point_id;
  }

  std::cout << std::endl << std::endl;
}

std::vector<int64_t> HNSWIndex::SearchLayer(
  const HNSWIndex::Point& q,
  int64_t ep,
  int64_t ef,
  int64_t lc) {

  std::cout << "SearchLayer " << lc << ", ep: " << ep << "\n";
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

  std::cout << "SelectNeighbours level: " << lc << "\n";
  std::vector<int64_t> R;
  std::priority_queue<HNSWIndex::PointDistCloser> W;
  for (int64_t c : C) {
    W.push({c, points[c].distance(q)});
  }

  while ((W.size() > 0) && (R.size() < M)) {
    int64_t e = W.top().id; W.pop();
    if (R.size() == 0) {
      std::cout << "\tchoosing closest e: ";
      std::copy(points[e].embedding.begin(), points[e].embedding.end(), std::ostream_iterator<float>(std::cout, ","));
      std::cout << std::endl << std::endl;
      R.push_back(e);
    }
    else {
      std::cout << "\tnext e: ";
      std::copy(points[e].embedding.begin(), points[e].embedding.end(), std::ostream_iterator<float>(std::cout, ","));
      // If 'e' is closer to any 'r' than it is to 'q', prune 'e'
      bool prune_e = false;
      for (int64_t r : R) {
        std::cout << ", r: ";
        std::copy(points[r].embedding.begin(), points[r].embedding.end(), std::ostream_iterator<float>(std::cout, ","));
        std::cout << ", q: ";
        std::copy(q.embedding.begin(), q.embedding.end(), std::ostream_iterator<float>(std::cout, ","));
        std::cout << std::endl;
        std::cout << "\tdist(e, r): " << points[e].distance(points[r]) << std::endl;
        std::cout << "\tdist(e, q): " << points[e].distance(q) << std::endl;
        if (points[e].distance(points[r]) < points[e].distance(q)) {
          prune_e = true;
          break;
        }
      }

      if (!prune_e) {
        std::cout << "\tchoosing ";
        std::copy(points[e].embedding.begin(), points[e].embedding.end(), std::ostream_iterator<float>(std::cout, ","));
        std::cout << std::endl;
        R.push_back(e);
      }
      else {
        std::cout << "\tpruning ";
        std::copy(points[e].embedding.begin(), points[e].embedding.end(), std::ostream_iterator<float>(std::cout, ","));
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
  }
  return R;
}

std::vector<std::vector<float>> HNSWIndex::KNNSearch(
  const std::vector<float>& q_embedding,
  int64_t K,
  int64_t ef) {

  Point q{-1, -1, q_embedding};
  std::vector<int64_t> W;
  int64_t ep = entry_point;
  for (int64_t lc = L; lc >= 1; lc--) {
    W = SearchLayer(q, ep, 1, lc);
    ep = W[W.size() - 1];
  }
  W = SearchLayer(q, ep, ef, 0);

  std::vector<std::vector<float>> ret;
  for (int64_t i = 0; i < K && i < W.size(); i++) {
    int64_t neighbour_id = W[W.size() - 1 - i];
    std::vector<float> neighbour_embedding = points[neighbour_id].embedding;
    ret.push_back(neighbour_embedding);

  }
  return ret;
}