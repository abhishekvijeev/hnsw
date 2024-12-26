#ifndef HNSW_HNSW_INDEX_H_
#define HNSW_HNSW_INDEX_H_

#include <cmath>
#include <cstdint>
#include <memory>
#include <queue>
#include <unordered_map>
#include <vector>

namespace hnsw {

// Implementation of the Hierarchical Navigable Small World (HNSW)
// algorithm based on:
// 
// Malkov, Yu A., and Dmitry A. Yashunin. "Efficient and robust
// approximate nearest neighbor search using hierarchical navigable
// small world graphs." IEEE transactions on pattern analysis and
// machine intelligence 42.4 (2018): 824-836.
struct HNSWIndex {

  /// @brief: A point in multi-dimensional space
  struct Point {
    int64_t id;                  // an internal ID that's unique to each point in the index
    int64_t level;               // the highest level containing this point
    std::vector<float> embedding; // the point's vector embedding

    float distance(const Point& other) const
    {
      float dist = 0.0;
      for (size_t i = 0; i < embedding.size(); i++) {
          dist += (embedding[i] - other.embedding[i]) * (embedding[i] - other.embedding[i]);
      }
      return std::sqrt(dist);
    }
  };

  struct PointDistCloser {
    int64_t id;
    float dist;

    bool operator<(const PointDistCloser& other) const {
      return dist > other.dist;
    }
  };

  struct PointDistFarther {
    int64_t id;
    float dist;

    bool operator<(const PointDistFarther& other) const {
      return dist < other.dist;
    }
  };

  /// @brief HNSW parameterized constructor
  /// @param ef_construction: The number of nearest neighbours to search for while inserting a new node (default = 40)
  /// @param M: The number of edges to create for a newly inserted node (default = 16)
  /// @param MMax: The maximum number of edges that a node in the graph can have at any point in time (default = 16)
  /// @param MMax0: The maximum number of edges that a node at level 0 can have at any point in time (default = 32)
  HNSWIndex(
    int64_t ef_construction = 40,
    int64_t M = 16,
    int64_t MMax = 16,
    int64_t MMax0 = 32):
      entry_point{0},
      L{0},
      ef_construction{ef_construction},
      M{M},
      MMax{MMax},
      MMax0{MMax0},
      mL{1 / log(M)} {
    PrintParameters();
  }

  /// @brief Insert a new point into the HNSW index
  /// @param embedding: The point's vector embedding
  void Insert(const std::vector<float>& embedding);

  /// @brief Searches for 'ef' points closest to 'q' at layer 'lc', beginning from node 'ep'
  /// @param q:
  /// @param ep:
  /// @param ef:
  /// @param lc:
  /// @returns std::vector containing the IDs of the closest points
  std::vector<int64_t> SearchLayer(
    const Point& q,
    int64_t ep,
    int64_t ef,
    int64_t lc);

  /// @brief Select 'M' neighbours
  /// @param q:
  /// @param W:
  /// @param M:
  /// @param lc:
  /// @param extend_candidates:
  /// @param keep_pruned_connections:
  /// @returns 
  std::vector<int64_t> SelectNeighbours(
    const Point& q,
    const std::vector<int64_t>& C,
    int64_t M,
    int64_t lc);


  void KNNSearch();
  void PrintParameters();
  int64_t GenerateLevel();

  int64_t entry_point;
  int64_t L;
  int64_t ef_construction;
  int64_t M;
  int64_t MMax;
  int64_t MMax0;
  double mL;

  std::vector<Point> points;
  std::vector<std::unordered_map<int64_t, std::vector<int64_t>>> graph;  // the graph consists of levels; each level contains points; each point has multiple neighbours
};

} // namespace hnsw

#endif  // HNSW_HNSW_INDEX_H_