#ifndef HNSW_HNSW_INDEX_H_
#define HNSW_HNSW_INDEX_H_

#include <cstdint>
#include <memory>
#include <vector>

namespace hnsw {

// Implementation of the Hierarchical Navigable Small World (HNSW)
// algorithm based on:
// 
// Malkov, Yu A., and Dmitry A. Yashunin. "Efficient and robust
// approximate nearest neighbor search using hierarchical navigable
// small world graphs." IEEE transactions on pattern analysis and
// machine intelligence 42.4 (2018): 824-836.
class HNSWIndex
{
  public:

  HNSWIndex();
  void Insert(const std::vector<float>& embedding);
  void SearchLayer();
  void SelectNeighbours();
  void KNNSearch();

  private:

  uint64_t GenerateLevel();

  struct HNSWNode
  {
    std::vector<float> embedding;
    std::vector<std::shared_ptr<HNSWNode>> neighbours;
    std::shared_ptr<HNSWNode> skip_link;
  };

  std::shared_ptr<HNSWNode> entry_point_;       // ep
  uint64_t highest_level_;                      // L
  uint64_t expansion_factor_construction_;      // efConstruction
  uint64_t edges_per_node_;                     // M
  uint64_t max_edges_per_node_;                 // Mmax
  uint64_t max_edges_per_node_level_0_;         // Mmax0
  double normalization_factor_;                 // mL
};

} // namespace hnsw

#endif  // HNSW_HNSW_INDEX_H_