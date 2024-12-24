#ifndef SHINX_HNSW_INDEX_H_
#define SHINX_HNSW_INDEX_H_

#include <cstdint>
#include <memory>
#include <vector>

namespace shinx {

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

  struct HNSWNode
  {
    std::vector<float> embedding;
    std::vector<std::shared_ptr<HNSWNode>> neighbours;
    std::shared_ptr<HNSWNode> skip_link;
  };

  std::shared_ptr<HNSWNode> entry_point_;

  uint32_t highest_level_;
  uint32_t expansion_factor_construction_;
  uint32_t level_generation_normalization_factor_;
  uint32_t established_connections_;
  uint32_t max_connections_;
};

} // namespace shinx

#endif  // SHINX_HNSW_INDEX_H_