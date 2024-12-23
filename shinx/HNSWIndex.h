#ifndef SHINX_HNSW_INDEX_H_
#define SHINX_HNSW_INDEX_H_

#include <cstdint>

namespace shinx {

// Implementation of the Hierarchical Navigable Small World graph:
// 
// Malkov, Yu A., and Dmitry A. Yashunin. "Efficient and robust
// approximate nearest neighbor search using hierarchical navigable
// small world graphs." IEEE transactions on pattern analysis and
// machine intelligence 42.4 (2018): 824-836.
class HNSWIndex
{
  public:

  HNSWIndex();
  void Insert();
  void SearchLayer();
  void SelectNeighbours();
  void KNNSearch();

  private:

  uint32_t entry_point_;
  uint32_t highest_level_;
  uint32_t expansion_factor_construction_;
  uint32_t level_generation_normalization_factor_;
  uint32_t established_connections_;
  uint32_t max_connections_;

};

} // namespace shinx

#endif  // SHINX_HNSW_INDEX_H_