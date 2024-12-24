#include "HNSWIndex.h"

#include <cmath>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

using namespace hnsw;

HNSWIndex::HNSWIndex() :
  highest_level_{0},
  expansion_factor_construction_{40},
  edges_per_node_{32},
  max_edges_per_node_{32},
  max_edges_per_node_level_0_{2 * max_edges_per_node_},
  normalization_factor_{1 / log(1.0 * max_edges_per_node_)}
{
  std::cout << "HNSWIndex()\n";
  std::cout << "normalization_factor_: " << normalization_factor_ << std::endl;
}

uint64_t HNSWIndex::GenerateLevel()
{
  std::mt19937 generator(std::random_device{}());
  return std::floor(-1 * std::log((std::uniform_real_distribution<>(0.0, 1.0))(generator) * normalization_factor_));
}

void HNSWIndex::Insert(const std::vector<float>& embedding)
{
  std::cout << "HNSWIndex::Insert()" << std::endl;
  
  for (int i = 0; i < 100; i++) {
    uint64_t level = GenerateLevel();
    printf("%lu\n", level);
  }

}