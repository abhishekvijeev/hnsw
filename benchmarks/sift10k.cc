#include "HNSWIndex.h"

#include "util.h"

#include <algorithm>
#include <set>

int main(int argc, char** argv) {
  std::string sift_base_path = "siftsmall/siftsmall_base.fvecs";
  std::string sift_groundtruth_path = "siftsmall/siftsmall_groundtruth.ivecs";
  std::string sift_learn_path = "siftsmall/siftsmall_learn.fvecs";
  std::string sift_query_path = "siftsmall/siftsmall_query.fvecs";

  std::vector<std::vector<float>> sift_base = fvecs_read(sift_base_path);
  std::vector<std::vector<int>> sift_groundtruth = ivecs_read(sift_groundtruth_path);
  std::vector<std::vector<float>> sift_learn = fvecs_read(sift_learn_path);
  std::vector<std::vector<float>> sift_query = fvecs_read(sift_query_path);

  // Print first base vector
  // std::copy(sift_base[0].begin(), sift_base[0].end(), std::ostream_iterator<float>(std::cout, " "));
  // std::cout << std::endl << std::endl;

  // Print first groundtruth vector
  // std::copy(sift_groundtruth[0].begin(), sift_groundtruth[0].end(), std::ostream_iterator<float>(std::cout, " "));
  // std::cout << std::endl;

  hnsw::HNSWIndex h;

  std::cout << "Building index\n";
  for (size_t i = 0; i < sift_base.size(); i++) {
    h.Insert(sift_base[i]);
    if ((i > 0) && (i % 1000 == 0)) {
      std::cout << "Inserted " << i << " vectors\n";
    }
  }
  std::cout << "Finished building index\n";

  float recall_sum = 0.0;
  for (size_t i = 0; i < sift_query.size(); i++) {
    std::vector<std::pair<int64_t, std::vector<float>>> ret = h.KNNSearch(sift_query[i], 100, 100);
    std::set<float> groundtruth_set(sift_groundtruth[i].begin(), sift_groundtruth[i].end());
    std::set<float> expt_set;
    std::vector<float> intersection;

    for (auto& point : ret) {
      expt_set.insert(point.first);
    }

    std::set_intersection(groundtruth_set.begin(), groundtruth_set.end(),
                          expt_set.begin(), expt_set.end(),
                          std::back_inserter(intersection));
    float recall = intersection.size() / static_cast<float>(groundtruth_set.size());
    recall_sum += recall;
    // std::cout << recall << std::endl;
  }

  std::cout << "Average recall: " << recall_sum / sift_query.size() << std::endl;
  return 0;
}