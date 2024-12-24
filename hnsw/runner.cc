#include "HNSWIndex.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

std::vector<std::vector<float>> fvecs_read(const std::string& file_path)
{
  std::vector<std::vector<float>> data;
  std::ifstream file(file_path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Unable to open file" + file_path);
  }

  while (!file.eof()) {
    int dim;
    file.read(reinterpret_cast<char*>(&dim), sizeof(int));
    if (file.eof()) {
      break;
    }

    std::vector<float> vec(dim);
    file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
    if (file.eof()) {
      break;
    }
    data.push_back(std::move(vec));
  }

  return data;
}

std::vector<std::vector<int>> ivecs_read(const std::string& file_path)
{
  std::vector<std::vector<int>> data;
  std::ifstream file(file_path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Unable to open file" + file_path);
  }

  while (!file.eof()) {
    int dim;
    file.read(reinterpret_cast<char*>(&dim), sizeof(int));
    if (file.eof()) {
      break;
    }

    std::vector<int> vec(dim);
    file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(int));
    if (file.eof()) {
      break;
    }
    data.push_back(std::move(vec));
  }

  return data;
}

int main(int argc, char** argv)
{
  std::string sift_base_path = "/home/avijeev/sift/sift_base.fvecs";
  std::string sift_groundtruth_path = "/home/avijeev/sift/sift_groundtruth.ivecs";
  std::string sift_learn_path = "/home/avijeev/sift/sift_learn.fvecs";
  std::string sift_query_path = "/home/avijeev/sift/sift_query.fvecs";

  std::vector<std::vector<float>> sift_base = fvecs_read(sift_base_path);
  std::vector<std::vector<int>> sift_groundtruth = ivecs_read(sift_groundtruth_path);
  std::vector<std::vector<float>> sift_learn = fvecs_read(sift_learn_path);
  std::vector<std::vector<float>> sift_query = fvecs_read(sift_query_path);

  // Print first base vector
  std::copy(sift_base[0].begin(), sift_base[0].end(), std::ostream_iterator<float>(std::cout, " "));
  std::cout << std::endl << std::endl;

  // Print first groundtruth vector
  // std::copy(sift_groundtruth[0].begin(), sift_groundtruth[0].end(), std::ostream_iterator<float>(std::cout, " "));
  // std::cout << std::endl;

  hnsw::HNSWIndex h;
  h.Insert(sift_base[0]);
  return 0;
}