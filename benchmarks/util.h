#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

std::vector<std::vector<float>> fvecs_read(const std::string& file_path) {
  std::vector<std::vector<float>> data;
  std::ifstream file(file_path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Unable to open file " + file_path);
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

std::vector<std::vector<int>> ivecs_read(const std::string& file_path) {
  std::vector<std::vector<int>> data;
  std::ifstream file(file_path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Unable to open file " + file_path);
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