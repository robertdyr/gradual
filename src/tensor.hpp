#pragma once

#include <sys/types.h>
#include <memory>
#include <vector>

// forward declaration
class IOperation;

class Tensor  : public std::enable_shared_from_this<Tensor> {
public:
  std::shared_ptr<IOperation> created_by;
  std::shared_ptr<Tensor> gradients;
  std::vector<float> data;
  std::vector<ssize_t> dimensions;
  bool requires_grad;
  Tensor(const std::vector<ssize_t>& dims, float initialValue = 0.0f, bool requires_grad = true);
  float& operator()(const std::vector<ssize_t>& indices);
  const float& operator()(const std::vector<ssize_t>& indices) const;
  void fill(float value);
  const std::vector<ssize_t>& get_dimensions() const;
  ssize_t size() const;
  void backwards();
  void randomize();
  std::shared_ptr<Tensor> transpose() const;

private:
  ssize_t calculate_size(const std::vector<ssize_t>& dims) const;
  ssize_t calculate_flat_index(const std::vector<ssize_t>& indices) const;
};
