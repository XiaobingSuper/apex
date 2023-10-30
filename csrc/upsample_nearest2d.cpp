#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/native/UpSample.h>

#include <c10/util/ArrayRef.h>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <iostream>


void upsample_nearest2d_out_cuda(const at::Tensor& input, c10::IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w, const at::Tensor& output);
void upsample_nearest2d_backward_out_cuda(const at::Tensor& grad_output, c10::IntArrayRef output_size, c10::IntArrayRef input_size, c10::optional<double> scales_h, c10::optional<double> scales_w, const at::Tensor& grad_input);

at::Tensor apex_upsample_nearest2d(
    const at::Tensor& input,
    c10::optional<std::vector<int64_t>> output_size,
    c10::optional<std::vector<double>> scale_factors) {
  c10::OptionalArrayRef<int64_t> ref_output_size = c10::nullopt;
  if (output_size) {
    ref_output_size = *output_size;
  }
  c10::optional<c10::ArrayRef<double>> ref_scale_factors = c10::nullopt;
  if (scale_factors) {
    ref_scale_factors = {scale_factors->data(), scale_factors->data() + scale_factors->size()};
  }
  auto osize = at::native::upsample::compute_output_size(input.sizes(), ref_output_size, ref_scale_factors);
  auto scale_h = at::native::upsample::get_scale_value(ref_scale_factors, 0);
  auto scale_w = at::native::upsample::get_scale_value(ref_scale_factors, 1);
  std::cout<<osize<<std::endl;
  auto output = at::zeros({input.size(0), input.size(1), osize[0], osize[1]}, input.options().memory_format(input.suggest_memory_format()));
  return output;
  //return at::upsample_nearest2d(input, osize, scale_h, scale_w);
  upsample_nearest2d_out_cuda(input, osize, scale_h, scale_w, output);
  return output;
}

at::Tensor apex_upsample_nearest2d_backward(
    const at::Tensor& grad_output,
    c10::IntArrayRef output_size,
    c10::IntArrayRef input_size,
    c10::optional<std::vector<double>> scale_factors) {
  c10::optional<c10::ArrayRef<double>> ref_scale_factors = c10::nullopt;
  if (scale_factors) {
    ref_scale_factors = {scale_factors->data(), scale_factors->data() + scale_factors->size()};
  }
  auto scale_h = at::native::upsample::get_scale_value(ref_scale_factors, 0);
  auto scale_w = at::native::upsample::get_scale_value(ref_scale_factors, 1);
  auto grad_input = at::zeros(input_size, grad_output.options().memory_format(grad_output.suggest_memory_format()));
  upsample_nearest2d_backward_out_cuda(grad_output, output_size, input_size, scale_h, scale_w, grad_input);
  return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &apex_upsample_nearest2d, "upsample_nearest2d forward (CUDA)");
  m.def("backward", &apex_upsample_nearest2d_backward, "upsample_nearest2d backward (CUDA)");
}
