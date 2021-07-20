#include "../util/ipu_interface.hpp"
#include "model_assets.hpp"

#include <poplar/Tensor.hpp>
#include <poplar/Program.hpp>
#include <poplar/Type.hpp>
#include <poplin/ConvParams.hpp>
#include <poplin/Convolution.hpp>
#include <poplin/Norms.hpp>
#include <poprand/RandomGen.hpp>
#include <popops/ElementWise.hpp>
#include <popnn/NonLinearity.hpp>

using namespace std;
using namespace poplar;
using namespace poplar::program;

Tensor transp_0123_0312(IPU_Interface &ipu, Graph &g, Tensor &in, Tensor &out, Sequence &seq) {
  Tensor transp = in.dimShuffle({0, 3, 1, 2});
  seq.add(Copy(transp, out));
  return out;
}
Tensor transp_0123_0231(IPU_Interface &ipu, Graph &g, Tensor &in, Tensor &out, Sequence &seq) {
  Tensor transp = in.dimShuffle({0, 2, 3, 1});
  seq.add(Copy(transp, out));
  return out;
}
