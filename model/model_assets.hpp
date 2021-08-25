
#ifndef MODEL_ASSETS
#define MODEL_ASSETS

#include <vector>
#include <string.h>
#include <stdio.h>

#include <poplar/Tensor.hpp>
#include <poplar/Program.hpp>
#include <poplar/Type.hpp>
#include <poplin/ConvParams.hpp>
#include <poplin/Convolution.hpp>
#include <poplin/Norms.hpp>
#include <poprand/RandomGen.hpp>
#include <popops/ElementWise.hpp>
#include <popnn/NonLinearity.hpp>

#include "../util/ipu_interface.hpp"
#include "../util/arguments.hpp"
#include "../data/data_utility.hpp"

using namespace std;
using namespace poplar;
using namespace poplar::program;

void layer_entry_note     (IPU_Interface &ipu, string layer, string scope, string notes="");
void layer_exit_note      (IPU_Interface &ipu, string layer, string scope, string notes="");

Tensor transp_0123_0231   (IPU_Interface &ipu, Graph &g, Tensor &in, Tensor &out, Sequence &seq);
Tensor transp_0123_0312   (IPU_Interface &ipu, Graph &g, Tensor &in, Tensor &out, Sequence &seq);

Tensor gconv              (IPU_Interface &ipu, Graph &g, Tensor &src, Tensor &dst, Tensor &theta, size_t Ks, size_t c_in, size_t c_out, Sequence &seq, Sequence &bwd, string scope="unscoped");
Tensor layer_norm         (IPU_Interface &ipu, Graph &g, Tensor &src, Tensor &dst, Sequence & Seq, Sequence &bwd, string scope="unscoped");
Tensor temporal_conv_layer(IPU_Interface &ipu, Graph &g, Tensor &src, Tensor &dst, size_t Kt, size_t c_in, size_t c_out, Sequence &seq, Sequence &bwd, string scope="unscoped", string act_func="relu");
Tensor spatio_conv_layer  (IPU_Interface &ipu, Graph &g, Tensor &src, Tensor &dst, size_t Ks, size_t c_in, size_t c_out, Sequence &seq, Sequence &bwd, string scope="unscoped");
Tensor st_conv_block      (IPU_Interface &ipu, Graph &g, Tensor &src, size_t Ks, size_t Kt, size_t channels[3], Sequence &seq, Sequence &bwd, string scope="unscoped", string act_func="GLU");
Tensor fully_con_layer    (IPU_Interface &ipu, Graph &g, Tensor &src, Tensor &out, size_t n, size_t channel, Sequence &seq, Sequence &bwd, string scope="unscoped");
Tensor output_layer       (IPU_Interface &ipu, Graph &g, Tensor &src, Tensor &x2, size_t T, Sequence &seq, Sequence &bwd, string scope, string act_func="GLU");

Tensor fully_con_w_bwd    (IPU_Interface &ipu, Graph &g, Tensor &x, Tensor &out, size_t n, size_t channel, Sequence &seq, Sequence &bwd, string scope="unscoped");

tuple<Program, Program, Tensor> build_model(size_t blocks[2][3], Arguments args, IPU_Interface &ipu, Graph &g);

string _shapestring(vector<size_t> shape);
vector<size_t> out_shape(vector<size_t> input_shape, vector<size_t> kernel_shape, vector<size_t> padding = vector<size_t>{1, 1, 1, 1}, vector<size_t> stride  = vector<size_t>{1, 1, 1, 1});
Tensor conv2d(IPU_Interface &ipu, Graph &g, Tensor &input, vector<size_t> filter_size, string filter_scope, string bias_scope, Tensor &output, Sequence &prog, bool biased=false);
Tensor conv2d_SAME(IPU_Interface &ipu, Graph &g, Tensor &input, vector<size_t> filter_shape, string filter_scope, string bias_scope, Tensor &output, Sequence &prog, bool biased=false);
Tensor conv2D_w_bwd(IPU_Interface &ipu, Graph &g, Tensor &input, vector<size_t> filter_shape, string filter_scope, string bias_scope, Tensor &output, Sequence &seq, Sequence &bwd, bool biased=false, bool same=false);
pair<size_t, size_t> __calculate_SAME_padding(size_t k);

Tensor verify_same_2(Graph &g, Tensor &res, Tensor &inp, Sequence &seq, Sequence &bwd, string label = "UNTITLED_VER", bool correct = false);
Tensor verify_same(IPU_Interface &ipu, Graph &g, Tensor &res, string file, Sequence &seq, bool correct = false);
Tensor verification_pass(IPU_Interface &ipu, Graph &g, Tensor &inp, string file, Sequence &seq);
Tensor derandomize(IPU_Interface &ipu, Graph &g, Tensor &rnd, string file);

#endif
