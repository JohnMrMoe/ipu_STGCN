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



vector<size_t> zs_re(size_t z) {return vector<size_t>(z, 0);}
vector<size_t> lmt_re(size_t n, size_t lm, vector<size_t> mx, bool one_zero=false) {
  vector<size_t> ret(n, lm);
  for (int i = 0; i < n; i ++) ret[i] = mx[i] < lm ? mx[i] : lm;
  if (one_zero) {ret[0]=1;}
  return ret;
}
Program quick_print_re(string title, Tensor &x, size_t lm = 2, bool one_zero=false) {
  Tensor slc = x.slice(zs_re(x.shape().size()), lmt_re(x.shape().size(), lm, x.shape(), one_zero));
  return PrintTensor(title, slc);
}

string _shapestring(vector<size_t> shape) {
  string x = "[";
  for (size_t i = 0; i < shape.size(); i++)  x += (i==0?"":",") + to_string(shape[i]);
  x += ']';
  return x;
}

pair<size_t, size_t> pad_dim(size_t input, size_t stride, size_t filter) {
  if (filter==0) return pair<size_t, size_t>(0, 0);
  size_t cmp = (input % stride == 0) ? (filter - stride) : (filter - (input % stride));
  size_t pad = cmp > 0 ? cmp : 0;
  size_t top = pad / 2;
  size_t bot = pad - top;
  return pair<size_t, size_t>(top, bot);
}
vector<size_t> out_shape(vector<size_t> input_shape,  // format: [batch_size, in_height, in_width, in_channels]
                         vector<size_t> kernel_shape, // format: [filter_height, filter_width, in_channels, out_channels]
                         vector<size_t> padding,
                         vector<size_t> stride
                       )
{
  size_t in_height      = input_shape[1];
  size_t in_width       = input_shape[2];
  size_t kernel_height  = kernel_shape[0];
  size_t kernel_width   = kernel_shape[1];
  size_t in_channels    = kernel_shape[2];
  size_t out_channels   = kernel_shape[3];

  size_t out_height = (size_t) ceil((float) (in_height - kernel_height + 1) / (float) stride[1]);
  size_t out_width  = (size_t) ceil((float) (in_width - kernel_width + 1)   / (float) stride[2]);

  return vector<size_t>{input_shape[0], out_height, out_width, out_channels};
}

pair<size_t, size_t>               __calculate_SAME_padding(size_t k) {
  pair<size_t, size_t> pad;
  pad.first  = k / 2;
  pad.second = k / 2 + k % 2;
  return pad;
}
pair<vector<size_t>, vector<size_t>> calculate_SAME_padding(vector<size_t> filter, int skip, int min) {
  vector<size_t> avant(filter.size() - min, 0);
  vector<size_t> post (filter.size() - min, 0);
  pair<size_t, size_t> pad;
  for (size_t i = min; i < filter.size(); i++) {
    if (i < skip) continue;
    pad = __calculate_SAME_padding(filter[i]);
    avant[i] = pad.first;
    post [i] = pad.second;
  }
  return pair<vector<size_t>, vector<size_t>>(avant, post);
}

Tensor /* TODO:: DEFINE RETURN */
conv2d(IPU_Interface &ipu,
       Graph &g,
       Tensor &input,  // with shape [batch_size, in_height, in_width, in_channels]
       vector<size_t> filter_shape, // with shape [filter_height, filter_width, in_channels, out_channels]
       string filter_scope,
       string bias_scope,
       Tensor &output, // with shape [batch_size, out_height, out_width, out_channels]
       Sequence &seq,
       bool biased
     )
{
  layer_entry_note(ipu, "CONV VALID", filter_scope);
  size_t cnvg = 1; // conv groups
  //
  // Program transp_0123_0312(IPU_Interface &ipu, Graph &g, Tensor in, Tensor out);
  //       output[b, i, j, k] =
  //                         sum_{di, dj, q}  input[b, strides[1] * i + di, strides[2] * j + dj, q] *
  //                                          filter[di, dj, q, k]
  //
  // filter_shape = [filter_height, filter_width, in_channels, out_channels]
  // convGroups x outChansPerConvGroup x inChansPerConvGroup x H x W

  vector<size_t> ipu_filter_shape = vector<size_t>{cnvg,
                                                   filter_shape[3]/cnvg, //
                                                   filter_shape[2]/cnvg, //
                                                   filter_shape[0],
                                                   filter_shape[1]};
  vector<size_t> inputFieldShape = vector<size_t>{ input.shape()[1],
                                                   input.shape()[2]};
  vector<size_t> kernelShape = vector<size_t>{     filter_shape[0],
                                                   filter_shape[1]};

  poplin::ConvParams cp(poplar::FLOAT, input.shape()[0], inputFieldShape, kernelShape, input.shape()[3], filter_shape[3], cnvg);

  Tensor input_t = createInput(g,   cp, "conv2d <" + filter_scope + "> input");
  Tensor input_port = input_t.reshape(vector<size_t>{input.shape()[0], input.shape()[3], input.shape()[1], input.shape()[2]});
  Tensor filter  = createWeights(g, cp, "conv2d <" + filter_scope + "> weights");

  size_t filter_size = filter.flatten().shape()[0];
  FileFedVector kernel(filter_scope, filter_size);

  Tensor refilter = filter.reshape({filter_shape[3], filter_shape[2], filter_shape[0], filter_shape[1]}).dimShuffle({2, 3, 1, 0});

  ipu.addVariable(filter_scope, refilter, 4, kernel.permanent_pointer);

  input_port = transp_0123_0312(ipu, g, input, input_port, seq);// ????


  Tensor outpt_t = poplin::convolution(g, input_t, filter, cp, false, seq);

  if (biased) {
    Tensor bias = poplin::createBiases(g, outpt_t, "conv2d <" + filter_scope + "> bias");
    FileFedVector bias_feed(bias_scope, bias.numElements());
    ipu.addVariable(bias_scope, bias, 4, bias_feed.permanent_pointer);
    poplin::addBias(g, outpt_t, bias, seq);
  }

  output = transp_0123_0231(ipu, g, outpt_t, output, seq);

  layer_exit_note(ipu, "CONV VALID", filter_scope);

  return output;
}


Tensor conv2d_SAME(IPU_Interface &ipu,
                 Graph &g,
                 Tensor &input,
                 vector<size_t> filter_shape,
                 string filter_scope,
                 string bias_scope,
                 Tensor &output,
                 Sequence &seq,
                 bool biased
              )
{
  size_t cnvg = 1;
  size_t ch = input.shape()[3];

  // input   = [B x H x W x inChans]
  // input_t = [B x inChans x H x W],
  // filter_shape = [filter_height, filter_width, in_channels, out_channels]
  //convGroups x outChansPerConvGroup x inChansPerConvGroup x H x W

  pair<size_t, size_t> padH;
  pair<size_t, size_t> padW;

  if (filter_shape[0] == 1 && filter_shape[1] == 1) {
    padH.first = padH.second = padW.first = padW.second = 0;
  } else {
    padH = __calculate_SAME_padding(filter_shape[0]);
    padW = __calculate_SAME_padding(filter_shape[1]);
  }


  vector<size_t> ipu_filter_shape = vector<size_t>{cnvg,
                                                   filter_shape[3]/cnvg, //
                                                   filter_shape[2]/cnvg, //
                                                   filter_shape[0],
                                                   filter_shape[1]};
  vector<size_t> inputFieldShape = vector<size_t>{ input.shape()[1] + padH.first + padH.second,
                                                   input.shape()[2] + padW.first + padW.second};
  vector<size_t> kernelShape = vector<size_t>{     filter_shape[0],
                                                   filter_shape[1]};

  poplin::ConvParams cp(poplar::FLOAT, input.shape()[0], inputFieldShape, kernelShape, input.shape()[3], filter_shape[3], cnvg);

  Tensor input_t = createInput(g,   cp, "conv2d.s <" + filter_scope + "> same");
  Tensor filter  = createWeights(g, cp, "conv2d.s <" + filter_scope + "> weights");


  ipu.addVariable(filter_scope + "_conv_IN", input_t, 0);

  Tensor ungrouped = input_t.reshape({input.shape()[0], ch, input_t.shape()[2], input_t.shape()[3]});
  Tensor tf_shaped = ungrouped.dimShuffle({0, 2, 3, 1});
  Tensor port = tf_shaped.slice({0, padH.first, padW.first, 0},
                                {input.shape()[0], input.shape()[1] + padH.first,
                                 input.shape()[2] + padW.first, ch});
  seq.add(Copy(input, port));

  size_t filter_size = filter.numElements();
  FileFedVector kernel(filter_scope, filter_size);
  Tensor refilter = filter.reshape({filter_shape[3], filter_shape[2], filter_shape[0], filter_shape[1]}).dimShuffle({2, 3, 1, 0});
  ipu.addVariable(filter_scope, refilter, 4, kernel.permanent_pointer);

  Tensor output_t = poplin::convolution(g, input_t, filter, cp, false, seq, DebugContext(filter_scope + ":<CONV>"));

  // std::cout << filter_scope << '\n';
  // std::cout << "input.shape    = " << input.shapeToString() << '\n';
  // std::cout << "input_t.shape  = " << input_t.shapeToString() << '\n';
  // std::cout << "output_t.shape = " << output_t.shapeToString() << '\n';


  if (biased) {
    Tensor bias = poplin::createBiases(g, output_t, "conv2d.s <" + filter_scope + "> bias");
    FileFedVector bias_feed(bias_scope, bias.flatten().shape()[0]);
    ipu.addVariable(bias_scope, bias, 4, bias_feed.permanent_pointer);
    poplin::addBias(g, output_t, bias, seq);
  }


  output = transp_0123_0231(ipu, g, output_t, output, seq);

  return output;
}
