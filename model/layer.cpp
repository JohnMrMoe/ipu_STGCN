#ifndef __LAYER
#define __LAYER

#include "../util/ipu_interface.hpp"
#include "../util/Logger.hpp"
#include "model_assets.hpp"
//#include "convolution.cpp"


#include <poplar/Tensor.hpp>
#include <poplar/Program.hpp>
#include <poplar/Type.hpp>
#include <poplin/ConvParams.hpp>
#include <poplin/Convolution.hpp>
#include <poplin/Norms.hpp>
#include <poplin/MatMul.hpp>
#include <poprand/RandomGen.hpp>
#include <popnn/NonLinearity.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Rearrange.hpp>
#include <popops/Reduce.hpp>
#include <popnn/Pooling.hpp>
#include <popnn/PoolingDef.hpp>
#include <popops/Fill.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace std;

vector<size_t> zs(size_t z) {return vector<size_t>(z, 0);}
vector<size_t> lmt(size_t n, size_t lm, vector<size_t> mx, bool one_zero=false) {
  vector<size_t> ret(n, lm);
  for (int i = 0; i < n; i ++) ret[i] = mx[i] < lm ? mx[i] : lm;
  if (one_zero) {ret[0]=1;}
  return ret;
}
Program quick_print(string title, Tensor &x, size_t lm = 2, bool one_zero=false) {
  Tensor slc = x.slice(zs(x.shape().size()), lmt(x.shape().size(), lm, x.shape(), one_zero));
  return PrintTensor(title, slc);
}

void layer_entry_note(IPU_Interface &ipu, string layer, string scope, string notes) {
  ipu.model_log.log("\t\t\t" + layer + "\t\tENTRY <" + scope + ">, " + notes + "", "\n", true);
}
void layer_exit_note(IPU_Interface &ipu, string layer, string scope, string notes) {
  ipu.model_log.log("\t\t\t" + layer + "\t\tEXIT  <" + scope + ">, " + notes, "\n", true);
}

size_t minus_one_val(vector<size_t> full_shape, vector<size_t> accounted_for) {
  size_t tot = 1;
  for (size_t i = 0; i < full_shape.size(); i++) {
    tot *= full_shape[i];
  }
  for (size_t i = 0; i < accounted_for.size(); i++) {
    tot /= accounted_for[i];
  }
  return tot;
}

Tensor gconv              (IPU_Interface &ipu, Graph &g, Tensor &x, Tensor &dst, Tensor &theta, size_t Ks, size_t c_in, size_t c_out, Sequence &seq, string scope) {
  layer_entry_note(ipu, "GCONV", scope);
  ipu.shape_display(x, "x", "\n");
  // writes to a tensor of size [batch_size, n, c_out]
  Sequence gconv_layer(ipu.notification(g, "gconv-layer {"+scope+"}"));

  Tensor kernel = ipu.getVariable_OLD(g, "graph_kernel");
  size_t n = kernel.shape()[0];

  // x_tmp = tf.reshape(tf.transpose(x, [0, 2, 1]), [-1, n])
  Tensor t_x = x.dimShuffle({0, 2, 1});
  Tensor x_tmp = t_x.reshape(vector<size_t>{minus_one_val(t_x.shape(), vector<size_t>{n}), n});

  // x_mul = tf.reshape(tf.matmul(x_tmp, kernel), [-1, c_in, Ks, n])
  Tensor x_mul = poplin::matMul(g, x_tmp, kernel, gconv_layer, poplar::FLOAT);
  Tensor x_mlr = x_mul.reshape(vector<size_t>{minus_one_val(x_mul.shape(), vector<size_t>{c_in, Ks, n}),
                                              c_in, Ks, n});


  // TRANSPOSE [0, 1, 2, 3] -> [0, 3, 1, 2] //ROTATION OF EACH INNER CUBE
  vector<size_t> s = x_mlr.shape();
  Tensor x_rt = ipu.getVariable(g, "Transposer", vector<size_t>{s[0], s[3], s[1], s[2]}, 0,0);
  x_rt = transp_0123_0312(ipu, g, x_mlr, x_rt, gconv_layer);
  Tensor x_rr = x_rt.reshape(vector<size_t>{minus_one_val(x_rt.shape(), vector<size_t>{c_in, Ks}), c_in * Ks});
  //

  Tensor x_fmul = poplin::matMul(g, x_rr, theta, gconv_layer, poplar::FLOAT);
  Tensor x_final = x_fmul.reshape({minus_one_val(x_fmul.shape(), vector<size_t>{n, c_out}), n, c_out});

  gconv_layer.add(Copy(x_final, dst));
  // FINISH METHOD

  // Backward Pass
  Sequence bwd;
  Tensor re_x_fmul = dst.reshape(x_fmul.shape());

  // get gradients over latter multiplication
  Tensor deltaWeights  = poplin::matMul(g, x_rr.transpose(), re_x_fmul,   bwd, poplar::FLOAT);
  Tensor bwError       = poplin::matMul(g, re_x_fmul, theta.transpose(), bwd, poplar::FLOAT);

  // apply gradients
  popops::addInPlace(g, theta, deltaWeights, bwd);

  //some fucking flip flops
  Tensor bwError_R  = bwError.reshape(x_rt.shape());

  Tensor bwError_xms = bwError_R.dimShuffle({0, 2, 3, 1}).reshape(x_mul.shape());

  // we DON't Change the kernel? I think, so we'll just leave it as is
  bwError = poplin::matMul(g, bwError_xms, kernel.transpose(), bwd, poplar::FLOAT);

  // final flips and flops
  bwError = bwError.reshape(t_x.shape()).dimShuffle({0, 2, 1});
  bwd.add(Copy(bwError, x));
  // BW DONE`?

  layer_exit_note(ipu, "GCONV", scope);
  seq.add(gconv_layer);
  return dst;
}
Tensor layer_norm         (IPU_Interface &ipu, Graph &g, Tensor &x, Tensor &dst, Sequence &seq, string scope) {

  layer_entry_note(ipu, "LAYER_NORM", scope);
  ipu.shape_display(x, "x", "\n");
  std::vector<std::size_t> shape = x.shape();
  size_t N = shape[2];
  size_t C = shape[3];

  Sequence norm_layer(ipu.notification(g, "layer-norm {"+scope+"}"));
  Tensor _NCHW = ipu.getVariable(g, "nchw(x)", vector<size_t>{x.shape()[0], x.shape()[3], x.shape()[1], x.shape()[2]});
  _NCHW = transp_0123_0312(ipu, g, x, _NCHW, norm_layer);


  float  epsilon = .1;// 0.001, // epsilon?

  // PAIR:  avg,    std-dev
  // std::pair<Tensor, Tensor> moments = poplin::normStatistics(g,
  //                                                    _NCHW,
  //                                                    epsilon,
  //                                                    norm_layer, // program
  //                                                    true //unbiasedVarEstimate
  //                                                  );

  Tensor mu = popnn::pooling::pool(g,
                    popnn::pooling::PoolParams(
                     popnn::PoolingType::AVG,
                     vector<size_t>{x.shape()[2], x.shape()[3]},
                     vector<size_t>{x.shape()[2], x.shape()[3]},
                     vector<unsigned>{1, 1},
                     vector<int>{0, 0},
                     vector<int>{0, 0},
                     x.shape()[1],
                     x.shape()[0],
                     poplar::FLOAT
                   ),
                   x,
                   norm_layer
                 );

  //
  Tensor divn = g.addConstant<float>(poplar::FLOAT, {1}, {(float) (x.shape()[2] * x.shape()[3]) /*-1*/}); g.setTileMapping(divn, 0);
  Tensor xmmu = popops::sub(g, x, mu, norm_layer);
  Tensor xmmu_sq = popops::square(g, xmmu, norm_layer);

  Tensor sigma = popnn::pooling::pool(g,
                    popnn::pooling::PoolParams(
                     popnn::PoolingType::SUM,
                     vector<size_t>{x.shape()[2], x.shape()[3]},
                     vector<size_t>{x.shape()[2], x.shape()[3]},
                     vector<unsigned>{1, 1},
                     vector<int>{0, 0},
                     vector<int>{0, 0},
                     x.shape()[1],
                     x.shape()[0],
                     poplar::FLOAT
                   ),
                   xmmu_sq,
                   norm_layer
                 );
  //
  popops::divInPlace(g, sigma, divn, norm_layer);

  ipu.addVariable(scope + "_avg", mu);
  ipu.addVariable(scope + "_inv-std-dev", sigma);

  FileFedVector ffv_g(scope + "=Gamma", N*C);
  FileFedVector ffv_b(scope + "=Beta", N*C);

  Tensor gamma = ipu.getVariable(g, "gamma." + scope, std::vector<size_t>{1, 1,  N,  C}, FILEFD_TENSOR, ffv_g.permanent_pointer);// TODO ASSERT CORRECT FILL?
  Tensor beta  = ipu.getVariable(g, "beta",  std::vector<size_t>{1, 1,  N,  C}, FILEFD_TENSOR, ffv_b.permanent_pointer);// TODO ASSERT CORRECT FILL?
  Tensor _1eminsix = g.addConstant<float>(poplar::FLOAT, {1}, {0.000001});
  g.setTileMapping(_1eminsix, 0);
  ipu.addVariable("_1eminsix", _1eminsix);

  //Tensor norm = popops::sub(g, x, mu, norm_layer);
  Tensor norm = ipu.getVariable(g, scope + "xmmu", xmmu.shape());
  norm_layer.add(Copy(xmmu, norm));

  ipu.addVariable(scope + "_norm", norm);

  // // Original interpreatation:
  popops::addInPlace (g, sigma, _1eminsix, norm_layer);
  Tensor sqrt_sigma = popops::sqrt(g, sigma, norm_layer);

  popops::divInPlace(g, norm, sqrt_sigma, norm_layer); // -- Problem
  popops::mulInPlace(g, norm, gamma, norm_layer);

  popops::addInPlace(g, norm, beta, norm_layer);

  norm_layer.add(Copy(norm, dst));


  // Backward pass over this!
    // with help from https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    // get the gradients?
  Sequence bwd;

  popnn::pooling::PoolParams pooler(
    popnn::PoolingType::SUM,
    vector<size_t>{dst.shape()[2], dst.shape()[3]}, // field shape
    vector<size_t>{dst.shape()[2], dst.shape()[3]}, // kernel shape
    vector<unsigned>{1, 1},                        // stride
    vector<int>{0, 0},    // pad
    vector<int>{0, 0},    // pad
    dst.shape()[1],     // channels
    dst.shape()[0],     // batch size
    poplar::FLOAT
  );

  Tensor deltaBias = popnn::pooling::pool(g, pooler, dst, bwd);

  Tensor deltaWeights  = popops::mul(g, xmmu, dst,   bwd/*, poplar::FLOAT*/);
  Tensor bwError       = popops::mul(g, dst, gamma,  bwd/*, poplar::FLOAT*/);

  /* ALT
  Tensor deltaWeights  = popops::matMul(g, xmmu.dimShuffle({0, 1, 3, 2}), dst,   bwd , poplar::FLOAT);
  Tensor bwError       = popops::matMul(g, dst, gamma.dimShuffle({0, 1, 3, 2}),  bwd , poplar::FLOAT);
  */

    // Update beta/gamma
  // popops::addInPlace(g, beta, deltaBias, bwd);     overlapping writes? TODO
  // popops::addInPlace(g, gamma, deltaWeights, bwd); overlapping writes? TODO

    //
  Tensor  one = g.addConstant<float>(poplar::FLOAT, {1}, { 1});
  Tensor mone = g.addConstant<float>(poplar::FLOAT, {1}, {-1});
  Tensor   p5 = g.addConstant<float>(poplar::FLOAT, {1}, {.5});
  g.setTileMapping(one, 1471); // just map them to a tile, not trying to put all on zero here.
  g.setTileMapping(mone, 1471);
  g.setTileMapping(p5, 1471);
  Tensor invSigma = popops::div(g, one, sqrt_sigma, bwd);
  Tensor divar = popops::mul(g, bwError, xmmu, bwd);
  Tensor dxmu1 = popops::mul(g, bwError, invSigma, bwd);


  // #step6 var=sigma
  // dsqrtvar = -1. /(sqrtvar**2) * divar
  Tensor dsqrtvar = popops::div(g, mone, sigma, bwd);
  // popops::mulInPlace(g, dsqrtvar, divar, bwd); OVerlapping write, TODO


  // #step5
  // dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar
  Tensor dvar = popops::div(g, p5, sqrt_sigma, bwd);
  popops::mulInPlace(g, dvar, dsqrtvar, bwd);

  cout << scope << "\t dvar.shape= " << dvar.shapeToString() << "\n";

  // #step4
  // dsq = 1. /N * np.ones((N,D)) * dvar
  FeedinVector fv(dvar.numElements()); fv.valfill(1. / (x.shape()[2] * x.shape()[3]));
  Tensor flector = ipu.getVariable(g, "flector", dvar.shape(), FILEFD_TENSOR, fv.permanent_pointer);

  Tensor dsq = popops::mul(g, flector, dvar, bwd);

  // #step3
  // dxmu2 = 2 * xmu * dsq
  Tensor two = ipu.getVariable_OLD(g, "two");
  popops::mulInPlace(g, xmmu, two, bwd);
  popops::mulInPlace(g, xmmu, dsq, bwd);
  // dxmu2 = xmmu;


  // #step2
  // dx1 = (dxmu1 + dxmu2)
  // dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
  popops::addInPlace(g, dxmu1, xmmu, bwd);
  Tensor dx1 = dxmu1;

  Tensor dmu = popnn::pooling::pool(g,
                    popnn::pooling::PoolParams(
                     popnn::PoolingType::SUM,
                     vector<size_t>{dx1.shape()[2], dx1.shape()[3]},
                     vector<size_t>{dx1.shape()[2], dx1.shape()[3]},
                     vector<unsigned>{1, 1},
                     vector<int>{0, 0},
                     vector<int>{0, 0},
                     dx1.shape()[1],
                     dx1.shape()[0],
                     poplar::FLOAT
                   ),
                   dx1,
                   bwd
                 );
  popops::mulInPlace(g, dmu, mone, bwd);

  // #step1
  // dx2 = 1. /N * np.ones((N,D)) * dmu
  bwd.add(Copy(flector, dsq));
  popops::mulInPlace(g, dsq, dmu, bwd);
  Tensor dx2 = dsq;

  // #step0
  // dx = dx1 + dx2
  bwd.add(Copy(dx1, x));
  popops::addInPlace(g, x, dx2, bwd);

  // end of backwards pass

  seq.add(norm_layer);
  return dst;
  // return dst;
}
Tensor temporal_conv_layer(IPU_Interface &ipu, Graph &g, Tensor &src, Tensor &dst, size_t Kt, size_t c_in, size_t c_out, Sequence &seq, string scope, string act_func) {
  layer_entry_note(ipu, "TEMPORAL_CONV_LAYER", scope, ("af="+act_func+", Kt=") +to_string(Kt));

  std::vector<std::size_t> shape = src.shape();
  std::vector<std::size_t> filter_shape;
  size_t T = shape[1];
  size_t n = shape[2];

  Sequence tconv_layer(ipu.notification(g, "temporal convolution layer {"+scope+"}"));
  // verification_pass(ipu, g, src, scope + "[TIN]["+act_func+"]", tconv_layer);


  Tensor x_input;
  Tensor x;


  Sequence bwd_ch;
  if (c_in > c_out) {
    vector<size_t> kernel_shape = {1, 1,  c_in,  c_out};
    Tensor feed = ipu.getAlternatingSpace("layer", out_shape(src.shape(), kernel_shape));
    x_input = conv2d_SAME(ipu, g, src, kernel_shape, scope + "=w_input", "N/A", feed, tconv_layer);
  } else if (c_in < c_out) {
    Tensor padding = ipu.getExistingVariableSlice("PAD", {shape[0], T, n,  (c_out-c_in)}, {0, 0, 0, 0}, 0);
    x_input = poplar::concat(src, padding, 3);
  } else {
    x_input = src;
  }

  x_input = x_input.slice(
     std::vector<size_t>{0, Kt-1, 0, 0},
     std::vector<size_t>{x_input.shape()[0], T, x_input.shape()[2], x_input.shape()[3]}
  );

  Tensor wt, bt, x_conv, filter, conv_error;
  // ACT FUNC DEPENDENT CODE:
  string s_wt_token = scope + "=wt[" + act_func + "]";
  string s_bt_token = scope + "=bt[" + act_func + "]";

  filter_shape = {Kt, 1, c_in, c_out * (act_func=="GLU"? 2 : 1)};
  Tensor output = ipu.getVariable(g, "output", out_shape(src.shape(), filter_shape));
  Sequence bwd_conv;
  x_conv = conv2d(ipu, g, src, filter_shape, s_wt_token, s_bt_token, output, tconv_layer, true);

  Sequence bwd_activation;
  if (act_func == "GLU") {
    vector<size_t> x_shape = x_conv.shape();
    x_shape[3] = c_out;
    Tensor sig = ipu.getAlternatingSpace("layer", x_shape);
    tconv_layer.add(Copy(x_conv.slice(std::vector<std::size_t>{0, 0, 0, x_conv.shape()[3]-c_out}, x_conv.shape()), sig));
    nonLinearityInPlace(g, popnn::NonLinearityType::SIGMOID, sig, tconv_layer);
    Tensor x_con_slice = x_conv.slice(vector<size_t>{0, 0, 0, 0},
                                      vector<size_t>{x_conv.shape()[0],
                                                     x_conv.shape()[1],
                                                     x_conv.shape()[2],
                                                     c_out});
    popops::addInPlace(g, x_con_slice, x_input, tconv_layer);
    popops::mulInPlace(g, x_con_slice, sig, tconv_layer);
    tconv_layer.add(Copy(x_con_slice, dst));

    // Backwards
    bwd_activation.add(Copy(dst, x_con_slice));
      // verify this maybe
    Tensor conv_error = popops::mul(g, x_con_slice, sig, bwd_activation);

      // slice error propagate
    popops::subInPlace(g, conv_error, x_input, bwd_activation);

  } else if (act_func=="linear") {
    // nothing
    tconv_layer.add(Copy(x_conv, dst));

    bwd_activation.add(Copy(dst, x_conv));
    conv_error = x_conv;
  } else if (act_func=="sigmoid") {
    popops::sigmoidInPlace(g, x_conv, tconv_layer);
    tconv_layer.add(Copy(x_conv, dst));

    // Backwards pass
    bwd_activation.add(Copy(dst, x_conv));
    Tensor ones = ipu.getVariable(g, "sig."+scope, x_conv.shape());
    popops::fill<float>(g, ones, bwd_activation, 1);
    popops::sigmoidInPlace(g, x_conv, bwd_activation);
    popops::subInPlace(g, ones, x_conv, bwd_activation);
    popops::mulInPlace(g, x_conv, ones, bwd_activation);
    conv_error = x_conv;

  } else if (act_func=="relu") {
    std::cout << "Employing RELU at " << scope << '\n';

    popops::addInPlace(g, x_conv, x_input, tconv_layer);
    popnn::reluInPlace(g, x_conv, tconv_layer);
    tconv_layer.add(Copy(x_conv, dst));

    // Backward over RELU
      // -- x_conv holds the error
    bwd_activation.add(Copy(dst, x_conv));
    popnn::reluInPlace(g, x_conv, bwd_activation); // set to zero if less than <=0
    popops::divInPlace(g, x_conv, dst, bwd_activation); // now they should all be 0s or 1s
    popops::subInPlace(g, x_conv, x_input, bwd_activation);
    conv_error = x_conv;

  } else {
    cout << "ERROR: Invalid Activation function:  << " << act_func << "\n";
  }



  Sequence bwd;
  bwd.add(bwd_activation);
  bwd.add(bwd_conv);


  layer_exit_note(ipu, "TEMPORAL_CONV_LAYER", scope);

  seq.add(tconv_layer);
  return dst;
}
Tensor spatio_conv_layer  (IPU_Interface &ipu, Graph &g, Tensor &src, Tensor &dst, size_t Ks, size_t c_in, size_t c_out, Sequence &seq, string scope) {
  layer_entry_note(ipu, "SPATIO_CONV_LAYER", "", " Funnel:: {" + to_string(c_in) + "->" + to_string(c_out) + "}");
  ipu.shape_display(src, "x", "\n");


  std::vector<std::size_t> shape = src.shape();
  size_t T = shape[1];
  size_t n = shape[2];

  Sequence spatio_conv_layer(ipu.notification(g, "Spatio Temporal Layer {"+scope+"}"));

  Tensor x_input;

  // PREPARE x_INPUT
  if (c_in > c_out) {
    // BOTTLENECK DOWN SAMPLING

    Tensor output = ipu.getAlternatingSpace("tmp_output", out_shape(src.shape(), {1, 1,  c_in,  c_out}));
    x_input = conv2d(ipu, g, src, std::vector<std::size_t>{1, 1,  c_in,  c_out},
                    scope + "=ws_in", "N/A", output,
                    spatio_conv_layer
                    );

  } else if (c_in < c_out) {
    // PADDING
    Tensor padding = ipu.getExistingVariableSlice("PAD", {shape[0], T, n,  (c_out-c_in)});
    x_input = poplar::concat(src, padding, 3);
  } else/* if (c_in == c_out)*/ {
    // just use x;
    x_input = src;
  }



  FileFedVector _ws(scope+"=ws", Ks*c_in*c_out);
  FileFedVector _bs(scope+"=bs", c_out);

  // add ws to collection as weight_decay
  // "variable_summaries(ws, 'theta')"
  Tensor ws = ipu.getVariable(g, "ws", { (Ks * c_in),  (c_out)}, 4, _ws.permanent_pointer); // GLOROT
  Tensor bs = ipu.getVariable(g, "bs", { c_out}, 4, _bs.permanent_pointer);

  Tensor gconv_out = ipu.getVariable(g, "gconv_out", {shape[0]*shape[1], n,  c_out});
  Tensor re_x = src.reshape(vector<size_t>{minus_one_val(src.shape(), vector<size_t>{n, c_in}),
                             n,
                             c_in});

  Sequence bwd_gconv;
  gconv_out = gconv(ipu, g, re_x, gconv_out, ws, Ks,  c_in,  c_out, spatio_conv_layer, "ST-CONV-LAYER");
  popops::addInPlace (g, gconv_out, bs, spatio_conv_layer);



  Tensor re_gconv = gconv_out.reshape(vector<size_t>{minus_one_val(gconv_out.shape(), vector<size_t>{T, n, c_out}), T, n, c_out});
  Tensor outslice = re_gconv.slice(vector<size_t>{0, 0, 0, 0}, vector<size_t>{minus_one_val(re_gconv.shape(), vector<size_t>{T, n, c_out}), T, n,  c_out});
  popops::addInPlace(g, outslice, x_input, spatio_conv_layer);
  popnn::reluInPlace(g, outslice, spatio_conv_layer);

  spatio_conv_layer.add(Copy(outslice, dst));

  // BACKWARD PASS
  Sequence bwd;

    // backward over relu!
  popops::fill<float>(g, outslice, bwd, 1);
  popnn::reluInPlace(g, dst, bwd);
  popops::subInPlace(g, outslice, dst, bwd);
  popops::mulInPlace(g, dst, outslice, bwd);
    // Backward over addition!
  popops::subInPlace(g, dst, x_input, bwd);
  // we should not have to do anything about the reshape and slice...
  // bias (bs) adjustments
  Tensor deltaBias = popnn::pooling::pool(g, popnn::pooling::PoolParams(
    popnn::PoolingType::SUM,
    vector<size_t>{dst.shape()[2], dst.shape()[3]}, // field shape
    vector<size_t>{dst.shape()[2], dst.shape()[3]}, // kernel shape
    vector<unsigned>{1, 1},                        // stride
    vector<int>{0, 0},    // pad
    vector<int>{0, 0},    // pad
    dst.shape()[1],     // channels
    dst.shape()[0],     // batch size
    poplar::FLOAT), dst, bwd);
  popops::addInPlace(g, bs, deltaBias, bwd);
    // add the reverse convolution
    // should we address x_input???
  bwd.add(bwd_gconv);

  layer_exit_note(ipu, "SPATIO_CONV_LAYER", "");
  seq.add(spatio_conv_layer);

  return dst;
}
Tensor st_conv_block      (IPU_Interface &ipu, Graph &g, Tensor &x, size_t Ks, size_t Kt, size_t channels[3], Sequence &seq, string scope, string act_func) {
  layer_entry_note(ipu, "ST_CONV_BOCK", scope);
  ipu.shape_display(x, scope + ":x", "\n");

  size_t c_si =  channels[0];
  size_t c_t  =  channels[1];
  size_t c_oo =  channels[2];

  vector<size_t> shp = x.shape();
  Tensor temp = ipu.getVariable(g, scope + " temporal_conv_layer-BLOCK", vector<size_t>{shp[0], (shp[1]-Kt+1), shp[2], (c_t>c_oo?c_t:c_oo)});
  Tensor t_out_1 = temp.slice(vector<size_t>{0, 0, 0, 0}, vector<size_t>{temp.shape()[0], temp.shape()[1],        temp.shape()[2], c_t});
  Tensor t_out_2 = temp.slice(vector<size_t>{0, 0, 0, 0}, vector<size_t>{temp.shape()[0], temp.shape()[1]-(Kt-1), temp.shape()[2], c_oo});
  Tensor sout = ipu.getVariable(g, scope + " spatio_conv_layer.OUT", t_out_1.shape());

  Sequence exe_st_conv_block;

  // exe_st_conv_block.add(ipu.notification(g, "ST-CONV-BLOCK ["+scope+"]{"+to_string(channels[0])+","+to_string(channels[1])+","+to_string(channels[2])+"}"));
  Sequence bwd_t1, bwd_st, bwd_t2, bwd_ln;


  t_out_1 = temporal_conv_layer(ipu, g, x,    t_out_1, Kt, c_si, c_t, exe_st_conv_block, "stn_block_" + scope + "_in", act_func);
  sout    = spatio_conv_layer  (ipu, g, t_out_1, sout, Ks, c_t, c_t,  exe_st_conv_block, "stn_block_" + scope + "_spt");
  t_out_2 = temporal_conv_layer(ipu, g, sout, t_out_2, Kt, c_t, c_oo, exe_st_conv_block, "stn_block_" + scope + "_out");
  t_out_2 = layer_norm         (ipu, g, t_out_2, t_out_2,             exe_st_conv_block, "layer_norm_" + scope);
  // verification_pass(ipu, g, t_out_2, scope+"[predrop]", exe_st_conv_block);



  Tensor drop = poprand::shapedDropout(g,    // graph
                                  NULL, // Seed, make random?
                                  0, // Seed modifier?
                                  t_out_2,    // input
                                  t_out_2,    // tilemapping refrence
                                  0.1, // double with one value: 0.1?
                                  (double) 10,//Scale:: typically inverse dropout probability???, keep_prob=0.1?, 1/0.1, = 10
                                  exe_st_conv_block
                                );
  ipu.addVariable("St-Conv: OUT", drop);

  #ifdef _DERANDOMIZE
    cout << "\t\tDerandomized postdrop " << scope << "\n";
    Tensor corr = derandomize(ipu, g, drop, scope+"[postdrop]");
    exe_st_conv_block.add(Copy(corr, drop));
  #endif

  layer_exit_note(ipu, "ST_CONV_BOCK", scope);
  seq.add(exe_st_conv_block);
  return drop;
}
Tensor fully_con_layer    (IPU_Interface &ipu, Graph &g, Tensor &x, Tensor &out, size_t n, size_t channel, Sequence &seq, string scope) {
  layer_entry_note(ipu, "FULLY_CON_LAYER", scope);

  vector<size_t> kernel_shape = {1, 1, channel, 1};

  Sequence full_con_layer;

  out = conv2d_SAME(ipu, g, x, kernel_shape, scope + "=w", "", out, full_con_layer, false);

  // verification_pass(ipu, g, out, scope + "=FC", full_con_layer);
  // verified to accuracy range = {-6.55651e-07, 8.34465e-07}

  FileFedVector bias_feed(scope+"=b", n);
  Tensor bias = ipu.getVariable(g, scope+"=b", {n, 1}, 4, bias_feed.permanent_pointer);

  // verified to last digit rounding errors

  popops::addInPlace(g, out, bias, full_con_layer);

  layer_exit_note(ipu, "FULLY_CON_LAYER", scope);
  seq.add(full_con_layer);
  return out;
}
Tensor output_layer       (IPU_Interface &ipu, Graph &g, Tensor &x, Tensor &out, size_t T, Sequence &seq, string scope, string act_func) {
  layer_entry_note(ipu, "OUTPUT_LAYER", scope);

  vector<size_t> shp{x.shape()[0], x.shape()[1]-T+1, x.shape()[2], x.shape()[3]};
  Tensor x1 = x.slice({0, 0, 0, 0}, shp);
  Tensor x2 = ipu.getVariable(g, scope + " temporal_conv_layer.X2", shp);

  size_t n       = shp[2];
  size_t channel = shp[3];

  Sequence bwd_t1, bwd_ln, bwd_t2, bwd_fc;


  x2  = temporal_conv_layer(ipu, g, x,  x2, T, channel, channel, seq, scope+"_in", act_func);

  x1  = layer_norm         (ipu, g, x2, x1, seq, "layer_norm_" + scope);

  x2  = temporal_conv_layer(ipu, g, x1, x2, 1, channel, channel, seq, scope+"_out", "sigmoid");

  out = fully_con_layer    (ipu, g, x2, out, n, channel, seq, scope);

  bwd.add(bwd_fc); bwd.add(bwd_x2); bwd.add(bwd_ln); bwd.add(bwd_x1);


  layer_exit_note(ipu, "OUTPUT_LAYER", scope);
  return out;
}


#endif
