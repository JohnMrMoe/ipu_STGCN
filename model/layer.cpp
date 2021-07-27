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
  ComputeSet cs = g.addComputeSet("transpose");
  // Tensor t_x = popops::rearrange::partialTranspose(g, x, cs);
  // gconv_layer.add(Execute(cs));
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
  std::pair<Tensor, Tensor> moments = poplin::normStatistics(g,
                                                     _NCHW,
                                                     epsilon,
                                                     norm_layer, // program
                                                     true //unbiasedVarEstimate
                                                   );

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

  Tensor gamma = ipu.getVariable(g, "gamma", std::vector<size_t>{1, 1,  N,  C}, FILEFD_TENSOR, ffv_g.permanent_pointer);// TODO ASSERT CORRECT FILL?
  Tensor beta  = ipu.getVariable(g, "beta",  std::vector<size_t>{1, 1,  N,  C}, FILEFD_TENSOR, ffv_b.permanent_pointer);// TODO ASSERT CORRECT FILL?
  Tensor _1eminsix = g.addConstant<float>(poplar::FLOAT, {1}, {0.000001});
  g.setTileMapping(_1eminsix, 0);
  ipu.addVariable("_1eminsix", _1eminsix);

  //Tensor norm = popops::sub(g, x, mu, norm_layer);
  Tensor norm = xmmu;
  ipu.addVariable(scope + "_norm", norm);

  // // Original interpreatation:
  popops::addInPlace (g, sigma, _1eminsix, norm_layer);
  popops::sqrtInPlace(g, sigma, norm_layer);

  popops::divInPlace(g, norm, sigma, norm_layer); // -- Problem
  popops::mulInPlace(g, norm, gamma, norm_layer);
  popops::addInPlace(g, norm, beta, norm_layer);

  norm_layer.add(Copy(norm, dst));

  layer_exit_note(ipu, "LAYER_NORM", scope);
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

  if (c_in > c_out) {
    vector<size_t> kernel_shape = {1, 1,  c_in,  c_out};
    Tensor feed = ipu.getAlternatingSpace("layer", out_shape(src.shape(), kernel_shape));
    x_input = conv2d_SAME(ipu, g, src, kernel_shape, scope + "=w_input", "N/A", feed, tconv_layer);
  } else if (c_in < c_out) {
    // PADDING
    Tensor padding = ipu.getExistingVariableSlice("PAD", {shape[0], T, n,  (c_out-c_in)}, {0, 0, 0, 0}, 0);
    x_input = poplar::concat(src, padding, 3);
  } else { /*if (c_in == c_out)*/
    x_input = src;
  }

  x_input = x_input.slice(
     std::vector<size_t>{0, Kt-1, 0, 0},
     std::vector<size_t>{x_input.shape()[0], T, x_input.shape()[2], x_input.shape()[3]}
  );

  verification_pass(ipu, g, x_input, scope + "[x_i]["+act_func+"]", tconv_layer);

  Tensor wt, bt, x_conv, filter;
  // ACT FUNC DEPENDENT CODE:
  string s_wt_token = scope + "=wt[" + act_func + "]";
  string s_bt_token = scope + "=bt[" + act_func + "]";


  if (act_func == "GLU") {

    filter_shape = {Kt, 1, c_in, 2*c_out};

    Tensor output = ipu.getVariable(g, "output", out_shape(src.shape(), filter_shape));



    Tensor x_conv = conv2d(ipu, g, src, filter_shape, s_wt_token, s_bt_token, output, tconv_layer, true);


    verification_pass(ipu, g, x_conv, scope + "[x_conv]", tconv_layer);

    // bt = ipu.getVariable(g, "bt", std::vector<std::size_t>{2*c_out}, 0);
    // popops::addInPlace (g, x_conv, bt, tconv_layer);

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

  } else {
    filter_shape = {Kt, 1, c_in, c_out};


    vector<size_t> shape_out = out_shape(src.shape(), filter_shape);

    Tensor x_conv = ipu.getVariable(g, "output", shape_out);

    x_conv = conv2d(ipu, g, src, filter_shape, s_wt_token, s_bt_token, x_conv, tconv_layer, true);

    //
    // bt = ipu.getVariable(g, "bt", std::vector<std::size_t>{c_out}, 0);
    //
    // popops::addInPlace (g, x_conv, bt, tconv_layer);

    if (act_func=="linear") {
      // nothing
    } else if (act_func=="sigmoid") {
      popops::sigmoidInPlace(g, x_conv, tconv_layer);
    } else if (act_func=="relu") {
      popops::addInPlace(g, x_conv, x_input, tconv_layer);
      popnn::reluInPlace(g, x_conv, tconv_layer);
    } else {
      cout << "ERROR: Invalid Activation function:  <<" << act_func << "\n";
    }

    // copy solution to destination.
    tconv_layer.add(Copy(x_conv, dst));
  }
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

  verification_pass(ipu, g, src, scope + "[SIN]", spatio_conv_layer);

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

  gconv_out = gconv(ipu, g, re_x, gconv_out, ws, Ks,  c_in,  c_out, spatio_conv_layer, "ST-CONV-LAYER");
  popops::addInPlace (g, gconv_out, bs, spatio_conv_layer);



  Tensor re_gconv = gconv_out.reshape(vector<size_t>{
                                    minus_one_val(gconv_out.shape(), vector<size_t>{T, n, c_out}),
                                    T, n, c_out});


  Tensor outslice = re_gconv.slice( vector<size_t>{       0, 0, 0,     0},
                                    vector<size_t>{minus_one_val(re_gconv.shape(), vector<size_t>{T, n, c_out}),
                                                   T, n,  c_out});
  popops::addInPlace(g, outslice, x_input, spatio_conv_layer);


  popnn::reluInPlace(g, outslice, spatio_conv_layer);

  spatio_conv_layer.add(Copy(outslice, dst));

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

  exe_st_conv_block.add(ipu.notification(g, "ST-CONV-BLOCK ["+scope+"]{"+to_string(channels[0])+","+to_string(channels[1])+","+to_string(channels[2])+"}"));

  t_out_1 = temporal_conv_layer(ipu, g, x,    t_out_1, Kt, c_si, c_t, exe_st_conv_block, "stn_block_" + scope + "_in", act_func);
  sout    = spatio_conv_layer  (ipu, g, t_out_1, sout, Ks, c_t, c_t,  exe_st_conv_block, "stn_block_" + scope + "_spt");
  t_out_2 = temporal_conv_layer(ipu, g, sout, t_out_2, Kt, c_t, c_oo, exe_st_conv_block, "stn_block_" + scope + "_out");


  t_out_2 = layer_norm         (ipu, g, t_out_2, t_out_2,             exe_st_conv_block, "layer_norm_" + scope);
  verification_pass(ipu, g, t_out_2, scope+"[predrop]", exe_st_conv_block);



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

  Sequence full_con_layer(ipu.notification(g, "Fully Con Layer {"+scope+"}"));

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




  verification_pass(ipu, g, x, scope+"[F-T0]", seq);
  x2  = temporal_conv_layer(ipu, g, x,  x2, T, channel, channel, seq, scope+"_in", act_func);
  verification_pass(ipu, g, x2, scope+"[F-T1]", seq);

  x1  = layer_norm         (ipu, g, x2, x1, seq, "layer_norm_" + scope);
  // verification_pass(ipu, g, x1, scope+"[F-LN]", seq);

  x2  = temporal_conv_layer(ipu, g, x1, x2, 1, channel, channel, seq, scope+"_out", "sigmoid");
  // verification_pass(ipu, g, x2, scope+"[F-T2]", seq);

  out = fully_con_layer    (ipu, g, x2, out, n, channel, seq, scope);
  // verification_pass(ipu, g, out, scope+"[F-OUT]", seq);

  layer_exit_note(ipu, "OUTPUT_LAYER", scope);
  return out;
}


#endif
