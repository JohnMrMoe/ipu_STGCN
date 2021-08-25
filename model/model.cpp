
#include "model_assets.hpp"

#include <popnn/Loss.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Program.hpp>
#include <popops/Reduce.hpp>
#include <popops/Expr.hpp>
#include <popops/Operation.hpp>
#include <popops/ElementWise.hpp>

using namespace poplar;
using namespace poplar::program;

#define B_OUT 2
#define B_IN 3

#define min(y, x) (y < x ? y : x)

tuple<Program, Program, Tensor> build_model(size_t blocks[B_OUT][B_IN], Arguments args, IPU_Interface &ipu, Graph &g) {
  // Tensor &input, size_t n_his, size_t Ks, size_t Kt, size_t blocks[B_OUT][B_IN], IPU_Interface &ipu, Graph &g)  {
  std::vector<size_t> shape {args.batch_size,
                               args.n_his + 1,
                               args.n_route,
                               1};
  std::vector<std::size_t> shape_end = shape;
  shape_end[1] = args.n_his;

  Tensor input = ipu.getVariable(g, (string) "input",     shape, INPUT_TENSOR); // doesn't really matter!
  Tensor kb    = ipu.getVariable(g, (string) "keep_prob", {1},   0);

  // we use two other places too
  Tensor two = (Tensor) g.addConstant<float>(FLOAT, {1}, {2.0});
  g.setTileMapping(two, 0);
  ipu.addVariable("two", two);


  Sequence sq(ipu.notification(g, "VERIFICATION NOTIFICATION DELIBERATION " + ipu.shape_display(shape, "") + ", " + to_string(args.n_his)));

  Tensor x = input.slice({0,0,0,0},       //{0,                     0,        0,        0},
                         shape_end);       //{shape[0], (size_t) n_his, shape[2], shape[3]});


  size_t largest_channel = 0;
  for (size_t i = 0; i < B_OUT; i++) for (size_t j = 0; j < B_IN; j++) {
    largest_channel = largest_channel > blocks[i][j] ? largest_channel : blocks[i][j];
  }

  vector<size_t> tmp_sections{shape[0], args.n_his, shape[2], largest_channel};
  Tensor block_t1 = ipu.getVariable(g, "layer0", tmp_sections);
  Tensor block_t2 = ipu.getVariable(g, "layer1", tmp_sections);

  Tensor next_in = x;
  Sequence model(sq);
  vector<Sequence> bwd;
  size_t Ko = args.n_his;

  // ST BLOCK:
  for (size_t i = 0; i < 2; i++) {
    string scope = "st-conv[" + to_string(i) + "]";
    Sequence bwd_stcb;
    next_in = st_conv_block(ipu, g, next_in, args.ks, args.kt, blocks[i], model, bwd_stcb, to_string(i), "GLU");
    bwd.push_back(bwd_stcb);
    Ko-= 2 * (args.ks - 1);
  }
  if (Ko <= 1) {// output layer
    printf("ERROR: Kernel size Ko must be greater than 1, but recieved %ld\n", Ko);
    return tuple<Program, Program, Tensor>{model, model, next_in};
  }
  // y = [batch_size, 1, n_route, 1]
  Tensor y = ipu.getVariable(g, "-Y-block", {next_in.shape()[0], 1, next_in.shape()[2], 1}, 0);
  Sequence bwd_oula;
  y = output_layer(ipu, g, next_in, y, Ko, model, bwd_oula, "output_layer");
  bwd.push_back(bwd_oula);



  Sequence loss_prog;

  Tensor single_pred = ipu.getVariable(g, "y_pred", {y.shape()[0], 1, y.shape()[2], y.shape()[3]}, 0);
  Tensor copy_loss  = ipu.getVariable(g, "copy_loss", {input.shape()[0], 1, input.shape()[2], input.shape()[3]}, 0);
  Tensor train_loss = ipu.getVariable(g, "train_loss", y.shape(), 0);

  model.add(Copy(y.slice({0, 0, 0, 0}, {y.shape()[0], 1, y.shape()[2], y.shape()[3]}), single_pred));

  loss_prog.add(Copy(input.slice({0, args.n_his - 1, 0, 0}, {input.shape()[0], args.n_his, input.shape()[2], input.shape()[3]}), copy_loss));
  loss_prog.add(Copy(y, train_loss));

  popops::minInPlace(g, copy_loss , input.slice({0, args.n_his, 0, 0}, {input.shape()[0], args.n_his+1, input.shape()[2], input.shape()[3]}), loss_prog);
  popops::minInPlace(g, train_loss, input.slice({0, args.n_his, 0, 0}, {input.shape()[0], args.n_his+1, input.shape()[2], input.shape()[3]}), loss_prog);

  // L2 Loss:
  popops::squareInPlace(g, copy_loss, loss_prog);
  popops::squareInPlace(g, train_loss, loss_prog);
  popops::divInPlace(g, copy_loss,  two, loss_prog);
  popops::divInPlace(g, train_loss, two, loss_prog);

  loss_prog.add(Copy(train_loss, y));

  Sequence bwd_pass(loss_prog);
  for (int elem = bwd.size() - 1; elem>=0; elem--) {
    bwd_pass.add(bwd[elem]);
  }

  bwd_pass.add(loss_prog);

  return tuple<Program, Program, Tensor>(model, bwd_pass, y);
  // return pair<Program, Program>(model, bwd_pass);
  // return model;
}
