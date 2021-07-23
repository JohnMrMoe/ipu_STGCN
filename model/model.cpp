
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

Sequence build_model(Tensor &input, size_t n_his, size_t Ks, size_t Kt, size_t blocks[B_OUT][B_IN], IPU_Interface &ipu, Graph &g)  {
  std::vector<std::size_t> shape = input.shape();
  std::vector<std::size_t> shape_end = shape;
  shape_end[1] = n_his;


  Sequence sq(ipu.notification(g, "VERIFICATION NOTIFICATION DELIBERATION " + ipu.shape_display(shape, "") + ", " + to_string(n_his)));

  Tensor x = input.slice({0,0,0,0},       //{0,                     0,        0,        0},
                         shape_end);       //{shape[0], (size_t) n_his, shape[2], shape[3]});


  size_t largest_channel = 0;
  for (size_t i = 0; i < B_OUT; i++) for (size_t j = 0; j < B_IN; j++) {
    largest_channel = largest_channel > blocks[i][j] ? largest_channel : blocks[i][j];
  }


  vector<size_t> tmp_sections = {shape[0], n_his, shape[2], largest_channel};
  Tensor block_t1 = ipu.getVariable(g, "layer0", tmp_sections);
  Tensor block_t2 = ipu.getVariable(g, "layer1", tmp_sections);
  // Tensor layer_t1 = ipu.getVariable(g, "block0", tmp_sections);
  // Tensor layer_t2 = ipu.getVariable(g, "block1", tmp_sections);

  Tensor next_in = x;

  Sequence model(sq);

  size_t Ko = n_his;

  // ST BLOCK:
  for (size_t i = 0; i < 2; i++) {

    string scope = "st-conv[" + to_string(i) + "]";
    next_in = st_conv_block(ipu, g, next_in, Ks, Kt, blocks[i], model, to_string(i), "GLU");

    Ko-= 2 * (Ks - 1);
  }

  // Output layer:
  if (Ko <= 1) {
    printf("ERROR: Kernel size Ko must be greater than 1, but recieved %ld\n", Ko);
    //???exit();
    return model;
  }
  // y = [batch_size, 1, n_route, 1]
  Tensor y = ipu.getVariable(g, "block", {next_in.shape()[0], 1, next_in.shape()[2], 1}, 0);
  y = output_layer(ipu, g, next_in, y, Ko, model, "output_layer");

  model.add(PrintTensor("Y :", y.reshape({y.shape()[0], y.shape()[2]}).slice({0, 0}, {6, 6}) ));


  Sequence loss_prog;

  Tensor single_pred = ipu.getVariable(g, "y_pred", {y.shape()[0], 1, y.shape()[2], y.shape()[3]}, 0);
  Tensor copy_loss  = ipu.getVariable(g, "copy_loss", {input.shape()[0], 1, input.shape()[2], input.shape()[3]}, 0);
  Tensor train_loss = ipu.getVariable(g, "train_loss", y.shape(), 0);

  loss_prog.add(Copy(input.slice({0, n_his - 1, 0, 0}, {input.shape()[0], n_his, input.shape()[2], input.shape()[3]}), copy_loss));
  loss_prog.add(Copy(y, train_loss));

  popops::minInPlace(g, copy_loss , input.slice({0, n_his, 0, 0}, {input.shape()[0], n_his+1, input.shape()[2], input.shape()[3]}), loss_prog);
  popops::minInPlace(g, train_loss, input.slice({0, n_his, 0, 0}, {input.shape()[0], n_his+1, input.shape()[2], input.shape()[3]}), loss_prog);

  Tensor two = (Tensor) g.addConstant<float>(FLOAT, {1}, {2.0});
  g.setTileMapping(two, 0);
  // L2 Loss:
  popops::squareInPlace(g, copy_loss, loss_prog);
  popops::squareInPlace(g, train_loss, loss_prog);
  popops::divInPlace(g, copy_loss,  two, loss_prog);
  popops::divInPlace(g, train_loss, two, loss_prog);

  model.add(Copy(y.slice({0, 0, 0, 0}, {y.shape()[0], 1, y.shape()[2], y.shape()[3]}), single_pred));

  // VERIFIVICATION
  verification_pass(ipu, g, y, "STGCN_OUT", model);

  model.add(loss_prog);

  return model;
}
