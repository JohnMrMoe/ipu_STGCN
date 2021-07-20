
#include <vector>
#include <string.h>
#include <stdio.h>
#include <tuple>

#include "../util/ipu_interface.hpp"
#include "../util/arguments.hpp"
#include "../data/data_utility.hpp"
#include "model_assets.hpp"

#include <poplar/Tensor.hpp>
#include <poplar/Program.hpp>



 tuple<Program, Tensor, Tensor> model_train(Dataset &input, size_t blocks[2][3], Arguments args, IPU_Interface &ipu, Graph &g, string path) {
  std::vector<size_t> shape_x {args.batch_size,
                               args.n_his + 1,
                               args.n_route,
                               1};
  std::vector<size_t> shape_keep_prob{1};

  FileFedVector ff_in("input", shape_x[0]*shape_x[1]*shape_x[2]);
  Tensor x         = ipu.getVariable(g, (string) "data_input", shape_x, 4, ff_in.permanent_pointer);
  //Tensor x         = ipu.getVariable(g, (string) "data_input", shape_x, input.raw

  Tensor keep_prob = ipu.getVariable(g, (string) "keep_prob", shape_keep_prob);

  Sequence model;
  model.add(build_model(x, args.n_his, args.ks, args.kt, blocks, ipu, g));


  Tensor train_loss = ipu.getVariable_OLD(g, "copy_loss");
  Tensor pred = ipu.getVariable_OLD(g, "y_pred");

  ipu.shape_display(train_loss.shape(), "train_loss");
  ipu.shape_display(pred.shape(), "pred");

  model.add(PrintTensor("Input     ", x.flatten().slice({0}, {6})));
  model.add(PrintTensor("Train_loss", train_loss.flatten().slice({0}, {6})));

  pred = pred.reshape({pred.shape()[0], pred.shape()[2]});

  std::cout << "..." << ipu.shape_display(pred, "", "") << '\n';
  model.add(PrintTensor("Prediction", pred.slice({0, 0}, {6, 6})));

  return tuple<Program, Tensor, Tensor>(model, train_loss, pred);

}
