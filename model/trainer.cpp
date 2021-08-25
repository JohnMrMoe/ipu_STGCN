//
// #include <vector>
// #include <string.h>
// #include <stdio.h>
// #include <tuple>
//
// #include "../util/ipu_interface.hpp"
// #include "../util/arguments.hpp"
// #include "../data/data_utility.hpp"
// #include "model_assets.hpp"
//
// #include <poplar/Tensor.hpp>
// #include <poplar/Program.hpp>
//
//
//
//  tuple<Program, Program, Tensor, Tensor> model_train(Dataset &input, size_t blocks[2][3], Arguments args, IPU_Interface &ipu, Graph &g, string path) {
//   std::vector<size_t> shape_x {args.batch_size,
//                                args.n_his + 1,
//                                args.n_route,
//                                1};
//   std::vector<size_t> shape_keep_prob{1};
//
//   FileFedVector ff_in("input", shape_x[0]*shape_x[1]*shape_x[2]);
//   Tensor x         = ipu.getVariable(g, (string) "data_input", shape_x, 4, ff_in.permanent_pointer);
//   //Tensor x         = ipu.getVariable(g, (string) "data_input", shape_x, input.raw
//
//   Tensor keep_prob = ipu.getVariable(g, (string) "keep_prob", shape_keep_prob);
//
//   tuple<Program, Program, Program> progset = build_model(x, args.n_his, args.ks, args.kt, blocks, ipu, g);
//
//   Program model =      get<0>(progset);
//   Program bwd =        get<1>(progset);
//   Program calc_error = get<2>(progset);
//
//
//   return tuple<Program, Program, Tensor, Tensor>(model, Sequence(calc_error, bwd), keep_prob, keep_prob);
//
// }
