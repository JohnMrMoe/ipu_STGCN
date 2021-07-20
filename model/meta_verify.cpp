#ifndef __META_VERIFICATION
#define __META_VERIFICATION

#include <poplar/Tensor.hpp>
#include <poplar/Program.hpp>
#include <popops/Reduce.hpp>
#include <popops/Expr.hpp>
#include <popops/Operation.hpp>
#include <popops/ElementWise.hpp>

#include <string>

#include "../util/ipu_interface.hpp"
#include "model_assets.hpp"


using namespace poplar;
using namespace poplar::program;
using namespace std;



Tensor verify_same_2(Graph &g, Tensor &res, Tensor &inp, Sequence &seq, string label, bool correct) {
  Tensor delta = popops::sub(g, res, inp, seq);
  Tensor delta_min = popops::reduce(g, delta, vector<size_t>{0, 1, 2, 3}, popops::ReduceParams(popops::Operation::MIN), seq);
  Tensor delta_max = popops::reduce(g, delta, vector<size_t>{0, 1, 2, 3}, popops::ReduceParams(popops::Operation::MAX), seq);
  seq.add(PrintTensor("\t\t" + label + " verified abs.MIN(DELTA)", delta_min));
  seq.add(PrintTensor("\t\t" + label + " verified abs.MAX(DELTA)", delta_max));

  for (size_t i = 0; i < delta.shape().size(); i++) {
    #ifndef _DEEP_VERIFICATION
      continue;
    #endif
    if (delta.shape()[i]==1) continue;
    string reducS = "";
    size_t reducT = 1;
    vector<size_t> reducV(0, 0);
    for (size_t j = 0; j < delta.shape().size(); j++) {
      if (i!=j) {
        reducV.push_back(j);
        reducS += (reducS.size()==0?"":",") + to_string(j);
      } else {
        reducT *= delta.shape()[j];
      }
    }

    Tensor reduc = popops::reduce(g, delta, reducV, popops::ReduceParams(popops::Operation::MIN), seq );

    if (reduc.shape()[0]==50)  reduc = reduc.reshape({ 5, 10});
    if (reduc.shape()[0]==128) reduc = reduc.reshape({ 8, 16});
    if (reduc.shape()[0]==228) reduc = reduc.reshape({19, 12});

    seq.add(PrintTensor("\t\t\t" + label + " verified MIN("+reducS+")." + to_string(reducT) + ":" + to_string(reduc.numElements()), reduc));

    reduc = popops::reduce(g, delta, reducV, popops::ReduceParams(popops::Operation::MAX), seq );

    if (reduc.shape()[0]==50)  reduc = reduc.reshape({ 5, 10});
    if (reduc.shape()[0]==128) reduc = reduc.reshape({ 8, 16});
    if (reduc.shape()[0]==228) reduc = reduc.reshape({19, 12});

    seq.add(PrintTensor("\t\t\t" + label + " verified MAX("+reducS+")." + to_string(reducT) + ":" + to_string(reduc.numElements()), reduc));
  }

  if (correct) {
    printf("\t### Replaced with Correction!\n");
    return inp;
  } else {
    return res;
  }
}
Tensor verify_same(IPU_Interface &ipu, Graph &g, Tensor &res, string file, Sequence &seq, bool correct) {
  FileFedVector ffv(file, res.numElements());
  Tensor cmp = ipu.getVariable(g, file, res.shape(), 4, ffv.permanent_pointer);
  g.setTileMapping(cmp, g.getTileMapping(res));
  return verify_same_2(g, res, cmp, seq, file, correct);
}
Tensor verification_pass(IPU_Interface &ipu, Graph &g, Tensor &inp, string file, Sequence &seq) {
  #ifdef _VERIFICATION
    cout << "\t\tVerifying: " << file << ipu.shape_display(inp.shape(), " ", "") << endl;
    seq.add(ipu.notification(g, "VERIFICATION " + file + ipu.shape_display(inp, "", "") + ":"));
    return verify_same(ipu, g, inp, file, seq, false);
  #else
    return inp;
  #endif
}
Tensor derandomize(IPU_Interface &ipu, Graph &g, Tensor &rnd, string file) {
  FileFedVector ffv(file, rnd.numElements());
  Tensor cmp = ipu.getVariable(g, file, rnd.shape(), 4, ffv.permanent_pointer);
  return cmp;
}

#endif
