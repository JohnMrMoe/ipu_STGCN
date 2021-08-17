#include <vector>
#include <string.h>
#include <stdio.h>

#include "data/csv_reader.hpp"
#include "util/arguments.hpp"
#include "util/Matrices.cpp"
#include "util/util.hpp"
#include "util/ipu_interface.hpp"
#include "time/time_run.cpp"

#include "data/data_utility.hpp"

#include "model/model_assets.hpp"


#include <poplar/DeviceManager.hpp>

#include <poplar/Engine.hpp>


using namespace std;
using namespace poplar;

int calculate_Lk(Arguments args, Matrix_2D Lk, bool no_write);
int read_Lk(Arguments args, Matrix_2D Lk);

int main(int argc, char const *argv[]) {
  Arguments args(argc, argv);
  //Collection cl();

  //bottleneck design  --
  size_t blocks [2][3] = {{1, 32, 64}, {64, 32, 128}};

  vector<size_t > Lk_shape = { args.n_route,
                              (args.n_route*args.ks) };
  size_t Lk_size = Lk_shape[0] * Lk_shape[1];
  Matrix_2D Lk(Lk_size, Lk_shape);

  int status;
  #ifdef _QUICK_READ
    status = read_Lk(args, Lk);
  #else
    status = calculate_Lk(args, Lk, true);
  #endif
  if (status) return 0;


  string data_file = "./data/dataset/V_" + to_string(args.n_route) + ".csv";
  std::cout << "Started reading " << data_file << "\n";;
  CSVReader reader(data_file, ",");
  reader.read();
  Matrix_2D v_set(reader.outer()*reader.inner(),
                  std::vector<size_t >{reader.outer(),
                                            reader.inner()},
                  "Raw");

  for (size_t  i = 0; i < reader.outer(); i++) {
    for (size_t  j = 0; j < reader.inner(); j++) {
      v_set[i][j] = stof(reader.raw[i][j]);
    }
  }
  std::cout << "\nFinished reading " << data_file << '\n';


  size_t n_frame = args.n_his + args.n_pred;
  int n_slot = args.d_slot - n_frame + 1;

  //int config[3] = {34, 5, 5};
  // int inner_shape[3] = {args.n_his + args.n_pred, args.n_route, args.c_w};

  int config[3] = {34, 5, 5};
  vector<vector<size_t >> shapes = {
      { (config[0] *(args.d_slot-n_frame+1)),
        n_frame,  args.n_route,  1},
      { (config[1] *(args.d_slot-n_frame+1)),
        n_frame,  args.n_route,  1},
      { (config[2] *(args.d_slot-n_frame+1)),
        n_frame,  args.n_route,  1}};

  Dataset dataset(v_set,
                  config,
                  (size_t) args.n_route,
                  n_frame,
                  (size_t) args.d_slot,
                  shapes
                );

  std::cout << "Sizeof(Training)   [B]:    " << (dataset.train.size * sizeof(float)) << '\n';
  std::cout << "Sizeof(Validation) [B]:  " << (dataset.valid.size * sizeof(float)) << '\n';
  std::cout << "Sizeof(Testing)    [B]:     " << (dataset.test.size * sizeof(float)) << '\n' << '\n';

  // CREATE
  IPU_Interface ipu;
  // ADD ANY CODELETS:
  // graph.addCodelets("matrix-mul-codelets.cpp");

  // add the Lk to the graph
  Graph graph(ipu.target);

  ipu.getVariable(graph, "NULL", NULL_VECTOR, NULL_TENSOR);
  Tensor p = graph.addConstant<float>(poplar::FLOAT, {1}, {1});
  graph.setTileMapping(p, 0);
  ipu.addVariable("pluss", p);
  FileFedVector ffv("Kernel", Lk_size);
  Tensor graph_kernel = ipu.getVariable(graph,
                                        "graph_kernel",
                                        std::vector<std::size_t>{Lk.outer, Lk.inner},
                                        FEEDIN_TENSOR,
                                        ffv.permanent_pointer //Lk.data
                                       );


  // zero_tensor padding
  ipu.getVariable(graph,
                  "PAD",
                  vector<size_t> {args.batch_size,
                                  args.n_his + 1,
                                  args.n_route,
                                  128-1},
                  0); // magic number, highest in the blocks - 1


  //poputil  ::addCodelets(g);
  popops   ::addCodelets(graph);
  poplin   ::addCodelets(graph);
  popsparse::addCodelets(graph);
  popnn    ::addCodelets(graph);
  poprand  ::addCodelets(graph);

  //  WITH THIS IT RUNS EFFORTLESSLY!
  // Sequence seq;
  // ipu.finalize_and_run(graph, seq);
  //
  // return 0;

  tuple<Program, Tensor, Tensor> assets = model_train(dataset, blocks, args, ipu, graph);
  Program model     = get<0>(assets);
  Tensor train_loss = get<1>(assets);
  Tensor pred_y     = get<2>(assets);

  Engine engine = ipu.finalize_and_run(graph, model, false);

  double ms = time_run(engine, "STGCN");

  size_t runs = 10; // should be an odd number
  if (runs%2==0) runs++;

  double * times = (double *) malloc(sizeof(double) * runs);
  double total;

  string timeunit = "ms";
  std::cout << "Preliminary Run timed at " << ms << " " << timeunit << '\n';
  // cout << "Press Enter to Continue";
  // cin.ignore();
  // cout << "Executing " << runs << " runs.\n";

  // for (size_t i = 0; i < runs; i++) {
  //   times[i] = time_run(engine, "STGCN #" + to_string(i));
  //   if (i%20==0) std::cout << "Run: " << (i<100?" ":"") << (i<10?" ":"") << i << '\n';
  //   total += times[i];
  // }
  //
  // size_t step = 5;
  // cout << " +-----------------------------------------------------------\n";
  // cout << " | TIMES:\n";
  // cout << " |\tListing:\n";
  // for (size_t i = 0; i < runs; i+=step) {
  //   if (i+step>runs) {
  //     cout << " |\t\t";
  //     for (i=i; i < runs; i++) std::cout << times[i] << (i==runs-1?"\n":", ");
  //   } else {
  //     cout << " |\t\t" << times[i] << ", " << times[i+1] << ", " << times[i+2] << ", " << times[i+3] << ", " << times[i+4] << "\n";
  //   }
  // }
  //
  // sort(times, times+runs);
  //
  // cout << " |\tMedian:  " << times[(runs - (runs%2))/2] << timeunit << "\n";
  // cout << " |\tAverage: " << total/runs << timeunit << "\n";
  // cout << " |\tMinimum: " << times[0] << timeunit << "\n";
  // cout << " |\tMaximum: " << times[runs-1] << timeunit << "\n";
  // cout << " +-----------------------------------------------------------\n";



  free(times);

  return 0;
}


int read_Lk(Arguments args, Matrix_2D Lk) {
  CSVReader reader("quick.csv", ",");
  printf("Started read of Lk.\n");
  reader.read();
  for (size_t  i = 0; i < reader.outer(); i++) {
    for (size_t  j = 0; j < reader.inner(); j++) {
      Lk[i][j] = stof(reader.raw[i][j]);
    }
  }
  printf("Completed read of Lk.\n");
  return 0;
}
int calculate_Lk(Arguments args, Matrix_2D Lk, bool no_write) {
  Matrix_2D data(args.n_route*args.n_route,
                 vector<size_t >{ args.n_route,
                                  args.n_route});

  printf("Calculating LK\n");
  char failure = 2;
  string graphkey;
  if (args.graph.compare("default")==0) {
    graphkey = "./data/dataset/W_" + to_string(args.n_route) + ".csv";
  } else {
    graphkey = args.graph;
  }

  failure = read_csv(graphkey, data);
  if (failure) {
    std::cout << "Failed to read file \'"<<graphkey<<"\', exiting.\n";
    return -1;
  }

  weight_matrix(data);

  Matrix_2D l(args.n_route*args.n_route,
              vector<size_t >{ args.n_route,
                                    args.n_route});

  scaled_laplacian(l, data);
  printf("Laplacian Calculated.\n");

  // PROGRESS

  if (args.ks<1) {
    printf("You dummy, the Spatial Kernel was given an invalid size.\n");
    return 1;
  } else if (args.ks==1) {
    memcpy(Lk.data, l.data, l.size*sizeof(float));
  } else if (args.ks>1) {
    cheb_poly_approx(l, args.ks, args.n_route, Lk);
  }

  printf("Chebyshev Polynomials Approximated\n");
  // LK here

  if (no_write) return 0;
  fopen("quick.csv", "w");
  for (size_t i = 0; i < Lk.outer; i++) {
    for (size_t j = 0; j < Lk.inner; j++) {
      printf("%s%lf", j==0?"":",", Lk[i][j]);
    }
    printf("\n");
  }

  return 0;
}
