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
  tuple<Program, Program, Tensor> assets = build_model(blocks, args, ipu, graph);
  Program model   = get<0>(assets);
  Program bwd_p   = get<1>(assets);
  Tensor  product = get<2>(assets);

  vector<Program> progs{model, bwd_p};

  Engine engine = ipu.finalize(graph, progs);

  size_t batches = (dataset.train.W - (dataset.train.W % args.batch_size)) / args.batch_size;
  size_t runs = args.epoch * batches;


  double * times_run = (double *) malloc(sizeof(double) * runs);
  double * times_trn = (double *) malloc(sizeof(double) * runs);
  double total_run;
  double total_trn;

  string timeunit = "ms";

  std::vector<size_t> shape {args.batch_size,
                               args.n_his + 1,
                               args.n_route,
                               1};

  size_t plane = (args.n_his + 1) * args.n_route;
  size_t * permutation = (size_t *) malloc (sizeof(size_t) * dataset.train.W);
  float  * input_buffr = (float  *) malloc (sizeof(float)  * args.batch_size * plane);

  for (size_t i = 0; i < dataset.train.W; i++) permutation[i] = i;

  // #include <algorithm>

  cout << "Epochs: " << args.epoch << ", batches: " << batches << '\n';
  for (size_t   epoch = 0; epoch < args.epoch; epoch++) {
    cout << "\tEpoch #" << epoch <<  '\n';
    for (size_t batch = 0; batch < batches   ; batch++) {
      size_t batch_idx =   batch * args.batch_size;
      for (size_t cube= 0; cube  < args.batch_size; cube++) {
        memcpy(
          (void *) (dataset.train.data + dataset.train.ZYX * permutation [batch_idx + cube]),
          (void *) (input_buffr        + dataset.train.ZYX * cube), plane
        );
      }
      // WRITE to input!
      engine.writeTensor(StringRef("input"), input_buffr, &input_buffr[args.batch_size * plane  ]);

      // forward and backward pass
      total_run +=
        times_run[epoch * batches + batch]
          = time_run(engine, "Forward  Pass #"+to_string(epoch)+":"+to_string(batch), 0);
      total_trn +=
        times_trn[epoch * batches + batch]
          = time_run(engine, "Backward Pass #"+to_string(epoch)+":"+to_string(batch), 1);

      //
    }
  }

  sort(times_run, times_run+runs);
  sort(times_trn, times_trn+runs);
  cout << " +-----------------------------------------------------------\n";
  cout << " | TIMES:\n";
  cout << " |\tForward Pass";
  cout << " |\t\tMedian:  " << times_run[(runs - (runs%2))/2] << timeunit << "\n";
  cout << " |\t\tAverage: " << total_run/runs << timeunit << "\n";
  cout << " |\t\tMinimum: " << times_run[0] << timeunit << "\n";
  cout << " |\t\tMaximum: " << times_run[runs-1] << timeunit << "\n";
  cout << " |\tBackward Pass";
  cout << " |\t\tMedian:  " << times_trn[(runs - (runs%2))/2] << timeunit << "\n";
  cout << " |\t\tAverage: " << total_trn/runs << timeunit << "\n";
  cout << " |\t\tMinimum: " << times_trn[0] << timeunit << "\n";
  cout << " |\t\tMaximum: " << times_trn[runs-1] << timeunit << "\n";
  cout << " +-----------------------------------------------------------\n";


  free(times_run);
  free(times_trn);

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
