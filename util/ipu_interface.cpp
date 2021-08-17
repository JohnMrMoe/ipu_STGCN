#ifndef IPU_INTERFACE_C
#define IPU_INTERFACE_C

#include "ipu_interface.hpp"
#include "Logger.hpp"

#include <iostream>
#include <vector>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <chrono>

#ifdef _IRL_IPU
  #include <poplar/DeviceManager.hpp>
#else
  #include <poplar/IPUModel.hpp>
#endif

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Type.hpp>
#include <popops/ElementWise.hpp>

//#include <poputil/codelets.hpp>
#include <popops/codelets.hpp>
#include <poplin/codelets.hpp>
#include <popsparse/codelets.hpp>
#include <popnn/codelets.hpp>
#include <poprand/codelets.hpp>


using namespace std;
using namespace poplar;
using namespace poplar::program;

string IPU_Interface::title(int i)    {return get<0>(items[i]);}
Tensor IPU_Interface::tensor(int i)  {return get<1>(items[i]);}
int    IPU_Interface::type(int i)    {return get<2>(items[i]);}
float* IPU_Interface::data(int i)    {return get<3>(items[i]);}

void progress(int v, int of) {
  auto s = std::string("Progress: 001%");
  for (size_t i = 0; i < s.size(); i++) printf("\b");

  printf("Progress: %3d%%", (int)(v * 100 / (double)of));
  if (v == of) {
    std::cout << endl;
  }
  fflush(stdout);
}

IPU_Interface::IPU_Interface() : tensor_rq("log/tensor_requests.txt", "Tensor Requests"  ),
                                 feedin_dt("log/feedin_reports.txt" , "Feedin Report"    ),
                                 model_log("log/model.txt" ,          "Model Declaration")
{

  // get the things
  #ifdef _IRL_IPU
    DeviceManager manager = DeviceManager::createDeviceManager();

    // attempt to attach to ipu
    bool success = false;
    size_t skips = 4;
    size_t ipus = 1;

    for (auto &hwDevice : manager.getDevices(poplar::TargetType::IPU, ipus)) {
      if (skips) {skips--; continue;}
      device = std::move(hwDevice);
      std::cerr << "Trying to attach to IPU " << device.getId() << "\n";
      if ((success = device.attach())) {
        std::cerr << "Attached to IPU " << device.getId() << "\n";
        break;
      }
    }

    if (!success) {
      std::cerr << "Error attaching to device!\n";
      failed_to_attach = 1;
    }
  #else
    device = ipuModel.createDevice();
  #endif

  target = device.getTarget();

}
IPU_Interface::~IPU_Interface() {
  tensor_rq.log("IPU_Interface DECONSTRUCTOR:");
  tensor_rq.log("\tTotal Tensor Floats:\t" + to_string(total_floats));
  tensor_rq.log("\tTotal Tensor Bytes: \t" + to_string(total_floats* sizeof(float)));
}

void IPU_Interface::progress_print(string title, size_t step, size_t max, size_t title_buffer, size_t acc) {
  size_t perc = ((step * 100.0) / (max * 1.0));

  size_t deci = perc / acc;
  size_t digi = perc % acc;

  size_t x = title.size();

  string s = ":";
  for (size_t i = x; i < 20; i++) s+="";

  s = "\r\t" + title + s + "[";

  for (size_t i = 0; i < 100/acc; i++) {
    if (i<deci) {
       s += "█";
    } else if (i==deci) {
      if (digi < acc/4.0) {
        s += "░";
      } else if (digi < 2*(acc/4.0)) {
        s += "▒";
      } else if (digi < 3*(acc/4.0)) {
        s += "▓";
      } else {
        s += "█";
      }
    } else {
      s += " ";
    }
  }

  std::cout << s << "]";
  if (step==max) {
    std::cout << endl;
  } else {
    std::cout << ends;
  }
}
size_t IPU_Interface::tensor_size(vector<size_t> shape) {
  size_t size = 1;
  for (size_t i = 0; i < shape.size(); i++) size*=shape[i];
  return size;
}
string IPU_Interface::shape_display(std::vector<size_t> shape, string name, string end) {
  size_t total_size = 1;
  string x = name + '\t';
  for (size_t i = 0; i < shape.size(); i++) {
    x += (i==0?"{":", ") + to_string(shape[i]);
    total_size *= shape[i];
  }
  x += "}, size=" + to_string(total_size) + end;
  return x;
}
string IPU_Interface::shape_display(Tensor &t, string name, string end) {
  return shape_display(t.shape(), name, end);
}
size_t IPU_Interface::exists(std::string s) {
  for (size_t i = 0; i < items.size(); i++) {
    if (title(i)==s) return i;
  }
  return -1;
}
Tensor IPU_Interface::expandTensor(Graph &g, Tensor &t, vector<size_t> new_shape, string name) {
  vector<size_t> old_shape = t.shape();
  vector<size_t> exp_shape = t.shape();

  for (size_t i = 0; i < t.shape().size(); i++) exp_shape[i] = old_shape[i];

  for (size_t i = 0; i < t.shape().size(); i++) {
    if (old_shape[i] >= new_shape[i]) continue;
    exp_shape[i] = new_shape[i] - old_shape[i];
    Tensor expansion = getVariable(g, "[unq:exp(" + name + "):" + to_string(unique_exp_adr()) + "]", exp_shape);
    t = poplar::concat(t, expansion, i);
    exp_shape[i] = new_shape[i];
  }

  return t;
}
Tensor IPU_Interface::padTensor(Graph &g, Tensor &core, vector<size_t> avant_padding, vector<size_t> post_padding) {
  vector<size_t> shape      = core.shape();
  vector<size_t> top_exp_shape  = core.shape();
  vector<size_t> bot_exp_shape  = core.shape();

  for (size_t i = 0; i < shape.size(); i++) top_exp_shape[i] = shape[i];
  for (size_t i = 0; i < shape.size(); i++) bot_exp_shape[i] = shape[i];

  size_t top_pad, bot_pad;
  Tensor top_tensor, bot_tensor;

  Tensor padded = core;

  for (size_t i = 0; i < shape.size(); i++) {
    top_pad = avant_padding[i];
    bot_pad = post_padding[i];
    if (top_pad==0 && bot_pad==0) continue;
    top_exp_shape[i] = top_pad;
    bot_exp_shape[i] = bot_pad;
    shape_display(top_exp_shape, "PAD_bef", "\n");
    top_tensor = getVariable(g, "ZERO", vector<size_t>{tensor_size(top_exp_shape)}).reshape(top_exp_shape);
    shape_display(bot_exp_shape, "PAD_aft", "\n");
    bot_tensor = getVariable(g, "ZERO", vector<size_t>{tensor_size(bot_exp_shape)}).reshape(bot_exp_shape);
    padded = poplar::concat(top_tensor, padded, i);
    padded = poplar::concat(padded, top_tensor, i);
    top_exp_shape[i] = padded.shape()[i];
    bot_exp_shape[i] = padded.shape()[i];
  }

  return padded;
}
Tensor IPU_Interface::getVariable(Graph &g, std::string name, std::vector<size_t> shape, int type, float *ptr) {

  //std::cout << "\t\tFetching variable: " << name << " .. " << shape.size() << '\n';

  if (shape.size()==0) return getVariable_OLD(g, name, shape);


  size_t precurrence = 0; string key;
  for (size_t i = 0; i < items.size(); i++) {
    key = title(i);
    size_t c = key.find(":", 0);
    if (c!=-1) key = key.substr(0, c);
    if (name.compare(key)==0) precurrence+=1;
  }

  if (precurrence) name = name + ":" + to_string(precurrence);

  Tensor new_t = g.addVariable(poplar::FLOAT, shape, VariableMappingMethod::LINEAR, name);

  update_allocated_bytes(new_t.shape());

  addVariable(name, new_t, type, ptr);

  return new_t;
}
Tensor IPU_Interface::getVariable_OLD(Graph &g, std::string name, std::vector<size_t> shape, int type, float *ptr) {
  size_t index = exists(name);
  tensor_rq.log("\t\t*\tRequesting Tensor: " + shape_display(shape, name, ""));

  if (-1 != index) {

    char m = shape.size() == tensor(index).shape().size() ? 1 : 0;
    for (size_t i = 0; i < shape.size() && m; i++) {//Check if shape matches
      m &= shape[i]==tensor(index).shape()[i];
    }
    if (shape.size() == 0) m=1;  // This is a fetch-only request, assumes existing
    if (!m) {// IF NOT MATCH
      char l = 1;
      vector<size_t> fit_shape(shape.size());
      for (size_t i = 0; i < shape.size(); i++) {//Check if shape matches
        l &= ( (shape[i] <= tensor(index).shape()[i]) ? 1 : 0);
        fit_shape[i] = tensor(index).shape()[i];
        fit_shape[i] = shape[i] < fit_shape[i] ? shape[i] : fit_shape[i];
      }
      if (l) {// all dimensions smaller/equal , may attempt simple slicing
        return tensor(index).slice(vector<size_t>(shape.size(), 0), shape);
      }
      // Attempt to expand the Tensor to match
      Tensor t = tensor(index).slice(vector<size_t>(fit_shape.size(), 0), fit_shape);
      return expandTensor(g, t, shape, name);
    }
    return tensor(index);
  }

  if (shape.size()==0) return getVariable(g, "NULL", {1});

  Tensor new_t = g.addVariable(poplar::FLOAT, shape, VariableMappingMethod::LINEAR, name);

  update_allocated_bytes(new_t.shape());

  //
  addVariable(name, new_t, type, ptr);

  return new_t;
}
Tensor IPU_Interface::getExistingVariableSlice(std::string name, std::vector<size_t> shape, std::vector<size_t> offset, int type, float *ptr) {
  // ONLY fetch a slice of an existing tensor
  size_t index = exists(name);

  if (index == -1) {
    printf("Couldn't find a tensor with name: ");
    cout << name << '\n';
    return tensor(0); // the Null tensor, will break the code downstream
  }
  if (shape.size() != offset.size()) {
    printf("Shape/Offset dimension mismatch: %ld != %ld.\n", shape.size(), offset.size());
    return tensor(0); // the Null tensor, will break the code downstream
  }
  vector<size_t> slice_sto = vector<size_t>(shape.size());
  for (int i = 0; i < slice_sto.size(); i++) {
    slice_sto[i] = shape[i] + offset[i];
  }
  return tensor(index).slice(offset, slice_sto);
}
void IPU_Interface::addVariable(std::string s, Tensor &t, int fill, float *ptr) {
  // string space = ""; for (size_t i = s.length(); i < 30; i++) space+=" ";
  // std::cout << "\tAdded new tensor: " << s << space << ", with shape:";
  // shape_display(t);
  // std::cout << '\n';
  items.push_back(new_entry(s, t, fill, ptr));
}
Tensor IPU_Interface::getAlternatingSpace(string type, std::vector<size_t> shape) {
  if (type == "layer") {
    type += to_string(layer_temp);
    layer_temp = !layer_temp;
  } else if (type == "block") {
    type += to_string(block_altr);
    block_altr = !block_altr;
  } else {
    printf("\tERROR! attempting to get Alternating Space,\n\ttrying to get variable: ");
    std::cout << type << '\n';
  }
  return getExistingVariableSlice(type, shape);
}
float IPU_Interface::random(float top, size_t div) {
  size_t raw = rand();
  size_t cap = raw % ((size_t)(top+.5 )); // i.e. ceil.
  size_t divd = (raw / (cap ? cap : 1));
  size_t tail = (divd % div);
  float value = (float) cap + (((float) tail) / div);

  return value;
}
void IPU_Interface::glorot_fill(float *ptr, size_t len, std::vector<std::size_t> shape) {

  // def init_weights(self):  # Glorot uniform
  // nin = self.ni; nout = self.nh
  // sd = np.sqrt(6.0 / (nin + nout))
  // for i in range(self.ni):
  //   for j in range(self.nh):
  //     x = np.float32(self.rnd.uniform(-sd, sd))
  //     self.ih_weights[i,j] = x

  // TODO; ACTUALLY FIND THESE VALUES
  size_t hack;
  if (shape.size()==4) {
    hack = shape[1] * shape[2];
  } else if (shape.size()>1) {
    hack = shape[0] * shape[1];
  } else {
    hack = shape[0];
  }
  size_t n_in  = hack,
         n_out = hack;
  float sd = sqrt(6 / (n_in + n_out));
  srand(time(NULL));
  for (size_t i = 0; i < len; i++) {
    ptr[i] = random(sd*2+1)-sd;
  }

}
Engine IPU_Interface::finalize_and_run(Graph &g, Program model, bool run) {
  printf("\n-----\nFINALIZING:\n");

  string label;
  string relabel;
  Tensor tensor;
  int    t_type;
  float* pointr;
  float* glorot;
  StringRef handle;

  size_t largest_zero = 0;
  vector<tuple<Tensor, size_t>> zerofills(0);
  size_t size_X;

  printf("\tLoading Engine\n");

  for (size_t item = 0; item < items.size(); item++) {
    // progress_print("Preload", item, items.size());
    // Just get me my variables
    label  = get<0>(items[item]);
    tensor = get<1>(items[item]);
    t_type = get<2>(items[item]);
    pointr = get<3>(items[item]);
    feedin_dt.log("" + label + "\t, type: " + to_string(t_type) + ", &" + to_string((size_t) pointr), "\n", false);

    switch(t_type) {
      case (NULL_TENSOR):   feedin_dt.log("NULL_TENSOR"); break;
      case (ZEROFL_TENSOR): feedin_dt.log("ZEROFL_TENSOR"); break;
      case (GLOROT_TENSOR): feedin_dt.log("GLOROT_TENSOR"); break;
      case (FEEDIN_TENSOR): feedin_dt.log("FEEDIN_TENSOR"); break;
      case (AUTOFL_TENSOR): feedin_dt.log("AUTOFL_TENSOR"); break;
      case (FILEFD_TENSOR): feedin_dt.log("FEEDFL_TENSOR"); break;
      default:              feedin_dt.log("No type for: " + (label + "\t, type: " + to_string(t_type) + ", &" + to_string((size_t) pointr)) + "!!!", "", true);
    }

    size_X = tensor.flatten().shape()[0];

    if (t_type == GLOROT_TENSOR || t_type == FEEDIN_TENSOR || t_type ==FILEFD_TENSOR) {
      g.createHostWrite(StringRef(label), tensor);
    } else if (t_type == ZEROFL_TENSOR) {
      zerofills.push_back(tuple<Tensor, size_t>(tensor, size_X));
      largest_zero = largest_zero > size_X ? largest_zero : size_X;
      relabel = "[" + to_string(zerofills.size()-1)+ "]{" +  to_string(get<1>(zerofills[zerofills.size()-1])) + "}-zerowrite";
      g.createHostWrite(relabel, get<0>(zerofills[zerofills.size()-1]));
      feedin_dt.log("\t\t\tIs ZEROFILL with label: " + relabel + '\n');
    }
  }
  // progress_print("Preload");

  float* empty = (float *) calloc (largest_zero, sizeof(float));

  // OptionFlags opts = {{"debug.allowOutOfMemory", "true"}};
  vector<Program> runnables{model /*, exports*/};
  OptionFlags opts = {};
  std::cout << "\t\tCOMPILING GRAPH...\n";

  auto run_start = std::chrono::high_resolution_clock::now();
  auto exes = poplar::compileGraph(g, runnables, opts, progress);
  auto run_end = std::chrono::high_resolution_clock::now();

  auto s = std::chrono::duration_cast<std::chrono::nanoseconds>(run_end - run_start).count() / 1e9;
  std::cout << "\t\tCompiled in " << to_string(s) << "s, " << to_string(s/60) << " min" << '\n';

  std::cout << "\t\tMAKING ENGINE...\n";
  run_start = std::chrono::high_resolution_clock::now();
  Engine engine(std::move(exes), opts);
  run_end = std::chrono::high_resolution_clock::now();
  s = std::chrono::duration_cast<std::chrono::nanoseconds>(run_end - run_start).count() / 1e9;
  printf("\t\tENGINE MADE!\n");
  std::cout << "\t\tEngine made in " << to_string(s) << "s, " << to_string(s/60) << " min" << '\n';

  std::cout << "\t\tENGINE PREPARING FOR DEVICE!\n";
  engine.prepare(device);

  std::cout << "\t\tENGINE DEPLOYING ON DEVICE!\n";
  engine.deploy();

  printf("\tENSURE TENSOR TRANSFERS:\n");
  for (size_t item = 0, zerox=0; item < items.size(); item++) {
    // progress_print("Transfer", item, items.size());

    // Just get me my variables
    label  = get<0>(items[item]);
    tensor = get<1>(items[item]);
    t_type = get<2>(items[item]);
    pointr = get<3>(items[item]);

    //std::cout << "\t-\t\"" << label << "\", " << item <<"/"<< items.size() << "\t\ttype [t_type]=";
    size_X = tensor.flatten().shape()[0];
    switch (t_type) {
      case (NULL_TENSOR):
        // skip, shouldn't be used anyway
        break;
      case (ZEROFL_TENSOR):

        relabel = "[" + to_string(zerox)+ "]{" +  to_string(get<1>(zerofills[zerox])) + "}-zerowrite";


        engine.writeTensor(relabel, &empty[0], &empty[get<1>(zerofills[zerox])]);
        zerox++;

        break;
      case (GLOROT_TENSOR):

        handle = StringRef(label);
        // create a matrix of the right shape
        glorot = (float *) malloc (size_X * sizeof(float));
        // fill it with glorot random numbers
        glorot_fill(glorot, size_X, tensor.shape());
        // transfer them to tensor
                    //g.createHostWrite(label, tensor);
        engine.writeTensor(label, glorot, &glorot[size_X]);
        // free matrix
        free(glorot);
        break;
      case (FEEDIN_TENSOR):case (FILEFD_TENSOR):
        // handle = StringRef(label);
        // engine.writeTensor(label, &pointr[0], &pointr[size_X]);
        // break;
        //
      //case (FILEFD_TENSOR):
        handle = StringRef(label);
        engine.writeTensor(label, &pointr[0], &pointr[size_X]);
        break;
      case (AUTOFL_TENSOR):
        break;
      default:
        printf("\tUNDEFINED WRITING: %d", t_type);
        break;
    }
  }
  // Run the control program
  if (run) {
    std::cout << "Running program\n";
    engine.run(0);
    std::cout << "Program complete\n";
  }

  return engine;

}
Program IPU_Interface::notification(Graph &g, string notification) {
  Tensor t = getVariable_OLD(g, "Notification", {1}, 0);
  Tensor p = getVariable_OLD(g, "pluss", {1});
  Sequence note;
  #ifdef   _NOTIFY
    note.add(PrintTensor(notification, t));
  #endif //_NOTIFY
  popops::addInPlace(g, t, p, note); // increment
  return note;
}
void IPU_Interface::retain(FileFedVector &ffv) {
 remember_pointer.push_back(ffv);
}
#endif
