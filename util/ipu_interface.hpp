#ifndef IPU_INTERFACE_H
#define IPU_INTERFACE_H

#include "Logger.hpp"

#include <iostream>
#include <vector>
#include <string.h>
#include <stdio.h>
#include <tuple>
#include <fstream>
#include <sstream>

//#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Type.hpp>
#ifdef _IRL_IPU
  #include <poplar/DeviceManager.hpp>
#else
  #include <poplar/IPUModel.hpp>
#endif

//#include <poputil/codelets.hpp>
#include <popops/codelets.hpp>
#include <poplin/codelets.hpp>
#include <popsparse/codelets.hpp>
#include <popnn/codelets.hpp>
#include <poprand/codelets.hpp>


#define NULL_TENSOR  -1   // this Tensor is for debug purposes
#define ZEROFL_TENSOR 0  // this Tensor should be filled with 0
#define GLOROT_TENSOR 1  // fill with glorot randoms
#define FEEDIN_TENSOR 2  // fill using the float pointer
#define AUTOFL_TENSOR 3  // This Tensor should not be fed
#define FILEFD_TENSOR 4
#define NULL_VECTOR std::vector<size_t>(0)

#define Tensor_Entry std::tuple<std::string, Tensor, int, float*>

using namespace std;
using namespace poplar;
using namespace poplar::program;


class FeedinVector {
public:
  size_t size;
  float *permanent_pointer;
  FeedinVector(size_t s) {
    size = s;
    permanent_pointer = (float *) malloc (sizeof(s) * size);
  }
  ~FeedinVector() {
    free(permanent_pointer);
  }

  void copy_in(float * data) {
    memcpy(data, permanent_pointer, size);
  }
  void valfill(float x) {
    for (size_t i = 0; i < size; i++) {
      permanent_pointer[i] = x;
    }
  }
};

class FileFedVector {
public:
  size_t size;
  float *permanent_pointer;
  FileFedVector(string filename, size_t _size) {
    size = _size;
    string path = "rawdata/" + filename + ".txt";
    permanent_pointer = (float *) malloc (sizeof(filename) * size);

    // ---
    ifstream file;
	  file.open(path, ios::out);
    if (!file) {
      cout << "-- Couldn't open file:" << path << "!" << endl;
      return;
    }

    // ---
    string s, ss;
    file >> s;
    file.close();
    stringstream stream(s);

    for (size_t i = 0; i < size; i++) {
      getline(stream, ss, ',');
      permanent_pointer[i] = stof(ss);
    }
    // ---
  }
};


class IPU_Interface {
private:
  static constexpr float *dummy = 0;
  Program p;

  bool failed_to_attach = 0;
  bool layer_temp = 0;
  bool block_altr = 0;

  size_t unique_exp_num = 0;
  size_t total_floats = 0;

  Tensor_Entry new_entry(string s, Tensor t, int type=0, float *ptr=dummy) {
    return Tensor_Entry(s, t, type, ptr);
  }
  size_t unique_exp_adr() {
    return unique_exp_num++;
  }
  void update_allocated_bytes(vector<size_t> shape) {
    size_t c = 1;
    for (int i = 0; i < shape.size(); i++) c *= shape[i];
    total_floats += c;
  }
public:
  #ifndef _IRL_IPU
    IPUModel ipuModel;
  #endif
  Device device;
  Target target;

  Logger tensor_rq;
  Logger feedin_dt;
  Logger model_log;

  std::vector<Tensor_Entry> items;
  std::vector<FileFedVector> remember_pointer;

  string title(int i);
  Tensor tensor(int i);
  int    type(int i);
  float* data(int i);

  IPU_Interface();
  ~IPU_Interface();
  void progress_print(string title, size_t step=100, size_t max=100, size_t title_buffer = 20, size_t acc = 5);
  size_t tensor_size(vector<size_t> shape);
  string shape_display(vector<size_t> shape, string name = "x", string end="\n");
  string shape_display(Tensor &t, string name = "x", string end="\n");
  Tensor expandTensor(Graph &g, Tensor &t, vector<size_t> shape, string name="");
  size_t exists(std::string s);
  Tensor getVariable(Graph &g, std::string name, std::vector<size_t> shape=NULL_VECTOR, int type=3, float *ptr=dummy);
  Tensor getVariable_OLD(Graph &g, std::string name, std::vector<size_t> shape=NULL_VECTOR, int type=3, float *ptr=dummy);
  Tensor padTensor(Graph &g, Tensor &core, vector<size_t> avant_padding, vector<size_t> post_padding);
  Tensor getExistingVariableSlice(std::string name, std::vector<size_t> shape=NULL_VECTOR, std::vector<size_t> offset=vector<size_t>{0, 0, 0, 0}, int type=0, float *ptr=dummy);
  void addVariable(std::string s, Tensor &t, int type=AUTOFL_TENSOR, float *ptr=dummy);
  Tensor getAlternatingSpace(string type, std::vector<size_t> shape);
  float random(float top, size_t div = 10000);
  void glorot_fill(float *ptr, size_t len, std::vector<std::size_t> shape);
  Engine finalize_and_run(Graph &g, Program model, bool run=false);
  Program notification(Graph &g, string notification);
  void retain(FileFedVector &ffv);
};

#endif
