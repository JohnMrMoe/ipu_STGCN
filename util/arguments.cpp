#ifndef ARGUMENTS_C_GUARD
#define ARGUMENTS_C_GUARD


#include <stdio.h>
#include <string>
#include "arguments.hpp"
using namespace std;


void Arguments::set(string s, string v) {
  if (0==s.compare("--n_route")   )  n_route     = (size_t) stoi(v);
  if (0==s.compare("--n_his")     )  n_his       = (size_t) stoi(v);
  if (0==s.compare("--n_pred")    )  n_pred      = (size_t) stoi(v);
  if (0==s.compare("--batch_size"))  batch_size  = (size_t) stoi(v);
  if (0==s.compare("--epoch")     )  epoch       = (size_t) stoi(v);
  if (0==s.compare("--save")      )  save        = (size_t) stoi(v);
  if (0==s.compare("--ks")        )  ks          = (size_t) stoi(v);
  if (0==s.compare("--kt")        )  kt          = (size_t) stoi(v);
  if (0==s.compare("--c_w")       )  c_w         = (size_t) stoi(v);
  if (0==s.compare("--lr")        )  lr          = stof(v);
  if (0==s.compare("--opt")       )  opt         = v;
  if (0==s.compare("--graph")     )  graph       = v;
  if (0==s.compare("--inf_mode")  )  inf_mode    = v;
}
Arguments::Arguments(int argc, char const *argv[]) {
  for (int i = 1; i < argc; i+=2) {
    set(argv[i], argv[i+1]);
    printf("Reading argument: %s %s\n", argv[i], argv[i+1]);
  }
}


#endif
