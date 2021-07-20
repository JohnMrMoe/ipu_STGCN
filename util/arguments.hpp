
#ifndef ARGUMENTS_H_GUARD
#define ARGUMENTS_H_GUARD


#include <stdio.h>
#include <string>
using namespace std;


class Arguments {
public:
  size_t n_route   = 228,
         n_his     = 12,
         n_pred    = 9,
         batch_size= 50,
         epoch     = 50,
         save      = 10,
         ks        = 3,
         kt        = 3,
         c_w       = 1,
         d_slot    = 288;    // DAY SLOT
  float  lr        = 1e-3;
  string opt       = "RMSProp",
         graph     = "default",
         inf_mode  = "merge";
  void set(string s, string v);
  Arguments(int argc, char const *argv[]);
};

#endif
