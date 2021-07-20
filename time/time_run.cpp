#include "../util/Matrices.cpp"
#include "../util/ipu_interface.hpp"

#include <chrono>


double time_run(Engine &engine, string label="--unlabeled run--", size_t prog=0) {

  auto run_start = std::chrono::high_resolution_clock::now();
  engine.run(prog, label);
  auto run_end = std::chrono::high_resolution_clock::now();

  // double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(run_end - run_start).count() / 1e3;
  // old ms: std::chrono::duration<double, std::milli> ms = run_end - run_start;
  double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(run_end - run_start).count() / 1e6;

  return ms;
}
