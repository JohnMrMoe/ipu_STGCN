#include <poplar/Vertex.hpp>


class Conv2D_dumb : public poplar::Vertex {
public:
  // Fields
  poplar::Input<Vector<int>> in_shape;
  poplar::Input<Vector<int>> fl_shape;

  poplar::Input<Vector<Vector<Vector<float>>>>         input;
  poplar::Input<Vector<Vector<Vector<Vector<float>>>>> filter;



  poplar::Output<float> out;

  // Compute function
  bool compute() {
    int di, dj, q;

    int in_height   = in_shape[0],
        in_width    = in_shape[1],
        in_channels = in_shape[2],
        fl_height   = fl_shape[0],
        fl_width    = fl_shape[1],
        fl_channels = fl_shape[3];

    // output[b, i, j, k] =
    //     sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
    //                     filter[di, dj, q, k]

    out = 0;

    for (size_t di = 0; di < count; di++) {
      for (size_t dj = 0; dj < count; dj++) {
        for (size_t q = 0; q < count; q++) {
          out += input[][][][] * filter[di][dj][q][];
    } } }




  }
};
