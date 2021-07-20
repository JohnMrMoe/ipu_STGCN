#ifndef ___DATA_UTILITY___
#define ___DATA_UTILITY___

#include "../util/Matrices.cpp"

class Dataset {
private:
  size_t tot_size(vector<size_t> &shape)
  {
    return shape[0] * shape[1] * shape[2] * shape[3];
  }
  void transfer(Matrix_2D &src,
                Matrix_4D &dst,
                int len_seq,
                int n_frame,
                int n_route,
                int day_slot,
                int offset
              )
  {
    float *raw_src, *raw_dst;
    size_t n_slot = day_slot - n_frame + 1;
    size_t sta, end;

    /*
    std::cout << "Transfers: " << src.title << " >> " << dst.title << '\n';
    std::cout << "=====================================================" << '\n';
    std::cout << "\t" << src.title << ": Shape=(" << src.outer << ", " << src.outer << "), ";
    std::cout << "Total Entries= " << src.size << ", Total Bytes= " << src.size * sizeof(float) << '\n';
    std::cout << "\t" << dst.title << ": Shape=(" << dst.W << ", " << dst.Z << ", " << dst.Y << ", " << dst.X << "),";
    std::cout << "Total Entries= " << dst.size << ", Total Bytes= " << dst.size * sizeof(float) << '\n';
    std::cout << "-----------------------------------------------------" << '\n';
    */

    for (size_t i = 0; i < len_seq; i++) {
      for (size_t j = 0; j < n_slot; j++) {

        sta = (i + offset) * day_slot + j;
        raw_src = src[sta];
        raw_dst = dst[i * n_slot + j];
        size_t size = n_frame * src.inner * sizeof(float);

        size_t v = (i*n_slot+j);

        /*if (i==17 || j==0) {
          std::cout << " * Src: [";
          std::cout << sta << ":" << (sta + n_frame) << "/" <<src.outer<<", :]";
          std::cout << "\t\tDst: @";
          std::cout << v << ":" << (v+(n_frame * src.inner)) << "/" << dst.size << '\n';
        }*/

        memcpy(raw_dst, raw_src, size);
      }
    }
  }
  void remember_shape(vector<size_t> a, vector <size_t> b)
  {
    for (size_t i = 0; i < a.size(); i++) {
      a[i] = b[i];
    }
  }
public:
  Matrix_4D train;
  Matrix_4D valid;
  Matrix_4D test;

  vector<size_t> shape_train;
  vector<size_t> shape_valid;
  vector<size_t> shape_test;

  float train_mean, train_std;

  Dataset(Matrix_2D &data_seq,
          int config[3],
          size_t n_route,
          size_t n_frame,
          size_t day_slot = 288,
          vector<vector<size_t>> shapes = vector<vector<size_t>> {{0}, {0}, {0}}
        ) : train(shapes[0], "Training Sequence"),
            valid(shapes[1], "Validation Sequence"),
            test (shapes[2], "Testing Sequence"),
            shape_train(4),   shape_valid(4),   shape_test(4)
  {
    // n_train, n_val, n_test = config

    remember_shape(shape_train, shapes[0]);
    remember_shape(shape_valid, shapes[1]);
    remember_shape(shape_test, shapes[2]);

          // src,      dst,   len_seq,   n_frame, n_route, day_slot, offset
    transfer(data_seq, train, config[0], n_frame, n_route, day_slot, 0);
    transfer(data_seq, valid, config[1], n_frame, n_route, day_slot, config[1]);
    transfer(data_seq, test,  config[2], n_frame, n_route, day_slot, config[1] + config[2]);

    //sequence size:  (n_frame * len_seq * (day_slot - n_frame +1))
    //train dimension



    train_mean = train.mean();
    train_std  = train.standard_deviation(train_mean);

    train.z_score(train_mean, train_std);
    valid.z_score(train_mean, train_std);
    test .z_score(train_mean, train_std);
  }

};


#endif
