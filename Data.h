#ifndef DATA_H
#define DATA_H 
#include <Eigen/Eigen>

struct Data {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Data(const double &y, const Eigen::VectorXd &x):
    y(y), x(x){}
  double y;
  Eigen::VectorXd x;
};

typedef std::vector<Data, Eigen::aligned_allocator<Data>> DataSet;

#endif // DATA_H
