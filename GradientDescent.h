#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H
#include "Data.h"
#include <Eigen/Eigen>
#include <iostream>

template <typename DataType, typename GradientLambda, typename CostLambda>
class GradientDescent {
public:
  static void evaluate(const DataType &data, Eigen::VectorXd &weights,
                       GradientLambda &&f, CostLambda &&c, int max_iter = 10000,
                       double learn_rate = 0.0001) {

    for (int i = 0; i < max_iter; ++i) {
      double prev_cost = c(data, weights);
      std::string out = "iter " + std::to_string(i);
      std::cout << '\r' << out;

      double step = learn_rate;

      Eigen::VectorXd gradient = f(data, weights);

      weights += -gradient * step;

      double cost = c(data, weights);
      std::cout<<" "<<(prev_cost - cost)<<" "<<cost<<" "<<step<<"                  ";
    }
    std::cout << "\n";
  }
};

#endif // GRADIENTDESCENT_H
