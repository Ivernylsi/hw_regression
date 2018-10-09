#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H
#include "Data.h"
#include <Eigen/Eigen>
#include <iostream>

template <typename DataType, typename GradientLambda, typename CostLambda>
class GradientDescent {
public:
  static auto evaluate(const DataType &data, const Eigen::VectorXd &w_,
                       GradientLambda &&f, CostLambda &&c,
                       int max_iter = 10000, double learn_rate = 0.0001) {

    Eigen::VectorXd weights = w_;
    double cost_change = 100;
    for (int i = 0; i < max_iter ; ++i) {
      double prev_cost = c(data, weights);
      std::string out = "iter " + std::to_string(i);
      std::cout << '\r' << out;

      Eigen::VectorXd gradient = f(data, weights);

      double step = learn_rate;
     
      weights -= gradient * step;

      double cost = c(data, weights);
      cost_change = prev_cost - cost;
      std::cout << " " << cost_change << " " << cost << " " << step
                << "                  ";
    }
    std::cout<<std::endl;
    return weights;
  }
};

#endif // GRADIENTDESCENT_H
