#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H
#include "Data.h"
#include <Eigen/Eigen>
#include <iostream>

template <typename DataType, typename GradientLambda, typename CostLambda,
          typename Gradient_1Dim>
class GradientDescent {
public:
  static void evaluate(const DataType &data, Eigen::VectorXd &weights,
                       GradientLambda &&f, CostLambda &&c, Gradient_1Dim &&g1,
                       int max_iter = 10000, double learn_rate = 0.0001) {

    double cost_change = 100;
    for (int i = 0; i < max_iter; ++i) {
      double prev_cost = c(data, weights);
      std::string out = "iter " + std::to_string(i);
      std::cout << '\r' << out;

      Eigen::VectorXd gradient = f(data, weights);

      double step = learn_rate;
      
      for(int i = 0; i < 100; ++i) {
          step = step -  0.1 * g1(data, weights, gradient, step);
          Eigen::VectorXd newY = (weights - step * gradient);
          double cost = c(data, newY);
          if(cost - prev_cost < learn_rate) break;
      }      
      
      weights += -gradient * step;

      double cost = c(data, weights);
      cost_change = prev_cost - cost;
      std::cout << " " << cost_change << " " << cost << " " << step
                << "                  ";
    }
    std::cout<<std::endl;
  }
};

#endif // GRADIENTDESCENT_H
