#include "LinearRegression.h"
#include "tbb/tbb.h"

double LinearRegression::calcR2(const DataSet &set) {
  double SStot = 0, SSres = 0;
  for (Data data : set) {
    double mean = data.x.mean();
    double y_ = predict(data);
    SStot += (data.y - mean) * (data.y - mean);
    SSres += (data.y - y_) * (data.y - y_);
  }
  return 1 - SSres / SStot;
}

double LinearRegression::predict(const Data &data) {
  Eigen::VectorXd currX = data.x;
  if (freeTerm)
    currX = currX.homogeneous();
  return currX.dot(weight);
}

double LinearRegression::calcRMSE(const DataSet &set) {
  double ans = 0;
  for (Data data : set) {
    auto ansTerm = data.y - predict(data);
    ans += ansTerm * ansTerm / set.size();
  }
  return sqrt(ans);
}

void LinearRegression::train(DataSet &set) {

  // f(w) = \sum (x.w + b - y)^2
  // df/dw_i = 2* sum y[i] * (x.w - y)
  //                       gradientCommonPart
  // df/db = 2 * sum (x.w + b - y) - homogeneous part
  auto lambda = [](bool freeTerm, const DataSet &data,
                   const Eigen::VectorXd &w) {
    Eigen::VectorXd gradient(w.rows());
    gradient.setZero();
    for (const auto &d : data) {
      Eigen::VectorXd x = freeTerm ? d.x.homogeneous() : d.x;
      double gradientCommonPart = 2 * (x.dot(w) - d.y);
      gradient += x * gradientCommonPart / data.size();
    }
    return gradient;
  };

  auto costLambda = [](bool freeTerm, const Eigen::VectorXd &w, DataSet &set) {
    double cost = 0;
    for (const auto &s : set) {
      Eigen::VectorXd x = freeTerm ? s.x.homogeneous() : s.x;
      double part = x.dot(w) - s.y;
      cost += (part * part) / set.size();
    }
    return cost;
  };

  auto gradient =
      std::bind(lambda, freeTerm, std::placeholders::_1, std::placeholders::_2);

  auto cost = std::bind(costLambda, freeTerm, std::placeholders::_1,
                        std::placeholders::_2);

  GradientDescent<DataSet, typeof(gradient), typeof(cost)>::evaluate(
      set, weight, std::move(gradient), std::move(cost));
}

void LinearRegression::solveQR(DataSet &set) {
  int size = featureSize + (freeTerm ? 1 : 0);
  Eigen::Matrix<double, -1, -1> data(set.size(), size);
  Eigen::VectorXd b(set.size());
  for (size_t i = 0; i < set.size(); ++i) {
    Eigen::VectorXd x = !freeTerm ? set[i].x : set[i].x.homogeneous();
    data.row(i) = x.transpose();
    b(i) = set[i].y;
  }

  weight = data.colPivHouseholderQr().solve(b);
}
