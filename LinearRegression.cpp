#include "LinearRegression.h"

double LinearRegression::calcR2(const DataSet &set) {
  double SStot = 0, SSres = 0;
  long double mean = 0;
  for (Data data: set) mean += data.y / set.size();
  for (Data data : set) {
    double y_ = predict(data);
    SStot += (data.y - mean) * (data.y - mean);
    SSres += (data.y - y_) * (data.y - y_);
  }

  std::cout<<"SStot = " << SStot << " SSres = "<< SSres <<std::endl;
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

void LinearRegression::train(DataSet &set,
                             const double &learn_rate,
                             const int &mxIter) {

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
    return gradient.normalized();
  };

  auto one_dim_gradient = [](bool freeTerm, const DataSet &data, 
                             const Eigen::VectorXd &w, 
                             const Eigen::VectorXd &g,
                             double lambda) {
    // f(X) = \sum ( x * (w - l*g) - y)^2
    // df(X)/dl = 2 \ sum * sum(x*g) * (x*(w-l*g) - y)
    Eigen::VectorXd curr_G = w - lambda*g;
    double ans = 0;
    for (const auto &d : data) {
      Eigen::VectorXd x = freeTerm ? d.x.homogeneous() : d.x;
      double common = x.dot(w) - x.dot(lambda*g) - d.y;
      if(!std::isfinite(common)) return 0.0;
      ans +=  -x.dot(g) * common / data.size(); 
    }
    if(!std::isfinite(ans)) return 0.0;
    return ans;

  };

  auto costLambda = [](bool freeTerm, const DataSet &set, Eigen::VectorXd &w) {
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

  auto gradient_1dim = 
    std::bind(one_dim_gradient, freeTerm, std::placeholders::_1, std::placeholders::_2,
              std::placeholders::_3, std::placeholders::_4);


  auto cost = std::bind(costLambda, freeTerm, std::placeholders::_1,
                        std::placeholders::_2);

  GradientDescent<DataSet, typeof(gradient), typeof(cost), typeof(gradient_1dim)>::evaluate(
      set, weight, std::move(gradient), std::move(cost), std::move(gradient_1dim),  mxIter, learn_rate);
}

void LinearRegression::solveQR(DataSet &set) {
  Eigen::Matrix<double, -1, -1> data(set.size(), size());
  Eigen::VectorXd b(set.size());
  for (size_t i = 0; i < set.size(); ++i) {
    Eigen::VectorXd x = set[i].x;
    if (freeTerm)
      x = x.homogeneous();
    data.row(i) = x.transpose();
    b(i) = set[i].y;
  }

  weight = data.colPivHouseholderQr().solve(b);
}

