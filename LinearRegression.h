#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H
#include <Eigen/Eigen>
#include <vector>
#include "Data.h"
#include "GradientDescent.h"
#include <random>

class LinearRegression {
public:
  LinearRegression(const int &featureSize, const bool &freeTerm = false) : freeTerm(freeTerm), featureSize(featureSize) {
  if(freeTerm) this->featureSize ++;
    weight = Eigen::VectorXd(size());
  std::normal_distribution<double> runif(-100, 100);
  std::mt19937 rng;

  for(int i = 0; i < weight.rows(); ++i) 
    weight[i] = runif(rng);
    
  }

  void train(DataSet &set, const double &learn_rate = 0.00001,
             const  int &mxIter = 10000);
  void trainStochastic(DataSet &set, const int &num, const double &learn_rate = 0.00001,
             const  int &mxIter = 10000);

  void solveQR(DataSet &set);
  void solveSVD(DataSet &set);
  double calcR2(const DataSet &set);
  double predict(const Data &data);
  double calcRMSE(const DataSet &set);


  Eigen::VectorXd getWeight() {
    return weight;
  }

  Eigen::VectorXd weight;
  int size() const { return featureSize;}
private:
  bool freeTerm;
  int featureSize;
};

#endif // LINEARREGRESSION_H
