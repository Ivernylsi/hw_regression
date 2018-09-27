#include <Eigen/Eigen>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <LinearRegression.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

static std::string EigentoString(const Eigen::MatrixXd &mat) {
  std::stringstream ss;
  ss << mat;
  return ss.str();
}

void normalizeData(DataSet &data, const std::vector<int> &bools) {
  Eigen::VectorXd means(data[0].x.rows());
  means.setZero();
  Eigen::VectorXd STD(means.rows());
  STD.setZero();
  double N = data.size();

  // for loop for means
  for (auto &d : data)
    for (int i = 0; i < d.x.rows(); ++i)
      if (!(std::find(bools.begin(), bools.end(), i) != bools.end()))

        means[i] += d.x(i) / N;

  // for loop for std
  for (auto &d : data)
    for (int i = 0; i < d.x.rows(); ++i)
      if (!(std::find(bools.begin(), bools.end(), i) != bools.end()))
        STD(i) += (d.x(i) - means(i)) * (d.x(i) - means(i)) / (N - 1);

  for (int i = 0; i < STD.rows(); ++i)
    STD(i) = sqrt(STD(i));

  for (auto &d : data)
    for (int i = 0; i < d.x.rows(); ++i)
      if (!(std::find(bools.begin(), bools.end(), i) != bools.end())) {
        d.x(i) = d.x(i) - means(i);
        double denominator = STD(i) == 0 ? 1 : STD(i);
        d.x(i) /= denominator;
      }
}

namespace py = pybind11;
PYBIND11_MODULE(PyRegression, m) {

  m.def("normalize_data", [](DataSet &data, const std::vector<int> &v) {
    normalizeData(data, v);
    return data;
  });

  py::class_<LinearRegression>(m, "LinearRegression")
      .def(py::init<const int &, const bool &>())
      .def("train", &LinearRegression::train)
      .def("calc_R2", &LinearRegression::calcR2)
      .def("predict", &LinearRegression::predict)
      .def("calc_RMSE", &LinearRegression::calcRMSE)
      .def("getWeight", &LinearRegression::getWeight)
      .def("solve_QR", &LinearRegression::solveQR)
      .def("__repr__", [](const LinearRegression &l) {
        std::string s = "<class LinearRegression ";
        s = s + " wieghts =\n" + EigentoString(l.weight.transpose());
        s = s + "\n>";
        return s;
      });

  py::class_<Data>(m, "Data")
      .def(py::init<const double &, const Eigen::VectorXd &>())
      .def_readwrite("x", &Data::x)
      .def_readwrite("y", &Data::y)
      .def("__repr__", [](const Data &d) {
        std::string s = "<class Data \n";
        s = s + "y : " + std::to_string(d.y) + "\n";
        s = s + "x : " + EigentoString(d.x.transpose());
        s = s + "\n>";
        return s;
      });
}
