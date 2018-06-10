#include "Util.h"
#include <dlib/global_optimization.h>

typedef polynomial_kernel<sample_type> poly_kernel;
typedef radial_basis_kernel<sample_type> rbf_kernel;
typedef one_vs_one_trainer<any_trainer<sample_type>> ovo_trainer;
typedef one_vs_one_decision_function<ovo_trainer, decision_function<poly_kernel> > MODEL;
typedef linear_kernel<sample_type> kernel_type;

void training(char *path);

// load model
MODEL loadModel(string filename);

// hàm dữ doán expression dữa vào ảnh đầu vào
double predict(cv::Mat &faceImage, MODEL& df);

void linear_training(char *path);

double linear_predict(cv::Mat &image, multiclass_linear_decision_function<kernel_type, double>& df);