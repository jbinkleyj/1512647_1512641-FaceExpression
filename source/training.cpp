#include "training.h"

// 
/*
* đọc các ảnh trong tập traning, xây dựng dữ liệu traning
* sử dụng thuật toán svm, 
* feature : 36000 chiều
* label   : 1 - happy (hạnh phúc)
		  : 2 - surprise(bất ngờ)
		  : 3 - fear(sỡ hãi)
		  : 4 - disgust(ghê tởm)
		  : 5 - sad(buồn)
		  : 6 - anger(giận dữ)
		  : 7 - neutral state(trạng thái trung lập không cảm xúc)
*/

void model_selection(std::vector<sample_type> samples, std::vector<double> labels, double &g1, double &c, double &d) {
	randomize_samples(samples, labels);
	auto cross_validation_score = [&](const double gg1,const double cc, const double dd) {
		ovo_trainer trainer;
		svm_nu_trainer<poly_kernel> poly_trainer;
		svm_nu_trainer<rbf_kernel> rbf_trainer;
		poly_trainer.set_kernel(poly_kernel(gg1, cc, dd));
		// mặc định với one_vs_one_trainer nên sử dụng rbf_trainer để giải quyết subproblem phân loại binary
		trainer.set_trainer(poly_trainer);

		matrix<double> result = cross_validate_multiclass_trainer(trainer, samples, labels, 10);
		cout << "gamma1 : " << setw(11) << gg1 << " cc : " << cc << " dd : " << dd <<  endl;

		return 2 * prod(result) / sum(result);
	};

	auto result = find_max_global(cross_validation_score, { 1e-5, 1e-5, 1}, { 25, 100, 2}, max_function_calls(50));

	g1 = result.x(0);
	c = result.x(1);
	d = result.x(2);
	ofstream out("params.txt");
	out << "best param : " << g1 << " "  << c << " " << d << endl;
	out.close();
	cout << "best param : " << g1 << " " << c << " " << d << endl;
}


void training(char * path)
{
	std::vector<sample_type> samples;
	std::vector<double> labels;

	get_training_data(path, samples, labels);

	cout << "samples.size(): " << samples.size() << endl;
	cout << "Labels size " << labels.size() << endl;

	/*
	* one - one strategy
	*/

	double gamma = 0.01, c = 2, d = 2;
	model_selection(samples, labels, gamma, c, d);
	ovo_trainer trainer;

	std::cout << "Bat dau traning " << endl;
	// tạo phân loại binary và thiết lập thông số

	svm_nu_trainer<poly_kernel> poly_trainer;
	poly_trainer.set_kernel(poly_kernel(gamma, c, d));

	// mặc định với one_vs_one_trainer nên sử dụng rbf_trainer để giải quyết subproblem phân loại binary
	trainer.set_trainer(poly_trainer);
	//trainer.set_trainer(rbf_trainer);

	// chúng ta sẽ thu được decision rule từ one_vs_one trainer và lưu chúng vào
	// one_vs_one_decision_funcition
	one_vs_one_decision_function<ovo_trainer> df = trainer.train(samples, labels);

	one_vs_one_decision_function<ovo_trainer>::binary_function_table functs;
	functs = df.get_binary_decision_functions();
	cout << "number of binary decision functions in df: " << functs.size() << endl;

	// lưu model xuống đĩa tránh retrain
	MODEL df2;

	df2 = df;
	serialize("df.dat") << df2;
	std::cout << "Ket thuc training" << endl;

}

//one_vs_one_decision_function<ovo_trainer, decision_function<poly_kernel>, decision_function<rbf_kernel> >
//loadModel(string filename) {
//	one_vs_one_decision_function<ovo_trainer, decision_function<poly_kernel>, decision_function<rbf_kernel> > df;
//	deserialize(filename) >> df;
//	return df;
//}

double predict(cv::Mat & faceImage, MODEL& df)
{
	if (!faceImage.data)
		return -1;

	std::vector<sample_type> sample;

	//build feature cho ảnh input
	build_feature(faceImage, sample);

	double label = df(sample[0]);
	return label;
}

void linear_training(char *path) {


	std::vector<sample_type> samples;
	std::vector<double> labels;

	get_training_data(path, samples, labels);

	svm_multiclass_linear_trainer<kernel_type, double> trainer;
	
	trainer.set_c(0);
	trainer.set_epsilon(0.0000001);

	multiclass_linear_decision_function<kernel_type, double> df = trainer.train(samples, labels);

	cout << "number of binary decision functions in df: " << df.number_of_classes() << endl;
	serialize("df_linear.dat") << df;

}

double linear_predict(cv::Mat &image, multiclass_linear_decision_function<kernel_type, double>& df) {
	if (!image.data)
		return -1;

	std::vector<sample_type> sample;

	//build feature cho ảnh input
	build_feature(image, sample);

	double label = df(sample[0]);
	return label;

}
