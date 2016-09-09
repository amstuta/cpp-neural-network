#ifndef NETWORK
#define NETWORK

#include <vector>
#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/function.hpp>

using namespace boost::numeric::ublas;


typedef boost::function<vector<double>* (const vector<double>&)> Function;


class Network
{
	double												learningRate;
	int 													nbLayers;
	std::vector<vector<double>*>	errors;
	std::vector<vector<double>*>	inputs;
	std::vector<vector<double>*>	outputs;
	std::vector<matrix<double>*>	weights;
	std::vector<vector<double>*>	biases;

	std::vector<Function*>				functions;
	std::vector<Function*>				derivates;

public:
	Network(const std::vector<int>&, double=0.01);
	virtual ~Network();

	void						fit(matrix<double>&, matrix<double>&, const int=100);
	vector<double> 	*predict(const vector<double>&);

	friend std::ostream& operator<<(std::ostream&, const Network&);

private:

	void 						initializeNetwork(const std::vector<int>&);
	void 						updateWeights(vector<double>*, vector<double>*);
	vector<double> 	*feedForward(const vector<double>&);
	vector<double> 	*row2vec(const matrix_row<matrix<double> >&) const;

	static vector<double>	*sigmoid(const vector<double>&);
	static vector<double> *sigmoidPrime(const vector<double>&);
	static vector<double>	*identity(const vector<double>&);
	static vector<double>	*identityPrime(const vector<double>&);
};

#endif
