#include "Network.hpp"
#include <stdexcept>
#include <math.h>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>


Network::Network(const std::vector<int> &layers, double lR):
	learningRate(lR),
	nbLayers(layers.size()),
	errors(),
	inputs(),
	outputs(),
	weights(),
	biases(),
	functions(),
	derivates()
{
	if (layers.size() < 3)
		throw std::invalid_argument("Network must contain at least one hidden layer");

	std::srand(std::time(0));
	this->initializeNetwork(layers);

	/* Setting activation functions & derivates */
	this->functions.push_back(0);
	this->derivates.push_back(0);
	for (int i=0; i < this->nbLayers - 2; ++i)
	{
		this->functions.push_back(new Function(&sigmoid));
		this->derivates.push_back(new Function(&sigmoidPrime));
	}
	this->functions.push_back(new Function(&identity));
	this->derivates.push_back(new Function(&identityPrime));
}


Network::~Network()
{
	for (int i=0; i < this->nbLayers; ++i)
	{
		delete this->errors[i];
		delete this->outputs[i];
		delete this->inputs[i];

		if (i != this->nbLayers - 1)
		{
			delete this->biases[i];
			delete this->weights[i];
		}

		if (this->functions[i])
			delete this->functions[i];
		if (this->derivates[i])
			delete this->derivates[i];
	}
}


/*
	Initializes all the components of the network
 	(weights, biases, inputs, outputs, errors)
*/
void 						Network::initializeNetwork(const std::vector<int> &layers)
{
	/* Initializing input & hidden layers */
	for (unsigned int layer=0; layer < layers.size() - 1; ++layer)
	{
		int rows = layers[layer + 1];
		int columns = layers[layer];

		matrix<double> 	*currentWeights = new matrix<double>(rows, columns);
		vector<double> 	*currentErrors = new vector<double>(columns);
		vector<double> 	*currentInputs = new vector<double>(columns);
		vector<double> 	*currentOutputs = new vector<double>(columns);
		vector<double>	*currentBiases = new vector<double>(rows);

		for (unsigned int i=0; i < currentWeights->size1(); ++i)
		{
			for (unsigned j=0; j < currentWeights->size2(); ++j)
			{
				(*currentWeights)(i, j) = static_cast<double>(std::rand()) / RAND_MAX;
			}
		}

		for (int i=0; i < rows; ++i)
		{
			(*currentBiases)[i] = static_cast<double>(std::rand()) / RAND_MAX;
		}

		std::fill(currentErrors->begin(), currentErrors->end(), 0.0);
		std::fill(currentInputs->begin(), currentInputs->end(), 0.0);
		std::fill(currentOutputs->begin(), currentOutputs->end(), 0.0);

		this->weights.push_back(currentWeights);
		this->errors.push_back(currentErrors);
		this->inputs.push_back(currentInputs);
		this->outputs.push_back(currentOutputs);
		this->biases.push_back(currentBiases);
	}

	/* Output layer */
	int outputSize = layers[layers.size() - 1];

	vector<double> *outputErrors = new vector<double>(outputSize);
	vector<double> *outputInputs = new vector<double>(outputSize);
	vector<double> *outputOutputs = new vector<double>(outputSize);

	std::fill(outputErrors->begin(), outputErrors->end(), 0.0);
	std::fill(outputInputs->begin(), outputInputs->end(), 0.0);
	std::fill(outputOutputs->begin(), outputOutputs->end(), 0.0);

	this->errors.push_back(outputErrors);
	this->inputs.push_back(outputInputs);
	this->outputs.push_back(outputOutputs);
}


vector<double> *Network::predict(const vector<double> &X)
{
	if (X.size() != this->inputs[0]->size())
		throw std::length_error("Invalid number of features");

	return this->feedForward(X);
}


/* Propagates the input through the network and returns the output */
vector<double> 	*Network::feedForward(const vector<double> &X)
{
	delete this->inputs[0];
	delete this->outputs[0];

	this->inputs[0] = new vector<double>(X);
	this->outputs[0] = new vector<double>(X);

	for (int i=1; i < this->nbLayers; ++i)
	{
		delete this->inputs[i];
		delete this->outputs[i];

		this->inputs[i] = new vector<double>(prod(*(this->weights[i - 1]), *(this->outputs[i - 1])) + *(this->biases[i - 1]));
		this->outputs[i] = (*(this->functions[i]))(*(this->inputs[i]));
	}

	return this->outputs[this->outputs.size() - 1];
}


/* Fits the network using the data given in parameter */
void			Network::fit(matrix<double> &X, matrix<double> &Y, const int iterations)
{
	if (X.size1() != Y.size1())
		throw std::length_error("Features and targets must have the same length");

	for (int iter=0; iter < iterations; ++iter)
	{
		for (unsigned int idx=0; idx < X.size1(); ++idx)
		{
			matrix_row<matrix<double> > rowX(X, idx);
			matrix_row<matrix<double> > rowY(Y, idx);

			if (rowX.size() != this->inputs[0]->size())
				throw std::length_error("Invalid number of features");

			this->updateWeights(this->row2vec(rowX), this->row2vec(rowY));
		}
	}
}


/* Updates the weights in the network by using the backpropagation algorithm */
void			Network::updateWeights(vector<double> *features, vector<double> *target)
{
	Function				f(*(this->derivates[this->derivates.size() - 1]));
	vector<double> 	*output = this->feedForward(*features);
	vector<double> 	diff = *output - *target;
	vector<double>	*deriv = f(*(this->outputs[this->outputs.size() - 1]));
	vector<double>	*err = new vector<double>(element_prod(*deriv, diff));

	delete deriv;
	delete this->errors[this->errors.size() - 1];
	this->errors[this->errors.size() - 1] = err;

	for (int i=this->nbLayers - 2; i > 0; --i)
	{
		vector<double>	*deriv = (*(this->derivates[i]))(*(this->inputs[i]));
		vector<double>	mul(prod(trans(*(this->weights[i])), *(this->errors[i + 1])));
		matrix<double>	*oldW = this->weights[i];
		vector<double>	*oldB = this->biases[i];

		delete this->errors[i];
		this->errors[i] = new vector<double>(element_prod((*deriv), mul));
		this->weights[i] = new matrix<double>(*(this->weights[i]) - (outer_prod(*(this->errors[i + 1]), *(this->outputs[i])) * this->learningRate));
		this->biases[i] = new vector<double>(*(this->biases[i]) - (*(this->errors[i + 1]) * this->learningRate));

		delete oldW;
		delete oldB;
		delete deriv;
	}

	matrix<double>	*oldW = this->weights[0];
	vector<double>	*oldB = this->biases[0];
	this->weights[0] = new matrix<double>(*(this->weights[0]) - (outer_prod(*(this->errors[1]), *(this->outputs[0])) * this->learningRate));
	this->biases[0] = new vector<double>(*(this->biases[0]) - (*(this->errors[1])  * this->learningRate));

	delete features;
	delete target,
	delete oldW;
	delete oldB;
}


vector<double> 	*Network::sigmoid(const vector<double> &input)
{
	vector<double> *result = new vector<double>(input.size());

	for (unsigned int i=0; i < input.size(); ++i)
	{
		(*result)[i] = (1.0 / (1.0 + exp(-input[i])));
	}

	return result;
}


vector<double> 	*Network::sigmoidPrime(const vector<double> &input)
{
	vector<double>	*result = new vector<double>(input.size());

	for (unsigned int i=0; i < input.size(); ++i)
	{
		double sigmoid = (1.0 / (1.0 + exp(-input[i])));
		(*result)[i] = sigmoid * (1 - sigmoid);
	}

	return result;
}


vector<double>	*Network::identity(const vector<double> &input)
{
	return new vector<double>(input);
}


vector<double>	*Network::identityPrime(const vector<double> &input)
{
	vector<double>	*result = new vector<double>(input.size());

	for (unsigned int i=0; i < input.size(); ++i)
	{
		(*result)[i] = 1;
	}

	return result;
}


vector<double>	*Network::row2vec(const matrix_row<matrix<double> > &row) const
{
	vector<double>	vec(row.size());

	std::copy(row.begin(), row.end(), vec.begin());

	return new vector<double>(vec);
}


std::ostream&	operator<<(std::ostream& os, const Network& net)
{
	for (unsigned int layer=0; layer < net.outputs.size(); ++layer)
	{
		os << "-> Layer " << layer << " : " << net.outputs[layer]->size() << " neuron(s)" << std::endl;
	}

    return os;
}
