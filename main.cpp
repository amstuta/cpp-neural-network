#include "Network.hpp"
#include <math.h>
#include <boost/numeric/ublas/io.hpp>


int	main()
{
	std::vector<int> layers;
	layers.push_back(1);
	layers.push_back(2);
	layers.push_back(1);

	Network net(layers, 0.001);
	std::cout << net << std::endl;

	/* Training */
	matrix<double>	X(200,1), Y(200,1);
	for (int i=-100; i < 100; i += 1)
	{
		X(i+100, 0) = i / 100.0;
		Y(i+100, 0) = pow(i, 2) / 1000.0;
	}
	net.fit(X, Y, 10000);

	/* Predict */
	vector<double> res(200);
	for (double i=-100; i < 100; ++i)
	{
		vector<double> a(1);
		a[0] = i / 100;
		res.insert_element(i + 100, (*(net.predict(a)))[0] * 1000.0);
	}
	std::cout << res << std::endl;

	return 0;
}
