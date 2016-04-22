/*
 *
 *
 *  \date June 2, 2010
 *  \author Stig Rune Jensen \n
 *          CTCC, University of Tromsø
 *
 */

//#include "ScalingBasis.h"
//#include "QuadratureCache.h"
//#include "LegendreBasis.h"
//#include "LegendrePoly.h"
//#include "eigen_disable_warnings.h"

//using namespace std;
//using namespace Eigen;


//LegendreBasis::LegendreBasis(int order) : ScalingBasis(order) {
//	this->type = Legendre;
//	initScalingBasis();
//	preEvaluate();
//}

//void LegendreBasis::initScalingBasis() {
//	for (int i = 0; i < scalingOrder + 1; i++) {
//		LegendrePoly *poly = new LegendrePoly(i, 0.0, 1.0);
//		this->addFunc(poly);
//		this->getFunc(i) *= sqrt(2.0 * i + 1.0); // exact normalization
//	}
//}

//void LegendreBasis::preEvaluate() {
//	getQuadratureCache(qc);
//	const VectorXd &pts = qc.getRoots(quadratureOrder);
//	const VectorXd &wgts = qc.getWeights(quadratureOrder);

//	int npts = scalingOrder + 1;
//	preVals = MatrixXd::Zero(quadratureOrder, npts);

//	assert(quadratureOrder == npts);

//	for (int k = 0; k < npts; k++) {
//		const Polynomial &poly = this->getFunc(k);
//		for (int i = 0; i < quadratureOrder; i++) {
//			preVals(i, k) = poly.evalf(pts(i)) * wgts(i);
//		}
//	}
//}

///****** WARNING! Ugliness ahead!!! ********************************/

//VectorXd LegendreBasis::calcScalingCoefs(int d,
//		const SeparableFunction<1> &func, int n, int l) const {
//	getQuadratureCache(qc);
//	const VectorXd &pts = qc.getRoots(quadratureOrder);

//	VectorXd scs(scalingOrder + 1);
//	RowVectorXd fvals(quadratureOrder);

//	double r;
//	double sfac;
//	if (n < 0) {
//		sfac = pow(2.0, n);
//	} else {
//		sfac = 1 << n;
//	}

//	for (int i = 0; i < quadratureOrder; i++) {
//		r = (pts(i) + l) / sfac;
//		fvals(i) = func.evalf(r, d);
//	}
//	scs = fvals * preVals;
//	scs *= sqrt(1.0 / sfac);
//	return scs;
//}

//VectorXd LegendreBasis::calcScalingCoefs(int d,
//		const SeparableFunction<2> &func, int n, int l) const {
//	getQuadratureCache(qc);
//	const VectorXd &pts = qc.getRoots(quadratureOrder);

//	VectorXd scs(scalingOrder + 1);
//	RowVectorXd fvals(quadratureOrder);

//	double r;
//	double sfac;
//	if (n < 0) {
//		sfac = pow(2.0, n);
//	} else {
//		sfac = 1 << n;
//	}

//	for (int i = 0; i < quadratureOrder; i++) {
//		r = (pts(i) + l) / sfac;
//		fvals(i) = func.evalf(r, d);
//	}
//	scs = fvals * preVals;
//	scs *= sqrt(1.0 / sfac);
//	return scs;
//}

//VectorXd LegendreBasis::calcScalingCoefs(int d,
//		const SeparableFunction<3> &func, int n, int l) const {
//	getQuadratureCache(qc);
//	const VectorXd &pts = qc.getRoots(quadratureOrder);

//	VectorXd scs(scalingOrder + 1);
//	RowVectorXd fvals(quadratureOrder);

//	double r;
//	double sfac;
//	if (n < 0) {
//		sfac = pow(2.0, n);
//	} else {
//		sfac = 1 << n;
//	}

//	for (int i = 0; i < quadratureOrder; i++) {
//		r = (pts(i) + l) / sfac;
//		fvals(i) = func.evalf(r, d);
//	}
//	scs = fvals * preVals;
//	scs *= sqrt(1.0 / sfac);
//	return scs;
//}

//void LegendreBasis::calcScalingCoefs(const SeparableFunction<1> &func,
//		int n, const int *l, MatrixXd &scs) const {
//	getQuadratureCache(qc);
//	const VectorXd &pts = qc.getRoots(quadratureOrder);

//	int dim = scs.cols();
//	RowVectorXd fvals(quadratureOrder);

//	double r;
//	double sfac;
//	if (n < 0) {
//		sfac = pow(2.0, n);
//	} else {
//		sfac = 1 << n;
//	}

//	for (int d = 0; d < dim; d++) {
//		for (int i = 0; i < quadratureOrder; i++) {
//			r = (pts(i) + l[d]) / sfac;
//			fvals(i) = func.evalf(r, d);
//		}
//		scs.col(d) = fvals * preVals;
//	}

//	scs *= sqrt(1.0 / sfac);
//}

//void LegendreBasis::calcScalingCoefs(const SeparableFunction<2> &func,
//		int n, const int *l, MatrixXd &scs) const {
//	getQuadratureCache(qc);
//	const VectorXd &pts = qc.getRoots(quadratureOrder);

//	int dim = scs.cols();
//	RowVectorXd fvals(quadratureOrder);

//	double r;
//	double sfac;
//	if (n < 0) {
//		sfac = pow(2.0, n);
//	} else {
//		sfac = 1 << n;
//	}

//	for (int d = 0; d < dim; d++) {
//		for (int i = 0; i < quadratureOrder; i++) {
//			r = (pts(i) + l[d]) / sfac;
//			fvals(i) = func.evalf(r, d);
//		}
//		scs.col(d) = fvals * preVals;
//	}

//	scs *= sqrt(1.0 / sfac);
//}

//void LegendreBasis::calcScalingCoefs(const SeparableFunction<3> &func,
//		int n, const int *l, MatrixXd &scs) const {
//	getQuadratureCache(qc);
//	const VectorXd &pts = qc.getRoots(quadratureOrder);

//	int dim = scs.cols();
//	RowVectorXd fvals(quadratureOrder);

//	double r;
//	double sfac;
//	if (n < 0) {
//		sfac = pow(2.0, n);
//	} else {
//		sfac = 1 << n;
//	}

//	for (int d = 0; d < dim; d++) {
//		for (int i = 0; i < quadratureOrder; i++) {
//			r = (pts(i) + l[d]) / sfac;
//			fvals(i) = func.evalf(r, d);
//		}
//		scs.col(d) = fvals * preVals;
//	}
//	scs *= sqrt(1.0 / sfac);
//}