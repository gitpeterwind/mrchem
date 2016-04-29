#ifndef TREEBUILDER_H
#define TREEBUILDER_H

#include "mwrepr_declarations.h"

template<int D>
class TreeBuilder {
public:
    TreeBuilder(const MultiResolutionAnalysis<D> &mra, int iter);
    virtual ~TreeBuilder();

    void setMaxIter(int iter) { this->maxIter = iter; }

protected:
    int maxIter;
    TreeAdaptor<D> *adaptor;
    TreeCalculator<D> *calculator;
    const MultiResolutionAnalysis<D> MRA;

    void clearCalculator();
    void clearAdaptor();

    double calcScalingNorm(MWNodeVector &vec) const;
    double calcWaveletNorm(MWNodeVector &vec) const;

    void build(MWTree<D> &tree) const;
    bool maxIterReached(int iter) const {
        if (this->maxIter < 0) return false;
        if (this->maxIter > iter) return false;
        return true;
    }
};

#endif // TREEBUILDER_H