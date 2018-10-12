#pragma once

#include "trees/MWNode.h"

namespace mrcpp {

template<int D>
class TreeCalculator {
public:
    TreeCalculator() { }
    virtual ~TreeCalculator() = default;

    virtual MWNodeVector<D>* getInitialWorkVector(MWTree<D> &tree) const {
        return tree.copyEndNodeTable();
    }

    virtual void calcNodeVector(MWNodeVector<D> &nodeVec) {
#pragma omp parallel shared(nodeVec)
{
        int nNodes = nodeVec.size();
#pragma omp for schedule(guided)
        for (int n = 0; n < nNodes; n++) {
            MWNode<D> &node = *nodeVec[n];
            calcNode(node);
        }
}
        postProcess();
    }
protected:
    virtual void calcNode(MWNode<D> &node) = 0;
    virtual void postProcess() { }
};

}