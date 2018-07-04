import numpy as np
from collections import Counter
from count_histories import LabelledUnorderedRootedTree, count_histories

def sampler(data, params):
    pass

def Q_combIS(partial_path, theta, wq = (1.0,) ,wp = (1.0,), histDict = dict()):
    """Returns a tuple (p,w) consisting of a path and a weight, sampled according to the combinatorial importance
    sampler.
    Input is a partial path ans partial weights"""
    # NOTE THE WEIGHT wq SHOULD BE THE SAME FRO ALL PATHS STARTED FROM THE SAME POINT!

    # raise NotImplementedError('Not implemented yet!')

    state = partial_path[0]
    ancestors = compute_ancestors(state)
    if len(ancestors) == 0:
        assert state.nodes == 1
        return partial_path, wq, wp

    # compute weights
    histCounts = np.zeros(len(ancestors), dtype=float)
    transitionProbs = np.zeros(len(ancestors), dtype=float)
    trans_coal = coalescence_prob(state.nodes,theta)
    trans_mut  = mutation_prob(state.nodes,theta)
    for i,x in enumerate(ancestors):
        histCounts[i], histDict = count_histories(x, histDict)
        transitionProbs[i] = trans_coal if x.leaves < state.leaves else trans_mut

    normalized_weights = histCounts / sum(histCounts)
    i_next = np.random.choice(len(ancestors), p=normalized_weights)

    partial_path = (ancestors[i_next],) + partial_path
    wq = (normalized_weights[i_next]*wq[0],) + wq
    wp = (transitionProbs[i_next]*wp[0],) + wp

    return Q_combIS(partial_path,theta,wq,wp,histDict)


def Q_GT(x0,theta):
    """Returns a tuple (p,wo,wq) consisting of a path and a weight, sampled according to the Griffiths and Tavare sampler"""
    raise NotImplementedError('Not implemented yet!')

def Q_comb_GT(x0):
    """Returns a tuple (p,w) consisting of a path and a weight, sampled according to the Griffiths and Tavare sampler
    with combinatorial correction"""

def coalescence_prob(n,theta):
    p_next_event_is_coalescense = (n - 1) / (n - 1 + theta)
    p_right_two_lineages_merge = 2.0 / (n * (n - 1))
    # TODO: Verify by hand that the below is not off by a factor of two or not.
    return p_next_event_is_coalescense * p_right_two_lineages_merge

def mutation_prob(n,theta):
    p_next_event_is_mutation = theta / (n - 1 + theta)
    prob_right_lineage_affected = 1 / float(n)
    return prob_right_lineage_affected * p_next_event_is_mutation

def compute_ancestors(rootedTree, atOriginalRoot = True):
    """computes all trees which can be reached from rooted tree, by either merging two leaves below the same node,
    or by merging a singleton leaf with the node above it (who to be clear is allowed to have no other children!) """

    ancestors = []
    if rootedTree.is_leaf():
        return ancestors

    # ancestors can be obtained by merging leaves below the root.
    leaves_below_root = rootedTree.getLeavesBelowRoot()
    if atOriginalRoot and len(leaves_below_root)==rootedTree.leaves==2:
        #special case: we merge the last two leaves below the root
        leaf1 = leaves_below_root[0]
        leaf2 = leaves_below_root[1]
        label = mergeLabels(leaf1.rootLabel, leaf2.rootLabel)
        label = mergeLabels(label,rootedTree.rootLabel)
        return [LabelledUnorderedRootedTree(rootLabel=label)]

    for i in xrange(len(leaves_below_root)-1):
        l1 = leaves_below_root[i]
        ancestralSubtreeCounter_base = Counter(rootedTree.subtree_counts)
        ancestralSubtreeCounter_base[l1] -= 1
        for j in xrange(i+1,len(leaves_below_root)):
            l2 = leaves_below_root[j]

            ancestralSubtreeCounter = Counter(ancestralSubtreeCounter_base)
            ancestralSubtreeCounter[l2] -= 1

            assert l1.rootLabel != None
            assert l2.rootLabel != None

            newLeafLabel = mergeLabels(l1.rootLabel, l2.rootLabel)
            newLeaf = LabelledUnorderedRootedTree(rootLabel=newLeafLabel)
            ancestralSubtreeCounter[newLeaf] += 1

            ancestralSubtreeCounter = removeElementsWithCountZero(ancestralSubtreeCounter)

            ancestor = LabelledUnorderedRootedTree(rootLabel=rootedTree.rootLabel, subtree_counts=ancestralSubtreeCounter)

            ancestors.append(ancestor)

    if len(leaves_below_root) == 1 and rootedTree.rootDegree == 1:
        leafLabel = rootedTree.subtree_counts.keys()[0].rootLabel
        # if rootedTree.rootLabel is None:
        #     newLabel = leafLabel
        # else:
        #     newLabel = tuple(rootedTree.rootLabel, leafLabel)
        newLabel = mergeLabels(rootedTree.rootLabel, leafLabel)
        ancestor = LabelledUnorderedRootedTree(rootLabel=newLabel, subtree_counts=Counter())
        ancestors.append(ancestor)


    # now apply the whole thing to each sub-tree
    for subtree in rootedTree.getSubtrees_unique():

        ancestralSubtreeCounter_base = Counter(rootedTree.subtree_counts)
        ancestralSubtreeCounter_base[subtree] -= 1
        ancestralSubtreeCounter_base = removeElementsWithCountZero(ancestralSubtreeCounter_base)

        # for tree,count in rootedTree.subtree_counts.iteritems():
        #     if tree == subtree:
        #         if count > 1:
        #             ancestralSubtreeCounter_base[tree] = count - 1
        #     else:
        #         ancestralSubtreeCounter_base[tree] = count

        subtree_ancestors = compute_ancestors(subtree, atOriginalRoot=False)
        for subtree_ancestor in subtree_ancestors:
            ancestralCounter = Counter(ancestralSubtreeCounter_base)
            ancestralCounter[subtree_ancestor] += 1
            ancestor = LabelledUnorderedRootedTree(rootLabel=rootedTree.rootLabel, subtree_counts=ancestralCounter)
            ancestors.append(ancestor)

    assert all(T.nodes == rootedTree.nodes - 1 for T in ancestors)
    return ancestors

def mergeLabels(lab1,lab2):
    if lab1 is None:
        return lab2
    if lab2 is None:
        return lab1
    return '-'.join(sorted((str(lab1),str(lab2))))


def removeElementsWithCountZero(c):
    """Takes a counter c, and removes any key with count 0"""
    assert isinstance(c,Counter)
    c_new = Counter()
    for key,count in c.iteritems():
        if count > 0:
            c_new[key] = count
    return c_new


if __name__ == '__main__':
    run_tests()

def run_tests():
    pass