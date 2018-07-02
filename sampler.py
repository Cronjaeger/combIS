from collections import Counter
from count_histories import LabelledUnorderedRootedTree

def sampler(data, params):
    pass

def Q_combIS(x0):
    """Returns a tuple (p,w) consisting of a path and a weight, sampled according to the combinatorial importance
    sampler"""
    # NOTE THE WEIGHT SHOULD BE THE SAME FRO ALL PATHS STARTED FROM THE SAME POINT!
    raise NotImplementedError('Not implemented yet!')

def Q_GT(x0):
    """Returns a tuple (p,w) consisting of a path and a weight, sampled according to the Griffiths and Tavare sampler"""
    raise NotImplementedError('Not implemented yet!')

def Q_comb_GT(x0):
    """Returns a tuple (p,w) consisting of a path and a weight, sampled according to the Griffiths and Tavare sampler
    with combinatorial correction"""


def compute_ancestrs(rootedTree):
    """computes all trees which can be reached from rooted tree, by either merging two leaves below the same node,
    or by merging a singleton leaf with the node above it (who to be clear is allowed to have no other children!) """

    ancestors = []
    if rootedTree.is_leaf():
        return ancestors

    # ancestors can be obtained by merging leaves below the root.
    leaves_below_root = rootedTree.getLeavesBelowRoot()
    for i in xrange(len(leaves_below_root)-1):
        l1 = leaves_below_root[i]
        ancestralSubtreeCounter_base = Counter(rootedTree.subtree_counts)
        ancestralSubtreeCounter_base[l2] -= 1
        for j in xrange(i+1,len(leaves_below_root)):
            l2 = leaves_below_root[j]
            newLeafLabel = (l1.rootLabel, l2.rootLabel)
            if rootedTree.rootDegree == 2:
                ancestor = LabelledUnorderedRootedTree(rootLabel=newLeafLabel)
            else:
                ancestor = LabelledUnorderedRootedTree(rootLabel=rootedTree.rootLabel, subtree_counts=ancestralSubtreeCounter_base)
            ancestors.append(ancestor)

    if len(leaves_below_root) == 1 and rootedTree.rootDegree == 1:
        leafLabel = rootedTree.subtree_counts.keys()[0].rootLabel
        if rootedTree.rootLabel is None:
            newLabel = leafLabel
        else:
            newLabel = tuple(rootedTree.rootLabel, leafLabel)
        ancestor = LabelledUnorderedRootedTree(rootLabel=newLabel, subtree_counts=Counter())
        ancestors.append(ancestor)


    # now apply the whole thing to each sub-tree
    for subtree in rootedTree.getSubtrees_unique():

        ancestralSubtreeCounter_base = Counter()
        for tree,count in rootedTree.subtree_counts():
            if tree == subtree:
                if count > 1:
                    ancestralSubtreeCounter_base[tree] = count - 1
            else:
                ancestralSubtreeCounter_base[tree] = count

        subtree_ancestors = compute_ancestrs(subtree)
        for subtree_ancestor in subtree_ancestors:
            ancestralCounter = Counter(ancestralSubtreeCounter_base)
            ancestralCounter[subtree_ancestor] += 1
            ancestor = LabelledUnorderedRootedTree(rootLabel=rootedTree.rootLabel, subtree_counts=ancestralCounter)
            ancestors.append(ancestor)


    return ancestors
