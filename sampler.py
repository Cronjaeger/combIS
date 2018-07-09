from time import clock
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from count_histories import LabelledUnorderedRootedTree, count_histories, unordered_rooted_leaf_labelled_tree_from_haplotype_matrix

def IS(data, theta, n_particles, Q, histDict = dict(), return_diagnostics = False):
    T = unordered_rooted_leaf_labelled_tree_from_haplotype_matrix(data)
    qs = np.zeros(n_particles)
    ps = np.zeros(n_particles)
    paths = []
    ps_full = []
    qs_full = []
    times = np.zeros(n_particles)
    t1_total = clock()
    for i in xrange(n_particles):
        t1 = clock()
        path,wq,wp,histDict = Q((T,),theta, histDict=histDict)
        t2 = clock()
        qs[i] = wq[0]
        ps[i] = wp[0]
        ps_full.append(wp)
        qs_full.append(wq)
        times[i] = t2 - t1
        paths.append(path)
    result = np.mean(ps/qs)
    t2_total = clock()
    ess = effective_sample_size(qs)
    diagnostics = {'ess':ess,
                   'histDict':histDict,
                   'paths':paths,
                   'weights':qs,
                   'path_probs':ps,
                   'path_prob_sequences':ps_full,
                   'weight_sequences':qs_full,
                   'times':times,
                   'time_total':t2_total - t1_total}
    if return_diagnostics:
        return result, diagnostics
    else:
        return result

def effective_sample_size(w):
    n = float(len(w))
    var = sum(w - 1/n) / (n - 1)
    return n / (1 + var)

def Q_combIS(partial_path, theta, wq = (1.0,) ,wp = (1.0,), histDict = dict(), symbolic_path = None):
    """ Sample according to the combinatorial importance sampler.
    Input is a partial path ans partial weights.
    Returns a tuple (p,wq,wp,histDict) consisting of a path, weight-chains, and a histDict"""

    # raise NotImplementedError('Not implemented yet!')

    state = partial_path[0]
    ancestors = compute_ancestors(state)
    if len(ancestors) == 0:
        assert state.nodes == 1
        if symbolic_path is None:
            return partial_path, wq, wp, histDict
        else:
            return partial_path, wq, wp, histDict, symbolic_path

    # compute weights
    histCounts = np.zeros(len(ancestors), dtype=float)
    transitionProbs = np.zeros(len(ancestors), dtype=float)
    mutationIndicator = np.zeros(len(ancestors), dtype=bool)
    trans_coal = coalescence_prob(state.leaves,theta)
    trans_mut  = mutation_prob(state.leaves,theta)
    for i,x in enumerate(ancestors):
        histCounts[i], histDict = count_histories(x, histDict)
        transitionProbs[i] = trans_coal if x.leaves < state.leaves else trans_mut
        mutationIndicator[i] = not x.leaves < state.leaves

    normalized_weights = histCounts / sum(histCounts)
    i_next = np.random.choice(len(ancestors), p=normalized_weights)

    if not (symbolic_path is None):
        symbolic_path = (mutationIndicator[i_next],) + symbolic_path
    partial_path = (ancestors[i_next],) + partial_path
    wq = (normalized_weights[i_next]*wq[0],) + wq
    wp = (transitionProbs[i_next]*wp[0],) + wp

    return Q_combIS(partial_path,theta,wq,wp,histDict, symbolic_path)


def Q_GT(partial_path, theta, wq=(1.0,), wp=(1.0,), histDict=dict()):
    """ Sample according to the Griffiths and Tavare sampler (with labelled leaves).
    Input is a partial path and partial weights.
    Returns a tuple (p,wq,wp,histDict) consisting of a path, weight-chains, and a histDict"""

    state = partial_path[0]
    ancestors = compute_ancestors(state)
    if len(ancestors) == 0:
        assert state.nodes == 1
        return partial_path, wq, wp, histDict

    # compute weights
    trans_coal = coalescence_prob(state.leaves, theta)
    trans_mut = mutation_prob(state.leaves, theta)

    # histCounts = np.zeros(len(ancestors), dtype=float)
    transitionProbs = np.zeros(len(ancestors), dtype=float)
    for i, x in enumerate(ancestors):
        # histCounts[i], histDict = count_histories(x, histDict)
        transitionProbs[i] = trans_coal if x.leaves < state.leaves else trans_mut

    weights = transitionProbs
    normalized_weights = weights / sum(weights)

    i_next = np.random.choice(len(ancestors), p=normalized_weights)

    partial_path = (ancestors[i_next],) + partial_path
    wq = (normalized_weights[i_next] * wq[0],) + wq
    wp = (transitionProbs[i_next] * wp[0],) + wp

    return Q_GT(partial_path, theta, wq, wp, histDict)


def Q_comb_GT(partial_path, theta, wq = (1.0,) ,wp = (1.0,), histDict = dict()):
    """ Sample according to the Griffiths and Tavare sampler with combinatorial correction.
    Input is a partial path and partial weights.
    Returns a tuple (p,wq,wp,histDict) consisting of a path, weight-chains, and a histDict"""

    # raise NotImplementedError('Not implemented yet!')

    state = partial_path[0]
    ancestors = compute_ancestors(state)
    if len(ancestors) == 0:
        assert state.nodes == 1
        return partial_path, wq, wp, histDict

    # compute weights
    trans_coal = coalescence_prob(state.leaves,theta)
    trans_mut  = mutation_prob(state.leaves,theta)

    histCounts = np.zeros(len(ancestors), dtype=float)
    transitionProbs = np.zeros(len(ancestors), dtype=float)
    for i,x in enumerate(ancestors):
        histCounts[i], histDict = count_histories(x, histDict)
        transitionProbs[i] = trans_coal if x.leaves < state.leaves else trans_mut

    weights = histCounts * transitionProbs
    normalized_weights = weights / sum(weights)

    i_next = np.random.choice(len(ancestors), p=normalized_weights)

    partial_path = (ancestors[i_next],) + partial_path
    wq = (normalized_weights[i_next]*wq[0],) + wq
    wp = (transitionProbs[i_next]*wp[0],) + wp

    return Q_comb_GT(partial_path, theta, wq, wp, histDict)

def Q_SD(partial_path, theta, wq=(1.0,), wp=(1.0,), histDict=dict()):
    """ Sample according to a Stephens and Donnelly-style sampler (labelled leaves; pick ancestors uniformly at random).
    Input is a partial path and partial weights.
    Returns a tuple (p,wq,wp,histDict) consisting of a path, weight-chains, and a histDict"""

    state = partial_path[0]
    ancestors = compute_ancestors(state)
    if len(ancestors) == 0:
        assert state.nodes == 1
        return partial_path, wq, wp, histDict

    # compute weights
    trans_coal = coalescence_prob(state.leaves, theta)
    trans_mut = mutation_prob(state.leaves, theta)

    # histCounts = np.zeros(len(ancestors), dtype=float)
    transitionProbs = np.zeros(len(ancestors), dtype=float)
    for i, x in enumerate(ancestors):
        # histCounts[i], histDict = count_histories(x, histDict)
        transitionProbs[i] = trans_coal if x.leaves < state.leaves else trans_mut

    weights = np.ones(len(ancestors), dtype=float)/len(ancestors)
    normalized_weights = weights / sum(weights)

    i_next = np.random.choice(len(ancestors), p=normalized_weights)

    partial_path = (ancestors[i_next],) + partial_path
    wq = (normalized_weights[i_next] * wq[0],) + wq
    wp = (transitionProbs[i_next] * wp[0],) + wp

    return Q_SD(partial_path, theta, wq, wp, histDict)

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


def plot_simple_likelihoods(N_particles = 10, N_replications = 30):
    S = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0]
    ])
    theta_watterson = 4 / (1 + 1.0/2 + 1.0/3 )
    theta_range = np.linspace(0.1, 15.1, 30, endpoint=False)
    likelihoods_GT = np.zeros((N_replications, len(theta_range)))
    likelihoods_SD = np.zeros((N_replications, len(theta_range)))
    likelihoods_combIS = np.zeros((N_replications, len(theta_range)))
    likelihoods_comb_GT = np.zeros((N_replications, len(theta_range)))
    hist_dict = dict()
    for j,theta in enumerate(theta_range):
        for i in xrange(N_replications):
            likelihoods_GT[i,j] = IS(S, theta, N_particles, Q_GT)
            likelihoods_SD[i,j] = IS(S, theta, N_particles, Q_SD)
            likelihoods_combIS[i,j], diagnostics = IS(S, theta, N_particles, Q_combIS, histDict=hist_dict, return_diagnostics=True)
            hist_dict = diagnostics['histDict']
            likelihoods_comb_GT[i,j], diagnostics = IS(S, theta, N_particles, Q_comb_GT, histDict=hist_dict, return_diagnostics=True)
            hist_dict = diagnostics['histDict']
    # print likelihoods

    means_GT = np.mean(likelihoods_GT, axis=0)
    means_SD = np.mean(likelihoods_SD, axis=0)
    means_combIS = np.mean(likelihoods_combIS, axis=0)
    means_comb_GT = np.mean(likelihoods_comb_GT, axis=0)
    vars_GT = np.var(likelihoods_GT, axis=0)
    vars_SD = np.var(likelihoods_SD, axis=0)
    vars_combIS = np.var(likelihoods_combIS, axis=0)
    vars_comb_GT = np.var(likelihoods_comb_GT, axis=0)

    filename_base = 'csv/likelihoods_Nparticles_%i__Nreplications_%i'%(N_particles,N_replications)
    np.savetxt(filename_base+'_thetas.csv', theta_range, delimiter=', ')
    np.savetxt(filename_base+'_GT.csv',likelihoods_GT, delimiter=',')
    np.savetxt(filename_base + '_SD.csv', likelihoods_SD, delimiter=',')
    np.savetxt(filename_base + '_combIS.csv', likelihoods_combIS, delimiter=',')
    np.savetxt(filename_base + '_comb_GT.csv', likelihoods_comb_GT, delimiter=',')

    plt.figure(1,)
    # plt.subplots_adjust(top=0.7)
    plt.subplot(2,2,1)
    plt.plot(theta_range, means_GT)
    plt.plot(theta_range, means_GT + np.sqrt(vars_GT), '--', color = 'grey')
    plt.plot(theta_range, means_GT - np.sqrt(vars_GT), '--', color = 'grey')
    plt.title('GT, N=%i' % N_particles)
    plt.ylabel(r'$\hat L(S \mid \theta)$')

    plt.subplot(2, 2, 2)
    plt.plot(theta_range, means_SD)
    plt.plot(theta_range, means_SD + np.sqrt(vars_SD), '--', color='grey')
    plt.plot(theta_range, means_SD - np.sqrt(vars_SD), '--', color='grey')
    plt.title('SD, N=%i' % N_particles)

    plt.subplot(2, 2, 3)
    plt.plot(theta_range, means_combIS)
    plt.plot(theta_range, means_combIS + np.sqrt(vars_combIS), '--', color='grey')
    plt.plot(theta_range, means_combIS - np.sqrt(vars_combIS), '--', color='grey')
    plt.title('CombIS, N=%i' % N_particles)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\hat L(S \mid \theta)$')

    plt.subplot(2, 2, 4)
    plt.plot(theta_range, means_comb_GT)
    plt.plot(theta_range, means_comb_GT + np.sqrt(vars_comb_GT), '--', color='grey')
    plt.plot(theta_range, means_comb_GT - np.sqrt(vars_comb_GT), '--', color='grey')
    plt.title('GT_comb, N=%i' % N_particles)
    plt.xlabel(r'$\theta$')

    #plt.suptitle('Approxiamte likelihoods fom IS with N_particles=%i particles'%N_particles)
    plt.show()

def run_tests():
    pass


if __name__ == '__main__':
    run_tests()
    plot_simple_likelihoods(N_particles = 10, N_replications=1000)

