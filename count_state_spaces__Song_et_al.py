class configuration(object):
    '''Implements a configuration in the manner outlined in Song, Hein and Lyngsoe 2006 (DOI: 10.1109/TCBB.2006.31)'''

    def __init__(self, haplotypes, multiplicities):
        '''
        :param haplotypes: A series of equal length binary sequences fulfilling the three gamete test.
        :param multiplicities: A sequence of positive integers
        '''

        if len(haplotypes) != len(multiplicities):
            msg = 'List of haplotypes and multiplicities supplied with different lengths'
            raise ValueError(msg)
        if min(multiplicities) < 1:
            msg = 'Multiplicities <1 are invalid.'
            raise ValueError(msg)

        self.haplotypes = tuple(tuple(seq) for seq in haplotypes)
        if len(self.haplotypes) > len(set(self.haplotypes)):
            msg = 'Duplicate set of haplotypes supplied'
            raise ValueError(msg)

        if len(set(len(seq) for seq in self.haplotypes)) > 1:
            msg = 'All haplotypes must have equal length'
            raise ValueError(msg)

        self.multiplicities = tuple(multiplicities)
        self.haplotypes_with_counts__sorted = sorted( zip(self.haplotypes, self.multiplicities))

        #Sort haplotypes and multiplicities (according to haplotypes)
        self.haplotypes, self.multiplicities = zip(*self.haplotypes_with_counts__sorted)

        self.nSeq = sum(multiplicities)
        self.nSites = len(self.haplotypes[0])
        self.singletonSites = self.__computeSingletonSites__()

    def __hash__(self):
        return hash(self.haplotypes_with_counts__sorted)

    def __eq__(self, other):
        if type(self) == type(other):
            return self.haplotypes_with_counts__sorted == other.haplotypes_with_counts__sorted
        else:
            return False

    def __str__(self):
        return '\n'.join(['%s x %i'%(str(seq), n) for seq, n in self.haplotypes_with_counts__sorted])

    def __computeSingletonSites__(self):
        return filter(lambda i: sum(seq[i] != 0 for seq in self.haplotypes) == 1, xrange(self.nSites))


def __removeEntry__(l, i):
    if i >= len(l):
        raise IndexError('Index %i out of range for input list (which has %i elements)'%(i,len(l)))
    return tuple(filter(lambda x: l.index(x) != i, l))

def __replaceEntry__(l, i, newEntry):
    if i >= len(l):
        raise IndexError('Index %i out of range for input list (which has %i elements)'%(i,len(l)))
    return tuple( (x if l.index(x) != i else newEntry) for x in l)

# def D(conf, i):
#     new_haplotypes = __removeEntry__(conf.haplotypes, i)
#     new_multiplicities = __removeEntry__(conf.multiplicities, i)
#     return configuration(new_haplotypes, new_multiplicities)

def M(conf,i):
    if i not in conf.singletonSites:
        msg = 'Site %i is not a singleton site in the input configuration (remember, indexing starts form 0)' % i
        raise ValueError(msg)

    #find which haplotype is segregating at position i
    index_of_segregating_haplotype = 0
    while conf.haplotypes[index_of_segregating_haplotype][i] == 0 and index_of_segregating_haplotype < len(conf.haplotypes):
        index_of_segregating_haplotype += 1
    if index_of_segregating_haplotype == len(conf.haplotypes): ##check if we terminated because no seg. site was found
        msg = 'Site %i is not segregating in the input configuration'%i
        raise ValueError(msg)

    new_haplotype = __replaceEntry__(conf.haplotypes[index_of_segregating_haplotype], i, 0)

    #scan existing haplotypes to see if any of them match the new haplotype
    match_index = 0
    while conf.haplotypes[match_index] != new_haplotype and match_index < len(conf.haplotypes):
        match_index += 1

    if match_index == len(conf.haplotypes): #no match found
        new_multiplicities = __replaceEntry__(conf.multiplicities, index_of_segregating_haplotype, 1)
        new_haplotype_list = __replaceEntry__(conf.haplotypes, index_of_segregating_haplotype, new_haplotype)
    else: #The new haplotype already exists
        new_multiplicities = __replaceEntry__(conf.multiplicities, match_index, conf.multiplicities[match_index]+1)
        new_multiplicities = __removeEntry__(new_multiplicities, index_of_segregating_haplotype)
        new_haplotype_list = __removeEntry__(conf.haplotypes, index_of_segregating_haplotype)

    return configuration(new_haplotype_list, new_multiplicities)

def compute_product(l):
    return reduce(lambda x,y: x*y, l, 1)

def generate_out_neighbours(conf):
    return set( M(conf, i) for i in conf.singletonSites )

def song_et_al_algorithm(starting_configuration):

    #initialize Graph
    # (we encode it as a dictionary where each node is a key, and the value of each node is a list of all nodes in its
    # out-neighbourhood, i.e. the graph
    #         --->---
    #        /       \
    #  X -> Y -> Z -> W
    #
    # would be encoded as
    # { X : [Y],
    #   Y : [Z, W],
    #   Z : [W],
    #   W : [] }
    G = {starting_configuration: []}

    unresolved_nodes = {starting_configuration}
    while len(unresolved_nodes) > 0:
        node = unresolved_nodes.pop()
        out_neighbourhood = generate_out_neighbours(node)
        G[node] = out_neighbourhood

        for neighbour in out_neighbourhood:
            if neighbour not in G:
                G[neighbour] = []
                unresolved_nodes.add(neighbour)

    return sum(map(compute_product, [conf.multiplicities for conf in G.keys()]))