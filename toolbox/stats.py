import numpy as np
from fast_histogram import histogram1d

class MultBinner(object):
    def __init__(self, bin_edges, nchannels):
        self.binners = [None] * nchannels
        self.nchannels = nchannels
        if type(bin_edges) == type([]):
            assert (len(bin_edges) == nchannels)
            for i in range(nchannels):
                self.binners[i] = BINNER(bin_edges[i])
        else:
            for i in range(nchannels):
                self.binners[i] = BINNER(bin_edges)

    def bin(self, arr, right=True):
        assert (arr.shape[0] == self.nchannels)
        for i in range(self.nchannels):
            self.binners[i].bin(arr[i], right=right)

    def get_info(self):
        ret = {}
        for i in range(self.nchannels):
            ret[i] = {"bin_centers": self.binners[i].bin_center,
                      "hist": self.binners[i].storage,
                      "bin_edges": self.binners[i].bin_edges}
        return ret


class BINNER(object):
    def __init__(self, bin_edges):
        bin_lower = bin_edges[:-1].copy()
        bin_upper = bin_edges[1:].copy()
        bin_lower = bin_lower[:len(bin_upper)]

        self.bin_edges = bin_edges
        self.bin_lower = bin_lower
        self.bin_upper = bin_upper
        self.bin_center = (bin_lower + bin_upper) / 2.
        self.bin_sizes = bin_upper - bin_lower + 1
        self.nbin = len(bin_lower)
        self.storage = np.zeros(len(self.bin_center))

        assert (self.bin_sizes > 0).all()

    def bin(self, arr, right=True):
        digitized = np.digitize(arr.flatten(), self.bin_edges, right=right)
        for i in range(self.nbin):
            self.storage[i] += np.sum(digitized == i)
        return self.bin_center, self.storage


class FastMultBinner(object):
    def __init__(self, ranges, nbins, nchannels):
        self.binners = [None] * nchannels
        self.nchannels = nchannels
        if type(ranges) == type([]):
            assert (len(ranges) == nchannels)
            assert (len(ranges) == len(nbins))
            for i in range(ranges):
                self.binners[i] = FastBINNER(ranges[i][0], ranges[i][1], nbins[i])
        else:
            for i in range(nchannels):
                self.binners[i] = FastBINNER(ranges[0], ranges[1], nbins)

    def bin(self, arr, weights=None, verbose=False):
        assert (arr.shape[0] == self.nchannels)
        for i in range(self.nchannels):
            if verbose: print("binning {}".format(i))
            self.binners[i].bin(arr[i], weights=weights)

    def get_info(self):
        ret = {}
        for i in range(self.nchannels):
            ret[i] = {"bin_centers": self.binners[i].bin_center,
                      "hist": self.binners[i].storage,
                      "bin_edges": self.binners[i].bin_edges}
        return ret


class FastBINNER(object):
    def __init__(self, minval, maxval, nbins):
        self.minval = minval
        self.maxval = maxval
        bin_edges = np.linspace(minval, maxval, nbins + 1)
        bin_lower = bin_edges[:-1].copy()
        bin_upper = bin_edges[1:].copy()
        bin_lower = bin_lower[:len(bin_upper)]

        self.bin_edges = bin_edges
        self.bin_lower = bin_lower
        self.bin_upper = bin_upper
        self.bin_center = (bin_lower + bin_upper) / 2.
        self.bin_sizes = bin_upper - bin_lower + 1
        self.nbin = int(len(bin_lower))
        self.storage = np.zeros(len(self.bin_center))

        assert (self.bin_sizes > 0).all()

    def bin(self, arr, weights=None):
        self.storage += histogram1d(arr, self.nbin, [self.minval, self.maxval], weights=weights)
        return self.bin_center, self.storage
