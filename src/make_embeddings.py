#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import itertools
import math
import numpy as np
import os
import random
import sys
import time
from collections import Counter
from collections import defaultdict
from scipy.sparse import csr_matrix, dok_matrix, lil_matrix
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

__version__ = "2021-01-26"

# Globals. All meta-symbols must have a string and integer form.
# Integers must be negative! 
PADDING = {'str': "#",      'int': -3}
STOP =    {'str': "<stop>", 'int': -2}
LACUNA =  {'str': "_",      'int': -1}

# If you use more symbols, add them here as well in their string form.
META_SYMBOLS = (PADDING['str'], LACUNA['str'], STOP['str'])

# Pretty print dividers 
DIV = "> " + "--" * 24

""" *´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*

 Word embeddings for cuneiform languages                           asahala 2020

 This script combines findings in several PMI+SVD and related research papers 
 to test their applicability in fragmentary and repetitive low-resource 
 languages such as Sumerian and Akkadian.

  + Dirichlet Smoothing                  (Jungmaier et al. 2020)
  + Context similarity weighting (CSW)   (Sahala & Linden 2020)
  + Shifted PMI                          (Levy et al. 2015)
  + Dynamic Context Window (DCW)         (Sahlgren 2006)
  + Subsampling                          (Mikolov et al. 2013)
  + Dirty and clean stopwords            (Mikolov et al. 2013)
  + Context Distribution Smoothing (CDS) (Levy et al. 2015)
  + Eigenvalue weighting                 (Caron 2001)
  + Window scaling                       (Church & Hanks 1990)
  + NPMI                                 (Bouma 2009)
  + PMI^2                                (Daille 1994)
  
 Input format:
   Lemmatized text one word per line. Use symbol ´#´ to set window
   span constraints (i.e. text or paragraph boundaries) and ´_´
   to indicate stop words and lacunae in Cuneiform text.

 Output format:
   Word2vec compatible raw text word vector files.

 TODO:
  - Asymmetric co-occurrences for stopwords

 Credits:
  Code cannibalized from Jacob Jungmaier and Omer Levy is credited where
  it is due.

 Version history:
   2021-01-26       Add PMI^2 and NPMI
   2021-01-23       Filter non-zero vectors to pacify Gensim warnings.
   2021-01-20       Add dirty stop words
   2023-06-29       Change PMI shift formula *= --> -= and avoid log2(0)

*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`* """ 

def get_size(obj, seen=None):
    """ Recursively finds size of objects
        copied from: https://stackoverflow.com/questions/449560/ """

    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


class CSW:

    """ Context Similarity Weighting as in Sahala & Linden (2020). 
    ´Improving Word Association Measures with Context Similarity Weighting´
    KDIR 2020.

    Improves results in repetitive corpora such as Oracc.

         2020-12-09        Improved memory usage. Uses now lookup instead 
                           of storing all the windows. Still a better 
                           solution is needed for reasonable space 
                           complexity.

                           Lookup indices should be stored as generators 
                           but I haven't yet found an efficient way to do 
                           it because the bigrams would have to be extracted 
                           in sorted order. Using itertools for chaining 
                           generators is not helpful.
                            
                           contexts = [[c1], [c2], ... [cn]]
                           lookup = {(bigram): [i, ...], ...}

                           where i refers to index of the contexts where bigram
                           was seen. 

                           TODO: - removed duplicate contexts
                                 - store bigrams only from a -> b """

    def __init__(self):
        self.contexts = []
        self.context_lookup = defaultdict(list)
        self.s = 0

    @property
    def size(self):
        return get_size(self.contexts) + get_size(self.context_lookup)
        
    def add_lookup(self, bigram, id_, discard):
        """ Add bigram to lookup if the requirements are met

        :param bigram              bigram in (int, int) format
        :param id_                 window index id
        :param discard             set of words to be discarded

        :type bigram               (int, int)
        :type id_                  int
        :type discard              {int, ...} """

        keyword, collocate = bigram
        if collocate < 0:
            return
        if keyword in discard:
            return
        if collocate in discard:
            return
        self.context_lookup[bigram].append(id_)
        return True

    def add_context(self, context):
        """ Add context window to lookup
        :param context            context window
        :type context             [int, ...] """
        self.contexts.append(context)

    def _get_phi(self, contexts):
        """ Return context similarity weight φ over the
        given co-occurrence contexts 

        :param contexts           list of context windows
        :type contexts            [[int, ...], ...]"""

        if len(contexts) == 1:
            return 1.0
        else:
            prop_vector = []
            for index in zip(*contexts[::-1]):
                Vi = len(set(index) - set([-1, -2, -3]))
                Wi = len([i for i in index if i >= 0])
                proportion = Vi / max(Wi, 1)
                if proportion > 0:
                    prop_vector.append(proportion)
            if len(prop_vector) == 0:
                return 1
            return sum(prop_vector) / len(prop_vector)

    def compute(self, dim, k_factor=0, verbose=False):
        """ Returns Context similarity Weights as a sparse
        matrix 

        :param dim            vocabulary size
        :param k_factor       CSW k-factor
        
        :type dim             int
        :type k_factor        int / float """
        
        i = 0
        weight_matrix = lil_matrix((dim, dim), dtype=float)
        items = len(self.context_lookup)

        for bigram, indices in self.context_lookup.items():
            w1, w2 = bigram
            contexts = [self.contexts[i] for i in indices]
            d = self._get_phi(contexts) ** k_factor
            weight_matrix[w1, w2] = d
            
            if i % 100000 == 0 and verbose:
                print("   %i / %i" % (i, items), end="\r")
            i += 1

        if verbose:
            print("   %i / %i" % (i, items), end="\r")

        return weight_matrix.tocsr()


class Cooc:

    """ Co-occurrence calculations. Takes in a file and returns a
    Context Similarity Weighted PMI matrix """

    def __init__(self, filename, chunksize=400000, dynamic_window=False,
                 min_count=1, subsampling_rate=0.0, window_size=5,
                 k_factor=0, window_scaling=False, dirty_stopwords=False,
                 verbose=True):

        """ 
        :param filename            input file
        :param chunksize           lines per chunk
        :param min_count           discard word below this frequency
        :param subsampling_rate    subsampling rate as in word2vec
        :param verbose             print processing info
        :param distance_scaling    take distance in to account in cooc counts
        :param window_size         window size (keyword included)
        :param k_factor            CSW k-factor
        :param dirty_stopwords     remove disallowed words from corpus
        :param window_scaling      scale co-occurrences with window size
        
        :type filename             str
        :type chunksize            int
        :type min_count            int
        :type subsampling_rate     float
        :type verbose              bool
        :type distance_scaling     bool
        :type window_size          int
        :type k_factor             int / float
        :type dirty_stopwords      bool
        :type window_scaling       bool """
    
        self.filename = filename
        self.chunksize = chunksize
        self.min_count = min_count
        self.subsampling_rate = subsampling_rate
        self.verbose = verbose
        self.dynamic_window = dynamic_window
        self.k_factor = k_factor
        self.window_size = window_size
        self.window_scaling = window_scaling
        self.csw = self.k_factor > 0
        self.dirty_stopwords = dirty_stopwords

        self.rand = random.Random(0)
        self.word_count = defaultdict(int)
        self.pad = [PADDING['int']] * self.window_size 

        self.context_weights = CSW()
        self.window_id = -1

        self.chunks = 0
        self.time = 0

        """ Scale co-occurrence freqs by window size to ensure
        Σ f(a,b) = Σ f(a) = Σ f(b) = N over PMI(*,*). Church & Hanks 1990
        only do this for asymmetric windows. This attempts to generalize
        it for symmetric windows as well. """

        self.scale = 1
        if self.window_scaling:
            if self.window_size > 1:
                self.scale = (self.window_size - 1) * 2
            else:
                self.scale = 2

    @property
    def pmi_matrix(self):
        return self.pmi

    def _read_file(self, only_vocab=False):
        """ Read file line-by-line to save memory. The file is read into chunks
        and spans. Chunks are simply N line long parts of the file and spans 
        are lines, paragraphs, sentences or texts within the chunks. When
        co-occurrences are calculated, the window is not allowed to extend
        from span to another!

        To customize the span symbol, change global variable ´PADDING´

        :param only_vocab            Ignore words that do not satisfy
                                     given constratins such as min_count

        :type only_vocab             bool """
        
        span = []
        chunks = []
        i = 0
        with open(self.filename, encoding='utf-8') as corpus:
            word = corpus.readline().rstrip()
            while word:
                i += 1
                if word != PADDING['str']:
                    if only_vocab:
                        if self.dirty_stopwords:
                            """ If dirty stop words, remove all disallowed
                            words from the corpus as in Levy et al. 2015;
                            For cuneiform languages, we remove only stop
                            words and < min_counts, but not lacunae! """
                            if word == STOP['str']:
                                pass
                            elif word in self.vocabulary_set:
                                span.append(self.word_to_id[word])
                        else:
                            """ If clean stop words, add as they are
                            and replace non-vocabulary items with negative
                            indices """
                            if word in self.vocabulary_set:
                                span.append(self.word_to_id[word])
                    else:
                        span.append(word)
                else:
                    """ append spans into chunks and yield """
                    if i >= self.chunksize:
                        chunks.append(span)
                        yield chunks
                        span = []
                        chunks = []
                        i = 0
                    else:
                        chunks.append(span)
                        span = []
                word = corpus.readline().rstrip()
            chunks.append(span)
            yield chunks        

    def _count_freqs(self):
        """ Count word freqs, make vocabulary and subsampling dictionary """
        spans = 0
        corpus_size = 0

        if self.verbose:
            print(DIV)
            print("> Reading %s..." % self.filename)

        for chunks in self._read_file(only_vocab=False):
            self.chunks += 1
            for span in chunks:
                spans += 1
                for word in span:
                    self.word_count[word] += 1
                    corpus_size += 1

        """ Make vocabulary of words satisfying freq constraints """
        self.vocabulary = [w for w, c in sorted(self.word_count.items(),
                                                key=lambda x:x[1],
                                                reverse=True)
                           if c >= self.min_count]

        """ Represent vocabulary as set for faster filtering """
        self.vocabulary_set = frozenset(self.vocabulary)

        """ Make word->id mapping ordered by word frequency. This allows
        representing the vocabulary as sparse matrix row and column indices
        and saves some memory in CSW """
        self.word_to_id = {w: i for i, w in
                           enumerate(self.vocabulary) if w not in META_SYMBOLS}

        """ Give negative index for all meta symbols """
        self.word_to_id[STOP['str']] = STOP['int']
        self.word_to_id[LACUNA['str']] = LACUNA['int']

        """ Make a set of words that are discarded in CSW, by default
        all words that cannot possibly have more than one co-occurrence with
        any other word (i.e. we want to only save windows that matter) """
        self.csw_discard = frozenset([self.word_to_id[w] for w in self.vocabulary_set
                        if self.word_count[w] < 2])

        """ Prepare subsampling: calculate discard probabilities. """
        if self.subsampling_rate:
            ssr = self.subsampling_rate * corpus_size
            self.subsampling_dict = {self.word_to_id[w]: 1-math.sqrt(ssr/c)
                                     for w, c in self.word_count.items()
                                     if c > ssr}

        if self.verbose:
            lacunae = self.word_count.get(LACUNA['str'], 0)
            stops = self.word_count.get(STOP['str'], 0)
            print("   Corpus statistics:")
            print("      spans       {}".format(spans))
            print("      tokens      {}".format(corpus_size))
            print("      types       {}".format(len(self.vocabulary_set)))
            print("      lacunae     {}".format(lacunae))
            print("      stopwords   {}".format(stops))
            print("      frag. rate: %.2f" % (lacunae/corpus_size))

            if self.subsampling_rate:
                print("> Subsampled words: {}".format(ssr))
                print("> Words in subsampling dictionary: {}"\
                      .format(len(self.subsampling_dict)))

    def _extract_windows(self, chunk):
        """ Extract symmetric oc-occurrence windows from text chunks"""
        wz = self.window_size
        distance = 1

        """ Split input text into windows; note that at this point
        the word are already represented as integers! 0 is the most
        common word 1 the second etc. Negative indices are meta-symbols """
        for w in zip(*[chunk[i:] for i in range(1+wz*2)]):
            center = w[wz]
            
            """ Skip paddings, stopwords and lacunae """
            if center < 0:
                continue
                    
            """ If CSW is on, save co-occurrence context """
            if self.csw:
                if center not in self.csw_discard:
                    context = w[0:wz] + w[wz+1:]
                    self.context_weights.add_context(context)
                    self.window_id += 1

            for i, bigram in enumerate(itertools.product([center], w)):
                """ Disallow center word occurring with itself """
                if i == wz:
                    continue

                """ Add window id to lookup if CSW is on. The lookup
                contains all contexts where (a, b) is attested """
                if self.csw:
                    self.context_weights.add_lookup(bigram,
                                                    self.window_id,
                                                    self.csw_discard)
                """ Add co-occurrence to sparse matrix if not stopword,
                lacuna or padding. If dynamic window is set, reduce
                co-occurrence count based on words' mutual distance """
                if bigram[-1] >= 0:
                    self.row.append(center)
                    self.col.append(bigram[-1])
                    if self.dynamic_window:
                        distance = 1 / abs(i - wz)
                    self.data.append(1 * distance)

    def count_cooc(self):
        """ This method creates a co-occurrence matrix of
        the given corpus """
        st = time.time()

        """ Initialize word counts to set dimensions, subsampling,
        word indices etc. """
        self._count_freqs()

        """ Initialize vocab x vocab sized sparse co-occurrence matrix """
        dim = len(self.vocabulary)
        self.cooc_matrix = csr_matrix((dim, dim), dtype=int)

        et = time.time() - st
        self.time += et

        if self.verbose:
            print("    (%.2f seconds)" % (et))
            print(DIV)
            print("> Extracting bigrams...")
        i = 1

        """ Extract windows and count co-occurrences. Read the input text
        second time but discard subsampled and too rare words if min_count
        is set.  """
        ctime = 0
        ttime = 0
        for chunks in self._read_file(only_vocab=True):

            """ Initialize matrix row, col and data """
            self.row = []
            self.col = []
            self.data = []

            st = time.time()
            if self.verbose:
                print('   %i / %i (%.2f)' % (i, self.chunks, ctime), 
                      sep='\t', end='\r')

            """ Skip empty spans (full of lacunae), do subsampling if set and
            pad the windows to equal length for CSW """
            for span in chunks:
                if not span or len(span) == 1:
                    continue

                # Would be faster if done in _read_file()
                """ Filter out subsampled words """
                if self.subsampling_rate:
                    span = [w for w in span if (w not in self.subsampling_dict
                                or self.rand.random() > self.subsampling_dict[w])]

                """ Pad spans to properly align peripheral windows in CSW """
                padded = self.pad + span + self.pad
                self._extract_windows(padded)

            """ Update co-occurrence matrix """
            self.cooc_matrix += csr_matrix((self.data, (self.row, self.col)),
                                           shape=(dim, dim), dtype=float)
            
            ctime = round(time.time() - st, 3)
            ttime += ctime
            i += 1

        if self.verbose:
            print('> Matrix size: {} ({} kB)'\
                  .format(np.prod(self.cooc_matrix.shape),
                          round(self.cooc_matrix.data.nbytes/1000)))
            print('> Non-zero elements: %i' % (self.cooc_matrix.nnz)) 
            print('    (%.2f seconds)' % (ttime))
            print(DIV)

        self.time += ttime

        """ Calculate CSW and take Hadamard product of the co-occurrence matrix
        and the context similiarity weight matrix. """
        if self.csw:
            st = time.time()
            if self.verbose:
                print('> Calculating context similarities...')
            self.cooc_matrix = self.cooc_matrix.multiply(
                self.context_weights.compute(dim, self.k_factor, self.verbose)
            )

            et = time.time() - st
            if self.verbose:
                print('    (%.2f seconds)' % (et))
                print(DIV)
                # getting the size of the lookup is very slow, only for debugging
                #print(self.context_weights.size)
            del self.context_weights

        self.time += et

    def calculate_pmi(self, shift_type=0, alpha=None, lambda_=None, threshold=0,
                      pmi_variant=None):
        """ Calculate Shifted PMI matrix with various modifications

        :param lambda_                  Dirichlet smoothing
        :param alpha                    Context distribution smoothing
        :param threshold                PMI shift value k (see formulae below)
        :param shift_type               set PMI shift type
                                          0: Jungmaier et al. 2020
                                          1: Levy & Goldberg 2014
                                          2: Experimental: My old def of L&G???
                                          3: Experimental: linear addition
        :param variant                  set PMI variant

        :type lambda_                   float (recommended: 0.0001)
        :type alpha                     float (recommended: 0.75)
        :type threshold                 integer (useful values: 0-10)
        :type shift_type                integer (0, 1, 2)
                                          0: PMI(a,b) < -k = 0
                                          1: max(PMI(a,b) - log2(k), 0)
                                          2: max(PMI(a,b) * log2(k), 0)
                                          3: max(PMI(a,b) + k, 0)
        :type variant                   str (pmi2, npmi)

        For PMI(*,*) it is more efficient to calculate the scores using
        elementwise matrix multiplication thanks to optimization in 
        numpy/scipy as in Levy et al. 2015. For each element (a,b) in the 
        matrix:

                         p(a,b)                               1
        PMI(a,b) = log2 --------   =  log2 ( N * f(a,b) * -------- )
                        p(a)p(b)                          f(a)f(b)    

        """

        st = time.time()
        if self.verbose:
            print("> Calculating PMI...")

        sum_a = np.array(self.cooc_matrix.sum(axis=1))[:, 0]
        sum_b = np.array(self.cooc_matrix.sum(axis=0))[0, :]

        """ Dirichlet and context distribution smoothing as in
           Jacob Jungmaier (Accessed: 2020-12-01):
              https://github.com/jungmaier/dirichlet-smoothed-word-embeddings/
           Omer Levy (Accessed: 2019-05-30)
              https://bitbucket.org/omerlevy/hyperwords/ """
        if lambda_ is not None:
            if self.verbose:
                print("> Dirichlet smoothing λ={}".format(lambda_))
            sum_a += (lambda_ * self.cooc_matrix.shape[0])
            sum_b += (lambda_ * self.cooc_matrix.shape[0])
            # TODO: do not modify the cooc_matrix directly! May cause
            #       problems if someone runs this method several times
            #       without recalculating the matrix!
            self.cooc_matrix.data = self.cooc_matrix.data + lambda_
        if alpha is not None:
            if self.verbose:
                print("> Context distribution smoothing α={}".format(alpha))
            sum_b = sum_b ** alpha

        sum_total = sum_b.sum()
        self.pmi = csr_matrix(self.cooc_matrix)
        
        """ Scale co-oc frequencies by window size; this has to
        be done after row and column summation """
        if self.scale != 1:
            self.pmi *= self.scale
        
        """ Calculate PMI (take reciprocals and multiply all
        columns and rows with reciprocal * sum to get the
        products of marginal probabilities. """
        with np.errstate(divide='ignore'):
            sum_a = np.reciprocal(sum_a)
            sum_b = np.reciprocal(sum_b)
        self.pmi = self.pmi.multiply(sum_a)\
                           .multiply(sum_b[:, None]) * sum_total
        self.pmi.data = np.log2(self.pmi.data)

        """ Various PMI derivations:
        
        NPMI          (Gerlof Bouma 2009): PMI / -log2 p(a,b)
        PMI^2         (Daille 1994): I define PMI^2 here as 
                      PMI - ((1+x) * -log2 p(a,b)), where x is a small smoothing
                      factor to make sure perfect dependencies are not
                      confused with null co-occurrences in the sparse matrix, 
                      as PMI^2 has bounds of 0 > log2 p(a,b) > -inf. x = 0.0001
                      should be enough for corpora of few million words 
                      to avoid any bigram getting a score of 0.0 """

        if pmi_variant == 'npmi':
            joint_dist_matrix = self.cooc_matrix.data * (1/sum_total)
            self.pmi.data = self.pmi.data / -np.log2(joint_dist_matrix.data)
        elif pmi_variant == 'pmi2':
            joint_dist_matrix = self.cooc_matrix.data * (1/sum_total)
            self.pmi.data -= (1.0001 * -np.log2(joint_dist_matrix.data))

        """ Apply threshold for Shifted PMI. """
        if threshold is not None:
            if shift_type == 0:
                self.pmi.data[self.pmi.data < -threshold] = 0
            elif shift_type == 1:
                self.pmi.data -= np.log2(min(threshold, 1))
                self.pmi.data[self.pmi.data < 0] = 0
            elif shift_type == 2:
                self.pmi.data *= np.log2(max(threshold, 2))
                self.pmi.data[self.pmi.data < 0] = 0                                
            elif shift_type == 3:
                self.pmi.data += threshold
                self.pmi.data[self.pmi.data < 0] = 0
    
        et = time.time() - st
        self.time += et

        if self.verbose:
            print('    (%.2f seconds)' % (et))
            print(DIV)

    def factorize(self, dimensions, eigenvalue_weighting=0.0):
        """ Perform truncated SVD for the PMI matrix. 
        
        Adapted from Jungmaier 
        htps://github.com/jungmaier/dirichlet-smoothed-word-embeddings """
        
        st = time.time()

        # Do not modify pmi_matrix directly
        matrix = self.pmi_matrix

        if self.verbose:
            print("> SVD...")
        
        if eigenvalue_weighting == 1:
            """ Traditional SVD as in pmizer's pmi2vec """
            svd = TruncatedSVD(n_components=dimensions, random_state=0)
            matrix = svd.fit_transform(matrix)
        elif not eigenvalue_weighting:
            """ Ignore eigen value matrix. Improves performance """
            matrix, _, _ = randomized_svd(matrix,
                                     n_components=dimensions,
                                     random_state=0)
        else:
            """ Weight eigen value matrix """
            matrix, s, _ = randomized_svd(matrix, 
                                     n_components=dimensions,
                                     random_state=0)
            sigma = np.zeros((matrix.shape[0], matrix.shape[1]))
            sigma = np.diag(s ** eigenvalue_weighting)
            matrix = matrix.dot(sigma)

        if self.verbose:
            print("> Normalizing vectors...")

        """ Normalize vectors to unit length; reported to increase
        performance in Levy et al 2015 and Wilson and Schakel 2015.
        After normalization over all v elements of the vector:
        Σ v^2 = 1 """
        self.svd_matrix = normalize(matrix, norm='l2', axis=1, copy=False)
        del matrix

        et = time.time() - st
        self.time += et

        if self.verbose:
            print('    (%.2f seconds)' % (et))
            print(DIV)

    def save_vectors(self, filename):        
        """ Save non-zero word vectors (i.e. ignore words that only
        occur in completely broken contexts, which may cause zero-
        division errors in certain word vector tools). 

        :param filename              vector file name
        :type filename               str                       """

        st = time.time()

        if self.verbose:
            print("> Filtering zero-vectors...")

        # This is a temporary fix. Would be better to do in the fly
        # E.g. by deleting isolated words in the middle of broken
        # passages.
        def get_nonzero(count_only):
            for i, word in enumerate(self.vocabulary, start=1):
                vector = self.svd_matrix[self.word_to_id[word], :]
                if sum(vector) != 0 and word not in META_SYMBOLS:
                    if count_only:
                        yield 1
                    else:
                        yield word + " " + " ".join(map(str, vector))

        vocab_size = sum(get_nonzero(count_only=True))

        if self.verbose:
            print("> Saving %i non-zero vectors (%i discarded)... " \
                  % (vocab_size, self.svd_matrix.shape[0]-vocab_size))

        with open(filename, "w", encoding="utf-8") as vector_file:
            """ Vector file header """
            vector_file.write("%i %i\n" % (vocab_size,
                                           self.svd_matrix.shape[1]))
            for vector in get_nonzero(count_only=False):
                vector_file.write(vector + '\n')

        if self.verbose:
            et = time.time() - st
            print('    (%.2f seconds)' % (et))
            print(DIV)


if __name__ == "__main__":

    desc = "Calculates word embeddings by using Shifted PPMI \
    (Levy, Dagan & Goldberg 2015), SVD, Dirichlet Smoothing \
    (Jungmaier, Kassner & Roth 2020) and Context Similarity Weighting \
    (Sahala & Linden 2020)."

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('corpus_file',
                        help='Text file with word tokens separated \
                              by newlines.')
    parser.add_argument('word_vector_filename',
                        help='Desired name of the word vector file.')
    parser.add_argument('--window_size', '-w', type=int, default=3,
                        help='Size of the window around a word for \
                              co-occurrence counts, integer. Default: 5.')
    parser.add_argument('--cds', type=float, default=None,
                        help='Context Distribution Smoothing\
                        (if used, recommended α = 0.75)')
    parser.add_argument('--window_scaling', action='store_true', default=False,
                        help='Scale co-oc by window size')
    parser.add_argument('--dirty_stopwords', action='store_true', default=False,
                        help='Remove unwanted words without placeholders')
    parser.add_argument('--dynamic_window', action='store_true', default=False,
                        help='Dynamic Window as in GloVE and Word2vec')
    parser.add_argument('--k_value', '-k', type=float, default=0.0,
                        help='CSW k-value')
    parser.add_argument('--min_count', '-m', type=int, default=1,
                        help='Minimal word count for words to process, \
                              integer, useful range 1-10. Default: 1.')
    parser.add_argument('--subsampling', '-s', type=float, default=0.0,
                        help='Subsampling rate (similar to word2vec). \
                              Default: 0.0 (no subsampling).')
    parser.add_argument('--chunk_size', '-c', type=int, default=400000,
                        help='Chunk size in bytes for chunkwise processing \
                              of the corpus. Default: 3000000. If memory \
                              overflow encountered, choose lower value.')
    parser.add_argument('--dimensions', '-d', type=int, default=300,
                        help='Size of word embeddings, integer. Default: 300.')
    parser.add_argument('--pmi_variant', type=str, default=None,
                        help='PMI variant: npmi, pmi2')
    parser.add_argument('--smoothing_factor', '-l', type=float, default=None,
                        help='Smoothing factor lambda, float, \
                              useful range: 0-1. Default: 0.0001.')
    parser.add_argument('--threshold', '-t', type=int, default=None,
                        help='Shifted PPMI threshold:\
                              Default: 0 -> PPMI.')
    parser.add_argument('--shift_type', type=int, default=0,
                        help='PMI matrix shift type. \
                        0 = replace PMI(a,b) < -k with 0, \
                        1 = max(PMI(a,b) - log(k), 0), \
                        2 = max(PMI(a,b) + k, 0).')
    parser.add_argument('--eigenvalue_weighting', '-e', type=float,
                        default=0.0, help='Weighting of singular values \
                                           (Sigma), float, useful range: 0-1. \
                                           Default: 0.0 (singular values are \
                                           not considered).')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output.')
    args = parser.parse_args()

    print(DIV)
    print(args)
    """ Warnings """
    if args.pmi_variant is not None:
        print(DIV)
        if args.threshold is not None:
            print('> Warning: --threshold and --pmi_variant are '\
                  'generally incompatible.')

    """ Init embeddings object """
    embeddings = Cooc(args.corpus_file,
                      chunksize=args.chunk_size,
                      window_size=args.window_size,
                      min_count=args.min_count,
                      subsampling_rate=args.subsampling,
                      k_factor=args.k_value,
                      dynamic_window=args.dynamic_window,
                      window_scaling=args.window_scaling,
                      dirty_stopwords=args.dirty_stopwords,
                      verbose=args.verbose)

    """ Make CSW and Cooc matrix """
    embeddings.count_cooc()

    """ Make PMI matrix """
    embeddings.calculate_pmi(shift_type=args.shift_type,
                             alpha=args.cds,
                             lambda_=args.smoothing_factor,
                             threshold=args.threshold,
                             pmi_variant=args.pmi_variant)

    """ Factorize sparse matrix  """
    embeddings.factorize(dimensions=args.dimensions,
                         eigenvalue_weighting=args.eigenvalue_weighting)

    """ Save word vectors from embeddings object """
    embeddings.save_vectors(filename=args.word_vector_filename)
