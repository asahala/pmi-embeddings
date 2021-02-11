# pmi-embeddings
State-of-the-art count-based word vectors for low-resource languages with a special focus on historical languages, especially Akkadian and Sumerian.

1. ```src/make_embeddings.py``` creates PMI-based word vectors from the source corpus.
2. ```src/explore_embeddings.py``` make simple searches from embeddings. Requires a vector file and a dictionary (included in ```corpora/akkadian.zip```)
3. ```src/hypertune.py``` tests hyperparameters (by using brute force) to find the best settings for the given data set (requires a gold standard).
4. ```corpora/extract_corpus.py``` a script for extracting sense-disambiguated corpora from Korp-Oracc VRT files.
5. ```corpora/akkadian.zip``` a zipped test corpora and a dictionary of Akkadian language (use these to generate new embeddings and to explore them).
6. ```eval/gold.tsv``` an initial version of the [Akkadian gold standard](https://www.helsinki.fi/en/news/language-culture/creating-a-gold-standard-for-akkadian-word-embeddings).

## Jupyter tutorials
For those who like to use Jupyter Notebooks, ```src/jupyter_embeddings.ipynb``` instructs how to build your own word embeddings with just a few lines of code. ```scr/jypyter_explore_embeddings.ipynb``` guides how to make queries from embeddings.

For setting up Jupyter environment, read [this guide](https://github.com/niekveldhuis/compass/blob/master/1_Preliminaries/install_packages.ipynb) by Niek Veldhuis. Note that you only need the packages listed below (of which most are likely preinstalled in Conda/Jypyter).

## Requirements
```Python 3.6``` or newer, [numpy](https://numpy.org/), [scipy](https://www.scipy.org/) and [sklearn](https://scikit-learn.org/stable/). For evaluation scripts you will also need [gensim](https://pypi.org/project/gensim/).

## Features
```make_embeddings.py``` is a fast and efficient way to build PMI-based word embeddings from small (a few million words) text corpora. It combines findings from several recent research papers:

+ Dirichlet Smoothing (Turney & Pantel 2010; Jungmaier et al. 2020)
+ Context Similarity Weighting (Sahala & Linden 2020)
+ Shifted PMI (Levy et al. 2015) with different variants
+ Dynamic Context Windows (Sahlgren 2006; Mikolov et al. 2013; Pennington et al. 2014)
+ Subsampling (Mikolov et al. 2013)
+ Context Distribution Smoothing (Levy et al. 2015)
+ Eigenvalue Weighting (Caron 2001; Levy et al. 2015)
+ Dirty and clean stopwords (Mikolov et al. 2013; Levy et al. 2015)

### Input format
Lemmatized UTF-8 encoded text one word per line. Use symbol ´#´ to set window span constraints (i.e. text or paragraph boundaries) and ´_´ to indicate lacunae (breakages in cuneiform text) and ´\<stop\>´ to indicate stop words.

### Output format
Word2vec compatible raw text word vectors.

### Parameters and usage
Run script from the commmand line ```python3 make_embeddings.py corpusfile vectorfile [parameters]```. See [this document](https://docs.google.com/document/d/1TjVWqrhalCDjkOQf-JLk1jmC6N83MWGUIEVjbJpm9Es) for detailed information about the parameters and references. 

### Runtime performance
On Intel i5-2400 3.10GHz using a corpus of 1M words and a window size of 3, takes ca. 35 seconds to make basic embeddings and 50 seconds to make CSW-embeddings. On 2.1GHz Xeon Gold 6230 the runtimes are ca. 6 and 10 seconds respectively. Although this is quite fast, testing hundreds or thousands of combinations using ```hypertune.py``` may take a while.

### Version history
- 2021-01-26 -- ```make_embeddings.py``` added pmi-variants.
- 2021-01-23 -- ```make_embeddings.py``` no longer saves vectors for words that occur in completely broken contexts (Gensim doesn't like them).

### TODO:
- Add parsing directly from Oracc using Niek's script
- PMI-delta
- TF-IDF filtering
