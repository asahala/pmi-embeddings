#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import gzip
from collections import defaultdict
from collections import Counter

__version__ = "2021-01-19"

""" Korp-Oracc VRT project extractor ===================================

                                                            asahala 2020

Extracts texts by project from Korp-Oracc VRT and disambiguates
lemmas by their given translations. The script generates corpus files
that can be used with Pmized and pmi-embeddings toolkits.

Lemmas found in the EBL lexicon are given Roman numerals following
CDA (e.g. rabû_I, rabû_II...), if the lemma is not found in EBL
lexicon, they will be enumerated by using Arabic numerals in ascending
order by their frequency (e.g. dummy_1, dummy_2 ...)

NOTE: 
The frequencies are based on ´lexicon.tsv´ that contains all Akkadian
words from Oracc as of 2020. Because Oracc is a constantly accruing
resource and new words may appear (and their frequency distributions
change), it is wise to save the dictionary file generated by this script
to keep track which numerals point to which meanings.

If lemmas with _? prefix occur, it means that the corpus contains
previously unseen lemmas and that the ´lexicon.tsv´ should be updated


OUTPUT FILE STRUCTURE:

   CORPUS:
      Text per line or word per line. Lemmas disambiguated, lacunae marked with
      underscores and stop words wrapped in brackets.

   DICTIONARY:
      TSV with the following fields:
          lemma   lemma_freq   translation1[freq]   translation2[freq] ...


HOW TO USE:

   (1) Add desired projects to ´PROJECTS´
   (2) Add desired stop words to ´STOP_WORDS´
   (3) Add desired input and output (without extension) file names
   (4) Set ´WPL´ to False if you want text per line instead of word per line
   (5) Set ´FLATTEN_STOPS´ to True if you want to consider stop words as lacunae
   (5) Run the script


TODO:

   - Possibility to extract data directly from Oracc JSON

"""

# Globals
OUTPUT_FILE = 'akk_corpus'         # Output filename
INPUT_FILE = 'oracc_akkadian.vrt'  # Input VRT-file

PROJECTS = ['adsd', 'saao', 'rinap', 'ribo', 'riao']
STOP_POS = ['pronoun',
            'number',
            'interjection',
            'particle',
            'conjunction',
            'prepositionpostposition']
FLATTEN_STOPS = False
WPL = True

# Lexicon files
EBL = 'lex/oracc_ebl.gz'
BYFREQ = 'lex/oracc_by_freq.gz'


def write(filename, content):
    print('> Writing %s...' % filename)
    with open(filename, 'w', encoding='utf-8') as data:
        data.write(content.replace(' \n ', '\n'))

def make_dictionary(dictionary, outfile):
    print('> Compiling dictionary...')
    lemma_counts = Counter(outfile)
    for lemma, translations in dictionary.items():
        tr_counts = Counter(translations)
        tr = '\t'.join(['%s[%i]' % (x,n) for x, n in sorted(tr_counts.items(),
                        key=lambda item: item[1], reverse=True)])

        yield lemma + '\t%i' % lemma_counts[lemma] + '\t' + tr    

def openmap(mapping={}):
    """ Frequency-based Oracc mappings """
    print('> Reading Oracc-mappings...')
    with gzip.open(BYFREQ, 'rt', encoding='utf-8', errors='ignore') as data:
        for line in data.read().splitlines():
            key, lemma, trans = line.split('\t')
            mapping[key] = lemma
        return mapping

def openebl(mapping={}):
    """ Dictionary-based EBL mappings """
    print('> Reading EBL mappings...')
    with gzip.open(EBL, 'rt', encoding='utf-8', errors='ignore') as data:
        for line in data.read().splitlines():
            lemma, no, trans, olemma, otrans = line.split('\t')
            """ Prefer Oracc-based translatins and lemmas """
            if olemma:
                key = olemma + '[' + otrans + ']'
            else:
                key = lemma + '[' + otrans + ']'                
            mapping[key] = lemma.replace("'", 'ʾ') + '_' + no
        return mapping

def openvrt(filename, projects):
    """ Read projects from Korp-Oracc VRT """

    map_base = openmap()
    map_ebl = openebl()

    output = []
    dictionary = defaultdict(list)
    project_texts = defaultdict(int)
    project_words = defaultdict(int)
    
    print('> Reading VRT...')
    with open(filename, 'r', encoding='utf-8', errors='ignore') as data:
        for line in data.read().splitlines():
            if line.startswith('<text'):
                proj = re.sub('.*cdlinumber=\"(.+?)\/.*', r'\1', line)
                if '-' in proj:
                    proj = proj.split('-')[0]
                if proj in projects:
                    getdata = True
                    project_texts[proj] += 1
                else:
                    getdata = False
            if not line.startswith('<') and getdata:
                """ Get VRT word attributes """
                project_words[proj] += 1
                data = line.split('\t')
                lemma = data[1]
                translation = data[2]
                sense = data[3]
                pos = data[5]
                normname = data[7]
                """ Use Korp-Oracc's normalized names if available """
                if pos == 'propernoun' and normname != '_':
                    lemma = normname
                if pos not in STOP_POS:
                    key = "%s[%s]" % (lemma, translation)
                    disamb_lemma = map_ebl.get(
                        key, map_base.get(key, lemma + '_?'))
                    output.append(disamb_lemma)
                    dictionary[disamb_lemma].append(translation)
                else:
                    """ Lacunae and stop words """
                    if lemma == '_' or FLATTEN_STOPS:
                        output.append(lemma)
                    else:
                        output.append('<%s>' % lemma)
                    
            if line.startswith('</text') and getdata:
                output.append('\n')

        for proj, count in project_texts.items():
            print('   %s: %i texts, %i words' \
                  % (proj, count, project_words[proj]))
        print('   TOTAL: %i texts, %i words' \
              % (sum(project_texts.values()), sum(project_words.values())))
            
    return output, dictionary
                
outfile, dictionary = openvrt(INPUT_FILE, PROJECTS)

""" Write corpus files """
if WPL:
    write(OUTPUT_FILE + '.txt', '\n'.join(
        [w.replace('\n', '#') for w in outfile]
        ))
else:
    write(OUTPUT_FILE + '.txt', ' '.join(outfile))

""" Write dictionaries """
data = make_dictionary(dictionary, outfile)
write(OUTPUT_FILE + '_dict.tsv', '\n'.join(sorted(list(data))))
