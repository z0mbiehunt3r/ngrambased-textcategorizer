#!/usr/bin/env python
#coding:utf-8
# Author: Alejandro Nolla - z0mbiehunt3r
# Purpose: Example for detecting language using a N-Gram-Based analysis approach
# Created: 19/05/13


import argparse
import sys

try:
    import nltk.corpus
except ImportError:
    print '[!] You need to install nltk (http://nltk.org/index.html)'
    sys.exit(-1)


import ngramfreq



if __name__=='__main__':
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='N-Gram-Based Text Categorization', add_help=False)
    gr1 = parser.add_argument_group('Main arguments')
    gr1.add_argument('-i', '--input', dest='textfile', required=True)
    args = parser.parse_args()
    
    
    unknown_language_text = open(args.textfile, mode='r').read()
    
    
    text_categorizer = ngramfreq.NGramBasedTextCategorizer()
    guessed_language = text_categorizer.guess_language(unknown_language_text)
    
    print '[*] %s seems to be written in %s' %(args.textfile, guessed_language)
    
    
    # To generate N-gram frequency profile from specific file
    #text_categorizer.generate_ngram_frequency_profile_from_file('./example_data/uuee_const.txt', 'english.dat')
    
    # To generate N-gram frequency profile from readed text (string)
    #elquijote_words = open('./example_data/elquijote.txt', mode='r').read()
    #text_categorizer.generate_ngram_frequency_profile_from_raw_text(elquijote_words, 'spanish.dat')
    #lesmiserables_words = open('./example_data/lesmiserables.txt', mode='r').read()
    #text_categorizer.generate_ngram_frequency_profile_from_raw_text(lesmiserables_words, 'french.dat')