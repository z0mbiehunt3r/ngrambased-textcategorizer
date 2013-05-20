#coding:utf-8

__license__= '''
ngrambased-textcategorizer - N-Gram-Based Text Categorization

Implementation of N-Gram-Based Text Categorization (1994) paper by William B. Cavnar and John M. Trenkle
download it from http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.53.9367 

Copyright (C) 2013  Alejandro Nolla Blanco - alejandro.nolla@gmail.com 
Nick: z0mbiehunt3r - Twitter: https://twitter.com/z0mbiehunt3r
Blog: blog.alejandronolla.com


This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
'''

'''
Frodo: It's a pity Bilbo didn't kill him when he had the chance. 
Gandalf: Pity? It was pity that stayed Bilbo's hand. Many that
live deserve death. Some that die deserve life. Can you give it
to them, Frodo? Do not be too eager to deal out death in judgment.
Even the very wise cannot see all ends.

            The Lord of the Rings: The Fellowship of the Ring
'''

__version__ = '0.4'


import glob
import operator
import os
import sys


try:
    from nltk.tokenize import RegexpTokenizer
    from nltk.util import ngrams
except ImportError:
    print '[!] You need to install nltk (http://nltk.org/index.html)'
    sys.exit(-1)


LANGDATA_FOLDER = './langdata/'

########################################################################
class NGramBasedTextCategorizer:
    """
    Class used to generate Ngrams frequency profiles as well as checking
    them agains pre-computed ones.
    
    Useful to guess text language, could also be used to categorize text
    only changing N-gram ranking kept (starting around rank 300 or so)
    """

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        
        self._languages_statistics = {}
        self._tokenizer = RegexpTokenizer("[a-zA-Z'`éèî]+") # keep only letters and apostrophes
        self._langdata_path = LANGDATA_FOLDER
    
    #----------------------------------------------------------------------
    def _load_ngram_statistics(self):
        """
        Load pre-computed profiles from local directory and store them
        at self._languages_statistics dictionary
        """
        
        languages_files = glob.glob('%s*.dat' %self._langdata_path)
        
        for language_file in languages_files:
            filename = os.path.basename(language_file)
            language = os.path.splitext(filename)[0]
            
            ngram_statistics = open(language_file, mode='r').readlines()
            ngram_statistics = map(str.rstrip, ngram_statistics) # remove edge trailing
            
            self._languages_statistics.update({language:ngram_statistics})
    
    #----------------------------------------------------------------------
    def _tokenize_text(self, raw_text):
        """
        Split the text into separate tokens consisting only of letters and
        apostrophes. Digits and punctuation are discarded. Pad the token
        with sufficient blanks before and after.
        
        @param raw_text: Text to be tokenized
        @type raw_text: str
        
        @return: List of tokens
        @rtype: list
        """
        
        tokens = self._tokenizer.tokenize(raw_text)
        
        return tokens
    
    #----------------------------------------------------------------------
    def _generate_ngrams(self, tokens):
        """
        Scan down each token, generating all possible N-grams, for N=1 to 5.
        Use positions that span the padding blanks, as well.
        
        @param tokens: List of tokens
        @type tokens: list
        
        @return: List of generated n-grams
        @rtype: list
        """
        
        generated_ngrams = []
        
        for token in tokens:            
            '''            
            In our system, we use N-grams of several different lengths
            simultaneously. We also append blanks to the beginning and ending
            of the string in order to help with matching beginning-of-word
            and ending-of-word situations. (We will use the underscore
            character ("_") to represent blanks.)
            Thus, the word "TEXT" would be composed of the following N-grams:
                bi-grams: _T, TE, EX, XT, T_
                tri-grams: _TE, TEX, EXT, XT_, T_ _
                quad-grams: _TEX, TEXT, EXT_, XT_ _, T_ _ _
            '''
            for x in xrange(1, 6): # generate N-grams, for N=1 to 5
                xngrams = ngrams(token, x, pad_left=True, pad_right=True, pad_symbol=' ')
                
                for xngram in xngrams:
                    # convert ('E', 'X', 'T', ' ') to 'EXT '
                    ngram = ''.join(xngram)
                    generated_ngrams.append(ngram)
        
        return generated_ngrams
    
    #----------------------------------------------------------------------
    def _count_ngrams_and_hash_them(self, ngrams):
        """
        Hash into a table to find the counter for the N-gram, and increment it.
        The hash table uses a conventional collision handling mechanism to
        ensure that each N-gram gets its own counter.
        
        @param ngrams: List of generated ngrams
        @type ngrams: list
        
        @return: Dictionary with ngrams occurrences {'going': 437, 'eas ': 487...}
        @rtype: dict
        """
        
        ngrams_statistics = {}
        
        for ngram in ngrams:
            if not ngrams_statistics.has_key(ngram):
                ngrams_statistics.update({ngram:1})
            else:
                ngram_occurrences = ngrams_statistics[ngram]
                ngrams_statistics.update({ngram:ngram_occurrences+1})
        
        return ngrams_statistics
        
    #----------------------------------------------------------------------
    def _calculate_ngram_occurrences(self, text):
        """
        Sort those counts into reverse order by the number of occurrences.
        Paper says to keep just the Ngrams themselves in reverse order but
        we store ngrams and occurrences to being able to store a profile.
        
        @param text: Text to analyze an compute Ngrams occurrences
        @type text: str
        
        @return: Ngrams with occurrences sorted by most occurrences
        @rtype: list
        """
        
        tokens = self._tokenize_text(text)
        ngrams_list = self._generate_ngrams(tokens)
        
        ngrams_statistics = self._count_ngrams_and_hash_them(ngrams_list)
        
        '''
        The top 300 or so N-grams are almost always highly correlated
        to the language.
        
        Starting around rank 300 or so, an N-gram frequency profile begins
        to show N-grams that are more specific to the subject of the document.
        These represent terms and stems that occur very frequently in
        documents about the subject.
        '''
        ngrams_statistics_sorted = sorted(ngrams_statistics.iteritems(),\
                                          key=operator.itemgetter(1),\
                                          reverse=True)[0:300]
        
        return ngrams_statistics_sorted
    
    #----------------------------------------------------------------------
    def _compare_ngram_frequency_profiles(self, category_profile, document_profile):
        """
        It merely takes two N-gram profiles and calculates a simple rank-order statistic
        we call the "out-of-place" measure. This measure determines how far out of place
        an N-gram in one profile is from its place in the other profile.
        
        @param category_profile: Ngrams statistics of pre-computed category
        @type category_profile: list
        
        @param document_profile: Ngrams statistics for document being analyzed
        @type document_profile: list
        
        @return: Distance measure from category profile to document being analyzed
        @rtype: int
        """
        
        document_distance = 0
        
        # convert [['eas ', 487], ['going', 437], ...] to ['eas', 'going', ...]
        category_ngrams_sorted = [ngram[0] for ngram in category_profile]
        document_ngrams_sorted = [ngram[0] for ngram in document_profile]
        
        maximum_out_of_place_value = len(document_ngrams_sorted)
        
        for ngram in document_ngrams_sorted:
            # pick up index position of ngram
            document_index = document_ngrams_sorted.index(ngram)
            try:
                # check if analyzed ngram exists in pre-computed category
                category_profile_index = category_ngrams_sorted.index(ngram)
            except ValueError:
                '''
                If an N-gram (such as "ED" in the figure) is not in the category
                profile, it takes some maximum out-of-place value.
                '''
                category_profile_index = maximum_out_of_place_value
            
            '''
            The sum of all of the out-of-place values for all N-grams is the
            distance measure for the document from the category.
            '''
            distance = abs(category_profile_index-document_index) # absolute value
            document_distance+=distance
        
        return document_distance
    
    #----------------------------------------------------------------------
    def guess_language(self, raw_text):
        """
        Will try guessing text's language by computing Ngrams and comparing
        them against pre-computed ones.
        
        @param raw_text: Text whose language want to guess
        @type raw_text: str
        
        @return: Guessed language
        @rtype: str
        """
        
        languages_ratios = {}
        self._load_ngram_statistics() # load pre-computed data
        
        '''
        Finally, the bubble labelled "Find Minimum`Distance" simply takes
        the distance measures from all of the category profiles to the
        document profile, and picks the smallest one.
        '''
        for language, ngrams_statistics in self._languages_statistics.iteritems():
            language_ngram_statistics = self._calculate_ngram_occurrences(raw_text)
            distance = self._compare_ngram_frequency_profiles(ngrams_statistics, language_ngram_statistics)
            
            languages_ratios.update({language:distance})
        
        nearest_language = min(languages_ratios, key=languages_ratios.get)
        
        return nearest_language
    
    
    #----------------------------------------------------------------------
    def generate_ngram_frequency_profile_from_raw_text(self, raw_text, output_filename):
        """
        Will compute Ngrams for given text, keep the 300 most common and
        save a profile to output file.
        
        @param raw_text: Text from which want to create a Ngram frequency profile
        @type raw_text: str
        
        @param output_filename: Filename to save Ngram frequency profile
        @type output_filename: str
        """
        
        output_filenamepath = os.path.join(self._langdata_path, output_filename)
        
        profile_ngrams_sorted = self._calculate_ngram_occurrences(raw_text)
        
        fd = open(output_filenamepath, mode='w')
        for ngram in profile_ngrams_sorted:
            fd.write('%s\t%s\n' % (ngram[0], ngram[1]))
        fd.close()
    
    #----------------------------------------------------------------------
    def generate_ngram_frequency_profile_from_file(self, file_path, output_filename):
        """
        Will compute Ngrams for given text, keep the 300 most common and
        save a profile to output file.
        
        @param raw_text: Textfile from which want to create a Ngram frequency profile
        @type raw_text: str
        
        @param output_filename: Filename to save Ngram frequency profile
        @type output_filename: str
        """
        
        raw_text = open(file_path, mode='r').read()
        self.generate_ngram_frequency_profile_from_raw_text(raw_text, 
                                                           output_filename)
        profile_ngrams_sorted = self._calculate_ngram_occurrences(raw_text)
        

#----------------------------------------------------------------------
def guess_language(raw_text):
    """
    Will try guessing text's language by computing Ngrams and comparing
    them against pre-computed ones.
    
    @param raw_text: Text whose language want to guess
    @type raw_text: str
    
    @return: Guessed language
    @rtype: str
    """
    
    return NGramBasedTextCategorizer().guess_language(raw_text)