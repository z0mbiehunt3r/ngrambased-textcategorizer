ngrambased-textcategorizer
==========================

Python implementation (PoC) of "N-Gram-Based Text Categorization (1994)" paper by William B. Cavnar and John M. Trenkle just for fun, learn and researching purposes.

How it works
-----
To perform N-Gram-Based Text Categorization we need to compute N-grams (with N=1 to 5) for each word - and apostrophes - found in the text, doing something like (being the word "TEXT"):
* bi-grams: \_T, TE, EX, XT, T\_  
* tri-grams: \_TE, TEX, EXT, XT\_, T\_ \_  
* quad-grams: \_TEX, TEXT, EXT\_, XT\_ \_, T\_ \_ \_  
  
With every N-Gram computed we just keep top 300 with most occurrences and save them as a "text category profile":  
``` bash
$ tail -n 10 langdata/spanish.dat 
  se   6826
nc  6788
su	6770
mi	6665
 con	6590
  con	6590
er 	6459
er   	6459
er  	6459
z	6379
```
  
Now, to check/guess text category we only need to generate N-grams in previous way and match against pre-computed profiles to calculate distance for every N-gram, choosing the nearest one (the profile with smallest total "distance")

Language detection example
-----
I have included three profiles - spanish, english and french - for a quick demo:
``` bash
$ python example.py -i example_data/El\ Hobbit\ -\ Una\ Tertulia\ Inesperada.txt 
[*] example_data/El Hobbit - Una Tertulia Inesperada.txt seems to be written in spanish
```

Creating category profile
-----
``` python
>>> import ngramfreq
>>> text_categorizer = ngramfreq.NGramBasedTextCategorizer()
>>> lesmiserables_words = open('./example_data/lesmiserables.txt', mode='r').read()
>>> text_categorizer.generate_ngram_frequency_profile_from_raw_text(lesmiserables_words, 'french.dat')
>>> # or we could just do
>>> text_categorizer.generate_ngram_frequency_profile_from_file('./example_data/uuee_const.txt', 'english.dat')
```

Categorizing text
-----
``` python
>>> import ngramfreq
>>> # create just one object for multiple checks
>>> text_categorizer = ngramfreq.NGramBasedTextCategorizer()
>>> text_categorizer.guess_language("Le temps est un grand maître, dit-on, le malheur est qu'il tue ses élèves.")
'french'
>>> # or for only one quick check
>>> ngramfreq.guess_language("The only thing necessary for the triumph of evil is that good men do nothing.")
'english'
```