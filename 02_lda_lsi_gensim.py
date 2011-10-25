"""
======================================================
Topic modeling using Gensim
======================================================
Topic modeling using Gensim library on the 
LDA, LSA, Similarities
"""

print __doc__

import os
import glob
from time import time
import logging

from gensim import corpora, models, similarities, utils

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

###############################################################################
# Preprocessing

# read all .txt in a folder into a string list
documents = [] # the intial of the list for all the paragraphs

print "reading all files"
t0 = time()
for files in glob.glob("171/*.txt"):
    g = open(files)
    for line in g:
        documents.append(line)
print

# stop word
print "processing stop words"
t0 = time()
# stoplist = set('for a of the and to in you your all information if or may by with be can from we about email & firefox this our how under will'.split())
stoplist = ['for', 'a', 'of', 'the', 'and', 'to', 'in', 'you', 'your', 'all', \
            'information', 'if', 'or', 'may', 'by', 'with', 'be', 'can', 'from', \
            'we', 'about', 'email', '&', 'firefox', 'this', 'our', 'how', 'under', \
            'will', 'any', 'that', 'on', 'as', 'is', 'have', 'has', 'are', \
            'new', 'yahoo!', 'verizon', 'other', 'please', 'us', 'it', 'web',  \
            'what', 'do', '|', '-', 'site', 'not', 'zynga', 'information?',\
            'at', 'such', 'these', 'time', 'privacy', 'policy', 'google', \
            'target', 'also', 'an', 'e-mail', 'personal', 'personally', \
            'does', 'network', 'sites', 'when', 'their', 'there', 'out', 'page',\
            'i', 'online',  'visa', 'should', 'policies', 'through', \
            ]

texts = [[word for word in document.lower().split() if word not in stoplist] \
            for document in documents]
print "done in %fs" % (time() - t0)
print

# tested == Too slow!
# # remove words that appear only once
# print "Removeing words that appear only once..."
# t0 = time()
# all_tokens = sum(texts, [])
# tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
# texts = [[word for word in text if word not in tokens_once] \
#             for text in texts]
# print "done in %fs" % (time() - t0)
# print            

# generate dictionary
print "preparing dictionary"
t0 = time()
dictionary = corpora.Dictionary(texts)
dictionary.save('tmp/ppdict.dict')
print "done in %fs" % (time() - t0)
print dictionary
print

# generate corpus
print "preparing corpus"
t0 = time()
corpus = [dictionary.doc2bow(text) for text in texts]
print "done in %fs" % (time() - t0)
# print corpus[:100]
print

# serialize into market matrix format
print "serializing"
t0 = time()
corpora.MmCorpus.serialize('tmp/corpus.mm', corpus)
# corpora.BleiCorpus.serialize('tmp/corpus.lda-c', corpus)
print "done in %fs" % (time() - t0)
print

# load back
diction = corpora.Dictionary.load('tmp/ppdict.dict')
corp = corpora.MmCorpus('tmp/corpus.mm')
print diction
print corp
print

# creat tfi-df model
print "transform to tfi-df"
t0 = time()
tfidf = models.TfidfModel(corp)
corp_tfidf = tfidf[corp]
print "done in %fs" % (time() - t0)
print



###############################################################################
# Topic modeling

# LDA
print "LDA modeling"
t0 = time()
lsi = models.ldamodel.LdaModel(corpus=corp, id2word=diction, num_topics=15)
# lsi.print_topics(10)
print "done in %fs" % (time() - t0)
print

# print 10 topics by LDA
# lsi.print_topics(10)
print "Top10 topics:"
topic_strings = lsi.show_topics(15)
for i in xrange(15):
    print 'Topic', i+1
    print topic_strings[i]
    print 
print


# LSI
print "LSI modeling"
t0 = time()
lsi = models.LsiModel(corp_tfidf, id2word=diction, num_topics=15)
corp_lsi = lsi[corp_tfidf]
print "done in %fs" % (time() - t0)
print

# print 10 topics by LSI
# lsi.print_topics(10)
print "Top10 topics:"
topic_strings = lsi.show_topics(15)
for i in xrange(15):
    print 'Topic', i+1
    print topic_strings[i]
    print 
print
