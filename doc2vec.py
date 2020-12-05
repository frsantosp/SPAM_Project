import os
import gensim
# Set file names for train and test data
#test_data_dir = os.path.join(gensim.__path__[0], 'test', 'test_data')
train_file = "train.txt"
#lee_test_file = os.path.join(test_data_dir, 'lee.cor')

###############################################################################
# Define a Function to Read and Preprocess Text
# ---------------------------------------------
#
# Below, we define a function to:
#
# - open the train/test file (with latin encoding)
# - read the file line-by-line
# - pre-process each line (tokenize text into individual words, remove punctuation, set to lowercase, etc)
#
# The file we're reading is a **corpus**.
# Each line of the file is a **document**.
#
# .. Important::
#   To train the model, we'll need to associate a tag/number with each document
#   of the training corpus. In our case, the tag is simply the zero-based line
#   number.
#
import smart_open

def read_corpus(fname, tokens_only=False):
    with smart_open.open(fname, encoding="utf-8") as f:
        for i, line in enumerate(f):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

train_corpus = list(read_corpus(train_file))
#test_corpus = list(read_corpus(lee_test_file, tokens_only=True))

###############################################################################
# Let's take a look at the training corpus
#
print(train_corpus[:2])

###############################################################################
# Notice that the testing corpus is just a list of lists and does not contain
# any tags.
#

###############################################################################
# Training the Model
# ------------------
#
# Now, we'll instantiate a Doc2Vec model with a vector size with 50 dimensions and
# iterating over the training corpus 40 times. We set the minimum word count to
# 2 in order to discard words with very few occurrences. (Without a variety of
# representative examples, retaining such infrequent words can often make a
# model worse!) Typical iteration counts in the published `Paragraph Vector paper <https://cs.stanford.edu/~quocle/paragraph_vector.pdf>`__
# results, using 10s-of-thousands to millions of docs, are 10-20. More
# iterations take more time and eventually reach a point of diminishing
# returns.
#
# However, this is a very very small dataset (300 documents) with shortish
# documents (a few hundred words). Adding training passes can sometimes help
# with such small datasets.
#
model = gensim.models.doc2vec.Doc2Vec(vector_size=100, min_count=2, epochs=40)

###############################################################################
# Build a vocabulary
model.build_vocab(train_corpus)

###############################################################################
# Next, train the model on the corpus.
# If optimized Gensim (with BLAS library) is being used, this should take no more than 3 seconds.
# If the BLAS library is not being used, this should take no more than 2
# minutes, so use optimized Gensim with BLAS if you value your time.
#
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

###############################################################################
# Now, we can use the trained model to infer a vector for any piece of text
# by passing a list of words to the ``model.infer_vector`` function. This
# vector can then be compared with other vectors via cosine similarity.
#
#vector = model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires'])
#print(vector)

###############################################################################
# Note that ``infer_vector()`` does *not* take a string, but rather a list of
# string tokens, which should have already been tokenized the same way as the
# ``words`` property of original training document objects.
#
# Also note that because the underlying training/inference algorithms are an
# iterative approximation problem that makes use of internal randomization,
# repeated inferences of the same text will return slightly different vectors.
#

###############################################################################
# Assessing the Model
# -------------------
#
# To assess our new model, we'll first infer new vectors for each document of
# the training corpus, compare the inferred vectors with the training corpus,
# and then returning the rank of the document based on self-similarity.
# Basically, we're pretending as if the training corpus is some new unseen data
# and then seeing how they compare with the trained model. The expectation is
# that we've likely overfit our model (i.e., all of the ranks will be less than
# 2) and so we should be able to find similar documents very easily.
# Additionally, we'll keep track of the second ranks for a comparison of less
# similar documents.
#
ranks = []
second_ranks = []
for doc_id in range(len(train_corpus)):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)
    second_ranks.append(sims[1])
###############################################################################
# Let's count how each document ranks with respect to the training corpus
#
# NB. Results vary between runs due to random seeding and very small corpus
import collections

#counter = collections.Counter(ranks)
#print(counter)

###############################################################################
# Basically, greater than 95% of the inferred documents are found to be most
# similar to itself and about 5% of the time it is mistakenly most similar to
# another document. Checking the inferred-vector against a
# training-vector is a sort of 'sanity check' as to whether the model is
# behaving in a usefully consistent manner, though not a real 'accuracy' value.
#
# This is great and not entirely surprising. We can take a look at an example:
#
doc_id = 436
inferred_vector = model.infer_vector(train_corpus[doc_id].words)
sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))

score = []
count_acc = 0
for doc_id in range(len(train_corpus)):
    sr=[doc_id]
    if doc_id > 791:
        sr.append(1)
    else:
        sr.append(0)
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    count= 0
    for i in range(10):
        if sims[index][0] > 791 :
            count+=1
    if count>=8:
        sr.append(1)
    else:
        sr.append(0)
    score.append(sr)
    if sr[1]==sr[2]:
        count_acc+=1

file = open("table.txt","w")
for i in score:
    ln  = str(i[0])+","+str(i[1])+","+str(i[2])+"\n"
    file.write(ln)
file.close()
print(count_acc)
