#-*- coding: utf-8 -*-

############### 2017.05.10 word2vec with visualizing embeddings on tensorboard
############### revised by YI INGU
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile
import codecs

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from nltk import word_tokenize






########## Step 1. load data
#
# filename = 'text8.zip' #31344016
#
# # Read the data into a list of strings.
# def read_data(filename):
#   """Extract the first file enclosed in a zip file as a list of words"""
#   with zipfile.ZipFile(filename) as f:
#     data = tf.compat.as_str(f.read(f.namelist()[0])).split()
#   return data
#
# words = read_data(filename)
# print(type(words), words[:10])
# print('Data size', len(words)) # we have 17005207 words in the variable 'words'

with codecs.open("korean_text", "r", encoding="utf-8") as fid:
  words = [word_tokenize(sentence) for sentence in fid.readlines()]
  real_words = []
  for word in words:
    for letter in word:
      real_words.append(letter.encode('utf-8'))
      print(letter.encode('utf-8'))

words = real_words


# fid_read = codecs.open(words, "r", encoding="utf-8")
# words = [word.rstrip() for word in fid_read.readlines()]


### Organizing (one word per one line format)
# fid = codecs.open("OUTPUT", "w") #encoding="utf-8"
# for i in range(len(words)):
#     for ii in range(len(words[i])):
#         fid.write(u"{}\n".format(words[i][ii]))
# fid.close()




########### Step 2: Build the dictionary and replace rare words with UNK token.

vocabulary_size = 5000
#
def build_dataset(words, vocabulary_size):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))

  #make dictionary with words
  dictionary = dict()

  for word, _ in count:
    dictionary[word] = len(dictionary)


    # print(word, dictionary[word])
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  print('data :', type(data), data[:10])
  print('count :', type(count), count[:10])
  print('dictionary :', type(dictionary), dictionary['UNK'],dictionary['the'],dictionary['of'])
  print('reverse_dictionary :', type(reverse_dictionary), reverse_dictionary.items()[:10])


  return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size)

# del words  # Hint to reduce memory.
data_index = 0

######################### make metadata.tsv

file = open('./log_dir/metadata.tsv','w')
file.write('label\n')
for tuple in reverse_dictionary.items():
  file.write('%s\n' %str(tuple[1]))
file.close()






########### Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  # print('batch :', type(batch),len(batch), batch[:10])
  # print('labels :', type(labels), len(labels), labels[:10]) :
  return batch, labels





########## Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  # valid_dataset = tf.constant(valid_examples, dtype=tf.int32)


  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),name='embeddings')
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))



  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size)) ## ,

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm

  # Add variable initializer.
  init = tf.global_variables_initializer()


########## Step 5: Begin training.

num_steps = 200001 #change to how many steps you want to train

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()

  LOG_DIR = '/Users/demiust/PycharmProjects/tensorflow/models-master/tutorials/embedding/log_dir/'

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      average_loss = 0

      # print('normalized embeddings', normalized_embeddings.eval()[:10])
      # we can see that this algorithm is updating the embedding vectors


      #save checkpoints!!!

      saver = tf.train.Saver()
      saver.save(session, os.path.join(LOG_DIR, "model.ckpt"), step)

  final_embeddings = normalized_embeddings.eval()
  print('final embeddings :', final_embeddings[:10])
  # final embedding shappe == 50000*128
  # one embedding vector dimension : 128
  # all 50000 embedding vectors included


############ go to tensorboard
