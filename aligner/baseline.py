#!/usr/bin/env python
import optparse, sys, os, logging
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
optparser.add_option("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
optparser.add_option("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
optparser.add_option("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="threshold for alignment (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)

if opts.logfile:
    logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]


f_count = defaultdict(float)
e_count = defaultdict(float)
fe_count = defaultdict(float)

fe_prob = defaultdict(float)
iter = 0
prev = 0
pres = 0
sum = 1.00000
prevsum = 1.00000
############Training the Model##################

for (n, (f,e)) in enumerate(bitext):
  for f_i in set(f):
    f_count[f_i] += 1.00000000000000

f_vs = len(f_count) # vocab. size of French input

for (n, (f,e)) in enumerate(bitext):
  for f_i in set(f):
    for e_j in set(e):
      fe_prob[f_i,e_j] = 1.00000000000000/f_vs
      e_count[e_j] = 0
      fe_count[f_i,e_j] = 0


while (iter < 40):
  iter += 1
  
  e_count = {x : 0 for x in e_count}
  fe_count = {x : 0 for x in fe_count}

  for (n,(f,e)) in enumerate(bitext):
    for f_i in set(f):
      Z = 0
      for e_j in set(e):
        Z += fe_prob[f_i,e_j] 
      for e_j in set(e):
        c = fe_prob[f_i,e_j] / Z
        fe_count[f_i,e_j] += c 
        e_count[e_j] += c

  for (f,e) in fe_count:
    fe_prob[f,e] = fe_count[f,e] / e_count[e]

    


##########Decoding###################### 
for (n,(f,e)) in enumerate(bitext):
  for (i, f_i) in enumerate(f):
    bestp = 0
    bestj = 0
    for (j, e_j) in enumerate(e):
      if fe_prob[f_i,e_j] > bestp:
        bestp = fe_prob[f_i,e_j]
        bestj = j
    sys.stdout.write("%i-%i " % (i,bestj))
  sys.stdout.write("\n")


# for (n, (f, e)) in enumerate(bitext):
#   for f_i in set(f):
#     f_count[f_i] += 1
#     for e_j in set(e):
#       fe_count[(f_i,e_j)] += 1
#   for e_j in set(e):
#     e_count[e_j] += 1
#   if n % 500 == 0:
#     sys.stderr.write(".")

# dice = defaultdict(int)
# for (k, (f_i, e_j)) in enumerate(fe_count.keys()):
#   dice[(f_i,e_j)] = 2.0 * fe_count[(f_i, e_j)] / (f_count[f_i] + e_count[e_j])
#   if k % 5000 == 0:
#     sys.stderr.write(".")
# sys.stderr.write("\n")

# for (f, e) in bitext:
#   for (i, f_i) in enumerate(f): 
#     for (j, e_j) in enumerate(e):
#       if dice[(f_i,e_j)] >= opts.threshold:
#         sys.stdout.write("%i-%i " % (i,j))
#   sys.stdout.write("\n")
