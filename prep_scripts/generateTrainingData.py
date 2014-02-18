import sys

from RBMDahlFunctions import *
from optparse import OptionParser

def writeTrainData(trainfile,ngram_size,vocab_size,output_training_file,output_probs_file,output_vocab_file) :
    vocab = collections.Counter()
    n_context = (ngram_size -1) # the number of start words to keep 
    num_ngrams = 0
    for line in list(open(trainfile)):
      words = line.split()
      num_ngrams += len(words)
      words[:0] = ["<s>"]*n_context
      words[len(words):] = ["</s>"]
      vocab.update(words)
    int_to_word = [word for word, _ in vocab.most_common(vocab_size-1)]
    int_to_word.append("<unk>")
    vocab_size = min(vocab_size, len(int_to_word))
    word_to_int = dict((word, i) for i, word in enumerate(int_to_word))
    p_word = collections.defaultdict(float)
    n = sum(vocab.itervalues())
    for word in vocab:
      if word in word_to_int:
        p_word[word] += float(vocab[word]) / n
      else:
        p_word["<unk>"] += float(vocab[word]) / n
    unigram_probs = [p_word[word] for word in int_to_word]

    OUTTRAINFILE = open(output_training_file,'w')
    #print 'size of data is ',len(data)
    g=lambda x:word_to_int[x] if x in word_to_int else word_to_int['<unk>']
    for line in list(open(trainfile)):
      words = line.split()
      num_ngrams += len(words)
      words[:0] = ["<s>"]*n_context
      words[len(words):] = ["</s>"]
      int_words = map(g,words)
      for i in xrange(0, len(int_words)-ngram_size+1):
        OUTTRAINFILE.write("%s\n"%(' '.join(map(repr,int_words[i:i+ngram_size]))))

    '''
    for words in data:
      for i, word in enumerate(words):
        if word not in word_to_int:
          #print 'unk is ',word
          words[i] = word_to_int["<unk>"]
        else:
          #print 'not unk is ',word
          words[i] = word_to_int[word]
        for i in xrange(0, len(words)-ngram_size+1):
          OUTTRAINFILE.write("%s\n"%(' '.join(map(repr,words[i:i+ngram_size]))))
    '''
    OUTTRAINFILE.close()

    #write unigram probs
    OUTPROBFILE = open(output_probs_file,'w') 
    for i,prob in enumerate(unigram_probs):
      OUTPROBFILE.write("%d\t%s\n"%(i,repr(prob)))
    OUTPROBFILE.close()
  
    OUTPUTVOCABFILE = open(output_vocab_file,'w')
    for word in int_to_word:
      OUTPUTVOCABFILE.write("%s\n"%word)
    OUTPUTVOCABFILE.close()
    return(word_to_int,int_to_word)

def writeValidationData(input_validation_file,output_validation_file,ngram_size,word_to_int,int_to_word):
      
    OUTVALIDFILE = open(output_validation_file,'w')
    g=lambda x:word_to_int[x] if x in word_to_int else word_to_int['<unk>']
    n_context = ngram_size-1
    for line in list(open(input_validation_file)):
      words = line.split()
      words[:0] = ["<s>"]*n_context
      words[len(words):] = ["</s>"]
      int_words = map(g,words)
      for i in xrange(0, len(int_words)-ngram_size+1):
        OUTVALIDFILE.write("%s\n"%(' '.join(map(repr,int_words[i:i+ngram_size]))))

    OUTVALIDFILE.close()
  
def main():
  parser = OptionParser()

  parser.add_option("--train_words_file ", action="store", type="string", dest="train_words_file",default = 'training.txt',help="The ngrams file")
  #parser.add_option("--unigram_probs_file", action="store", type="string", dest="unigram_probs_file",default = 'unigram.probs',help="The unigram probs file")
  parser.add_option("--ngram_size", action="store", type="int", dest="ngram_size",default = 3,help="The ngram size")
  parser.add_option("--vocab_size", action="store", type="int", dest="vocab_size",default =1000,help="Learn embeddings for only the top n-1 words. the nth word is <unk>")
  parser.add_option("--output_training_file", action="store", type="string", dest="output_training_file",default ='training.txt',help="the output training file")
  parser.add_option("--output_probs_file", action="store", type="string", dest="output_probs_file",default ='unigram.probs',help="the unigram probs file")
  parser.add_option("--input_validation_file", action="store", type="string", dest="input_validation_file",default ='validation.lst',help="the validation_file")
  parser.add_option("--output_validation_file", action="store", type="string", dest="output_validation_file",default ='validation.lst',help="the validation_file")
  parser.add_option("--output_vocab_file", action="store", type="string", dest="output_vocab_file",default ='output_vocab',help="The output vocab file")

  (options, args) = parser.parse_args()

  print "OPTIONS:"
  print options
  print "ARGS:"
  print args


  word_to_int,int_to_word = writeTrainData(options.train_words_file,options.ngram_size,options.vocab_size,options.output_training_file,options.output_probs_file,options.output_vocab_file)

  writeValidationData(options.input_validation_file,options.output_validation_file,options.ngram_size,word_to_int,int_to_word)

  '''
  probs_file = open(options.unigram_probs_file,'w')
  for i in xrange(len(unigram_probs)):
    probs_file.write("%s\t%s\n"%(i,repr(unigram_probs[i])))

  word_list_file = open(options.words_file,'w')
  for i in xrange(len(int_to_word)):
      word_list_file.write("%s\n"%int_to_word[i])
  word_list_file.close()

  probs_file.close()
  '''

if __name__=="__main__":
  main()
