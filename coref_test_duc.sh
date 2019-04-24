java -cp os.path.expanduser('~') + "/stanford-corenlp-full-2018-02-27/*" -Xmx100g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse,coref -parse.model edu/stanford/nlp/models/srparser/englishSR.ser.gz -filelist data/coref/xsum/corenlp_lists/all_test.txt -outputFormat json -threads 32 -tokenize.whitespace true -coref.algorithm statistical -outputDirectory data/coref/xsum/processed