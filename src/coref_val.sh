java -cp "/home/logan/stanford-corenlp-full-2018-02-27/*" -Xmx100g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse,coref -parse.model edu/stanford/nlp/models/srparser/englishSR.ser.gz -filelist /home/logan/data/corenlp_lists/all_val.txt -outputFormat json -threads 32 -tokenize.whitespace true -parse.maxlen 100 -coref.algorithm statistical -outputDirectory /home/logan/data/corenlp_corefs/processed/cnn_dm/val