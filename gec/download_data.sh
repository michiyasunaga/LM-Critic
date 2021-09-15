conda activate errant200


######################## Set up benckmarks ########################
mkdir -p benchmarks
cd benchmarks

#Prepare CoNLL2014
wget https://www.comp.nus.edu.sg/~nlp/conll14st/conll14st-test-data.tar.gz
tar -xf conll14st-test-data.tar.gz
python3 scripts/get_orig_from_m2.py conll14st-test-data/noalt/official-2014.combined.m2 \
      -out conll14st-test-data/noalt/official-2014.combined.orig.txt


#Prepare BEA2019
wget https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz
tar -xf wi+locness_v2.1.bea19.tar.gz
mv wi+locness wi+locness_v2.1.bea19
python3 scripts/get_orig_from_m2.py wi+locness_v2.1.bea19/m2/ABCN.dev.gold.bea19.m2 \
      -out wi+locness_v2.1.bea19/m2/ABCN.dev.bea19.orig.txt


#Prepare GMEG-wiki and -yahoo
git clone https://github.com/grammarly/GMEG.git
root=GMEG/data/test/wiki
errant_parallel -orig $root/source \
                -cor  $root/ref0 $root/ref1 $root/ref2 $root/ref3 \
                -out  $root/ref.m2

root=GMEG/data/test/yahoo
errant_parallel -orig $root/source \
                -cor  $root/ref0 $root/ref1 $root/ref2 $root/ref3 \
                -out  $root/ref.m2


#Download M2 scorer
git clone https://github.com/nusnlp/m2scorer.git


######################## Download training data ########################
cd ../
wget https://nlp.stanford.edu/projects/myasu/LM-Critic/data.zip
unzip data.zip
