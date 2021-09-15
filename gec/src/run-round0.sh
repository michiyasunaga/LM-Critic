exit 0;
################################################################################
# run the following commands one by one in the `gec/` directory of the repo
################################################################################
export CUDA_VISIBLE_DEVICES=0
conda activate lm-critic

############### Train the fixer ###############
dt=`date '+%Y%m%d_%H%M%S'`
outdir=data/round0__synthetic/model-fixer__${dt}
mkdir -p $outdir
python3.8 -u src/run_seq2seq.py \
    --model_name_or_path facebook/bart-base --task summarization --text_column bad_detoked --summary_column good_detoked \
    --do_train --num_train_epochs 1 --train_file data/round0__synthetic/synthetic_paired_data_9M.json \
    --preprocessing_num_workers 20 --overwrite_output_dir --output_dir $outdir --predict_with_generate --fp16 \
    --per_device_train_batch_size 64 --gradient_accumulation_steps 8 --max_source_length 64 --max_target_length 64 \
    --logging_first_step --logging_steps 20 --save_steps 2000 \
  |& tee $outdir/log.txt



############### Run the fixer on benchmarks ###############
model_path=data/round0__synthetic/model-fixer

#BEA2019
python src/run_fixer.py -m $model_path -i benchmarks/wi+locness_v2.1.bea19/m2/ABCN.dev.bea19.orig.txt -o $model_path/predictions/bea19dev.out.txt --bea19
#CoNLL2014
python src/run_fixer.py -m $model_path -i benchmarks/conll14st-test-data/noalt/official-2014.combined.orig.txt -o $model_path/predictions/conll14.out.txt
#GMEG-wiki
python src/run_fixer.py -m $model_path -i benchmarks/GMEG/data/test/wiki/source -o $model_path/predictions/gmeg.wiki.out.txt
#GMEG-yahoo
python src/run_fixer.py -m $model_path -i benchmarks/GMEG/data/test/yahoo/source -o $model_path/predictions/gmeg.yahoo.out.txt



############### Evaluate the fixer outputs ###############
#CoNLL2014
python2 benchmarks/m2scorer/scripts/m2scorer.py $model_path/predictions/conll14.out.txt \
    benchmarks/conll14st-test-data/noalt/official-2014.combined.m2 | tee $model_path/predictions/conll14.eval.txt
# Precision   : 0.5922
# Recall      : 0.2920
# F_0.5       : 0.4912


#BEA2019 and GMEG uses errant scorer, which needs its own environment
conda deactivate
conda activate errant200

#BEA2019
errant_parallel -orig benchmarks/wi+locness_v2.1.bea19/m2/ABCN.dev.bea19.orig.txt \
                -cor $model_path/predictions/bea19dev.out.txt \
                -out $model_path/predictions/bea19dev.outm2.txt && \
errant_compare  -hyp $model_path/predictions/bea19dev.outm2.txt -ref benchmarks/wi+locness_v2.1.bea19/m2/ABCN.dev.gold.bea19.m2 | tee $model_path/predictions/bea19dev.eval.txt
# =========== Span-Based Correction ============
# TP	FP	FN	Prec	Rec	F0.5
# 1337	1686	6124	0.4423	0.1792	0.3419
# ==============================================

#GEMG-wiki
errant_parallel -orig benchmarks/GMEG/data/test/wiki/source \
                -cor $model_path/predictions/gmeg.wiki.out.txt \
                -out $model_path/predictions/gmeg.wiki.outm2.txt && \
errant_compare  -hyp $model_path/predictions/gmeg.wiki.outm2.txt -ref benchmarks/GMEG/data/test/wiki/ref.m2 | tee $model_path/predictions/gmeg.wiki.eval.txt
# =========== Span-Based Correction ============
# TP	FP	FN	Prec	Rec	F0.5
# 352	323	973	0.5215	0.2657	0.4373
# ==============================================

#GEMG-yahoo
errant_parallel -orig benchmarks/GMEG/data/test/yahoo/source \
                -cor $model_path/predictions/gmeg.yahoo.out.txt \
                -out $model_path/predictions/gmeg.yahoo.outm2.txt && \
errant_compare  -hyp $model_path/predictions/gmeg.yahoo.outm2.txt -ref benchmarks/GMEG/data/test/yahoo/ref.m2 | tee $model_path/predictions/gmeg.yahoo.eval.txt
# =========== Span-Based Correction ============
# TP	FP	FN	Prec	Rec	F0.5
# 241	301	411	0.4446	0.3696	0.4273
# ==============================================
