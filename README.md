# xgboost_hla_imputation
This provides code to impute HLA alleles using an XGboost model

Only one big requirement now: all SNPs in both the reference sample and the target sample (i.e. the one to impute) need to be encoded this way: "chr:pos:a1:a2". Here, a1 and a2 are the two alleles for that SNP.

This is inconvenient but easy to fix with bcftools and a reference genome:
```
pathRef=GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz
pathVCF=your vcf
pathOut=output path of vcf.gz file

bcftools norm -m -any --check-ref w -f ${pathRef} ${pathVCF} -Ou | \
  bcftools annotate --set-id '%CHROM:%POS:%REF:%FIRST_ALT' -Oz > ${pathOut}
tabix -p vcf "${pathOut}"
```

## Usage

### Phasing the reference HLA alleles and SNPs
This can be done using the MakeReference tool here: https://github.com/DrGBL/snp2hla_redux

### Make the input files required by XGboost
```
pathRef=prefix of bgl.phased file created in above step (should also have been created with a bim file)
pathSNPs=prefix of bim file containing only SNPs (without HLA alleles)
gene_choice=name of gene (e.g. HLA_A) see below
pathOut=path of output directory

python -u xgboost_imputation.py \
  --ref ${pathRef} \
  --sample ${pathSNPs} \
  --gene ${gene_choice} \
  --allele_present T \
  --model-dir ${pathOut}xgboost_${gene_choice} \
  --use_pandas False \
  --algo_phase data_loading \
  --min_ac 0
```
The list of supported genes is:
Class I: HLA_A, HLA_B, HLA_C, HLA_E, HLA_F, HLA_G
Class II: HLA_DRB1, HLA_DRB3, HLA_DRB4, and HLA_DRB5, HLA_DPA1, HLA_DPB1, HLA_DQA1, HLA_DQB1, HLA_DRA, HLA_DOA, HLA_DOB, HLA_DMA, HLA_DMB
For HLA_DRB3, HLA_DRB4, and HLA_DRB5, the algorithm will work, but requires more validation.

### Tune hyperparameters with cross-validation
```
python -u xgboost_imputation.py \
  --gene ${gene_choice} \
  --model-dir ${pathOut}xgboost_${gene_choice} \
  --algo_phase hyper_opt \
  --use_gpu False \
  --nfolds 5 \
  --threads 40 \
  --cv_seed 250
```

### Train the algorithm using the best hyperparameters obtained above
```
python -u xgboost_imputation.py 
  --gene ${gene_choice} \
  --model-dir ${pathOut}xgboost_${gene_choice} \
  --algo_phase xgb_train
```

### Impute a target sample
```
beagle_for_imputation=bgl.phased file of the SNPs from the target sample
bim_for_imputation=bim file corresponding to the bgl.phased file above

python -u xgboost_imputation.py \
  --snps_for_imputation ${beagle_for_imputation} \
  --sample_for_imputation ${bim_for_imputation} \
  --gene ${gene_choice} \
  --model-dir ${pathOut}xgboost_${gene_choice} \
  --use_pandas False \
  --algo_phase impute
```
