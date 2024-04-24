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
