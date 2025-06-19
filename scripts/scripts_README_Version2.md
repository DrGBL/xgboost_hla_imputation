# launch_steps.sh

## Purpose

The `launch_steps.sh` script automates the entire pipeline for preparing data, building a reference set, and performing HLA imputation using XGBoost on genomic data from the MHC locus. It orchestrates all stages, from raw VCF and HLA files through pre-processing, data preparation, format conversions, reference creation, training, and imputation, step by step.

## How it works

The script executes sequentially through 10 main steps, each corresponding to an output directory and a key phase of the pipeline. You can re-run the script from any given step (by passing the step number as an argument), making it possible to resume interrupted processes or to iterate on specific pipeline segments.

### Usage

```bash
./launch_steps.sh [START_STEP]
```
- `START_STEP`: (optional) step number from which to start the pipeline (default: 0).
- Use `--help` for help.

---

## Input & Output Details

### **Inputs**

- **HLA file:**  
  `HLA_FILE` (default: `/data/1000g/20181129_HLA_types_full_1000_Genomes_Project_panel.txt`)  
  *Tab-delimited file containing HLA genotype information for each individual. Third column contains individual/sample IDs.*

- **VCF file:**  
  `VCF_FILE` (default: `/data/1000g/ALL.chr6.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz`)  
  *Compressed VCF file containing phased genomic variants for chromosome 6 (MHC region).*

- **Reference genome:**  
  `REF_GENOME` (default: `/data/hg38.fa.gz`)  
  *Compressed reference genome for normalisation.*

- **XGBoost Apptainer image:**  
  `XGBOOST_IMG` (default: `/data/REPOSITORIES/XGboost-HLA/assets/xgboost_image_sandbox`)  
  *Container image containing all necessary tools and dependencies.*

- **Gene lists:**  
  - For reference creation: `MAKE_REF_GENES_LIST` (default: `A,B,C,DQB1,DRB1`)
  - For imputation: `XGBOOST_GENES_LIST` (default: `HLA_A`)

- **Parameters:**  
  - MHC region: `MHC_REGION` (default: `6:20000000-40000000`)
  - Thread count: `NB_THREADS` (default: `62`)
  - Memory: `MEMORY_KB` (default: `64000`), `MEMORY_GB` (default: `64G`)

**All input file paths and parameters can be modified directly within the script.**

---

### **Outputs**

Each step generates its own output directory with key intermediate or final files:

1. **1.sample_check/**
   - `hla_samples.txt`, `vcf_samples.txt`, `common_samples.txt`: Sample ID lists
   - `hla_types.chped`: CHPED format HLA file
   - `hla_types_conversion.log`: Conversion log

2. **2.vcf_filtering/**
   - `filtered.vcf.gz`: Filtered, region-restricted VCF file (common samples, only SNPs, high quality)
   - Tabix index: `filtered.vcf.gz.tbi`

3. **3.normalization/**
   - `normalized.vcf.gz`: Normalised and indexed VCF
   - Tabix index: `normalized.vcf.gz.tbi`

4. **4.80-20_samples_list/**
   - `train_samples.txt`: IDs for training (80%)
   - `to_impute_samples.txt`: IDs for imputation (20%)

5. **5.80-20_vcfs/**
   - `train.vcf.gz`: VCF for training samples (+ index)
   - `impute.vcf.gz`: VCF for imputation samples (+ index)
   - `impute.bgl.gz`: Beagle format file for imputation samples
   - `impute.bgl.phased`: Phased SNPs file

6. **6.plink_train/**
   - `mhc_binaries_train.*`: PLINK binary files (`.bed`, `.bim`, `.fam`) for training set

7. **7.reference/**
   - `mhc_binaries_train.ATtrick.CLEANED.bim`, `mhc_binaries_train.CLEANED.bim`, `mhc_binaries_train.CLEANED.bgl.phased`: Cleaned reference files for HLA imputation
   - `tmp/`: Temporary files

8. **8.plink_to_inpute/**
   - `mhc_binaries_to_impute.*`: PLINK binary files for imputation set

9. **9.xgboost_imputation/**
   - `xgboost_<GENE>/`: Folder for each gene with model files, logs, and imputation results for that gene

10. **10.comparison/**
    - (To be completed: for performance evaluation and result comparison; structure depends on downstream scripts)

Additionally, a global `pipeline.log` file is created, containing timestamped logs of all major operations.

---

## Main Steps

1. **Sample verification**
   - Converts HLA file to CHPED, extracts sample IDs, finds intersection with VCF.
2. **VCF filtering**
   - Restricts VCF to MHC region and common samples, keeps only SNPs, applies quality filters.
3. **VCF normalisation**
   - Normalises filtered VCF (split alleles, check reference) and indexes.
4. **Sample splitting**
   - Random 80/20 split of common samples into training and imputation sets.
5. **Filtering TRAIN & TO_IMPUTE into separate VCFs**
   - Creates VCFs for each subset, converts imputation VCF to Beagle format.
6. **Conversion of TRAIN to PLINK**
   - Converts training VCF to PLINK binaries for use in reference construction.
7. **Reference database creation**
   - Builds HLA imputation reference from training set, cleans invalid HLA entries.
8. **Conversion of TO_IMPUTE to PLINK**
   - Converts imputation VCF to PLINK binaries.
9. **XGBoost execution**
   - For each gene in `XGBOOST_GENES_LIST`:
     - Loads data, performs hyperparameter optimisation, trains XGBoost model, imputes HLA types.
10. **Results comparison**
    - (Placeholder for downstream evaluation; not fully implemented in this script.)

---

## Additional Features

- **Logging:** All important messages are timestamped and saved in `pipeline.log`.
- **Idempotency:** If a step is already complete, the script will use existing files and skip redundant computation.
- **Robust error handling:** The script stops on errors, undefined variables, or pipeline failures.
- **Debug mode:** Controlled by `DEBUG_MODE` variable; enables command echoing for troubleshooting.

---

## Prerequisites

- **Software:**  
  - `awk`, `bcftools`, `tabix`, `shuf`, `apptainer` (or `singularity`), `plink`, Java (for `vcf2beagle`)
- **Data:**  
  - Paths to input files must be valid and may need adjustment.

---

## Summary

`launch_steps.sh` provides robust, reproducible, and modular automation for the entire HLA imputation pipeline using XGBoost, suitable for large-scale datasets. It simplifies complex multi-stage processing and ensures full traceability of all steps and files produced.
