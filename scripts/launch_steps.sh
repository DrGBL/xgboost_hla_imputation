#!/bin/bash

# Function to display a help message
usage() {
    # Prints usage instructions when the script is executed with --help
    echo "Usage: $0 [START_STEP]"
    echo "START_STEP: Optional. The step number to start the pipeline from (default: 0)."
    echo "            Use --help to display this message."
    exit 0
}

# Check if the --help argument is passed
if [ "$1" == "--help" ]; then
    usage
fi

# Define the starting step of the pipeline (default: 0)
if [ -z "$1" ]; then
    START_STEP=0
else
    START_STEP=$1
fi

# Enable strict error handling
set -euo pipefail  # Stops the script on errors, undefined variables, or pipe failures

# Function to add a timestamp to logs
timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

# Function to log messages to both console and a log file
log() {
    local msg="$1"
    local log_file="pipeline.log"
    echo "$(timestamp) - $msg" | tee -a "$log_file"
}

# Toggle debug mode
DEBUG_MODE=1  # Set to 0 to disable debug mode
if [[ $DEBUG_MODE -eq 1 ]]; then
    set -x  # Enables debug mode (displays executed commands)
fi

# Global variable definitions used throughout the pipeline
HLA_FILE="/data/1000g/20181129_HLA_types_full_1000_Genomes_Project_panel.txt"  # HLA file containing sample data
VCF_FILE="/data/1000g/ALL.chr6.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz"  # VCF file containing genomic variants
REF_GENOME="/data/hg38.fa.gz"  # Reference genome file
MHC_REGION="6:20000000-40000000"  # Extended MHC region for analysis
XGBOOST_IMG="/data/REPOSITORIES/XGboost-HLA/assets/xgboost_image_sandbox"  # Apptainer image for XGBoost
MAKE_REF_GENES_LIST="A,B,C,DQB1,DRB1"  # List of genes for reference database creation
XGBOOST_GENES_LIST="HLA_A"  # List of genes for imputation
BIND_OPTIONS="-B /data"  # Bind options for Apptainer
NB_THREADS=62  # Number of threads to use for parallel processing
MEMORY_KB="64000"  # Memory allocation in KB
MEMORY_GB="64G"  # Memory allocation in GB

# Define output directories for each step of the pipeline
STEP1_OUTPUTDIR="1.sample_check"
STEP2_OUTPUTDIR="2.vcf_filtering"
STEP3_OUTPUTDIR="3.normalization"
STEP4_OUTPUTDIR="4.80-20_samples_list"
STEP5_OUTPUTDIR="5.80-20_vcfs"
STEP6_OUTPUTDIR="6.plink_train"
STEP7_OUTPUTDIR="7.reference"
STEP8_OUTPUTDIR="8.plink_to_inpute"
STEP9_OUTPUTDIR="9.xgboost_imputation"
STEP10_OUTPUTDIR="10.comparison"

# Step 1: Sample verification
HLA_SAMPLES="${STEP1_OUTPUTDIR}/hla_samples.txt"  # File containing HLA sample IDs
CHPED_FILE="${STEP1_OUTPUTDIR}/hla_types.chped"  # CHPED file generated from HLA data
CHPED_LOGFILE="${STEP1_OUTPUTDIR}/hla_types_conversion.log"  # Log file for CHPED conversion
VCF_SAMPLES="${STEP1_OUTPUTDIR}/vcf_samples.txt"  # File containing VCF sample IDs
COMMON_SAMPLES="${STEP1_OUTPUTDIR}/common_samples.txt"  # File containing common sample IDs between HLA and VCF
if [ "$START_STEP" -le 1 ]; then
    log "INFO: Starting step 1 - Sample verification"
    rm -rf "${STEP1_OUTPUTDIR}"  # Remove existing directory for this step
    mkdir -p "${STEP1_OUTPUTDIR}" # Create a new directory for this step
    # Convert the HLA file to CHPED format
    log "INFO: Converting HLA file to CHPED"
    /data/REPOSITORIES/XGboost-HLA/generate_chped.awk "${HLA_FILE}" > "${CHPED_FILE}" 2> "${CHPED_LOGFILE}"
    # Extract valid sample IDs from the HLA file
    awk 'NR > 1 {print $3}' "$HLA_FILE" | sort > "${HLA_SAMPLES}"
    # Extract sample IDs present in the VCF file
    bcftools query -l "$VCF_FILE" | sort > "${VCF_SAMPLES}"
    # Find the intersection of HLA and VCF sample IDs
    comm -12 "${HLA_SAMPLES}" "${VCF_SAMPLES}" > "${COMMON_SAMPLES}"
    NB_HLA_SAMPLES=$(wc -l < "${HLA_SAMPLES}")  # Count total HLA samples
    NB_VCF_SAMPLES=$(wc -l < "${VCF_SAMPLES}")  # Count total VCF samples
    NB_COMMON_SAMPLES=$(wc -l < "${COMMON_SAMPLES}")  # Count common samples
    if [[ $NB_HLA_SAMPLES -eq $NB_COMMON_SAMPLES ]]; then
        echo "INFO: All HLA samples are present in the VCF [${NB_COMMON_SAMPLES}]."
    else
        echo "WARNING: Not all HLA samples are present in the VCF. Only a subset will be used. HLA[$NB_HLA_SAMPLES] vs VCF[${NB_VCF_SAMPLES}] -> [${NB_COMMON_SAMPLES}]"
    fi
else
    if [ ! -d "${STEP1_OUTPUTDIR}" ]; then
        log "ERROR: Step 1 directory does not exist. Please execute step 1."
        exit 1
    else
        log "INFO: Step 1 has already been executed. Using existing files."
    fi
fi

# Step 2: VCF filtering
VCF_FILTERED="${STEP2_OUTPUTDIR}/filtered.vcf.gz"  # Output file for filtered VCF
if [ "$START_STEP" -le 2 ]; then
    rm -f "${STEP2_OUTPUTDIR}"  # Remove existing directory for this step
    mkdir -p "${STEP2_OUTPUTDIR}" # Create a new directory for this step
    log "INFO: Starting step 2 - VCF filtering"
    # Filter the VCF file based on common samples and MHC region
    bcftools view -S "${COMMON_SAMPLES}" --regions "${MHC_REGION}" "$VCF_FILE" -Ou | \
        bcftools view -f 'FILTER=PASS' --types snps -Ou | \
        bcftools view -i 'ALT!="*"' -Ou | \
        bcftools plugin fill-tags -Ou | \
        bcftools annotate --rename-chrs <(echo "6 chr6") --set-id '%CHROM:%POS:%REF:%FIRST_ALT' -Oz > "${VCF_FILTERED}"
    tabix -p vcf "${VCF_FILTERED}"  # Index the filtered VCF file
else
    if [ ! -d "${STEP2_OUTPUTDIR}" ]; then
        log "ERROR: Step 2 directory does not exist. Please execute step 2."
        exit 1
    else
        log "INFO: Step 2 has already been executed. Using existing file."
    fi
fi

# Step 3: VCF normalisation
VCF_NORMALIZED="${STEP3_OUTPUTDIR}/normalized.vcf.gz"  # Output file for normalised VCF
if [ "$START_STEP" -le 3 ]; then
    log "INFO: Starting step 3 - VCF normalisation"
    rm -f "${STEP3_OUTPUTDIR}"  # Remove existing directory for this step
    mkdir -p "${STEP3_OUTPUTDIR}" # Create a new directory for this step
    # Normalise the VCF file using the reference genome
    bcftools norm -m -any --check-ref w -f "$REF_GENOME" "${VCF_FILTERED}" -Oz > "${VCF_NORMALIZED}"
    tabix -p vcf "${VCF_NORMALIZED}"  # Index the normalised VCF file
else
    if [ ! -d "${STEP3_OUTPUTDIR}" ]; then
        log "ERROR: Step 3 directory does not exist. Please execute step 3."
        exit 1
    else
        log "INFO: Step 3 has already been executed. Using existing file."
    fi
fi

# Step 4: Sample splitting
IMPUTE_SAMPLES="${STEP4_OUTPUTDIR}/to_impute_samples.txt"  # File containing samples for imputation
TRAIN_SAMPLES="${STEP4_OUTPUTDIR}/train_samples.txt"  # File containing samples for training
if [ "$START_STEP" -le 4 ]; then
    log "INFO: Starting step 4 - Sample splitting"
    rm -f "${STEP4_OUTPUTDIR}"  # Remove existing directory for this step
    mkdir -p "${STEP4_OUTPUTDIR}" # Create a new directory for this step
    # Randomly split common samples into training and imputation sets (80-20 split)
    shuf "${COMMON_SAMPLES}" | awk -v IMPUTE_SAMPLES="${IMPUTE_SAMPLES}" -v TRAIN_SAMPLES="${TRAIN_SAMPLES}" 'NR%5==0 {print > IMPUTE_SAMPLES} NR%5!=0 {print > TRAIN_SAMPLES}'
else
    if [ ! -d "${STEP4_OUTPUTDIR}" ]; then
        log "ERROR: Step 4 directory does not exist. Please execute step 4."
        exit 1
    else
        log "INFO: Step 4 has already been executed. Using existing files."
    fi
fi

# Step 5: Filtering TRAIN & TO_IMPUTE data into independent VCFs
VCF_TO_IMPUTE="${STEP5_OUTPUTDIR}/impute.vcf.gz"  # VCF file for imputation samples
VCF_TRAIN="${STEP5_OUTPUTDIR}/train.vcf.gz"  # VCF file for training samples
BGL_GZ_TO_IMPUTE="${STEP5_OUTPUTDIR}/impute.bgl.gz"  # Beagle file for imputation samples
SNPS_PHASED_FOR_IMPUTATION="${STEP5_OUTPUTDIR}/impute.bgl.phased"  # Phased SNPs for imputation
if [ "$START_STEP" -le 5 ]; then
    log "INFO: Starting step 5 - Filtering TRAIN & TO_IMPUTE data into independent VCFs"
    # Filter the VCF file for training samples
    bcftools view -S "${TRAIN_SAMPLES}" "${VCF_FILTERED}" -Oz > "${VCF_TRAIN}"
    # Filter the VCF file for imputation samples
    bcftools view -S "${IMPUTE_SAMPLES}" "${VCF_FILTERED}" -Oz > "${VCF_TO_IMPUTE}"
    tabix -p vcf "${VCF_TRAIN}"  # Index the training VCF file
    tabix -p vcf "${VCF_TO_IMPUTE}"  # Index the imputation VCF file
    # Convert the imputation VCF file to Beagle format
    apptainer exec "${BIND_OPTIONS}" "$XGBOOST_IMG" bash -c "zcat ${VCF_TO_IMPUTE} | java -jar  /opt/snp2hla_redux/dependency/vcf2beagle.jar 0 ${STEP5_OUTPUTDIR}/impute"
    gunzip "${BGL_GZ_TO_IMPUTE}"  # Uncompress the Beagle file
    mv "${BGL_GZ_TO_IMPUTE%.gz}" "${SNPS_PHASED_FOR_IMPUTATION}"  # Rename the phased SNPs file
else 
    if [ ! -d "${STEP5_OUTPUTDIR}" ]; then
        log "ERROR: Step 5 directory does not exist. Please execute step 5."
        exit 1
    else
        log "INFO: Step 5 has already been executed. Using existing files."
    fi
fi

# Step 6: Conversion of TRAIN data to PLINK
PLINK_TRAIN="${STEP6_OUTPUTDIR}/mhc_binaries_train"  # PLINK binary files for training data
if [ "$START_STEP" -le 6 ]; then
    log "INFO: Starting step 6 - Conversion of TRAIN data to PLINK"
    rm -rf "${STEP6_OUTPUTDIR}"  # Remove existing directory for this step
    mkdir -p "${STEP6_OUTPUTDIR}" # Create a new directory for this step
    # Convert the training VCF file to PLINK binary format
    apptainer exec "${BIND_OPTIONS}" "$XGBOOST_IMG" plink \
        --vcf "${VCF_TRAIN}" \
        --make-bed \
        --double-id \
        --out "${PLINK_TRAIN}" \
        --threads ${NB_THREADS} \
        --memory ${MEMORY_KB}
else
    if [ ! -d "${STEP6_OUTPUTDIR}" ]; then
        log "ERROR: Step 6 directory does not exist. Please execute step 6."
        exit 1
    else
        log "INFO: Step 6 has already been executed. Using existing file."
    fi
fi

# Step 7: Creation of reference database
MAKE_REFERENCE_OUTPUT="${STEP7_OUTPUTDIR}/mhc_binaries_train"  # Output files for reference database
MAKE_REFERENCE_TMP="${STEP7_OUTPUTDIR}/tmp"  # Temporary directory for intermediate files
if [ "$START_STEP" -le 7 ]; then
    log "INFO: Starting step 7 - Creation of reference database"
    rm -rf "${STEP7_OUTPUTDIR}"  # Remove existing directory for this step
    mkdir -p "${STEP7_OUTPUTDIR}" # Create a new directory for this step
    mkdir -p "${MAKE_REFERENCE_TMP}" # Create temporary directory
    # Generate the reference database using training data
    apptainer exec "${BIND_OPTIONS}" "$XGBOOST_IMG" MakeReference \
        --variants "${PLINK_TRAIN}" \
        --chped "${CHPED_FILE}" \
        --hg 38 \
        --mind 0.3 \
        --genes "${MAKE_REF_GENES_LIST}" \
        --hardy 0.00000005 \
        --maf 0.00000005 \
        --miss 0.05 \
        --hla_maf 0.00000005 \
        --out "${MAKE_REFERENCE_OUTPUT}" \
        --mem ${MEMORY_GB} \
        --burnin 20 \
        --iter 100 \
        --nthreads ${NB_THREADS} \
        --phasing \
        --window 10 \
        --overlap 1.8 \
        --save-intermediates \
        --tmp_folder "${MAKE_REFERENCE_TMP}"

    # Clean up the reference database files by removing invalid HLA entries
    awk 'BEGIN{FS=OFS="\t"} {if ($2 ~ /^HLA/ && $2 !~ /:/) next; print }' ${MAKE_REFERENCE_OUTPUT}.ATtrick.bim > ${MAKE_REFERENCE_OUTPUT}.ATtrick.CLEANED.bim

    awk 'BEGIN{FS=OFS="\t"} {if ($2 ~ /^HLA/ && $2 !~ /:/) next; print }' ${MAKE_REFERENCE_OUTPUT}.bim > ${MAKE_REFERENCE_OUTPUT}.CLEANED.bim

    awk 'BEGIN{FS=OFS=" "} {if ($2 ~ /^HLA/ && $2 !~ /:/) next; print }' ${MAKE_REFERENCE_OUTPUT}.bgl.phased > ${MAKE_REFERENCE_OUTPUT}.CLEANED.bgl.phased

else
    if [ ! -d "${STEP7_OUTPUTDIR}" ]; then
        log "ERROR: Step 7 directory does not exist. Please execute step 7."
        exit 1
    else
        log "INFO: Step 7 has already been executed. Using existing file."
    fi
fi

# Step 8: Conversion of TO_IMPUTE data to PLINK
PLINK_IMPUTE="${STEP8_OUTPUTDIR}/mhc_binaries_to_impute"  # PLINK binary files for imputation data
if [ "$START_STEP" -le 8 ]; then
    log "INFO: Starting step 8 - Conversion of TO_IMPUTE data to PLINK"
    rm -rf "${STEP8_OUTPUTDIR}"  # Remove existing directory for this step
    mkdir -p "${STEP8_OUTPUTDIR}" # Create a new directory for this step
    # Convert the imputation VCF file to PLINK binary format
    apptainer exec "${BIND_OPTIONS}" "$XGBOOST_IMG" plink \
        --vcf "${VCF_TO_IMPUTE}" \
        --make-bed \
        --double-id \
        --out "${PLINK_IMPUTE}" \
        --threads ${NB_THREADS} \
        --memory ${MEMORY_KB}
else 
    if [ ! -d "${STEP8_OUTPUTDIR}" ]; then
        log "ERROR: Step 8 directory does not exist. Please execute step 8."
        exit 1
    else
        log "INFO: Step 8 has already been executed. Using existing file."
    fi
fi

# Step 9: Execution of XGBoost for each gene with TRAIN data
log "INFO: Starting step 9 - Execution of XGBoost for each gene with TRAIN data"
for GENE in $XGBOOST_GENES_LIST; do
    log "INFO: Processing gene: ${GENE}"
    log "INFO: Phase 1 - data_loading"
    # Load data for the specified gene
    apptainer exec "${BIND_OPTIONS}" "$XGBOOST_IMG" xgboost_imputation \
        --ref_bgl "${MAKE_REFERENCE_OUTPUT}.CLEANED" \
        --ref_bim "${MAKE_REFERENCE_OUTPUT}.ATtrick.CLEANED" \
        --sample "${PLINK_TRAIN}" \
        --gene "${GENE}" \
        --model-dir "${STEP9_OUTPUTDIR}/xgboost_${GENE}" \
        --use_pandas False \
        --algo_phase data_loading \
        --min_ac 0
    log "INFO: Phase 2 - hyper_opt"
    # Perform hyperparameter optimisation for the gene
    apptainer exec "${BIND_OPTIONS}" "$XGBOOST_IMG" xgboost_imputation \
        --gene "${GENE}" \
        --model-dir "${STEP9_OUTPUTDIR}/xgboost_${GENE}" \
        --algo_phase hyper_opt \
        --use_gpu False \
        --nfolds 5 \
        --threads ${NB_THREADS} \
        --cv_seed 250
    log "INFO: Phase 3 - xgb_train"
    # Train the XGBoost model for the gene
    apptainer exec "${BIND_OPTIONS}" "$XGBOOST_IMG" xgboost_imputation \
        --gene "${GENE}" \
        --model-dir "${STEP9_OUTPUTDIR}/xgboost_${GENE}" \
        --algo_phase xgb_train
    log "INFO: Phase 4 - impute"
    # Perform imputation using the trained model
    apptainer exec "${BIND_OPTIONS}" "$XGBOOST_IMG" xgboost_imputation \
        --snps_for_imputation "${SNPS_PHASED_FOR_IMPUTATION}" \
        --sample_for_imputation "${PLINK_TRAIN}.bim" \
        --gene "${GENE}" \
        --model-dir "${STEP9_OUTPUTDIR}/xgboost_${GENE}" \
        --use_pandas False \
        --algo_phase impute
done

echo "Pipeline successfully completed."
