#!/usr/bin/awk -f

# Description:
# This script processes an input HLA file from 1000g and converts it into CHPED format, which is used for downstream genetic analysis.
# It validates HLA allele formats, replaces invalid or missing values with "0", and logs any issues to stderr.
# The output is a tab-delimited file in CHPED format, containing sample IDs and HLA allele data.

# Usage:
# Run this script with an input HLA file as follows:
# awk -f generate_chped.awk <input_hla_file> > <output_chped_file>
# Example:
# awk -f generate_chped.awk /path/to/hla_file.txt > /path/to/output.chped

# BEGIN block: Initialises variables and patterns used throughout the script
BEGIN{
    FS="\t"  # Input field separator (tab-delimited file)
    OFS="\t"  # Output field separator (tab-delimited output)
    
    # Mapping of column indices to gene names
    genes["4"]="A"
    genes["5"]="A"
    genes["6"]="B"
    genes["7"]="B"
    genes["8"]="C"
    genes["9"]="C"
    genes["10"]="DQB1"
    genes["11"]="DQB1"
    genes["12"]="DRB1"
    genes["13"]="DRB1"
    
    # Regular expression pattern to validate HLA allele formats
    pattern = "^(A\\*[0-9]{2,3}:[0-9]{2,3}|B\\*[0-9]{2,3}:[0-9]{2,3}|DRB1\\*[0-9]{2,3}:[0-9]{2,3}|C\\*[0-9]{2,3}:[0-9]{2,3}|DQB1\\*[0-9]{2,3}:[0-9]{2,3}|0)$"
} 

# Process each line of the input file, skipping the header (NR > 1)
NR > 1 { 
    # Initialise a counter for null or invalid values
    nb_null=0
    
    # Iterate over columns 4 to 13 (HLA allele data)
    for (i = 4; i <= 13; i++) {
        # Remove any text after a forward slash (e.g., "A*01:01/01:02" -> "A*01:01")
        sub(/\/.*/, "", $i)
        # Remove trailing asterisks (e.g., "A*" -> "A")
        sub(/\*$/, "", $i)
        
        # Replace "None" or empty values with "0"
        if($i=="None" || $i=="")
            $i = "0"
        else
            # Prepend the gene name to the allele (e.g., "01:01" -> "A*01:01")
            $i = genes[i] "*" $i
        
        # Validate the allele format against the pattern
        if ($i !~ pattern) {
            # Log invalid values to stderr with detailed information
            print "Line " NR ": Field " i " (" $i ") does not match pattern, replaced by '0'.\nOriginal Line content: " $0 > "/dev/stderr"
            $i = "0"  # Replace invalid values with "0"
        }
        
        # Count null values (represented as "0")
        if($i=="0")
            nb_null++
    }
    
    # If the number of null values is less than or equal to 5, output the line in CHPED format
    if (nb_null <= 5)
        print $3, $3, "0", "0", "0", "0", $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
    else
        # Log lines with too many null values to stderr
        print "Line " NR ": Too much empty values. Line content: " $0 > "/dev/stderr"
}
