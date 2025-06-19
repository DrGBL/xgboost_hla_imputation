#!/usr/bin/awk -f

BEGIN {
    # Define the regular expression pattern to match the required strings
    pattern = "^(A\\*[0-9]{2,3}:[0-9]{2,3}|B\\*[0-9]{2,3}:[0-9]{2,3}|DRB1\\*[0-9]{2,3}:[0-9]{2,3}|C\\*[0-9]{2,3}:[0-9]{2,3}|DQB1\\*[0-9]{2,3}:[0-9]{2,3}|0)$"
}

{
    # Iterate over fields 4 to 13
    for (i = 4; i <= 13; i++) {
        if ($i !~ pattern) {
            # Print the line number, field number, field value, and the entire line
            print "Line " NR ": Field " i " (" $i ") does not match pattern. Line content: " $0
        }
    }
}