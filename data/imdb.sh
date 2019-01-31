#!/bin/bash

wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar xf aclImdb_v1.tar.gz

dirname="aclImdb"
dtypes=("train" "test")
labels=("neg" "pos")

for dtype in "${dtypes[@]}";
do
    for label_index in 0 1;
    do
        cd "${dirname}/${dtype}/${labels[label_index]}"
        awk '{print $0}' *.txt >> "../../../${dtype}-data.txt"
        cd ../../../

        for ((i=0;i<12500;i++));
        do
            echo $label_index >> "${dtype}-label"
        done
    done
done

rm -rf aclImdb aclImdb_v1.tar.gz


