#!/bin/bash
# BSD License
#
# For SentEval software
#
# Copyright (c) 2017-present, Facebook, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither the name Facebook nor the names of its contributors may be used to
#    endorse or promote products derived from this software without specific
#    prior written permission.
#
# Cods to fetch datasets is based on SentEval by FAIR.
# https://github.com/facebookresearch/SentEval/blob/master/data/downstream/get_transfer_data.bash
# Please check https://github.com/facebookresearch/SentEval/blob/master/LICENSE .


SSTbin='https://raw.githubusercontent.com/PrincetonML/SIF/master/data'
SSTfine='https://raw.githubusercontent.com/AcademiaSinicaNLPLab/sentiment_dataset/master/data/'

# SST binary
data_path=./binary
rm -rf $data_path
mkdir $data_path
for split in train dev test
do
    wget -O $data_path/sentiment-$split $SSTbin/sentiment-$split
    perl ../wikifil.pl < $data_path/sentiment-$split > $data_path/sentiment-$split.normalized
    python preprocess.py $data_path/sentiment-$split.normalized 2
    rm $data_path/sentiment-$split.normalized $data_path/sentiment-$split
done

# SST fine-grained
data_path=./fine
rm -rf $data_path
mkdir $data_path

for split in train dev test
do
    wget -O ./fine/sentiment-$split $SSTfine/stsa.fine.$split
    perl ../wikifil.pl < $data_path/sentiment-$split > $data_path/sentiment-$split.normalized
    python preprocess.py $data_path/sentiment-$split.normalized 5
    rm $data_path/sentiment-$split.normalized $data_path/sentiment-$split
done
