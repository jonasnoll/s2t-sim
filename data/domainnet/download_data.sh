#!/bin/sh
mkdir data/domainnet/multi

cd data/domainnet/multi
wget http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip -O real.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip -O sketch.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip -O painting.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip -O clipart.zip
unzip real.zip
unzip sketch.zip
unzip painting.zip
unzip clipart.zip

