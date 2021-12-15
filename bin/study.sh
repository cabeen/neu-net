#! /bin/bash
################################################################################
#
#  A script for running a whole experiment
#
#  Author: Ryan Cabeen
#
################################################################################

bin="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

if [ $# -ne 2 ]; then
  echo "usage: $(basename $0) input output"
  exit
fi

echo "started $(basename $0)"

input=$1
output=$2

epochs=10
rescale=128
kernel=128

mkdir -p ${output}/{valid,test,train}/{out,mosaics}

if [ ! -e ${output}/models ]; then
  python3 ${bin}/train.py \
    --kernel ${kernel} \
    --rescale ${rescale} \
    --epochs ${epochs} \
    --largest \
    --images ${input}/train/images \
    --masks ${input}/train/labels \
    --output ${output}/models
fi

if [ ! -e ${output}/valid/out ]; then
  python3 ${bin}/validate.py \
    --models ${output}/models \
    --images ${input}/valid/images \
    --masks ${input}/valid/labels \
    --output ${output}/valid/out
fi

if [ ! -e ${output}/test/out ]; then
  python3 ${bin}/test.py \
    --model ${output}/valid/out/best-model \
    --images ${input}/test/images \
    --masks ${input}/test/labels \
    --output ${output}/test/out
fi

if [ ! -e ${output}/train/out ]; then
  python3 ${bin}/test.py \
    --model ${output}/valid/out/best-model \
    --images ${input}/train/images \
    --masks ${input}/train/labels \
    --output ${output}/train/out
fi

if [ -e ${input}/fail ]; then
  if [ ! -e ${output}/fail/out ]; then
    python3 ${bin}/test.py \
      --model ${output}/valid/out/best-model \
      --images ${input}/fail/images \
      --masks ${input}/fail/labels \
      --output ${output}/fail/out
  fi
fi

function postprocess 
{
  name=$1

  if [ ! -e ${output}/${name}/mosaics ]; then
    for s in $(cat ${input}/${name}/names.txt); do \
      echo ${bin}/mosaic.sh \
        ${input}/${name}/images/${s}.nii.gz \
        ${input}/${name}/labels/${s}.nii.gz \
        ${output}/${name}/mosaics/${s}-label.png
      echo ${bin}/mosaic.sh \
        ${input}/${name}/images/${s}.nii.gz \
        ${output}/${name}/out/${s}.nii.gz \
        ${output}/${name}/mosaics/${s}-pred.png
    done | parallel -j 20
  fi

  if [ ! -e ${input}/${name}/volumes.csv ]; then
    qit --verbose MaskMeasureBatch \
      --value data \
      --input ${input}/${name}/labels/%s.nii.gz \
      --names ${input}/${name}/names.txt \
      --output ${input}/${name}/volumes.csv
  fi

  if [ ! -e ${output}/${name}/volumes.csv ]; then
    qit --verbose MaskMeasureBatch \
      --value ${output} \
      --input ${output}/${name}/out/%s.nii.gz \
      --names ${input}/${name}/names.txt \
      --output ${output}/${name}/volumes.csv
    qit --verbose TableMerge \
      --field name \
      --left ${input}/${name}/volumes.csv \
      --right ${output}/${name}/volumes.csv \
      --output ${output}/${name}/volumes.csv
  fi
}

postprocess train
postprocess valid
postprocess test
postprocess fail

################################################################################
# End
################################################################################
