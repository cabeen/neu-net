#! /bin/bash
################################################################################
#
#  A script for running a whole experiment
#
#  Author: Ryan Cabeen
#
################################################################################

cd $(dirname $0)
cd ..

bin="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

epochs=10
rescale=128
kernel=128

mkdir -p results/{valid,test,train}/{out,mosaics}

if [ ! -e results/models ]; then
  python3 ${bin}/train.py \
    --kernel ${kernel} \
    --rescale ${rescale} \
    --epochs ${epochs} \
    --largest \
    --images data/train/images \
    --masks data/train/labels \
    --output results/models
fi

if [ ! -e results/valid/out ]; then
  python3 ${bin}/validate.py \
    --models results/models \
    --images data/valid/images \
    --masks data/valid/labels \
    --output results/valid/out
fi

if [ ! -e results/test/out ]; then
  python3 ${bin}/test.py \
    --model results/valid/out/best-model \
    --images data/test/images \
    --masks data/test/labels \
    --output results/test/out
fi

if [ ! -e results/train/out ]; then
  python3 ${bin}/test.py \
    --model results/valid/out/best-model \
    --images data/train/images \
    --masks data/train/labels \
    --output results/train/out
fi

if [ -e data/fail ]; then
  if [ ! -e results/fail/out ]; then
    python3 ${bin}/test.py \
      --model results/valid/out/best-model \
      --images data/fail/images \
      --masks data/fail/labels \
      --output results/fail/out
  fi
fi

function postprocess 
{
  name=$1

  if [ ! -e results/${name}/mosaics ]; then
    for s in $(cat data/${name}/names.txt); do \
      echo ${bin}/mosaic.sh \
        data/${name}/images/${s}.nii.gz \
        data/${name}/labels/${s}.nii.gz \
        results/${name}/mosaics/${s}-label.png
      echo ${bin}/mosaic.sh \
        data/${name}/images/${s}.nii.gz \
        results/${name}/out/${s}.nii.gz \
        results/${name}/mosaics/${s}-pred.png
    done | parallel -j 20
  fi

  if [ ! -e data/${name}/volumes.csv ]; then
    qit --verbose MaskMeasureBatch \
      --value data \
      --input data/${name}/labels/%s.nii.gz \
      --names data/${name}/names.txt \
      --output data/${name}/volumes.csv
  fi

  if [ ! -e results/${name}/volumes.csv ]; then
    qit --verbose MaskMeasureBatch \
      --value results \
      --input results/${name}/out/%s.nii.gz \
      --names data/${name}/names.txt \
      --output results/${name}/volumes.csv
    qit --verbose TableMerge \
      --field name \
      --left data/${name}/volumes.csv \
      --right results/${name}/volumes.csv \
      --output results/${name}/volumes.csv
  fi
}

postprocess train
postprocess valid
postprocess test
postprocess fail

################################################################################
# End
################################################################################
