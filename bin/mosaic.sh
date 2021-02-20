#! /bin/bash
################################################################################
#
# A program for creating image mosaics of 3D volumes 
#
# Author: Ryan Cabeen
#
################################################################################

if [ $# -ne 3 ]; then
  echo "usage: $(basename $0) image.nii.gz labels.nii.gz output.png"
  exit
fi

echo "started $(basename $0)"

image=$1
labels=$2
output=$3

echo "... using image: ${image}"
echo "... using labels: ${labels}"
echo "... using output: ${output}"

mkdir -p $(dirname ${output})

echo "... rendering volume"
qit VolumeRender \
  --background ${image} --labels ${labels} \
  --bglow 0 --bghigh threeup --discrete Red --alpha 0.25 \
  --output ${output}.nii.gz

echo "... rendering mosaic"
qit VolumeMosaic --rgb --axis j --input ${output}.nii.gz --output ${output} 

rm -rf ${output}.nii.gz

echo "finished $(basename $0)"

################################################################################
# End
################################################################################
