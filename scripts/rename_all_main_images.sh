#!/bin/bash

for SPLIT in train test val; do
  DIR="data/nerf_synthetic/lego/${SPLIT}"
  echo "Renaming images in $DIR..."

  cd "$DIR" || continue

  imgs=($(ls r_*.png 2>/dev/null | grep -v depth | grep -v normal | sort))
  i=0
  for img in "${imgs[@]}"; do
    new_name="r_${i}.png"
    echo "Renaming $img -> $new_name"
    mv "$img" "$new_name"
    ((i++))
  done

  cd - >/dev/null
done
