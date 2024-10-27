#!/bin/bash
for i in {1..15}; do
  java -cp out/production/NNPROJECT Main 3 $i
done

for i in {1..15}; do
  java -cp out/production/NNPROJECT Main 6 $i
done

for i in {1..15}; do
  java -cp out/production/NNPROJECT Main 9 $i
done

for i in {1..15}; do
  java -cp out/production/NNPROJECT Main 12 $i
done