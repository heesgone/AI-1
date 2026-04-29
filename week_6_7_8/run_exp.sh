#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=2
export TMPDIR=~/tmp

run_exp () {
  echo "===================================="
  echo "Running $1"
  df -h .
  echo "===================================="
  python ResNetEXP.py --exp "$1"
}

run_exp exp4_label_smoothing
run_exp exp5_stochastic_depth
run_exp exp6_randaugment
run_exp exp7_dropout
run_exp exp8_reduced_wd
run_exp exp9_se
run_exp exp10_resnetd