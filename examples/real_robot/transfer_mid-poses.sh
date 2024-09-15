#!/bin/bash

mkdir -p "data_dir/processed/$1_mid-poses/"

cp "data_dir/processed/$1/data_point_0.json" "data_dir/processed/$1_mid-poses/data_point_0.json"
cp "data_dir/processed/$1/data_point_156.json" "data_dir/processed/$1_mid-poses/data_point_156.json"
cp "data_dir/processed/$1/data_point_401.json" "data_dir/processed/$1_mid-poses/data_point_401.json"
cp "data_dir/processed/$1/data_point_630.json" "data_dir/processed/$1_mid-poses/data_point_630.json"
cp "data_dir/processed/$1/data_point_778.json" "data_dir/processed/$1_mid-poses/data_point_778.json"
cp "data_dir/processed/$1/data_point_908.json" "data_dir/processed/$1_mid-poses/data_point_908.json"
cp "data_dir/processed/$1/data_point_1000.json" "data_dir/processed/$1_mid-poses/data_point_1000.json"
