#!/bin/bash

mkdir ~/temp
mv ~/library/.azure-cleanup.sh ~/temp

cd ~/library
git reset --hard b4014f74c
git fetch
git pull

mv -t ~/temp ~/library/release/* ~/library/README.md ~/library/LICENSE
rm -rf ~/library/*
mv -t ~/library ~/temp/* ~/temp/.azure-cleanup.sh
rm -rf ~/temp