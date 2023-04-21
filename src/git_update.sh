#!/bin/bash

BRANCH =$1

cd ~chris\Desktop\Python\Harrison Crypto Algorithm\crypto-trading-algorithm

git fetch origin
git checkout $BRANCH
git fetch origin main
git merge origin/main