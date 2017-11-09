#!/bin/bash
rm -f noized/*
./make_noize.py
./main.py
