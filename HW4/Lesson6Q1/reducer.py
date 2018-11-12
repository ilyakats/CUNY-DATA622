#!/usr/bin/python

# CUNY MSDS Program, DATA 622, Homework 4
# Created: November 2018
# Ilya Kats
#
# This is a basic reducer code to implement Hadoop MapReduce method.
# Based on Udacity's Intro into Hadoop and MapReduce course.
# Answer to Question 1 of Lesson 6.

import sys

salesTotal = 0
oldKey = None

for line in sys.stdin:
    data_mapped = line.strip().split("\t")
    
    # Check for expected number of fields
    if len(data_mapped) != 2:
        continue

    thisKey, thisSale = data_mapped

    if oldKey and oldKey != thisKey:
        print oldKey, "\t", salesTotal
        oldKey = thisKey;
        salesTotal = 0

    oldKey = thisKey
    salesTotal += float(thisSale)

if oldKey != None:
    print oldKey, "\t", salesTotal
