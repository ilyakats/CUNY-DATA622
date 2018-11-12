#!/usr/bin/python

# CUNY MSDS Program, DATA 622, Homework 4
# Created: November 2018
# Ilya Kats
#
# This is a basic mapper code to implement Hadoop MapReduce method.
# Based on Udacity's Intro into Hadoop and MapReduce course.
# Answer to Question 1 of Lesson 6.

import sys

for line in sys.stdin:
    data = line.strip().split("\t")
    
    # Check for expected number of fields
    if len(data) != 6:
        continue
    
    date, time, store, item, cost, payment = data
    
    # Check that cost is a number
    try:
        cost = float(cost)
    except ValueError:
        continue
    
    print "{0}\t{1}".format(item, cost)
