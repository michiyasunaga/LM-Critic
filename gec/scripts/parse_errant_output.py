#!/usr/bin/env python

import sys
import json

for i, line in enumerate(sys.stdin):
    if i == 3:
        nums = line.split()
        P, R, F = nums[3:6]
        json.dump({'precision': float(P), 'recall': float(R), 'F0.5': float(F)}, open('stats.json', 'w'), indent=2)
        break
