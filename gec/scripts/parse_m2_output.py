#!/usr/bin/env python

import sys
import json

scores = []
for i, line in enumerate(sys.stdin):
    score = line.split(':')[1].strip()
    scores.append(float(score))

json.dump({'precision': scores[0], 'recall': scores[1], 'F0.5': scores[2]}, open('stats.json', 'w'), indent=2)
