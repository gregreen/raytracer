#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np

import json


def load_obj(fname):
    with open(fname, 'r') as f:
        txt = f.read()

    verts = []
    faces = []

    for line in txt.splitlines():
        line = line.strip()
        if not len(line):
            continue

        cols = line.split()
        print(cols)

        if cols[0] == 'v':
            v = [float(c) for c in cols[1:]]
            print(v)
            assert len(v) == 3
            verts.append(v)
        if cols[0] == 'f':
            f = [int(c)-1 for c in cols[1:]]
            assert len(f) == 3
            faces.append(f)

    d = {'vertices': [verts], 'faces': [faces]}
    return d


def main():
    d = load_obj('data/teapot.obj')
    with open('data/teapot.json', 'w') as f:
        json.dump(d, f, indent=2)

    return 0

if __name__ == '__main__':
    main()

