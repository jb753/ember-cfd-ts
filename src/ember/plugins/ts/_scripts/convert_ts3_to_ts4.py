#!/usr/bin/env python
"""Convert a TS3-format HDF5 file to a TS4 input file.

Usage: convert_ts3_to_ts4.py INPUT_HDF5 OUTPUT_STEM

Vendored, with light trimming, from turbigen's ``convert_ts3_to_ts4_native.py``.
This is glue around Turbostream Ltd's own proprietary ``ts.process``/VTK stack
and has no ember dependency: it must run inside the sourced Turbostream
environment, as a subprocess, not in the interpreter that drives the solver.

The unsteady inlet/outlet "group" tagging of the original is dropped, since
unsteady forcing is out of scope for this port.

The conversion must be a pure format translation, preserving nodes one-for-one
(including the duplicates at block interfaces). The plugin checks that on the
way back in and refuses a file where it does not hold.
"""

import os
import sys

import numpy
import ts.process.ts3_reader
import ts.process.pre.vtk_adaptor

print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("Numpy version:", numpy.__version__)
print("Numpy path:", numpy.__path__)

if not len(sys.argv) == 3:
    print("Usage: convert_ts3_to_ts4.py INPUT_HDF5 OUTPUT_STEM")
    exit(1)

# Make all paths absolute
input_hdf5, output_stem = sys.argv[1:3]
input_hdf5_full = os.path.abspath(input_hdf5)
output_stem_full = os.path.abspath(output_stem)

# Check for input and output files
if not os.path.exists(input_hdf5_full):
    print("Input %s not found" % input_hdf5_full)
    exit(1)
output_dir, _ = os.path.split(output_stem_full)
if not os.path.exists(output_dir):
    print("Output dir %s not found" % output_dir)
    exit(1)

# Read TS3
reader = ts.process.ts3_reader.TS3Reader()
vtk_grid = reader.read(input_hdf5_full, True)

# Write TS4
g = ts.process.pre.vtk_adaptor.create_ts_grid(vtk_grid)
g.write_hdf5(f"{output_stem_full}.hdf5")
