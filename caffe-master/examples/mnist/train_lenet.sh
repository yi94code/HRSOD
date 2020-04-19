#!/usr/bin/env sh
set -e

<<<<<<< HEAD
./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt 2>&1 | tee logi1.txt

=======
./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt $@
>>>>>>> 42cd785e4b5ed824a9b2a02a19aa534042b64325
