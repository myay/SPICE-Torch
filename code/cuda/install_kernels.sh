#!/bin/bash

for d in */ ; do
  cd ${d}
  if [ -f setup.py ]; then
    python3 setup.py install --user
  fi
  cd ..
done
