#!/bin/bash
# Download and compile SQLite 3.41.2
[server]
preLaunch = "./setup.sh"
wget https://www.sqlite.org/2023/sqlite-autoconf-3410200.tar.gz
tar xzf sqlite-autoconf-3410200.tar.gz
cd sqlite-autoconf-3410200
./configure --prefix=$HOME/sqlite
make
make install