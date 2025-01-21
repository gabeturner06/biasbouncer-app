#!/bin/bash

echo "Starting setup script for SQLite upgrade..."

# Update package lists and install dependencies
echo "Updating package lists and installing dependencies..."
sudo apt-get update
sudo apt-get install -y wget build-essential libsqlite3-dev

# Download and extract SQLite source code (version 3.41.1 used here as an example)
SQLITE_VERSION="3410100"
SQLITE_URL="https://www.sqlite.org/2023/sqlite-autoconf-${SQLITE_VERSION}.tar.gz"
echo "Downloading SQLite version 3.41.1 from ${SQLITE_URL}..."
wget "${SQLITE_URL}" -O sqlite.tar.gz
tar -xzf sqlite.tar.gz
cd sqlite-autoconf-${SQLITE_VERSION}

# Configure, build, and install SQLite
echo "Configuring, building, and installing SQLite..."
./configure
make
sudo make install

# Verify the installation
echo "Verifying SQLite installation..."
sqlite3 --version || { echo "SQLite installation failed"; exit 1; }

# Clean up temporary files
echo "Cleaning up..."
cd ..
rm -rf sqlite-autoconf-${SQLITE_VERSION} sqlite.tar.gz

echo "SQLite setup completed successfully!"
