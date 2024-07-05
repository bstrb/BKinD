#!/bin/bash

# Navigate to the directory where the script is located
cd "$(dirname "$0")"

# Pull the latest changes from the main branch
git pull origin main

# You can add any other commands you want to run after updating, like installing dependencies
# pip install -r setup/requirements.txt
