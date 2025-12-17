#!/bin/bash
cd /home/ubuntu/true-asi-system
export GH_TOKEN=$(gh auth token)
git push https://oauth2:${GH_TOKEN}@github.com/AICSSUPERVISOR/true-asi-system.git master
