#!/bin/bash

sed -i 's/requests.packages.urllib3.util.retry/urllib3.util.retry/g' ~/.local/lib/python3.8/site-packages/best_download/__init__.py
