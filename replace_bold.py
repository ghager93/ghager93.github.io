#!/usr/bin/env python

import re
import os
with open("enclosed-curve-shortening-flow.md") as f:
    text = f.read()

changed_text = re.sub(r'\\bf', r'\\boldsymbol', text)

with open("enclosed-curve-shortening-flow.md", "w") as f:
    f.write(changed_text)
