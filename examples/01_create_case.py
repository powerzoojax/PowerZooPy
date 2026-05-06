import os
import sys

# Ensure project root on sys.path when running from examples directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from powerzoo.case import load_case

# Create a default case and print basic info
c = load_case(5) # 5 or '5' or '33bw' ...
print(c)




