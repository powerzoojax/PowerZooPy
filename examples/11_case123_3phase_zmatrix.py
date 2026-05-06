import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from powerzoo.case import load_case

# Load case and get impedance matrices
c = load_case(123)
Z_matrix = c.get_line_impedance_matrices()

# Read and convert CSV to complex array
csv_path = os.path.join(os.path.dirname(__file__), 'x11_IEEE123_Z_3ph', 'IEEE123_Z_3ph.csv')
Z_read = pd.read_csv(csv_path)
Z_read = np.array(Z_read.applymap(complex).values, dtype=np.complex128).reshape(-1, 3, 3)

# Calculate difference and magnitude sum
diff = Z_read - Z_matrix
magnitude_sum = np.abs(diff).sum(axis=(1, 2))

print(f"Shapes: Z_read={Z_read.shape}, Z_matrix={Z_matrix.shape}")
print(f"Magnitude sum: {magnitude_sum}")
