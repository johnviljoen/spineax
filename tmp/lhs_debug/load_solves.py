import numpy as np
import re
from pathlib import Path

def parse_array(s):
    s = s.replace('\n', '').replace('] [', '], [')
    return np.array(eval(re.sub(r'(\d)\s+(-?\d)', r'\1, \2', s)))

def load_solve(path):
    text = Path(path).read_text()
    csr = parse_array(re.search(r'csr_values:\n(\[.*?\])\n\ndiag', text, re.S).group(1))
    diag = parse_array(re.search(r'diag_reordered:\n(\[.*?\])\npositive', text, re.S).group(1))
    return csr, diag

def load_all(directory="tmp/lhs_debug", indices=None):
    files = sorted(Path(directory).glob("solve_*.txt"))
    if indices is not None:
        files = [files[i] for i in indices]
    data = [load_solve(f) for f in files]
    return np.stack([d[0] for d in data]), np.stack([d[1] for d in data])

def find_outlier_indices(diag_batch):
    from collections import Counter
    rows_as_tuples = [tuple(row) for row in diag_batch]
    most_common_row = np.array(Counter(rows_as_tuples).most_common(1)[0][0])
    outliers = np.where(~np.all(diag_batch == most_common_row, axis=1))[0]
    return outliers, most_common_row

def find_sign_outliers(diag_batch):
    from collections import Counter
    signs = np.sign(diag_batch)
    signs_as_tuples = [tuple(row) for row in signs]
    most_common_signs = np.array(Counter(signs_as_tuples).most_common(1)[0][0])
    outliers = np.where(~np.all(signs == most_common_signs, axis=1))[0]
    return outliers, most_common_signs

def find_inertia_outliers(diag_batch):
    from collections import Counter
    pos_counts = np.sum(diag_batch > 0, axis=1)
    neg_counts = np.sum(diag_batch < 0, axis=1)
    inertias = list(zip(pos_counts, neg_counts))
    common_inertia = Counter(inertias).most_common(1)[0][0]
    outliers = np.where((pos_counts != common_inertia[0]) | (neg_counts != common_inertia[1]))[0]
    return outliers, common_inertia

if __name__ == "__main__":
    csr_batch, diag_batch = load_all()
    print(f"Loaded {len(csr_batch)} solves: csr{csr_batch.shape}, diag{diag_batch.shape}")
