"""

Implements the "double dynamic programming" sequence to structure
threading algorithm (Jones, 1998). The idea is:

1) For every candidate anchor pair (m, n) -- structure position m,
   sequence residue n -- run an inner ("low-level") Needleman & Wunsch
   alignment (L_matrix) that scores how well the rest of the sequence
   fits the rest of the structure given that fixed anchor, using DOPE
   pseudo-energies.
2) Combine the low-level results of every anchor into a single
   "high-level" matrix H (H_matrix).
3) Run a final ("high-level") Needleman & Wunsch alignment (F_matrix)
   using H as the substitution-score matrix, to obtain the best overall
   sequence to structure threading alignment and its global DOPE pseudo 
   energy (F[0][0]).

"""

import numpy as np
from DOPE import get_dope_score 

# Constant
U = 9999.0          # penalty for forbidden cells 
GAP_COIL = 2.0      # cost of a gap in the coil (C)
GAP_H_E = 20.0     # cost of a gap in a helix (H) or in a strand (E)

DICO_AA_ONE_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

DICO_AA_THREE_ONE = {three_letter: one_letter for one_letter, three_letter in DICO_AA_ONE_THREE.items()}

# Handle the amino acid code
def get_AA_three(sequence):
    """
    Convert a 1 letter amino acid sequence into a list of 3 letter
    codes.
    eg :  "ACDE..." -> ["ALA", "CYS", "ASP", "GLU" ...]
    """
    return [DICO_AA_ONE_THREE[aa.upper()] for aa in sequence]

def get_AA_one(three_letter):
    """
    Convert a 3 letter amino acid code into its 1 letter
    equivalent.
    eg : "ALA" -> "A"
    """
    return DICO_AA_THREE_ONE.get(three_letter, "X")

# Gap penalties
def gap_cost(ss_symbol):
    """
    Return the gap penalty for a structure
    position, depending on its secondary structure.
    """
    if ss_symbol == "C":
        return GAP_COIL
    else :
        return GAP_H_E

# Global alignment algorithm -> Needleman wunsch
def needleman_wunsch(n_stru, n_seq, ss_stru, cost_fn):
    """
    Needleman & Wunsch algorithm (global
    alignment, cost minimisation), shared by the low-level matrix L and
    the high-level matrix F. Only the local scoring function
    (cost_fn) differs between the two use cases.

    Inputs:
        n_stru (int): Number of structure positions (rows).
        n_seq (int): Number of sequence residues (columns).
        ss_stru (list of str): Secondary structure of each structure position.
        cost_fn : Function cost_fn(p, q) -> score (float) of aligning structure 
        position p with sequence residue q.

    Principle:
        Fills a (n_stru+1) x (n_seq+1) matrix with zeros,
        from the C-terminal end (last row/column) back to the
        N-terminal end (cell [0][0]). 
        The last row and last column are initialised as coil-gap boundaries.
        For each remaining cell (p, q), the cheapest option is kept
        among the diagonal continuation, or a gap of any length in
        the structure or in the sequence. Each gap is weighted by
        gap_cost() of the structure position being skipped.

    Outputs:
        matrix, moves (tuple):
            matrix (numpy.ndarray): minimal cost from each cell to the end.
            moves (list of list):moves[p][q] is a tuple (move_type, target) recording
                which choice ("diag", "gap_row" or "gap_col") was made at cell (p, q).
                Used by get_path() for traceback.
    """

    # Initialisation of the scoring matrix
    matrix = np.zeros((n_stru + 1, n_seq + 1)) 
    # Initialisation of the deplacement matrix
    moves = [] 
    for p in range(n_stru + 1):
        row = [None] * (n_seq + 1)
        moves.append(row)

    # Fill the margin (last column, last row)
    for q in range(n_seq - 1, -1, -1): 
        matrix[n_stru][q] = matrix[n_stru][q + 1] + GAP_COIL
    for p in range(n_stru - 1, -1, -1):
        matrix[p][n_seq] = matrix[p + 1][n_seq] + GAP_COIL

    # Fill the matrix by computing the min cost 
    for p in range(n_stru - 1, -1, -1):
        g = gap_cost(ss_stru[p])
        for q in range(n_seq - 1, -1, -1):
            best_val = matrix[p + 1][q + 1] # Initialisation 
            best_move = ("diag", None) # Initialisation 

            # Check the min cost for all possible number of gap
            for r in range(p + 2, n_stru + 1): # gap(s) in the structure
                val = matrix[r][q + 1] + g
                if val < best_val:
                    best_val, best_move = val, ("gap_row", r)

            for s in range(q + 2, n_seq + 1): # gap(s) in the seq
                val = matrix[p + 1][s] + g
                if val < best_val:
                    best_val, best_move = val, ("gap_col", s)

            # Fill the cell with the minimal value
            matrix[p][q] = cost_fn(p, q) + best_val
            moves[p][q] = best_move # Save the displacement made

    return matrix, moves

def get_path(moves, n_stru, n_seq):
    """
    Reconstruct the optimal alignment path from a moves matrix.

    inputs:
        moves (list of list): Moves matrix as returned by
            needleman_wunsch().
        n_stru (int): Number of structure positions.
        n_seq (int): Number of sequence residues.

    Ouputs:
        list of tuple [(p0,q0), (...)]: Each element is (p, q) where p is a
            structure position index (int) or None (gap in the
            structure), and q is a sequence residue index (int) or
            None (gap in the sequence). p and q are never None at
            the same time.
    """
        
    # Initialisation
    path = []
    p, q = 0, 0

    # Explore the moves matrix
    while p < n_stru and q < n_seq: # while we are not at the end of
        # the sequence OR at the end of the structure
        path.append((p, q))
        move, target = moves[p][q]

        if move == "diag":
            p, q = p + 1, q + 1
        elif move == "gap_row":
            for gap in range(p + 1, target):
                path.append((gap, None))
            p, q = target, q + 1
        elif move == "gap_col":
            for gap in range(q + 1, target):
                path.append((None, gap))
            p, q = p + 1, target

    while p < n_stru: # the sequence ended but not the structure
        path.append((p, None))
        p += 1
    while q < n_seq: # the structure ended but not the sequence
        path.append((None, q)) 
        q += 1
    return path


# Low level matrix L for an anchor (m,n)
def low_level_cost(m, n, matrix_dist, seq_3_letter, dope_dict):
    """
    Build the local scoring function used to fill the low-level
    matrix L for one fixed anchor pair (m, n).

    Inputs:
        m (int): Structure position index of the anchor.
        n (int): Sequence residue index of the anchor.
        matrix_dist (numpy.ndarray): CA-CA distance matrix of the
         structure (returned by Handle_PDB.compute_distance_matrix).
        seq_3_letter (list of str): sequence in 3 letter amino acid codes.
        dope_dict: DOPE dictionary, returned by DOPE.read_dope().

    Principle:
        Returns a function cost_function_L(p, q) that gives:
            - 0.0 at the anchor cell (p == m and q == n)
            - U (9999.0) anywhere else on the anchor row or column
            - the DOPE pseudo-energy between residue n and residue
              q
            - U when (p, q) would require the alignment
              path to cross the anchor.

    output:
        cost_function_L(p, q) to be passed to needleman_wunsch().
    """
        
    res_n = seq_3_letter[n]

    def cost_function_L(p, q): # function called by needleman_wunsch
        if p == m and q == n: # anchor (m,n)
            return 0.0
        if p == m or q == n: # anchor line or column but not on (m,n) -> FORBID 
            return U
        
        autorized_cells = (p > m and q > n) or (p < m and q < n)

        if autorized_cells:
            d = matrix_dist[m][p] # distance between position m and p
            return get_dope_score(dope_dict, res_n, seq_3_letter[q], d)
        
        return U  # (q<n and p>n) OR (q>n and p<m)
    
    return cost_function_L

def L_matrix(m, n, matrix_dist, ss_stru, seq_3_letter, dope_dict):
    """
    Run the low-level (inner) dynamic programme for one anchor
    (m, n). The best possible alignment given that residue n is 
    forced into position m.

    Inputs:
        m (int): Structure position index of the anchor.
        n (int): Sequence residue index of the anchor.
        matrix_dist (numpy.ndarray): CA-CA distance matrix of the structure.
        ss_stru (list of str): Secondary structure of each structure position.
        seq_3_letter (list of str): Sequence in 3 letter amino acid codes.
        dope_dict: DOPE dictionary.

    Returns:
        L, path (tuple):
            L (numpy.ndarray): Low-level score matrix for this anchor (L[0][0]
                is the score of the best low-level threading for this anchor).
            path (list of tuple): Optimal low-level alignment path
                for this anchor.
    """
    n_stru = matrix_dist.shape[0] # length of the structure (number of residues)
    n_seq = len(seq_3_letter) # length of the sequence 

    cost_function_L = low_level_cost(m, n, matrix_dist, seq_3_letter, dope_dict)

    L, moves = needleman_wunsch(n_stru, n_seq, ss_stru, cost_function_L) # Fill the L matrix
    path = get_path(moves, n_stru, n_seq) # save the optimal path 

    return L, path

# High level matrix H to accumulate the optimal paths/ alignments
# Formulas different from the paper
def H_matrix(matrix_dist, ss_stru, seq_3_letter, dope_dict, verbose=False):
    """
    Build the high-level matrix H by combining the
    low-level results of many anchors.

    Inputs:
        matrix_dist (numpy.ndarray): CA-CA distance matrix of the structure.
        ss_stru (list of str): Secondary structure of each structure position.
        seq_3_letter (list of str): Sequence in 3 letter amino acid codes.
        dope_dict: DOPE dictionary.
        verbose (bool): If True, print progress (one line per
            structure position processed) to the terminal.

    Principle:
        For each structure position m, only sequence positions n
        within a window around the expected diagonal position 
        are tried to limit the number of low-level alignments computed. 
        For each possible anchor (m, n), L_matrix() is run.
        If its score L[0][0] is below a fixed cutoff (500), the score L[0][0] is
        added into every cell (p, q) of H that lies on that anchor
        optimal path.

    Output:
        H (numpy.ndarray): the high-level matrix
    """
    n_stru = len(ss_stru)
    n_seq = len(seq_3_letter)  
    H = np.zeros((n_stru, n_seq)) # Initialisation of H

    # Searching window
    window = max(15, int(n_seq * 0.20))

    # Iterate from C-terminal to N-terminal
    for m in range(n_stru - 1, -1, -1):
        expected_n = int(m * (n_seq / n_stru)) # ideal n position

        for n in range(n_seq - 1, -1, -1):
            if abs(n - expected_n) > window: 
                continue # do not comput L if n is outsite the window

            L, path = L_matrix(m, n, matrix_dist, ss_stru, seq_3_letter, dope_dict)
            
            # Add the optimal path of L in H
            if L[0][0] < 500: # we do not consider bad optimal paths
                for p, q in path:
                   if p is not None and q is not None: # Ignore gaps
                         H[p][q] += L[0][0] # Accumulation 
                    
        if verbose:
            print(f"H : line {n_stru - m}/{n_stru} computed", end="\r")

    if verbose:
        print() # return to the line when completed 

    return H

# High level and final matrix F, contains the optimal alignment
def high_level_cost_fn(H):
    """
    Build the local scoring function used to fill the final
    (high-level) matrix F.

    Input:
        H (numpy.ndarray): High-level matrix.

    Output:
        cost_function_F(p, q) -> return H[p][q], to be passed
            to needleman_wunsch().
    """

    def cost_function_F(p, q):  # function called by needleman_wunsch
        return H[p][q]
    
    return cost_function_F

def F_matrix(H, ss_stru):
    """
    Run the final (high-level) dynamic programme: find the
    best sequence to structure threading alignment,
    using H as the scoring matrix.

    Inputs:
        H (numpy.ndarray): High-level matrix
        ss_stru (list of str): Secondary structure of each structure position.

    Outputs:
        F, path (tuple):
            F (numpy.ndarray): Final score matrix. F[0][0] is the
                threading energy of the best alignment (the lower the better).
            path (list of tuple): Final optimal alignment path.
    """

    n_stru, n_seq = H.shape # (lines, columns)

    cost_fn = high_level_cost_fn(H)

    F, moves = needleman_wunsch(n_stru, n_seq, ss_stru, cost_fn) # Fill F
    path = get_path(moves, n_stru, n_seq) # save THE optimal path

    return F, path