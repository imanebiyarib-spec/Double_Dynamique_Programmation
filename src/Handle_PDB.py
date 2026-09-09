"""

Extracts, from a PDB file, the alpha carbon (CA) coordinates and
per residue secondary structure (Helix / Strand / Coil) of a
single chain, and computes the corresponding CA-CA Euclidean
distance matrix. 

"""


import math
import numpy as np


def extract_CA_pdb(pdb_file, target_chain='A'):
    """
    Extract CA coordinates and secondary structure for one chain
    of a PDB file.

    Input:
        pdb_file (str): Path to the PDB file.
        target_chain (str): Chain identifier to extract (default
            "A").

    Principle:
        1) Scans HELIX ('H') and SHEET ('E') records to build a per-residue
        secondary structure map. Any residue not covered by the
        record is assigned to 'C' (coil).
        2) Scans ATOM records, keeping only CA atoms of the requested chain,
        and stores their coordinates with the secondary structure looked
        up from the map built before.

    Output:
        CA_dict (dictionnary): 
            Keys are residue numbers (int)
            Values are dictionnaries with keys:
                "res" (str): 3 letter amino acid code.
                "x", "y", "z" (float): CA coordinates in Angstrom.
                "ss" (str): secondary structure.
    """

    second_stru = {}
    CA_dict = {}

    with open(pdb_file, 'r') as file:
        for line in file:
            
            # helix 
            if line.startswith("HELIX"):
                chain = line[19:20].strip() # starting chain

                if chain == target_chain:
                    index_res_start = int(line[21:25].strip())
                    index_res_end = int(line[33:37].strip())
                    for i in range(index_res_start, index_res_end + 1):
                        second_stru[i] = 'H'
                        
            # Sheet
            elif line.startswith("SHEET"):
                chain = line[21:22].strip() # starting chain

                if chain == target_chain:
                    index_res_start = int(line[22:26].strip())
                    index_res_end = int(line[33:37].strip())
                    for i in range(index_res_start, index_res_end + 1):
                        second_stru[i] = 'E'

            # C alpha 
            elif line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                chain = line[21:22].strip()
                
                if atom_name == "CA" and chain == target_chain:
                    res_name = line[17:20].strip()       # 3 letters code
                    index_res = int(line[22:26].strip())     
                    x = float(line[30:38].strip())       
                    y = float(line[38:46].strip())       
                    z = float(line[46:54].strip())     
                    ss = second_stru.get(index_res, 'C') # secondary structure (by default 'C' for Coil)
                    
                    # Save
                    CA_dict[index_res] = {
                        "res": res_name,
                        "x": x,
                        "y": y,
                        "z": z,
                        "ss": ss
                    }
                    
    return CA_dict


def compute_distance_matrix(CA_dict):
    """
    Compute the pairwise CA-CA euclidean distance matrix of a
    structure.

    Input:
        CA_dict: Dictionary returned by extract_CA_pdb().

    Principle:
        1) Residue numbers are sorted 
        2) All pairwise Euclidean distances between CA
        coordinates are then computed once (the matrix is
        symmetric, diagonal is 0).

    Output:
        matrix_dist (numpy.ndarray): Square matrix of shape (M, M),
            where M is the number of residues in CA_dict.
    """
    
    index_res = sorted(CA_dict.keys())
    M = len(index_res)
    
    # matrix MxM
    matrix_dist = np.zeros((M, M)) # d(A,A) = 0 -> our default 
    
    # filling 
    for i in range(M):
        for j in range(M):
            if i < j:
                res_i = CA_dict[index_res[i]]
                res_j = CA_dict[index_res[j]]
                
                dx = res_i['x'] - res_j['x']
                dy = res_i['y'] - res_j['y']
                dz = res_i['z'] - res_j['z']
                dist = math.sqrt(dx**2 + dy**2 + dz**2)
                
                matrix_dist[i][j] = dist
                matrix_dist[j][i] = dist # d(A,B) = d(B,A)
                
    return matrix_dist

if __name__ == "__main__" : 
    # CA_dict = extract_CA_pdb("5PTI.pdb")
    # print(CA_dict[1])
    # print(CA_dict[1]["res"])
    # print(len(CA_dict))
    # print(CA_dict)
    # matrix_dist = compute_distance_matrix(CA_dict)
    # print(matrix_dist.min(), matrix_dist.max())
    pass