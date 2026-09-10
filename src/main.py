"""

Orchestrator for the sequence to structure threading
pipeline. 
Given a structure (PDB), a sequence (FASTA) and the true
structure of that sequence (PDB), it runs the double
dynamic programming alignment implemented in Alignment.py, writes
the resulting alignment to a text file, appends a summary row
(scores, RMSD, timing) to a csv log, and prints a short
report.

Command line:
    python main.py structure.pdb sequence.fasta true_structure.pdb
        --dope dope.par --seq_id PDB_ID CHAIN

        [--outdir ./results] [--verbose]
        [--stru_chain A] [--target_chain A]

"""

# Libraries 
import os
import csv
import random
import datetime
import argparse
import time

import numpy as np

# Modules 
from DOPE import read_dope, get_dope_score
from Handle_PDB import extract_CA_pdb, compute_distance_matrix
from Handle_FASTA import extract_sequences
from Alignment import get_AA_three, get_AA_one, H_matrix, F_matrix


# Read pdb structure file 
def handle_structure(pdb_file, chain="A"):
    """
    Load a structure PDB file and prepare all the inputs needed by
    the threading algorithm.

    Inputs:
        pdb_file (str): Path to the structure PDB file.
        chain (str): Chain to extract (default "A").

    Outputs:
        CA_dict, sorted_keys, dist_matrix, ss_stru, res_stru (tuple):
            CA_dict (dictionnary).
            sorted_keys (list of int): Residue numbers in ascending order.
            dist_matrix (numpy.ndarray): CA-CA distance matrix.
            ss_stru (list of str): Secondary structure per position.
            res_stru (list of str): 3 letter amino acid code per position.
    """

    CA_dict = extract_CA_pdb(pdb_file, target_chain=chain) 
    sorted_keys = sorted(CA_dict.keys()) # Residue index as key

    dist_matrix = compute_distance_matrix(CA_dict)

    ss_stru = [CA_dict[k]["ss"] for k in sorted_keys] 
    res_stru = [CA_dict[k]["res"] for k in sorted_keys] # 3 letter amino acid code 
    
    return CA_dict, sorted_keys, dist_matrix, ss_stru, res_stru

# Read fasta sequence file
def handle_sequence(fasta_file, seq_id):
    """
    Load a FASTA file and extract one target sequence.

    Inputs:
        fasta_file (str): Path to the FASTA file.
        seq_id (tuple of str): Key (pdb_id, chain_id) identifying
            which sequence to extract.

    Outputs:
        pdb_id, chain_id, seq_dict[seq_id] (tuple):
            pdb_id (str): PDB identifier of the sequence.
            chain_id (str): Chain identifier of the sequence.
            sequence (str): sequence.
    """

    seq_dict = extract_sequences(fasta_file)
    pdb_id, chain_id = seq_id

    return pdb_id, chain_id, seq_dict[seq_id]


# RESULTS
# Final Alignement
def final_alignment(path, seq, res_stru, ss_stru):
    """
    Format the final threading alignment as three lines.

    Inputs:
        path (list of tuple): Final alignment path (each
            element is (structure_index or None, sequence_index or
            None)).
        seq (str): Sequence in 1 letter amino acid codes.
        res_stru (list of str): 3 letter amino acid code of the structure.
        ss_stru (list of str): Secondary structure of the structure.

    Principle:
        Walks the alignment path column by column and builds three
        strings of equal length: 
        the  sequence residue ("-"  for gaps)
        the structure residue ("-"  for gaps)
        the secondary structure symbol ("-"  for gaps)

    Outputs:
        (seq_line, stru_line, ss_line)
    """
    seq_line, stru_line, ss_line = [], [], []

    for p, q in path:
        if p is not None and q is not None:
            seq_line.append(seq[q])
            stru_line.append(get_AA_one(res_stru[p]))
            ss_line.append(ss_stru[p])

        elif p is not None: # Gap in the sequence
            seq_line.append("-")
            stru_line.append(get_AA_one(res_stru[p]))
            ss_line.append(ss_stru[p])

        else: # Gap in the structure
            seq_line.append(seq[q])
            stru_line.append("-")
            ss_line.append("-")
            
    return "".join(seq_line), "".join(stru_line), "".join(ss_line)


# RMSD: Calculate Kabsch RMSD between aligned coordinates
def kabsch_rmsd(coords_a, coords_b):
    """
    Compute the RMSD between two sets of 3D coordinates
    after optimal superposition (Kabsch algorithm).

    Inputs:
        coords_a : First set of K coordinates (Angstrom).
        coords_b : Second set of K coordinates (Angstrom).

    Principle:
        Both coordinate sets are centred on their own centroid.
        The optimal rotation matrix minimising the sum of squared
        distances between the two sets is found via singular value
        decomposition (SVD), with a reflection correction to
        ensure a proper rotation. This rotation is applied to the
        first set before computing the RMSD.

    Outputs:
        RMSD in Angstrom after optimal superposition (float).
    """
    # gravity center
    coords_a = np.asarray(coords_a, dtype=float)
    coords_b = np.asarray(coords_b, dtype=float)
    ca = coords_a - coords_a.mean(axis=0)
    cb = coords_b - coords_b.mean(axis=0)

    H = ca.T @ cb #covariance matrix
    V, S, Wt = np.linalg.svd(H) # singular Value decomposition
    # Reflection correction
    d = np.sign(np.linalg.det(Wt.T @ V.T)) 
    D = np.diag([1.0, 1.0, d])

    R = Wt.T @ D @ V.T # Rotation matrix

    ca_rot = (R @ ca.T).T # we move the predicted structure
    return float(np.sqrt(np.mean(np.sum((ca_rot - cb) ** 2, axis=1)))) # compute the distance 


# Compute RMSD using the true target PDB structure (Ground Truth)
def compute_rmsd(path, sorted_keys, CA_dict, target_pdb, target_chain="A"):
    """
    Evaluate the threading result by comparing it to the known
    structure of the sequence.

    Inputs:
        path (list of tuple): Final alignment path.
        sorted_keys (list of int): Residue numbers of the structure.
        CA_dict: CA dictionary of the structure.
        target_pdb (str): Path to the known PDB structure of the sequence.
        target_chain (str): Chain identifier to extract from target_pdb.

    Principle:
        Extracts CA coordinates from the known structure of the sequence.
        For every aligned (non-gap) pair (p, q) in path, the
        structure CA coordinate at position p is paired with the
        sequence CA coordinate at position q. 
        The two coordinate lists are then compared with kabsch_rmsd().

    Outputs:
        RMSD in Angstrom between the threading result and the true structure (float).
    """

    CA_target = extract_CA_pdb(target_pdb, target_chain=target_chain)
    sorted_target_keys = sorted(CA_target.keys())

    coords_struct, coords_target = [], []
    for p, q in path:
        if p is None or q is None:
            continue
        if q >= len(sorted_target_keys): #Kabsch need 2 structures of same size
            continue
        res_s = CA_dict[sorted_keys[p]]
        res_t = CA_target[sorted_target_keys[q]]
        coords_struct.append([res_s["x"], res_s["y"], res_s["z"]])
        coords_target.append([res_t["x"], res_t["y"], res_t["z"]])
        
    return kabsch_rmsd(coords_struct, coords_target)


# Dynamic filename generator 
def make_filename(pdb_seq_id, pdb_stru_id, result_type, ext, outdir="./results"):
    """
    Build a dynamically named output file path.

    Inputs:
        pdb_seq_id (str): Identifier of the sequence
        pdb_stru_id (str): Identifier of the structure.
        result_type (str): File content.
        ext (str): File extension.
        outdir (str): Output directory.

    Output:
        Path formatted as
            "{outdir}/{today's date}_{pdb_seq_id}_{pdb_stru_id}_{result_type}.{ext}",
            with today's date in ISO format (YYYY-MM-DD).
    """

    date_str = datetime.date.today().isoformat()
    fname = f"{date_str}_{pdb_seq_id}_{pdb_stru_id}_{result_type}.{ext}"

    return os.path.join(outdir, fname)


# Complete Threading Pipeline
def run_threading(structure_pdb, fasta_file, target_pdb, dope_dict,
                   chain_structure="A", target_chain="A", fasta_key=None,
                   outdir="./results", verbose=False):
    """
    Run the full threading pipeline : load inputs, run
    the double dynamic programming alignment, evaluate it against
    the true structureof the sequence, and write the results.

    Inputs:
        structure_pdb (str): Path to the structure PDB file.
        fasta_file (str): Path to the FASTA file containing the sequence.
        target_pdb (str): Path to the true PDB structure of the sequence.
        dope_dict: DOPE dictionary.
        chain_structure (str): Chain to extract from structure_pdb.
        target_chain (str): Chain to extract from target_pdb.
        fasta_key (tuple of str): Key (pdb_id, chain_id) identifying
            which sequence to use from fasta_file.
        outdir (str): Output directory for all result files.
        verbose (bool): If True, print progress messages.

    Principle:
        Loads the structure (handle_structure) and sequence (handle_sequence).
        Converts the sequence to 3-letter codes.
        Runs H_matrix() then F_matrix() (the double dynamic programming threading
        itself) to get the score and best alignment
        Computes the RMSD against the true structure of the sequence (compute_rmsd)
        Formats the alignment (final_alignment) and writes it to a text file
        Appends one summary row to a CSV log file shared across runs
        Measures and reports the total execution time.

    Returns:
        results (dictionnary): Summary of the run, with keys:
            "Global_DOPE_Energy" (float): final threading pseudo energy (the lower the better).
            "RMSD" (float): RMSD in Angstrom.
            "alignment_file" (str): Path to the written alignment text file.
            "log_file" (str): Path to the CSV log file the run was appended to.
            "path" (list of tuple): Final alignment path.
            "execution_time" (float): Time (seconds) the run took.
    """

    start_time = time.time() 

    # Handle structure
    CA_dict, sorted_keys, dist_matrix, ss_stru, res_stru = \
        handle_structure(structure_pdb, chain=chain_structure)

    # Handle sequence
    pdb_seq_id, seq_chain_id, target_seq = handle_sequence(
        fasta_file, seq_id=fasta_key)
    seq_3_letter = get_AA_three(target_seq)

    # isolate the pdb id (eg "data/structures/5PTI.pdb" -> "5PTI" )
    pdb_stru_id = os.path.splitext(os.path.basename(structure_pdb))[0].upper()

    if verbose:
        print(f"Structure : {pdb_stru_id} ({len(sorted_keys)} residues, chain {chain_structure})")
        print(f"Target sequence : {pdb_seq_id} ({len(target_seq)} residues)")
        print("Constructing H matrix ...")

    # Double Dynamic Programming
    H = H_matrix(dist_matrix, ss_stru, seq_3_letter, dope_dict, verbose=verbose)
    F, path = F_matrix(H, ss_stru)
    energy_dope_global = float(F[0][0])

    if verbose:
        print(f"Global DOPE Energy (F[0][0]) = {energy_dope_global:.3f}")
        print("Computing RMSD ...")

    # RMSD Calculation
    rmsd = compute_rmsd(
        path, sorted_keys, CA_dict, target_pdb, target_chain=target_chain)

    # Output
    line_target, line_struct, line_ss = final_alignment(
        path, target_seq, res_stru, ss_stru)
    
    align_path = make_filename(pdb_seq_id, pdb_stru_id, "alignment", "txt", outdir)
    with open(align_path, "w") as f:
        f.write(f"# threading {pdb_seq_id} onto {pdb_stru_id}\n")
        f.write(f"# Global DOPE Energy = {energy_dope_global:.3f}\n")
        f.write(f"# RMSD = {rmsd:.3f} Angstroms\n")
        f.write(line_target + "\n")
        f.write(line_struct + "\n")
        f.write(line_ss + "\n")

    end_time = time.time()
    exec_time = end_time - start_time

    # Centralized logging to CSV file
    log_file = os.path.join(outdir, "threading_results_log.csv")
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Date & Time", "Sequence id", "Structure id", "Global DOPE score", "RMSD", "Execution time (sec)"])
        
        current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([
            current_time_str,  
            pdb_seq_id, 
            pdb_stru_id, 
            f"{energy_dope_global:.3f}", 
            f"{rmsd:.3f}", 
            f"{exec_time:.2f}"
        ])

    results = {
        "Global_DOPE_Energy": energy_dope_global,
        "RMSD": rmsd,
        "alignment_file": align_path,
        "log_file": log_file,
        "path": path,
        "execution_time": exec_time
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Protein threading pipeline (Double Dynamic Programming).")
    
    # Need 3  positional arguments
    parser.add_argument("structure_pdb", help="Path to the structure, pdb file")
    parser.add_argument("sequence_fasta", help="Path to the sequence, fasta file")
    parser.add_argument("sequence_pdb", help="Path to the true structure of the sequence, pdb file")
    
    parser.add_argument("--dope", default="./data/dope.par", help="Path to DOPE file")
    parser.add_argument("--outdir", default="./results", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Print progression in the terminal")
    
    # Chain selection options
    parser.add_argument("--stru_chain", default="A", help="Chain to extract from the structure")
    parser.add_argument("--target_chain", default="A", help="Chain to extract from the true structure")
    parser.add_argument("--seq_id", nargs=2, help="pdb ID and chain for the sequence")
    
    args = parser.parse_args() # arguments written in the terminale

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # Convert list to tuple -> our dictionary keys
    user_fasta_key = tuple(args.seq_id)


    # Run threading pipeline
    dope_dict_global = read_dope(args.dope)
    results = run_threading(
        structure_pdb=args.structure_pdb,
        fasta_file=args.sequence_fasta,
        target_pdb=args.sequence_pdb,

        dope_dict=dope_dict_global,
        chain_structure=args.stru_chain,
        target_chain=args.target_chain,
        fasta_key=user_fasta_key,
        outdir=args.outdir,
        verbose=args.verbose
    )
    
    print(f"Threading completed !")
    print(f"Alignment saved : {results['alignment_file']}")
    print(f"Calculated RMSD : {results['RMSD']:.3f} Å")
    print(f"Log updated     : {results['log_file']}")
    print(f"Execution time  : {results['execution_time']:.2f} seconds")