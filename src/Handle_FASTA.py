"""
Parses FASTA sequence files (headers such as ">1cpc_1|Chains A, C|description")
into a dictionary so that a specific chain sequence can be retrieved.
"""

def extract_sequences(fasta_file):
    """
    Read a FASTA file and return a dictionary of sequences indexed
    by (pdb_id, chain_id).

    Input:
        fasta_file (str): Path to the FASTA file.

    Output:
        seq_dict (dictionnary): 
            Keys are tuples (pdb_id, chain_id), both str. 
            Values are the corresponding sequence.
            If a header lists several chains, the same
            sequence is stored under each chain key.
    """

    seq_dict = {}
    current_keys = []
    current_seq = ""
    
    with open(fasta_file, "r") as file:
        for line in file:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            if line.startswith(">"):
                # Store the previous sequence in the dictionary before moving on
                if current_keys and current_seq:
                    for key in current_keys:
                        seq_dict[key] = current_seq
                        
                # Parse the new header to create the new keys
                parts = line.split('|')
                if len(parts) >= 2:
                    # Extract PDB ID (eg : ">1cpc_1" -> "1CPC")
                    pdb_id = parts[0].split('_')[0][1:].upper()
                    
                    # Clean the chains part (eg : "chains a, c[auth k]" -> " a  c")
                    chains_str = parts[1].split('[')[0] 
                    chains_str = chains_str.lower().replace("chains", "").replace("chain", "").replace(",", " ")
                    
                    # Initialize the list of keys for this header
                    current_keys = [(pdb_id, chain.upper()) for chain in chains_str.split()] # eg : [(1CPC, A), (1CPC, C)]
                else:
                    current_keys = [] # Security if the header doesn't contain "|"
                    
                # Reset the sequence for the new header 
                current_seq = ""
                
            else:
                # Build the sequence line by line
                if current_keys:
                    current_seq += line
                    
        # Store the very last sequence of the file
        if current_keys and current_seq:
            for key in current_keys:
                seq_dict[key] = current_seq
                
    return seq_dict

if __name__ == "__main__":
    # seq_dict_1CPC = extract_sequences("./data/sequences/rcsb_pdb_1CPC.fasta")
    # print(seq_dict_1CPC)
    # print()
    # seq_dict_1MBA = extract_sequences("./data/sequences/rcsb_pdb_1MBA.fasta")
    # print(seq_dict_1MBA)
    # print(seq_dict_1CPC[("1CPC", "A")])
    pass