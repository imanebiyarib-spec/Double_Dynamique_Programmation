"""

Handles the DOPE (Discrete Optimized Protein Energy) statistical
pseudo score file. 
Read the DOPE.par file and save its data into a dictionnary, 
where you can get the pseudo energy of a pair of residues by
giving the distance (Angstrom) between their CA.  

"""

def read_dope(dope_file):

    """
    Read a DOPE.par file and build dictionary of
    CA-CA pairwise pseudo-energies

    Inputs:
        dope_file (str): Path to the DOPE file. 
        Each line is expected to be whitespace separated with the
        format "RES1 ATOM1 RES2 ATOM2 e0 e1...e29". 
        RES1/RES2 are 3 letter amino acid codes.
        ATOM1/ATOM2 are atom names, only lines where ATOM1 == ATOM2 == "CA" are kept.
        e0...e29 are statistic pseudo energy (one floats per 0.5 A distance bin,
        from 0-0.5A to 14.5-15A). 

    Outputs:
        dope_dict (dictionnary): Keys are tuples (res1, res2) of 3 letter amino acid
            codes (str). Values are lists of 30 pseudo-energy (floats) for each 0.5 A distance bin.
            Both (res1, res2) and (res2, res1) are stored and point to
            the same list, since the score is symmetric.
    """

    dope_dict = {}
    
    with open(dope_file, 'r') as file:
        for line in file:
            bins = line.strip().split()
            
            res1, atom1, res2, atom2 = bins[0], bins[1], bins[2], bins[3]
            
            # Filter to only retain lines with res1 CA res2 CA
            if atom1 == 'CA' and atom2 == 'CA':
                dope_score = [float(x) for x in bins[4:34]]
                
                dope_dict[(res1, res2)] = dope_score 
                dope_dict[(res2, res1)] = dope_score # res1-res2 = res2-res1
                
    return dope_dict

def get_dope_score(dope_dict, res1, res2, distance):
    """
    Get the DOPE pseudo energy for a pair of residues at a
    given CA-CA distance.

    Input:
        dope_dict : Dictionary returned by read_dope().
        res1 (str): 3-letter amino acid code of the first residue.
        res2 (str): 3-letter amino acid code of the second residue.
        distance (float): CA-CA distance in Angstrom. Must be >= 0.

    Output:
        float: The pseudo-energy for this residue pair at this
            distance. Returns 0.0 if distance >= 15.0 A 
        If distance is negative, it raises a ValueError
    """

    if distance < 0 : 
        raise ValueError(f"The distance computed is negative ({distance}), please check your work")
    
    # if the distance is higher than 15A, we consider there is no interaction -> null score 
    if distance >= 15.0:
        return 0.0
        
    # Compute the column index by divading the distance by 0.5 (eg : 2.3/0.5 = 4.6 -> index 4)
    index = int(distance / 0.5)

    return dope_dict[(res1, res2)][index]


if __name__ == "__main__" :
    #dope_dict = read_dope("dope.par")
    # print(len(dope_dict))
    # print(dope_dict[("ALA", "ALA")])
    # print(len(dope_dict[("ALA", "ALA")]))
    # print(dope_dict[("ALA", "ALA")][29])
    # print(dope_dict)
    #dope_ALA_ALA_07 =  get_dope_score(dope_dict, "ALA", "ALA", 12)
    #print(dope_ALA_ALA_07)
    pass
        

