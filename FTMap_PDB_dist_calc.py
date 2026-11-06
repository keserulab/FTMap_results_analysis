# This script calculates the center of mass of each FTMap probe clusters, then calculates the distances compared to a reference input (ref_name variable).
# To use, just set the folder_path variable at the bottom to a folder containing FTMap (Atlas) result .pdb files (the ones containing the clusters and the protein structure).

from Bio.PDB import PDBParser
import numpy as np
import os
import math
import csv

# Define atomic masses (simplified version, can be extended as needed)
ATOMIC_MASSES = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "P": 30.974, "S": 32.06,
    # Add more elements as needed
}

ref_name = "reference.pdb" #change to actual file name of the .pdb file containing reference probe clusters

def calculate_center_of_mass(atoms):
    """Calculate the center of mass of a group of atoms."""
    total_mass = 0
    weighted_sum = np.zeros(3)
    
    for atom in atoms:
        element = atom["element"]
        mass = ATOMIC_MASSES.get(element, 0)  # Default mass 0 for unknown elements
        total_mass += mass
        weighted_sum += mass * atom["coord"]

    if total_mass == 0:
        return np.zeros(3)

    return weighted_sum / total_mass

def extract_ligands_and_calculate_com(pdb_file):
    """Extract ligands from PDB file and calculate their centers of mass."""
    ligand_atoms = {}
    current_ligand = None

    with open(pdb_file, "r") as file:
        for line in file:
            if line.startswith("HEADER"):
                current_ligand = line[10:].strip()
                if current_ligand.lower() != "protein":  # Skip protein entries
                    ligand_atoms[current_ligand] = []
                else:
                    current_ligand = None

            elif line.startswith("HETATM") and current_ligand:
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    element = line[76:78].strip()

                    # Fall back to element symbol from atom name if necessary
                    if not element or element.isnumeric():
                        element = line[12:14].strip()[0]

                    ligand_atoms[current_ligand].append({"coord": np.array([x, y, z]), "element": element})
                except ValueError:
                    continue  # Skip malformed lines

    ligand_coms = {}
    for ligand, atoms in ligand_atoms.items():
        if atoms:  # Only calculate for non-empty atom lists
            com = calculate_center_of_mass(atoms)
            ligand_coms[ligand] = com

    return ligand_coms

def calculate_distances(reference_coms, target_coms):
    """Calculate distances between reference COMs and target COMs."""
    distances = {}
    for ref_ligand, ref_com in reference_coms.items():
        distances[ref_ligand] = {}
        for target_ligand, target_com in target_coms.items():
            distance = np.linalg.norm(ref_com - target_com)
            distances[ref_ligand][target_ligand] = distance
    return distances

def process_folder(folder_path):
    """Process all PDB files in a folder, calculate COMs and distances."""
    pdb_files = [f for f in os.listdir(folder_path) if f.endswith(".pdb")]
    if ref_name not in pdb_files:
        raise FileNotFoundError("reference.pdb file is missing in the folder.")

    reference_file = os.path.join(folder_path, ref_name)
    reference_coms = extract_ligands_and_calculate_com(reference_file)

    all_distances = []

    for pdb_file in pdb_files:
        if pdb_file == ref_name:
            continue

        pdb_path = os.path.join(folder_path, pdb_file)
        target_coms = extract_ligands_and_calculate_com(pdb_path)
        distances = calculate_distances(reference_coms, target_coms)

        for ref_ligand, target_distances in distances.items():
            for target_ligand, distance in target_distances.items():
                all_distances.append([pdb_file, ref_ligand, target_ligand, distance])

    output_file = os.path.join(folder_path, "homo_distances.csv")
    with open(output_file, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["PDB File", "Reference Ligand", "Target Ligand", "Distance"])
        csvwriter.writerows(all_distances)

    print(f"Distances have been written to {output_file}")

# Main execution
if __name__ == "__main__":
    folder_path = "/path/to/folder"  #Input folder containing FTMap (Atlas) output files
    process_folder(folder_path)
