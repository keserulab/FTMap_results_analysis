# This script groups FTMap clusters into a single groups (cluster) based on proximity.
# To use, set the sdf_folder variable at the bottom to a folder containing FTMap (Atlas) result .sdf files (the ones containing the clusters)

import os
import glob
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import DBSCAN

def get_ligand_centroid(mol):
    """Compute the centroid of a ligand based on atomic coordinates."""
    conformer = mol.GetConformer()
    positions = np.array([list(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    return np.mean(positions, axis=0)

def load_sdf_files(sdf_folder):
    """Load molecules from .sdf files and store their centroids."""
    sdf_files = glob.glob(os.path.join(sdf_folder, "*.sdf"))
    ligands = []
    file_map = []
    
    for sdf_file in sdf_files:
        suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
        for mol in suppl:
            if mol is not None:
                centroid = get_ligand_centroid(mol)
                ligands.append((mol, centroid))
                file_map.append(sdf_file)  # Keep track of original files
    
    return ligands, file_map, sdf_files

def cluster_ligands(ligands, eps=2.0):
    """Cluster ligands based on their centroid distances using DBSCAN."""
    centroids = np.array([lig[1] for lig in ligands])
    clustering = DBSCAN(eps=eps, min_samples=1).fit(centroids)
    return clustering.labels_

def merge_sdfs_by_cluster(ligands, labels, sdf_files):
    """Merge ligands into cluster-based .sdf files."""
    clustered_mols = {}
    for (mol, _), cluster_id in zip(ligands, labels):
        if cluster_id not in clustered_mols:
            clustered_mols[cluster_id] = []
        clustered_mols[cluster_id].append(mol)
    
    # Write merged .sdf files
    output_folder = "new_structures_merged_sdfs"
    os.makedirs(output_folder, exist_ok=True)
    for cluster_id, mols in clustered_mols.items():
        output_file = os.path.join(output_folder, f"cluster_{cluster_id}.sdf")
        writer = Chem.SDWriter(output_file)
        for mol in mols:
            writer.write(mol)
        writer.close()
        print(f"Saved merged cluster {cluster_id} -> {output_file}")

if __name__ == "__main__":
    sdf_folder = "/path/to/folder"  # Change this to your directory containing .sdf files
    ligands, file_map, sdf_files = load_sdf_files(sdf_folder)
    labels = cluster_ligands(ligands, eps=2.0)  # Adjust `eps` based on proximity threshold
    merge_sdfs_by_cluster(ligands, labels, sdf_files)
