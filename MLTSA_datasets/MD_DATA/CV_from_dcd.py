"""

This module will encompass the code for analyzing the molecular dynamics generated on the simulations as dcd files.
It also takes care of the CV generation for further feeding into the MLTSA pipeline.


"""

import numpy as np
import mdtraj as md
from itertools import combinations


class CVs(object):

    def __init__(self, top, CV_type, custom_selection_string=None, CV_indices=None):

        self.CV_type = CV_type
        self.topology_file = md.load(top)
        self.top = self.topology_file.top


        if self.CV_type == "all":
            print("You selected the option 'all', this can be very computationally expensive")
            print("please, proceed with caution")
            relevant_atoms = list(self.top.atoms)
            CV_indices = []
            for a,b in combinations(relevant_atoms, 2):
                CV_indices.append([a.index, b.index])

            print(len(CV_indices), " CV features defined, is this the correct number?")

            self.CV_indices = CV_indices

        elif self.CV_type == "Calpha_water":
            relevant_atoms = list(self.top.select("name == CA or water or resname LIG"))
            CV_indices = []
            for a,b in combinations(relevant_atoms, 2):
                CV_indices.append([a, b])

            print(len(CV_indices), " CV features defined, is this the correct number?")

            self.CV_indices = CV_indices

        elif self.CV_type == "Calpha":
            relevant_atoms = list(self.top.select("name == CA or resname LIG"))
            CV_indices = []
            for a,b in combinations(relevant_atoms, 2):
                CV_indices.append([a, b])

            print(len(CV_indices), " CV features defined, is this the correct number?")

            self.CV_indices = CV_indices

        elif self.CV_type == "all_closest_atoms":
            relevant_residues = list(self.top.residues)
            CV_indices = []
            for a, b in combinations(relevant_residues, 2):
                CV_indices.append([a.index, b.index])
            dist, CV_indices = md.compute_contacts(self.topology_file,
                                                   contacts=relevant_residues,
                                                   scheme="closest")
            self.CV_indices = CV_indices

        elif self.CV_type == "all_closest_heavy_atoms":
            relevant_residues = list(self.top.residues)
            CV_indices = []
            for a, b in combinations(relevant_residues, 2):
                CV_indices.append([a.index, b.index])
            dist, CV_indices = md.compute_contacts(self.topology_file,
                                                   contacts=relevant_residues,
                                                   scheme="closest-heavy")
            self.CV_indices = CV_indices

        elif self.CV_type == "custom_selection":
            relevant_atoms = list(self.top.select(custom_selection_string))
            CV_indices = []
            for a,b in combinations(relevant_atoms, 2):
                CV_indices.append([a, b])
            self.CV_indices = CV_indices

        elif self.CV_type == "custom_CVs":

            self.CV_indices = CV_indices


class DCD_MDs(object):

    def __init__(self, topology_path, dcd_paths):

        self.topology = topology_path
        self.dcd_list = dcd_paths
        self.type = "dcd"
        self.length =

    def parser( topology, )

