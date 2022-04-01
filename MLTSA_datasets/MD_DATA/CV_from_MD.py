"""

This module will encompass the code for analyzing the molecular dynamics generated on the simulations as dcd files.
It also takes care of the CV generation for further feeding into the MLTSA pipeline.


"""

import numpy as np
import mdtraj as md
from itertools import combinations
from itertools import permutations
import re


class CVs(object):
    """

    This class is a container of the Collective Variables defined to calculate on mdtraj later.
    It contains different methods for CV definition to later on calculate for the MLTSA.

    """

    def __init__(self, top):
        """

        The definition of this class needs only the topology which will use later on to define the CVs needed.

        :param top: str Path to the relevant topology file that will be used (.pdb or .psf format, can also work on
            other topologies compatible with mdtraj)

        """

        #TODO Implement a way to use multiple topologies, for now only way to make it work is call the CV each time.

        self.topology_file = md.load(top)
        self.topology_path = top
        self.top = self.topology_file.top

        return

    def define_variables(self, CV_type, custom_selection_string=None, CV_indices=None):
        """

        This method defines the variables depending on the type of CV passed, labels such as 'all' for example, will
        calculate every interatomic distance between all atoms. Custom CVs can be passed in with CV_indices=list
        and CV_type = "custom_CVs" as pairs of atom indices to calculate distances on. A custom_selection_string
        atom selection using mdtraj's syntax can be passed with custom_selection_string= to select atoms to
        calculate all distances from by using CV_type="custom_selection" .


        :param CV_type: str Label to specify the CV definition, it can be "all" for all atoms, "Calpha_water" for ligand
            +water+Calpha atoms, "Calpha" for ligand+Calpha atoms, "all_closest_atoms" for all close atoms between
            residues, "all_closest_heavy_atoms" for all closest heavy inter-residue atoms, "bubble_ligand" for all
            distances between ligand and protein for a 6 Angstroms bubble around the ligand. "custom_CVs" for a
            selected set of CV_indices to be passed, and "custom_selection" to pass a custom_selection_string to use on mdtraj
            as an atom selection sytnax.
        :param custom_selection_string: str Atom selection from mdtraj's atom selection reference syntax which will
            select the atom indices and use them for CV definition.
        :param CV_indices: list, array CVs can be defined outside of this class and passed here as atom indices.

        :return:

        """

        self.CV_type = CV_type

        if self.CV_type == "all":
            # TODO Label yet to test if it works, please do consider double cheking that it works before using
            print("You selected the option 'all', this can be very computationally expensive")
            print("please, proceed with caution")
            relevant_atoms = list(self.top.atoms)
            CV_indices = []
            for a,b in combinations(relevant_atoms, 2):
                CV_indices.append([a.index, b.index])

            print(len(CV_indices), " CV features defined, is this the correct number?")

            self.CV_indices = CV_indices

        elif self.CV_type == "Calpha_water":
            # TODO Label yet to test if it works, please do consider double cheking that it works before using
            relevant_atoms = list(self.top.select("name == CA or water or resname LIG"))
            CV_indices = []
            for a,b in combinations(relevant_atoms, 2):
                CV_indices.append([a, b])

            print(len(CV_indices), " CV features defined, is this the correct number?")

            self.CV_indices = CV_indices

        elif self.CV_type == "Calpha":
            # TODO Label yet to test if it works, please do consider double cheking that it works before using
            relevant_atoms = list(self.top.select("name == CA or resname LIG"))
            CV_indices = []
            for a,b in combinations(relevant_atoms, 2):
                CV_indices.append([a, b])

            print(len(CV_indices), " CV features defined, is this the correct number?")

            self.CV_indices = CV_indices

        elif self.CV_type == "all_closest_atoms":
            # TODO Label yet to test if it works, please do consider double cheking that it works before using
            relevant_residues = list(self.top.residues)
            CV_indices = []
            for a, b in combinations(relevant_residues, 2):
                CV_indices.append([a.index, b.index])
            dist, CV_indices = md.compute_contacts(self.topology_file,
                                                   contacts=relevant_residues,
                                                   scheme="closest")
            self.CV_indices = CV_indices

        elif self.CV_type == "all_closest_heavy_atoms":
            # TODO Label yet to test if it works, please do consider double cheking that it works before using
            relevant_residues = list(self.top.residues)
            CV_indices = []
            for a, b in combinations(relevant_residues, 2):
                CV_indices.append([a.index, b.index])
            dist, CV_indices = md.compute_contacts(self.topology_file,
                                                   contacts=relevant_residues,
                                                   scheme="closest-heavy")
            self.CV_indices = CV_indices

        elif self.CV_type == "custom_selection":
            # TODO Label yet to test if it works, please do consider double cheking that it works before using


            """" For now if you wish to find all interatomic between multiple selections, then put all selections under 
            one string using 'or' to keep adding selections, otherwise if you want to define pairs manually give them as 
            lists of [[atom1,atom2], [atom3.atom4]] """

            if isinstance(custom_selection_string, list):
                CV_indices = []
                for pair in custom_selection_string:
                    idx1 = self.top.select(pair[0])[0]
                    idx2 = self.top.select(pair[1])[0]
                    CV_indices.append([idx1, idx2])

                self.CV_indices = CV_indices

            else:
                relevant_atoms = list(self.top.select(custom_selection_string))
                CV_indices = []
                for a, b in combinations(relevant_atoms, 2):
                    CV_indices.append([a, b])
                self.CV_indices = CV_indices

        elif self.CV_type == "bubble_ligand":
            # Checked label, works fine
            relevant_atoms = list(self.top.select("resname LIG"))
            #Cutoff distance is in Nanometers as mdtraj expects, 0.6nm == 6 Angstroms
            close_atoms = list(md.compute_neighbors(self.topology_file, 0.6, relevant_atoms)[0])
            all_combinations = [[i, j] for i in relevant_atoms for j in close_atoms]
            self.CV_indices = all_combinations

        elif self.CV_type == "custom_CVs":
            # TODO Label yet to test if it works, please do consider double checking that it works before using

            self.CV_indices = CV_indices

        return

class MDs(object):

    """

    Analyzer wrapper based on mdtraj, that can generate distances out of a previously defined CVs object with
    calculate_CVs(). It can also make use of a list of dcd files and topology along with a set of selection strings
    and upper/lower values to check for an automatic labeling of simulations with label_simulations().

    """


    def __init__(self):

        """

        It does not need anything to initialize.

        """

        print("Setting up MD analyzer")


    def calculate_CVs(self, CVs, dcd_paths, loading="normal", iter_chunk=None):
        """

        Method for calculating the Collective Variables previously defined by passing on a CVs object along the
        list of trajectories to use and calculate the data. It has different methods for loading depending on the
        complexity of the dataset to analyze.

        :param CVs: class CVs object class previously defined with a set of CVs already defined. It will be used to
            calculate the distances.
        :param dcd_paths: list List of strings containing the paths to the different .dcd/trajectory files.
        :param loading: str Label for the type of trajectory loading to use, it can affect the performance.
        :return:

        """
        self.topology = CVs.topology_path
        self.dcd_list = dcd_paths
        self.type = dcd_paths[0][-3:]

        if loading == "normal":
            distances = []
            for dcd in dcd_paths:
                traj = md.load(dcd, top=self.topology, stride=1)
                dists = md.compute_distances(traj, CVs.CV_indices)
                distances.append(dists)

            return distances

        if loading == "optimized":
            print()


        if loading == "iterload":
            distances = []
            for dcd in dcd_paths:
                gen = md.iterload(dcd, top=self.topology, chunk=int(iter_chunk))
                dists = md.compute_distances(next(gen), CVs.CV_indices)
                distances.append(dists)

            return distances


    def label_simulations(self, top, dcd_paths, selection_strings_to_label, upper_lim, lower_lim,
                          loading="normal", end_perc=0.25, get_sum=True, plotting_sum=False,
                          plotting_all=False, show_plots=False, save_labels=False, save_plots=False, save_path=""):
        """

        Method for the labeling of a given set of trajectory files on the desired string selections to check and the
        upper/lower limit to classify. It can also plot figures with the values for each of the distances throughout
        the trajectories and save them in the specified path.

        :param top: str Path to the topology file to use (.pdb/.psf) or any mdtraj compatible topology file.
        :param dcd_paths: list List containing the paths to the trajectory files (.dcd/other)
        :param selection_strings_to_label: str String selection using mdtraj's atom selection reference syntax.
        :param upper_lim: float Upper limit which sets the OUT label for the trajectories when labeled. Anything bigger
            than this will be considered as OUT. Anything smaller than this and bigger than lower_lim will be labeled as
            UCL.
        :param lower_lim: float Lower limit which sets the IN label for the trajectories when labeled. Anything smaller
            than this will be considered as IN. Anything biggerr than this and smaller than upper_lim will be labeled as
            UCL.
        :param loading: str Label to specify the loading procedure, affects performance.
        :param plotting: boolean Determines whether to plot in matplotlib the evolution of the labeling distances
            throughout the trajectories. Figures will be saved in the given save_path, one per simulation.
        :param show_plots: boolean Whether to show the plots in the current session or not. If this is False and
            plotting=True and save_plots=True it will still save them without showing them.
        :param save_labels: boolean Determines if the labels should be saved in a file on the desired destination with
            save_path.
        :param save_plots: boolean Determines whether to save the plots or not.Figures will be saved in the given
            save_path, one per simulation.
        :param save_path: str Path to save the figures generated by the labelling if plotting=True. If not specified it
            saves in the working directory.

        :return: list Returns the list of labelled simulations as ["IN", "OUT", etc.] for each trajectory passed in the
            dcd_paths list.
        """
        #Prepare everything to calculate the data needed.

        if isinstance(top, list):
            clf_distances = []
            for t, dcd in zip(top, dcd_paths):
                topo = md.load_pdb(str(t))
                pairs = []
                for sel1, sel2 in selection_strings_to_label:
                    idx1 = topo.topology.select(sel1)
                    idx2 = topo.topology.select(sel2)
                    pairs.append([idx1[0], idx2[0]])
                vars = CVs(t)
                CV_indices = np.array(pairs)
                vars.define_variables(CV_type="custom_CVs", CV_indices=CV_indices)
                dists = self.calculate_CVs(vars, [dcd])
                clf_distances.append(dists[0])
        else:
            topo = md.load(top)
            pairs = []
            for sel1, sel2 in selection_strings_to_label:
                idx1 = topo.topology.select(sel1)
                idx2 = topo.topology.select(sel2)
                pairs.append([idx1[0], idx2[0]])
            vars = CVs(top)
            CV_indices = np.array(pairs)
            vars.define_variables(CV_type="custom_CVs", CV_indices=CV_indices)
            clf_distances = self.calculate_CVs(vars, dcd_paths)


        if plotting_all == True or plotting_sum == True:
            print("Plotting values, make sure you want to do this to either show them or save them somewhere.")
            import matplotlib.pyplot as plt


        md_labels = []
        sums_traj = []
        for n, traj in enumerate(clf_distances):
            sums = []
            sum_plot = []
            if plotting_all == True:
                plt.figure()
            for d in range(len(selection_strings_to_label)):
                values = np.array(traj).T[d]
                if plotting_all == True:
                    filename = re.split('/', dcd_paths[n])[-1]
                    plt.title("Sim: "+filename)
                    plt.plot(values,
                             label="CV {}: {} - {}".format(d, selection_strings_to_label[d][0],
                                selection_strings_to_label[d][1] ))
                    plt.xlabel("Frame")
                    plt.ylabel("Distance(A)")
                    plt.legend()
                    if show_plots == True:
                        plt.show()
                    if save_plots == True:
                        plt.savefig(str(save_path)+filename+"CV"+str(d)+"_label.svg")
                    plt.close()
                sum_plot.append(values)

                #This fetches the mean value found on the "end_perc" of the trajectory
                sums.append(np.mean(values[int(len(values)*(1-end_perc)):]))

            if plotting_sum == True:
                filename = re.split('/', dcd_paths[n])[-1]
                plt.figure()
                plt.title("Sum of Distances Trajectory "+filename)
                plt.plot(np.sum(sum_plot, axis=0), label="Sum of distances")
                label_data = np.sum(sum_plot, axis=0)
                label_data[0:int(len(label_data)*(1-end_perc))] = np.NaN
                plt.plot(label_data, label="Range used for labelling", linewidth=2)
                plt.xlabel("Frame")
                plt.ylabel("Distance(A)")
                plt.legend()
                if show_plots == True:
                    plt.show()
                if save_plots == True:
                    plt.savefig(str(save_path) + filename + "_sum_label.svg")
                plt.close()


            #This evaluates the sum of the means of each value to classify
            sums = np.sum(sums)
            sums_traj.append(sums)
            if sums < lower_lim:
                md_labels.append("IN")
            elif sums > upper_lim:
                md_labels.append("OUT")
            elif lower_lim < sums < upper_lim:
                md_labels.append("UCL")

        if save_labels == True:
            print("Saving labels at ", str(save_path)+"labels.dat")
            with open(str(save_path)+"labels.dat", "w") as f:
                for d, label in zip(dcd_paths, md_labels):
                    f.write(str(d)+"\t"+str(label)+"\n")
            f.close()

        if get_sum == True:
            print("Returning labels and the sum of values for each trajectory")
            return md_labels, sums_traj
        else:
            return md_labels


if __name__ == '__main__':
    print()