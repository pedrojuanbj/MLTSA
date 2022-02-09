import numpy as np
import matplotlib.pyplot as plt
import CV_from_MD as cvs


dcd_path = ["test.dcd", "test.dcd"]
top_path = "string_cropped.pdb"

""" Testing the creation of CVs """
CV_system = cvs.CVs(top_path)
CV_system.define_variables("custom_selection", custom_selection_string=[["index 3824", "index 1"], ["index 4251", "index 6"]])
#CV_system.define_variables("bubble_ligand")
print(CV_system.CV_indices)

analyzer = cvs.MDs()
CVs_from_bubble = analyzer.calculate_CVs(CV_system, dcd_path)

dists = np.array(CVs_from_bubble[0]).T

plt.plot(dists[0])
plt.show()

""" Testing the labelling system """
#Multiple topologies working
top_path = ["string_cropped.pdb", "string_cropped.pdb"]

analyzer = cvs.MDs()
labels = analyzer.label_simulations(top_path, dcd_path, [["index 3824", "index 1"], ["index 4251", "index 6"]], 9, 9,
                                    plotting_all=True, plotting_sum=True, show_plots=True)

print(labels)