import numpy as np
import matplotlib.pyplot as plt
import CV_from_MD as cvs


#dcd_path = ["string_cropped.dcd"]
dcd_path = ["test.dcd"]
top_path = "string_cropped.pdb"


""" Testing the creation of CVs """
CV_system = cvs.CVs(top_path)
CV_system.define_variables("bubble_ligand")

analyzer = cvs.MDs()
CVs_from_bubble = analyzer.calculate_CVs(CV_system, dcd_path)

dists = np.array(CVs_from_bubble[0]).T

plt.plot(dists[0])
plt.show()

""" Testing the labelling system """

analyzer = cvs.MDs()
labels = analyzer.label_simulations(top_path, dcd_path, [["index 3824", "index 1"], ["index 4251", "index 6"]], 10, 5, plotting=True)

print(labels)

#print(variables)