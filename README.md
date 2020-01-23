# Segmentation-of-Lungs
3D Segmentation of Lungs on CT: tools to aid Radiotherapy Planning

In this project, "Segmentation of lungs on CT: tools to aid Radiotherapy planning", I studied two very different automatic approaches for the segmentation of the lungs: a "traditional" one, a Region Growing method, and a more recent one, the U-Net CNN model (a Deep Learning model), achieving results of 98% of DICE when compared to the true segmentation. The analysed images corresponded to CT scans (DICOM images) from children and adults. Both the algorithms were programmed in Python, using Keras and Tensorflow for the second model. The work resulted in a publication in the IbPRIA 2019: 9th Iberian Conference on Pattern Recognition and Image Analysis and published in the Springer Lecture Notes in Computer Science Series.


Abstract:

Radiotherapy planning plays a decisive role in cancer’s treatment management. Many side effects come from the exposure of normal tissue to radiation during therapy, all the way from small acute side effects such as tiredness, to long term sequelae, like another type of cancer. To minimize this exposure, it is necessary to contour the organs at risk. However, this task is typically performed manually on a slice-by-slice basis, being consequently very time-consuming and susceptible to high intra and inter-subject variance and human errors.

In this way, this line of work aims to help the clinicians in this difficult and repetitive task by implementing algorithms that delineate non-pathological lungs. In this project, two lung segmentation algorithms are presented for Computed Tomography scans: the Iterative Region Growing algorithm and a U-Net Convolutional Neural Network model. One relies on image processing techniques, as intensity projection and region growing. This pipeline starts by isolating each lung. Then, three techniques for seed placement are explored. Lastly, an update on the traditional region growing algorithm is developed, allowing it to automatically discover the best threshold parameter value for each case. The other algorithm is a U-Net deep learning architecture model, that takes advantage of the distinctive ability of Convolutional Neural Networks to find hidden patterns present in the lungs without requiring feature extraction and selection.

The results obtained for the three different techniques for seed placement were, respectively, 74%, 74% and 92% of DICE with the Iterative Region Growing algorithm. The results for the U-Net model were 91% for the same metric.

Future work includes more tests on bigger and more diverse databases, analyzing the effect of morphology operations on the results and the effect of the hyperparameter optimization techniques on the network. 

Key-words: Automatic Segmentation, Radiotherapy Planning, Organs at Risk, Lungs, 3D, Computerized Tomography.


For the 3D U-Net CNN model, the code of David G. Ellis (git id Ellisdg) was adapted (available at https://github.com/ellisdg/3DUnetCNN).
