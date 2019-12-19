# LTAT.02.002-Kepler-project
Train machine learning algorithms to detect transitions of exoplanets using light curve data provided by the Kepler Space telescope.

Using data from: https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data

Main project file: src/Project M9.ipynb
Poster file: M9_poster.pdf
Editable poster file: M9_poster.pptx
Initial project report: M9_report.pdf
Project readme: README.md

Prerequisites to run it: Python 3, Jupyter Notebook, matplotlib, seaborn, pandas, numpy, sklearn, tensorflow, keras.  
You also will need to have downloaded the data and unzipped the .csv files into the src/ directory.

Run it as any other notebook.  
All the imports for the neural network parts is in the very end, so you don't really need to have tensorflow and keras installed to make it through all the other models.

We will go through the file cell by cell and briefly explain what is going on.

1. Imports everything outside of tf and keras.
2. Loads in the data, fixes column names so they aren't so ugly and replaces labels with a standard 1-0 encoding. Then it separates the datasets into labels and data.
3. Adds the preprocessing function, which applies a Fourier transform to the data, smooths it out and rescales it.
4. Plots of the original data for a positive and negative case.
5. Calls processing function.
6. Plots of the processed data for a positive and negative case.
7. Several plots of the processed data for a positive case. They all look quite similar...
8. Same for the negative case. These are quite a lot more chaotic.
9. Defines evaluation function for sklearn models.
10. Makes and evaluates logistic regression model.
11. Makes and evaluates a linear SVM model.
12. Makes and evaluates a 9-Nearest Neighbours model.
13. Makes and evaluates a decision tree model.
14. Makes and evaluates a random forest model.
15. Makes and evaluates a gradient boost model.
16. Imports for using neural networks.
17. Changing the data to numpy arrays.
18. Adding another feature to satisfy convolution layer constraints.
19. Writing a generator for the data which helps balance input.
20. Builds CNN layer by layer.
21. Compiles the network.
22. Fits the network to the data.
23. Defines a simple evaluation function for ANNs.
24. Evaluates the network.
