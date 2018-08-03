# CMPT353 Project

## Data Collection

The walking data is in the "CMPT353/Data" folder and contains Linear Acceleration & Angular Velocity data in csv form. The data was collected using an Android app called PhysicsToolbox.

## Instructions:

Required libraries:

      pip install scipy matplotlib pandas numpy sklearn statsmodels

Simply clone the repository and run the following command:
      
       python main.py 
       
The file analysis2.py is run with no arguments. However, it must be in the same folder as Data, and the datafiles in the Data folders must not be renamed or moved to different subfolders.
       
## Output      

In this project we compute the Fourier Transform of the walking data and plot the frequency of the walking pattern. We also create a Machine Learning model to predict the height, level of activity, and gender of an individual based on their walking pattern.
