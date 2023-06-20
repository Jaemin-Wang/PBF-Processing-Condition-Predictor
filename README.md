# PBF-Processing-Condition-Predictor
## Project Description

Our project introduces a user-friendly GUI program designed to optimize powder bed fusion (PBF) manufacturing outcomes through intelligent process condition identification. Leveraging the power of machine learning, this application conducts an inverse prediction based on a pre-trained model, accounting for the distinct material properties of the powder used in the process.

Our main goal is to assist in determining the process conditions that would yield PBF products with high relative density—a crucial aspect determining the quality and performance of the end product. The tool, therefore, is geared towards professionals and hobbyists in the additive manufacturing domain seeking to enhance the effectiveness and efficiency of their PBF processes.

Furthermore, for best printing results, the software provides an essential recommendation: users are advised to adopt a layer thickness surpassing the d50 size of the employed powder. This guidance aids in realizing a superior level of precision and detail in the produced objects, improving overall print quality.

Whether you're an experienced professional or an enthusiast in the PBF manufacturing domain, our project provides an innovative, easy-to-use solution to maximize the quality of your production processes.

## Prerequisites

Ensure you have Python installed on your machine. This project was developed using Python 3.8, but other versions should work as well. 

## Installation and Setup

First, clone the repository to your local machine:

```
git clone https://github.com/Jaemin-Wang/PBF-Processing-Condition-Predictor.git
cd PBF-Processing-Condition-Predictor
```

To run this project, you will need to install several dependencies. Here's a list of Python libraries that you need to install:

- csv
- io
- sys
- pandas
- numpy
- os
- PyQt5
- xgboost
- random
- math

You can install these packages using pip:

```
pip install pandas numpy PyQt5 xgboost
```

**Note**: Some of these libraries (like os, csv, io, sys, random, and math) are built-in modules in Python, so you don't need to install them.

## Usage 

To run the program, input material properties of powder and etc. (See "pbf_rd_pred manual.pptx" file)

Expected output: process parameters

Expected run time: it may take up to several days depending on the conditions.

## Contact Information 

Jaemin Wang - wjmpinkiepie@postech.ac.kr

Project Link: https://github.com/Jaemin-Wang/PBF-Processing-Condition-Predictor

Feel free to contact me if you have any questions!

