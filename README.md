# Classification of Interpretation Differences in String Quartets Based on the Origin of Performers

## Introduction:
This repo provides supplementary information for the "Classification of Interpretation Differences in String Quartets Based on the Origin of Performers" paper.
It contains all information (except audio recordings) to reproduce experiments and the results. 

Paper: https://www.mdpi.com/2076-3417/13/6/3603

## Description:

Some of the functions are hand-crafted for specific data and their labels. To use them within your dataset, follow the same hierarchy. The recordings are divided into 'user' (composer), 'session' (composition), 'movement'. This unusual segmentation is due to the compatibility with a specific software, which is in the development right now.
In our case, audio files were stored in 'F:/\_KInG/\_king_database/user/composition/CDXXX/XXX\_NameOfQuartet\_movement' path. 

For example, if we use 'user' = Dvorak, 'session' = No.12, 'movement' = 1, we get:
F:/\_KInG/\_king\_database/Dvorak/No.12/CDXXX/XXX\_name\_of\_quartet\_1.wav for each quartet (up to 78 in this case); 
such as ['F:/\_KInG/\_king\_database/Dvorak/No.12/CD001/001\_Brodsky\_1.wav', 'F:/\_KInG/\_king\_database/Dvorak/No.12/CD002/002\_Duke\_1.wav', ...]

We analyzed all 4 movements (sonata form) of given string quartets for each composer:

Composers:

* Dvorak (No.12, No.13, No.14)
* Janacek (No.1, No.2)
* Smetana (No.1, No.2)

Other important parameters are 'labels' ('label\_CZ' denoting Czech or non-Czech origin and 'label\_random' with random distribution) for binary classification 
and 'scenarios' ('movements', 'motifs', and 'measures') for different resolutions of analysis. Remember that for the 'movements' scenario, only movement 4 will produce
classification results, as others are used for datamatrix computation.

For a better description and more details, see the [paper](https://www.mdpi.com/2076-3417/13/6/3603).


## Install

To install the dependencies, use:

```
python install -r requirements.txt
```
To load non-.wav audio files, install ffmpeg to your OS and add it to the SYS path. To output results to the .xlsx files using pandas, install openpyxl:

```
python install openpyxl
```


## Usage:

To run the analysis:

```
python run_analysis.py
```


To run the whole analysis with debug mode, use:

```
python run_analysis.py -d True
```

You can also run all the scripts separately.

## Note:

The code is not optimized or clean. It is a result of experimentation. The feedback and comments are appreciated! If you have any questions, feel free to open an issue or send a message to matej.istvanek@vut.cz.

## Acknowledgment:

This work was supported by the "Identification of the Czech origin of digital music recordings using machine learning" grant, which is realized within the project Quality Internal Grants of BUT (KInG BUT), Reg. No. CZ.02.2.69/0.0/0.0/19\_073/0016948 and financed from the OP RDE.


