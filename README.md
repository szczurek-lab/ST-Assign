# ST-Assign

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10685485.svg)](https://doi.org/10.5281/zenodo.10685485)


This directory contains the source code required to replicate analyses presented in the manuscript:

A. Geras, K. Domżał, E. Szczurek, **Joint cell type identification in spatial transcriptomics and single-cell RNA sequencing data**, in review.


## Running ST-Assign

ST-Assign can be run from the `code` directory as follows:

```
bash  ST-Assign-run.sh 'input_data' 'results' 
```
The arguments of the bash script:
* `input_data` - the directory containing input data,
* `results` - the directory dedicated to results.

### Expected input

ST-Assign takes as input the following files provided in the `input_data` directory:

* `param.txt` - a file containing run setting (hyperparameters, number of iterations, etc.),
* `matB.csv` - a binary matrix with prior knowledge about marker genes,
* `C_gs.csv` - ST gene expression data,
* `C_gs.csv` - scRNA-seq gene expression data,
* `n_cells.csv` - estimates for the number of cells in each ST spot,
* `rho.csv` - cross-platform factor computed based on gene expression data.

**Note**: The order of genes in C_gs.csv and C_gs.csv should match the order of genes in matB.csv. Moreover, the order of spots in the file C_gs.csv should match the order in the matB.csv file.

Refer to the `Expected Input and Output` file in this repository for more information about input formats.

### ST-Assing output

* `est_M.csv` - cell type mixture decomposition in ST spots (size: number of spots vs. number of cell types),
* `res_TC.csv` - cell type annotations of single cells (size: trajectories after burn in times number of cells).

Refer to the `Expected Input and Output` file in this repository for more information about input and output formats.

### Test example
The `test-example` directory contains exemplificatory data. After running ST-Assign on this dataset, results can be visualized using the `Visualisation.rmd` file.

## Packages

ST-Assign was developed under:
* python version 3.9.18
* tensorflow version 2.13.0,
* tensorflow-probability version 0.21.0,
* numpy version 1.24.3,
* pandas version 1.5.2,
* scipy version 1.10.0.



