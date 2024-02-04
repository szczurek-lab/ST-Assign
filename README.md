# ST-Assign

This directory contains the source code necessary to reproduce analyses presented in the manuscript:  

A. Geras, K. Domżał, E. Szczurek, **Joint cell type identification in spatial transcriptomics and single-cell RNA sequencing data**, in review.


## Running ST-Assign

ST-Assign can be run from the `code` directory as follows:

```
bash  ST-Assign-run.sh 'input_data' 'results' 
```
The arguments of the bash script:
* `input_data` - directory containing input data,
* `results` - directory dedicated to results.

### Expected input

ST-Assign takes as input the following files provided in the `input_data` directory:

* `param.txt` - file containing run setting (hyperparameters, number of iterations, etc.),
* `matB.csv` - binary matrix with prior knowledge about marker genes,
* `C_gs.csv` - ST gene expression data,
* `C_gs.csv` - scRNA-seq gene expression data,
* `n_cells.csv` - estimates for the number of cells in each ST spot,
* `rho.csv` - cross-platform factor computed based on gene expression data.

Important: the order of genes in `C_gs.csv` and `C_gs.csv` should match the order of genes in `matB.csv`; the order of spots in the file  `C_gs.csv` should match the order in  `matB.csv` file.

For details, see the Manual in this repository.

### ST-Assing output

* `est_M.csv` - cell type mixture decomposition in ST spots (size: number of spots vs. number of cell types),
* `res_TC.csv` - cell type annotations of single cells (size: trajectories after burn in times number of cells).

For details about input and output, see the `Expected input and output` file in this repository.

### Test example
Directory `test-example` contains data exemplificatory data. After running ST-Assing on this data set, results can be visualised with the use of `Visualisation.rmd` file.

## Packages

ST-Assign was developed under:
* tensorflow version 2.3.0,
* tensorflow-probability version 0.21.0,
* numpy version 1.24.3,
* pandas version 1.5.2,
* scipy version 1.10.0.


## Repository content

* `data` - real and simulated data used do generate results from the paper,
* `code` - model's implementation,
* `example` - exemplary input files ready to run on ST-Assign,
* `manual` - details about input and output.

