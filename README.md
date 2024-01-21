# ST-Assign

This directory contains the source code necessary to reproduce analyses presented in the manuscript:  

A. Geras, K. Domżał, E. Szczurek, **Joint cell type identification in spatial transcriptomics and single-cell RNA sequencing data**, in review


## Running ST-Assign

ST-Assign can be run from the `code` directory as follows:

```
bash  STAssign-run.sh 'input_data' 'results' 
```

## Expected input

ST-Assign takes as input the following files provided in the `input_data` directory:

* `param.txt` - file containing run setting (hyperparameters, number of iterations, etc.),
* `matB.csv` - binary matrix with prior knowledge about marker genes,
* `C_gs.csv` - ST gene expression data,
* `C_gs.csv` - ST gene expression data,
* `n_cells.csv` - estimates for the number of cells in each ST spot,
* `rho.csv` - XXX.

## Expected output

ST-Assing outups:
* XX,
* XX.

ST-Assign was devoloped under:
pakiety
