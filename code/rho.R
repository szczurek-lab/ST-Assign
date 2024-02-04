INPUT_DATA <- "path to input"

ST <- read.csv( paste0(INPUT_DATA, "C_gs.csv"), row.names = 1)
C_marker_genes <- read.csv(paste0(INPUT_DATA,"C_gc.csv"), row.names = 1)
marker_genes <- read.csv(paste0(INPUT_DATA,"matB.csv"))[,1]

n_cells <- read.csv(paste0(INPUT_DATA,"n_cells.csv"))[,"no.of.nuclei"]
n_cells[n_cells==0] <- mean(n_cells[n_cells!=0])

rho <- function(gen){
  temp1 <- unlist(C_marker_genes[gen, ])
  temp2 <- unlist(ST[gen,])
  temp2 <- temp2[which(temp2>0)]
  n_cells2 <- n_cells[which(temp2>0)]
  temp1 <- temp1[temp1>0]

  mean(temp1)/ (  mean(temp2/n_cells2)    )

}

res_rho <- sapply(marker_genes, rho)
write.table(unlist(res_rho), paste0(INPUT_DATA, "rho.csv"),  row.names = TRUE,sep=",")


rho_0 <- function(){
  temp1 <- unlist(C_marker_genes)
  temp2 <- unlist(ST)
  
  mean(temp1)/ (  mean(temp2/n_cells)    )
}
rho_0() # prior rho_0

