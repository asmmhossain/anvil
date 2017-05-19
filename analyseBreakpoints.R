library(ggplot2)
library(reshape2)

# get the name of the probability file

args <- commandArgs(trailingOnly = TRUE)

fName <- args[1]
pName <- args[2]
qid <- args[3]

probs <- read.delim(fName,check.names=FALSE)

mProbs <- melt(probs,id="Positions")


p <- ggplot(data=mProbs, aes(x=Positions, y=value, group=variable, colour=variable)) +
  geom_line() + xlab('Nucleotide positions') + ylab('Subtype probability') + 
  ylim(0,1) + labs(colour= 'Subtypes') + ggtitle(qid) +
  theme(plot.title = element_text(hjust = 0.5))


png(pName)
print(p)

dev.off()

