rm(list=ls())

QQ <- function(method,P,cp)
{
  observed <- sort(P)
  lobs <- -(log10(observed))

  expected <- c(1:length(observed))
  lexp <- -(log10(expected / (length(expected)+1)))

  if(cp)
  {
    fname <- paste(method,".QQ.jpeg",sep="")
    cat('Plotting with jpeg\n')
    jpeg(paste(fname,sep=""), quality = 20, bg = "white", res = 200, width = 7, height = 7, units = "in")
    plot(c(0,8), c(0,8), col="red", lwd=3, type="l", main=method,xlab="Expected (-logP)", ylab="Observed (-logP)", xlim=c(0,8), ylim=c(0,8), las=1, xaxs="i", yaxs="i", bty="l")
    points(lexp, lobs, pch=23, cex=.4, bg="black")
    dev.off()
  }
  else
  {
    fname <- paste(method,".QQ.pdf",sep="")
    cat('Plotting with pdf\n')
    pdf(paste(fname,sep=""), bg = "white", width = 7, height = 7)
    plot(c(0,8), c(0,8), col="red", lwd=3, type="l", main=method,xlab="Expected (-logP)", ylab="Observed (-logP)", xlim=c(0,8), ylim=c(0,8), las=1, xaxs="i", yaxs="i", bty="l")
    points(lexp, lobs, pch=23, cex=.4, bg="black")
    dev.off()
  }
}   

args=(commandArgs(TRUE))
if(length(args)==0)
{
  stop("No arguments")
}else
{
    for(i in 1:length(args))
    {
         eval(parse(text=args[[i]]))
    }
}
cat("prefix = ",prefix,"\n")
#cat("dir = ",dir,"\n")
cat("model = ",model,"\n")

#prefix <- "chr"
#dir <- "./data/"
#model <- "Summary"

GWiS <- NULL
Vegas <- NULL
BF <- NULL
MS <- NULL
MSG <- NULL
Gates <- NULL
SNP <- c()
for (chr in c(1:23))
{
   # Process GWiS
   #fname <- paste(dir,"/",prefix,chr,".GWiS.",model,".txt",sep="")
   fname <- paste(prefix,".chr",chr,".GWiS.",model,".txt",sep="")
   if(file.exists(fname)==TRUE)
   {
      cat('Processing GWiS file : ',fname,'\n')
      D <- read.table(fname,header=TRUE)
      X <- subset(D,D$SNP.id=="SUMMARY")
      X$SNP.id <- NULL 
      X$SNP.pos <- NULL 
      X$SNP.maf <- NULL 
      X$SNP.qual <- NULL
      X$SSM <- NULL
      X$R2 <- NULL
      X$F_stat <- as.numeric(as.character(X$F_stat))
      X$BIC <- as.numeric(as.character(X$BIC))
      if(is.null(GWiS))
      {
         GWiS <- X
      }
      else
      {
         GWiS <- rbind(GWiS,X)
      }
   } 
   # Process Vegas
   #fname <- paste(dir,"/",prefix,chr,".Vegas.",model,".txt",sep="")
   fname <- paste(prefix,".chr",chr,".Vegas.",model,".txt",sep="")
   if(file.exists(fname)==TRUE)
   {
      cat('Processing Vegas file : ',fname,'\n')
      D <- read.table(fname,header=TRUE)
      if(is.null(Vegas))
      {
         Vegas <- D
      }
      else
      {
         Vegas <- rbind(Vegas,D)
      }
   }
   # Process BF
   #fname <- paste(dir,"/",prefix,chr,".BF.",model,".txt",sep="")
   fname <- paste(prefix,".chr",chr,".BF.",model,".txt",sep="")
   if(file.exists(fname)==TRUE)
   {
      cat('Processing Bimbam file : ',fname,'\n')
      D <- read.table(fname,header=TRUE)
      if(is.null(BF))
      {
         BF <- D
      }
      else
      {
         BF <- rbind(BF,D)
      }
   }
   # Process Gates
   #fname <- paste(dir,"/",prefix,chr,".Gates.",model,".txt",sep="")
   fname <- paste(prefix,".chr",chr,".Gates.",model,".txt",sep="")
   if(file.exists(fname)==TRUE)
   {
      cat('Processing Gates file : ',fname,'\n')
      D <- read.table(fname,header=TRUE)
      if(is.null(Gates))
      {
         Gates <- D
      }
      else
      {
         Gates <- rbind(Gates,D)
      }
   }

   # Process minSNP
   #fname <- paste(dir,"/",prefix,chr,".minSNP.",model,".txt",sep="")
   fname <- paste(prefix,".chr",chr,".minSNP.",model,".txt",sep="")
   if(file.exists(fname)==TRUE)
   {
      cat('Processing minSNP file : ',fname,'\n')
      D <- read.table(fname,header=TRUE)
      X <- subset(D,D$IsBest==1)
      X$SNP.id <- NULL 
      X$SNP.pos <- NULL 
      X$SNP.maf <- NULL 
      X$SNP.qual <- NULL
      if(is.null(MS))
      {
         MS <- X
      }
      else
      {
         MS <- rbind(MS,X)
      }
   }
   # Process minSNP_Gene
   #fname <- paste(dir,"/",prefix,chr,".minSNP_Gene.",model,".txt",sep="")
   fname <- paste(prefix,".chr",chr,".minSNP_Gene.",model,".txt",sep="")
   if(file.exists(fname)==TRUE)
   {
      cat('Processing minSNP_Gene file : ',fname,'\n')
      D <- read.table(fname,header=TRUE)
      X <- subset(D,D$IsBest==1)
      X$SNP.id <- NULL 
      X$SNP.pos <- NULL 
      X$SNP.maf <- NULL 
      X$SNP.qual <- NULL
      if(is.null(MSG))
      {
         MSG <- X
      }
      else
      {
         MSG <- rbind(MSG,X)
      }
   }
   # Process single SNP
   #fname <- paste(dir,"/",prefix,chr,".allSNP.",model,".txt",sep="")
   fname <- paste(prefix,".chr",chr,".allSNP.",model,".txt",sep="")
   if(file.exists(fname)==TRUE)
   {
      cat('Processing allSNP file : ',fname,'\n')
      D <- read.table(fname,header=TRUE)
      D$Pval <- as.numeric(as.character(D$Pval))
      SNP <- c(SNP,D$Pval)
   }
}

if(is.null(GWiS)==FALSE)
{
  GWiS$F_stat <- as.numeric(as.character(GWiS$F_stat))
  GWiS$Pval <- as.numeric(as.character(GWiS$Pval))
  GWiS$K <- as.numeric(as.character(GWiS$K))
  GWiS$Tests <- as.numeric(as.character(GWiS$Tests))
  d <- GWiS$Pval==1
  GWiS$Pval[d] <- pchisq(GWiS$F_stat[d],1,lower.tail=FALSE) * GWiS$Tests[d]
  d <- GWiS$Pval>=1
  GWiS$Pval[d] <- 1
}

cp <- capabilities()

if(is.null(SNP)==FALSE) QQ('SNP',SNP,cp[1]);
if(is.null(GWiS)==FALSE) QQ('GWiS',GWiS$Pval,cp[1])
if(is.null(Vegas)==FALSE) QQ('Vegas',Vegas$Pval,cp[1])
if(is.null(BF)==FALSE) QQ('Bimbam',BF$Pval,cp[1])
if(is.null(Gates)==FALSE) QQ('Gates',Gates$Pval,cp[1])
if(is.null(MS)==FALSE) QQ('MinSNP',MS$Pval,cp[1])
if(is.null(MSG)==FALSE) QQ('MinSNP_Gene',MSG$Pval,cp[1])
   
  
