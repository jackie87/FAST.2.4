#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#define MAX_LINE_WIDTH 120000
#define MAX_N_INDIV 20000

int get_ind_count(char* fname)
{
    char command[100] = "";
    sprintf(command,"wc -l %s",fname);
    FILE * fp = popen(command,"r");
    int n = 0;
    fscanf(fp,"%d",&n);
    printf("# samples = %d\n",n);
    fclose(fp);
    return n;
}

/*
void copy_file(char *fname,char *newfile)
{
    char command[100] = "";
    sprintf(command,"cp %s %s",fname,newfile);
    FILE * fp = popen(command,"r");
    fclose(fp);
}
*/

void readGeno(int nsample, char * sline_geno, int * chr, long * bp, double * cm, char * name, double * FREQ1, double * geno, char * a1, char * a2)
{
  int i;
  char * nextGeno;
    
  int k = 0;//indexes genotypes read.
  *a1 = '0';
  *a2 = '0';

  char * pch = strtok(sline_geno," \t");
  //read chr
  sscanf(pch, "%d", chr);
  //printf("Read chr = %d\n",*chr);

  //read snp name
  pch = strtok(NULL," \t"); 
  sscanf(pch, "%s", name);
  //printf("Read snp = [%s]\n",name);

  //read cm
  pch = strtok(NULL," \t"); 
  sscanf(pch, "%lg", cm);
  //printf("Read cm = %lg\n",*cm);

  //read bp
  pch = strtok(NULL," \t"); 
  sscanf(pch, "%ld", bp);
  //printf("Read bp = %ld\n",*bp);

  *FREQ1 = 0;

  for (i=0; i<nsample; i++)
  {
    //read A1
    pch = strtok(NULL," \t"); 
    char A1;
    if(sscanf(pch, "%c", &A1) != 1){
      printf("-Error in tped file: genotype\n");
      abort();
    }

    //read A2
    pch = strtok(NULL," \t"); 
    char A2;
    if(sscanf(pch, "%c", &A2) != 1){
      printf("-Error in tped file: genotype\n");
      abort();
    }

    // *a1 *a1 = 0
    // *a1 *a2 = 1
    // *a2 *a1 = 1
    // *a2 *a2 = 2
    double g = -1;
    if(A1=='0' || A2=='0')
    {
       g = -1;
    }
    else if(A1!=A2) //fix
    {
       if(*a1=='0')
          *a1 = A1;

       if(*a1==A1 && *a2=='0')
          *a2 = A2;
       else if(*a1==A2 && *a2=='0')
          *a2 = A1;
       g = 1.0;
    }
    else //A1==A2
    {
       if(*a1=='0' || *a1==A1)
       {
          *a1 = A1;
          g = 0.0;
       }
       else if(*a2=='0')
       {
          *a2 = A1;
          g = 2.0;
       }
       else if(*a2==A1)
       {
          g = 2.0; //fix
       }
    }
    *geno = g;
    if(g>=0)
    {
      (*FREQ1) += (*geno);
    }
    geno++;
    k++;
  }
  (*FREQ1) /= (2*k);
  //printf("\n-------------\n");
}

bool file_exists(const char *filename){
  FILE *file;
  if ((file = fopen(filename, "r"))) 
  {
     fclose(file);
     return true;
  }
  return false;
}

int main(int nARG, char *ARGV[])
{
   if(nARG!=2)
   {
      printf("Please provide plink transposed file name to convert\n");
      exit(1);
   }
   char * fname = ARGV[1];
   char tped_file[100];
   char tfam_file[100];

   char tped_out[100];
   char snpinfo_out[100];
   char mlinfo_out[100];
   char id_out[100];
   char trait_out[100];

   sprintf(tped_file,"%s.tped",fname);
   sprintf(tfam_file,"%s.tfam",fname);

   if(!file_exists(tped_file))
   {
      printf("%s does not exist\n",tped_file);
      exit(1);
   }
   if(!file_exists(tfam_file))
   {
      printf("%s does not exist\n",tfam_file);
      exit(1);
   }

   printf("Starting conversion to FAST dosage format\n");
   
   sprintf(tped_out,"%s.fast.tped",fname);
   sprintf(snpinfo_out,"%s.fast.snp.info",fname);
   sprintf(mlinfo_out,"%s.fast.mlinfo",fname);
   sprintf(id_out,"%s.fast.id",fname);
   sprintf(trait_out,"%s.fast.tfam",fname);

   static char sline_geno[MAX_LINE_WIDTH];
   FILE * fp_tped = fopen(tped_file, "r");
   FILE * fp_tfam = fopen(tfam_file,"r");

   FILE * fp_tped_out = fopen(tped_out,"w");
   FILE * fp_snpinfo_out = fopen(snpinfo_out,"w");
   FILE * fp_mlinfo_out = fopen(mlinfo_out,"w");
   FILE * fp_id_out = fopen(id_out,"w");
   FILE * fp_tfam_out = fopen(trait_out,"w");

   int chr = 0;
   long bp = 0;
   double cm = 0;
   char name[40];
   double MAF = 0;
   double FREQ1 = 0;
   char a1 = '0';
   char a2 = '0';

   int nsample = get_ind_count(tfam_file);
   double * geno = (double*)malloc(sizeof(double)*MAX_N_INDIV);
   
   fprintf(fp_snpinfo_out,"#snp\tchr\tcm\tbp\n");
   fprintf(fp_mlinfo_out,"#snp\ta1\ta2\tmaf\tfreq1\tRsq\n");

   while(!feof(fp_tped))
   {
      strcpy(sline_geno, "");
      fgets(sline_geno, MAX_LINE_WIDTH, fp_tped);
      if(strcmp(sline_geno,"")==0) continue;
      readGeno(nsample, sline_geno, &chr, &bp, &cm, name, &FREQ1, geno, &a1, &a2);
      int i = 0;
      for(i=0;i<nsample;i++)
      {
         fprintf(fp_tped_out,"%.4g\t",geno[i]);
      }
      fprintf(fp_tped_out,"\n");
      fprintf(fp_snpinfo_out,"%s\t%d\t%g\t%ld\n",name,chr,cm,bp);
      MAF = FREQ1;
      if(MAF > 0.5)
         MAF = 1.0 - FREQ1;
      fprintf(fp_mlinfo_out,"%s\t%c\t%c\t%.4g\t%.4g\t1.0\n",name,a1,a2,MAF,FREQ1);
      printf("snp = %s\n",name);
   }
   fclose(fp_tped);
   fclose(fp_tped_out);
   fclose(fp_snpinfo_out);
   fclose(fp_mlinfo_out);

   fprintf(fp_tfam_out,"#fid\tiid\tmid\tdid\tsex\tpheno\n");
   while(!feof(fp_tfam))
   {
      strcpy(sline_geno, "");
      fgets(sline_geno, MAX_LINE_WIDTH, fp_tfam);
      if(strcmp(sline_geno,"")==0) break;
      char fid[40];
      char iid[40];
      char mid[40];
      char did[40];
      char sex[40];
      char pheno[40];
      sscanf(sline_geno,"%s %s %s %s %s %s",fid,iid,mid,did,sex,pheno);
      fprintf(fp_id_out,"%s\n",iid);
      fprintf(fp_tfam_out,"%s\t%s\t%s\t%s\t%s\t%s\n",fid,iid,mid,did,sex,pheno);
   }
   fclose(fp_tfam);
   fclose(fp_id_out);
   fclose(fp_tfam_out);
}
