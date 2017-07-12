#include "GWiS.h"

int partition_linear(SNP** y, int f, int l)
{
     int up,down;
     SNP* temp;
     SNP* piv = y[f];
     up = f;
     down = l;
     goto partLS;
     do { 
         temp = y[up];
         y[up] = y[down];
         y[down] = temp;
     partLS:
         while (y[up]->pval_linear <= piv->pval_linear && up < l)
         {
             up++;
         }
         while (y[down]->pval_linear > piv->pval_linear  && down > f )
         {
             down--;
         }
     } while (down > up);
     y[f] = y[down];
     y[down] = piv;
     return down;
}

int partition_logistic(SNP** y, int f, int l)
{
     int up,down;
     SNP* temp;
     SNP* piv = y[f];
     up = f;
     down = l;
     goto partLS;
     do { 
         temp = y[up];
         y[up] = y[down];
         y[down] = temp;
     partLS:
         while (y[up]->pval_logistic <= piv->pval_logistic && up < l)
         {
             up++;
         }
         while (y[down]->pval_logistic > piv->pval_logistic  && down > f )
         {
             down--;
         }
     } while (down > up);
     y[f] = y[down];
     y[down] = piv;
     return down;
}

int partition_summary(SNP** y, int f, int l)
{
     int up,down;
     SNP* temp;
     SNP* piv = y[f];
     up = f;
     down = l;
     goto partLS;
     do { 
         temp = y[up];
         y[up] = y[down];
         y[down] = temp;
     partLS:
         while (y[up]->metaP <= piv->metaP && up < l)
         {
             up++;
         }
         while (y[down]->metaP > piv->metaP  && down > f )
         {
             down--;
         }
     } while (down > up);
     y[f] = y[down];
     y[down] = piv;
     return down;
}

void quicksort_linear(SNP** x, int first, int last)
{
     int pivIndex = 0;
     if(first < last) {
         pivIndex = partition_linear(x,first, last);
         quicksort_linear(x,first,(pivIndex-1));
         quicksort_linear(x,(pivIndex+1),last);
     }
}

void quicksort_logistic(SNP** x, int first, int last)
{
     int pivIndex = 0;
     if(first < last) {
         pivIndex = partition_logistic(x,first, last);
         quicksort_logistic(x,first,(pivIndex-1));
         quicksort_logistic(x,(pivIndex+1),last);
     }
}

void quicksort_summary(SNP** x, int first, int last)
{
     int pivIndex = 0;
     if(first < last) {
         pivIndex = partition_summary(x,first, last);
         quicksort_summary(x,first,(pivIndex-1));
         quicksort_summary(x,(pivIndex+1),last);
     }
}


/*
//gcc -lgsl -lgslcblas -lm QuickSort.c
int main()
{
  SNP s0;strcpy(s0.name,"rs0");s0.f_stat=0.3;
  SNP s1;strcpy(s1.name,"rs1");s1.f_stat=10.3;
  SNP s2;strcpy(s2.name,"rs2");s2.f_stat=0.3;
  SNP s3;strcpy(s3.name,"rs3");s3.f_stat=13.3;
  SNP s4;strcpy(s4.name,"rs4");s4.f_stat=2.3;
  SNP* x[5];
  x[0] = &s0; 
  x[1] = &s1; 
  x[2] = &s2; 
  x[3] = &s3; 
  x[4] = &s4;
  int i = 0;
  for(i=0;i<5;i++) printf("%s %g\n",x[i]->name,x[i]->f_stat);
  printf("------------\n");
  quicksort(x,0,4);
  for(i=0;i<5;i++) printf("%s %g\n",x[i]->name,x[i]->f_stat);
  return 0;
}
*/
