/*
 * Logistic.c
 *
 *  Created on: Jul 8, 2010
 *  Updated on July 26, 2010
 *      Author: pchanda
 */

#include "GWiS.h"

//check of a SNP has high correlation w/ SNP in the model
bool isSNPCorrelated_logistic(SNP* this_snp, SNP* curr_model[], int k,  gsl_matrix *LD,int n_covariates)
{
  int i;
  //printf("isSNPCorrelated_logistic called with %s and model size = %d\n",this_snp->name,k);
  for(i=n_covariates; i<k; i++)
  {
    //printf("Got %s %d\n",curr_model[i]->name,curr_model[i]->gene_id);fflush(stdout);
    double corr = fabs(gsl_matrix_get(LD, this_snp->gene_id, curr_model[i]->gene_id));
    if(LOGISTIC_DEBUG_LVL>=3)
       printf("Corr [*%s,%s] = %g\n",this_snp->name,curr_model[i]->name,corr);fflush(stdout);
    if(corr > VIF_R)
      return true;
  }
  return false;
}
 
double loglikelihood(LOGISTIC_SCRATCH* LG, gsl_vector* phenotype, int N, gsl_vector* W)
{
   //compute the log likelihood.
   double LL = 0;
   int p = W->size - 1;
   int i = 0;
   for(i=0;i<N;i++)
   {
      double o = gsl_vector_get(LG->OMIT,i);
      if(o==0) //no missing data in this sample.
      { 
        double w_x = gsl_vector_get(W,0); //intercept * w[0] = w[0].
        int j = 0;
        double pheno = gsl_vector_get(phenotype,i);
        for(j=0;j<p;j++)
        {
            double w_j = gsl_vector_get(W,j+1);
	    SNP* snp_j = LG->curr_model[j];
	    double x_j_i = snp_j->geno[i];
            w_x += w_j*x_j_i;
        }
        if(pheno<0)
        	LL += log(1 + exp(w_x));
        else
        	LL += log(1 + exp(-w_x));
      }
    }
    return -LL; //return the log likehood, a negative numner.
}

bool stop_iter(gsl_vector* W, gsl_vector* old_W)
{
   double max = 0;
   int p = W->size;
   int j = 0;
   for(j=0;j<p;j++)
   {
      double x = fabs(gsl_vector_get(W,j) - gsl_vector_get(old_W,j));
      if(x>max)
        max = x;
   }
   if(max<1e-5)
     return true;
   else
     return false;
}

int solve_svd(gsl_matrix* h, gsl_vector* g, gsl_vector* x) //FIX V.1.2
{
   int p = h->size1;
   gsl_matrix * V = gsl_matrix_alloc(p,p);
   gsl_vector * S = gsl_vector_alloc(p);
   gsl_vector * work = gsl_vector_alloc(p);
   int err = gsl_linalg_SV_decomp (h, V, S, work);
   if(err==0)
   {
      err = gsl_linalg_SV_solve (h, V, S, g, x);
   }
   return err;
}

int solve(gsl_matrix* h, gsl_vector* g, gsl_vector* x)
{
   // x = h\g is the solution to the equation hx = g.
   int s;
   int p = h->size1;
   int err = 0;
   if(p==1)
   {
     printf("Logistic : Error in solve, size of matrix should be > 1\n");
     exit(1); 
   }
   else if(p==2)
   {
       err = gsl_linalg_LU_decomp (h, P2, &s);
       if(err==0)
       {
         err = gsl_linalg_LU_solve (h, P2, g, x);
       }
   }
   else if(p==3)
   {
       err = gsl_linalg_LU_decomp (h, P3, &s);
       if(err==0)
       {
         err = gsl_linalg_LU_solve (h, P3, g, x);
       }
   }
   else if(p==4)
   {
       err = gsl_linalg_LU_decomp (h, P4, &s);
       if(err==0)
       {
         err = gsl_linalg_LU_solve (h, P4, g, x);
       }
   }
   else if(p==5)
   {
       err = gsl_linalg_LU_decomp (h, P5, &s);
       if(err==0)
       {
         err = gsl_linalg_LU_solve (h, P5, g, x);
       }
   }
   else if(p==6)
   {
       err = gsl_linalg_LU_decomp (h, P6, &s);
       if(err==0)
       {
         err = gsl_linalg_LU_solve (h, P6, g, x);
       }
   }
   else 
   {
     //very rare 6 snp model.
     gsl_permutation * P = gsl_permutation_alloc (p);
     err = gsl_linalg_LU_decomp (h, P, &s);
     if(err==0)
     {
       err = gsl_linalg_LU_solve (h, P, g, x);
     }
     gsl_permutation_free (P);
   }

   return err;
}

void alloc_P()
{
   P2 = gsl_permutation_alloc (2);
   P3 = gsl_permutation_alloc (3);
   P4 = gsl_permutation_alloc (4);
   P5 = gsl_permutation_alloc (5);
   P6 = gsl_permutation_alloc (6);
}

void free_P()
{
   if(P2!=NULL) gsl_permutation_free (P2);
   if(P3!=NULL) gsl_permutation_free (P3);
   if(P4!=NULL) gsl_permutation_free (P4);
   if(P5!=NULL) gsl_permutation_free (P5);
   if(P6!=NULL) gsl_permutation_free (P6);
}

void print1(double v[], int n)
{
	int i = 0;
	for(i=0;i<n;i++)
		printf("%.3f ",v[i]);
	printf("\n-------\n");
}

void print2(gsl_matrix* M)
{
	int n1 = M->size1;
	int n2 = M->size2;
	int i = 0;
	for(i=0;i<n1;i++)
	{
		int j = 0;
		for(j=0;j<n2;j++)
		{
			printf("%.6f ",gsl_matrix_get(M,i,j));
		}
		printf("\n");
	}
	printf("\n-------\n");
}

void print_model(SNP* model[], int size,double LL,double increment)
{
   printf("----Model----\nLL = %f-----inc = %f---------\n",LL,increment);
   int i = 0;
   for(i=0;i<size;i++)
   {
      printf("%s ",(model[i])->name);
   }
   printf("\n-------------\n");
}

void print_model_1(SNP* model[], int size)
{
   printf("----Model---------\n");
   int i = 0;
   for(i=0;i<size;i++)
   {
      printf("%s ",(model[i])->name);
   }
   printf("\n-------------\n");
}

void fprint_model(FILE* fp, SNP* model[], int size)
{
   fprintf(fp,"----Model---------\n");
   int i = 0;
   for(i=0;i<size;i++)
   {
      fprintf(fp,"%s ",(model[i])->name);
   }
   fprintf(fp,"\n-------------\n");
}

void print_data(LOGISTIC_SCRATCH* LG, gsl_vector* phenotype, int N)
{
	printf("Printing data....");
	int p = LG->curr_model_size;
	int j = 0;
	int i = 0;
	for(j=0;j<p;j++)
	{
		for(i=0;i<N;i++)
		{
			 SNP* snp_j = LG->curr_model[j];
			 double x_j_i = snp_j->geno[i];
			 printf("%f ",x_j_i);
		}
		printf("\n");
	}
	printf("\n---------------------------------\n");
	printf("Printing phenotype....");
	for(i=0;i<N;i++)
	{
		printf("%f ",gsl_vector_get(phenotype,i));
	}
	printf("\n---------------------------------\n");
}

/*
void write_error_data(char* snp)
{
        static int cnt = 0;
        char fname[40];
        sprintf(fname,"Logistic.%s.dump.txt",snp);
        char x[40];
        if(!PERM)
          sprintf(x,"%s.chr%d.%d.txt",fname,CHR,cnt);
        else
	  sprintf(x,"%s.chr%d.%d.perm.txt",fname,CHR,cnt);

        FILE *fp = fopen(x,"w");
	int p = LG_error.LG->curr_model_size;
	int j = 0;
	int i = 0;
        printf("Dumping...%d\n",cnt);
        fprint_model(fp, LG_error.LG->curr_model, LG_error.LG->curr_model_size);
        
	for(j=0;j<p;j++)
	{
		for(i=0;i<LG_error.N;i++)
		{
			 SNP* snp_j = LG_error.LG->curr_model[j];
			 double x_j_i = snp_j->geno[i];
			 fprintf(fp,"%f ",x_j_i);
		}
		fprintf(fp,"\n");
	}
	fprintf(fp,"\n---------------------------------\n");
	for(i=0;i<LG_error.N;i++)
	{
		fprintf(fp,"%f ",gsl_vector_get(LG_error.phenotype,i));
	}
	fprintf(fp,"\n---------------------------------\n");
        fclose(fp);
        cnt++;
        printf("Done : Dumping...%d\n",cnt);
}
*/

//V.1.2
void my_error_handler (const char * reason,
                        const char * file,
                        int line,
                        int gsl_errno)//ADDED PRITAM LATEST commented out the content of this function. Nothing to be done.
{
  /*
  printf("-FAST error handler invoked for Logistic Regression\n");
  printf("-reason = %s\n",reason);
  printf("-file = %s\n",file);
  printf("-line = %d\n",line);
  printf("-errno = %d\n",gsl_errno);
  //write_error_data();
  printf("-Continuing with remaining SNPs...\n");
  */
}

//V.1.2
void my_error_handler_1 (const char * reason,
                        const char * file,
                        int line,
                        int gsl_errno) //ADDED PRITAM LATEST commented out the content of this function. Nothing to be done.
{
  //printf("-FAST error handler invoked for Logistic Regression : %s\n",reason);
}

double gradient_descent(LOGISTIC_SCRATCH* LG, gsl_vector* phenotype, int N, bool PERM, int * err) //V.1.2
{
        *err = 0; //V.1.2
	//phenotype is +/- 1.
	int p = LG->curr_model_size; // count of snps in the model, intercept is an extra dimension.
	if(LOGISTIC_DEBUG_LVL>=2 && !PERM)
	{
		printf("In gradient descent..............%d %d...\n",N,p);
		print_model_1(LG->curr_model,LG->curr_model_size);
                fflush(stdout);
	}

	//Initialize scratch space.
	gsl_vector_view g = gsl_vector_subvector(LG->g, 0, p+1);
	gsl_vector_view W = gsl_vector_subvector(LG->W, 0, p+1);
	gsl_vector_view old_W = gsl_vector_subvector(LG->old_W, 0, p+1);
	gsl_vector_view W_delta = gsl_vector_subvector(LG->W_delta, 0, p+1);
	gsl_matrix_view h = gsl_matrix_submatrix(LG->h,0,0,p+1,p+1);
	gsl_matrix_view h_temp = gsl_matrix_submatrix(LG->h_temp,0,0,p+1,p+1);

	gsl_vector_set_zero(&g.vector); //g = [0 0 0 ... 0]';
	gsl_vector_set_zero(&old_W.vector); //old_W = [0 0 0 ... 0]';
	gsl_vector_set_zero(&W_delta.vector); //W_delta = [0 0 0 ... 0]';
	gsl_matrix_set_zero(&h.matrix); //h = [0];

	int i = 0;
	int j = 0;
	for(i=0;i<p+1;i++)
	{
		gsl_vector_set(&W.vector,i,0);
	}

	if(LOGISTIC_DEBUG_LVL>=2 && !PERM)
	{
		printf("Done initializing scratch space\n");
                fflush(stdout);
	}

	int iter = 0;
	for(iter=0;iter<MAX_LOGISTIC_ITER;iter++)
	{
	  //old_W = W.
	  for(j=0;j<p+1;j++)
	      gsl_vector_set(&old_W.vector,j,gsl_vector_get(&W.vector,j));

	   //s1 = 1./(1+exp(w'*x)).
	   //a = s1.*(1-s1).
	   double s1[N];
	   double a[N];
	   double g_0 = 0;
	   for(i=0;i<N;i++)
	   {
              double o = gsl_vector_get(LG->OMIT,i);
              if(o==0) //no missing data in this sample.
              { 
		  double pheno = gsl_vector_get(phenotype,i);
		  double w_x = gsl_vector_get(&W.vector,0);
		  for(j=0;j<p;j++)
		  {
			 double w_j = gsl_vector_get(&W.vector,j+1);
			 SNP* snp_j = LG->curr_model[j];
			 double x_j_i = snp_j->geno[i];
			 w_x += w_j*x_j_i;
		  }
		  if(pheno<0)
		  {
			  s1[i] = 1/(1 + exp(-w_x));
			  g_0 -= s1[i];
		  }
		  else
		  {
			  s1[i] = 1/(1 + exp(w_x));
			  g_0 += s1[i];
		  }
		  a[i] = s1[i]*(1-s1[i]);
               }
	   }

	   if(LOGISTIC_DEBUG_LVL>=3 && !PERM)
	   {
		   printf("s1 = "); print1(s1,N);
		   printf("a = "); print1(a,N);
                   fflush(stdout);
	   }

           //g = x*s1'.
	   //g_0 is already calculated above as \Sigma_i (intercept*s[i]*y[i]).
	   gsl_vector_set(&g.vector,0,g_0);
           for(j=0;j<p;j++)
           {
             double g_j = 0;
             for(i=0;i<N;i++)
             {
               double o = gsl_vector_get(LG->OMIT,i);
               if(o==0) //no missing data in this sample.
               { 
        	  double pheno = gsl_vector_get(phenotype,i);
                  //if(i<20 || i>7980)
                  //   printf("%d %g\n",i,pheno);
 		  SNP* snp_j = LG->curr_model[j];
 		  double x_j_i = snp_j->geno[i];
 		  if(pheno<0)
 			  g_j -= x_j_i*s1[i];
 		  else
 			 g_j += x_j_i*s1[i];
               }
             }
             gsl_vector_set(&g.vector,j+1,g_j);
          }
          if(LOGISTIC_DEBUG_LVL>=2 && !PERM)
          {
    	     printf("g = "); gsl_vector_fprintf (stdout, &g.vector, "%g");
          }

          //h = scale_cols(x,a)*x'.
          //do not need phenotype as it is squared (so is = 1) for each element of h.
          int j1 = 0;
          int j2 = 0;
          for(j1=0;j1<p+1;j1++)
          {
            for(j2=0;j2<p+1;j2++)
            {
              double val = 0;
              if(j1==0 && j2==0)
              {
		for(i=0;i<N;i++)
		{
                  double o = gsl_vector_get(LG->OMIT,i);
                  if(o==0)
		     val += a[i];
		}
              }
              else if(j1==0 && j2>0)
              {
		 for(i=0;i<N;i++)
		 {
                    double o = gsl_vector_get(LG->OMIT,i);
                    if(o==0)
                    {
			 SNP* snp_j2 = LG->curr_model[j2-1];
			 double x_j2_i = snp_j2->geno[i];
			 val += a[i]*x_j2_i;
                    }
		 }
              }
              else if(j1>0 && j2==0)
              {
		 for(i=0;i<N;i++)
		 {
                    double o = gsl_vector_get(LG->OMIT,i);
                    if(o==0)
                    {
			 SNP* snp_j1 = LG->curr_model[j1-1];
			 double x_j1_i = snp_j1->geno[i];
			 val += a[i]*x_j1_i;
                    }
		 }
		 gsl_matrix_set(&h.matrix,j1,j2,val);
              }
              else
              {
		 for(i=0;i<N;i++)
		 {
                    double o = gsl_vector_get(LG->OMIT,i);
                    if(o==0)
                    {
			 SNP* snp_j1 = LG->curr_model[j1-1];
			 double x_j1_i = snp_j1->geno[i];
			 SNP* snp_j2 = LG->curr_model[j2-1];
			 double x_j2_i = snp_j2->geno[i];
			 val += a[i]*x_j1_i*x_j2_i;
                    }
		 }
              }
              gsl_matrix_set(&h.matrix,j1,j2,val);
           }
         }
         if(LOGISTIC_DEBUG_LVL>=2 && !PERM)
         {
    	   printf("h = \n");
    	   print2(&h.matrix);
         }

         //to do h\g.
         // X = A\B is the solution to the equation AX = B.
         //So solve : h W_delta = g.
         gsl_matrix_memcpy (&h_temp.matrix, &h.matrix); //create a copy of h for solve to overwrite.

         //V.1.2
         *err = solve(&h_temp.matrix,&g.vector,&W_delta.vector);

         //FIX V.1.2
         if(*err > 0)
         {
            gsl_matrix_memcpy (&h_temp.matrix, &h.matrix); //create a copy of h for solve to overwrite.
            *err = solve_svd(&h_temp.matrix,&g.vector,&W_delta.vector);
         }

         if((*err>0) && (!PERM))
         {
             printf("-Logsitic regression gradient descent error : %s\n",gsl_strerror(*err));
             printf("-Logsitic regression gradient descent : cannot compute likelihood for model = ");
             int m = 0;
             for(m=0;m<LG->curr_model_size;m++)
             {
                printf("%s ",(LG->curr_model[m])->name);
             }
             printf("\n");
             //write_error_data(LG);
         }

         if(LOGISTIC_DEBUG_LVL>=2 && !PERM)
         {
    	   printf(" del W = "); gsl_vector_fprintf (stdout, &W_delta.vector, "%g");
         }
         //w = w + h\g.
         gsl_vector_add(&W.vector,&W_delta.vector);
         if(LOGISTIC_DEBUG_LVL>=2 && !PERM)
         {
    	   printf("\n W = "); gsl_vector_fprintf (stdout, &W.vector, "%g");
         }

         if(LOGISTIC_DEBUG_LVL>=2 && !PERM)
         {
    	   //compute negative loglikelihood.
           double LL = loglikelihood(LG,phenotype,N,&W.vector);
    	   printf("Iter = %d  :  LL = %.5f\n",iter,LL);
    	   printf("==========================================\n");
         }

         //if max(abs(w - old_w)) < 1e-8, break.
         bool flag = stop_iter(&W.vector,&old_W.vector);

         if(flag || *err>0) //V.1.2
         {
           break;
         }
       }//for each gradient descent iteration.
       
       double LL = loglikelihood(LG,phenotype,N,&W.vector);

       if(LOGISTIC_DEBUG_LVL>=3 && !PERM)
       {
    	   printf("h = \n");
    	   print2(&h.matrix);
       }

       //Also return the hessian matrix and the weight vector (betas).
       //We will use the inverse of the hessian to compute the standard errors, use these with the betas 
       //to compute the Wald test statistics (chi2).
       //Also need to handle missing data.
       //printf("*******************************************************\n");
       return LL;
}

//get the change of GWiS statistics when adding a SNP into the model
double get_Increment(double LL_new, double LL_old, double eSNP, int k, int nsample, bool PERM)
{
	//the change in logProb from k-1 to k = BIC(k-1) - BIC(k).
	double increment = LL_old - LL_new;
	increment += log(eSNP-k+1) - log(k);
	increment += log(nsample)/2;

	if(LOGISTIC_DEBUG_LVL>=1)
	{
                //commented because gsl_sf_lnbeta sometimes throws error!!!
		//double BIC_abs = LL_new - (k/2.0)*log(nsample) + gsl_sf_lnbeta (k+1,eSNP-k+1);
		//printf("BIC for (k=%d, eSNP=%g, L = %g) = %g\n",k,eSNP,LL_new,BIC_abs);
                if(!PERM)
		    printf("Increment = %g\n",-increment);
                else if(VERBOSE)
		    printf("PERM : Increment = %g\n",-increment);
	}
	return -increment;
}

/*
//commented because gsl_sf_lnbeta sometimes throws error!!!
//get the absolute of GWiS statistics for this model
double get_BIC(double LL, double eSNP, int k, int nsample)
{
	double BIC_abs = LL - (k/2.0)*log(nsample) + gsl_sf_lnbeta(k+1,eSNP-k+1);
	if(LOGISTIC_DEBUG_LVL>=1 && !PERM)
	{
		printf("BIC for (k=%d, eSNP=%g, L = %g) = %g\n",k,eSNP,LL,BIC_abs);
	}
	return BIC_abs;
}
*/

double loglik_0()
{
  double N = count_1+count_2;
  printf("Log lik for model 0 : counts = %d %d %g\n",count_1,count_2,N);
  return count_1*log(count_1/N) + count_2*log(count_2/N); //log likelihood, a negative number.
}

void get_wald_statistic(LOGISTIC_SCRATCH* LG,SNP* snp, double LL, LOGISTIC_Z * LZ,bool perm,int errnum)//V.1.2
{
   if(LOGISTIC_DEBUG_LVL>=2)
      printf("Computing wald \n");
   int s;
   int p = LG->curr_model_size; // count of covariates + 1 snp in the model, intercept is an extra dimension.
   gsl_matrix_view h = gsl_matrix_submatrix(LG->h,0,0,p+1,p+1);
   gsl_matrix_view hinv = gsl_matrix_submatrix(LG->h_inv,0,0,p+1,p+1);
   gsl_permutation * P = gsl_permutation_alloc (p+1);
   gsl_vector_view W = gsl_vector_subvector(LG->W, 0, p+1);

   if(LOGISTIC_DEBUG_LVL>=2)
   {
      printf("h = \n");
      print2(&h.matrix);
   }

   //V.1.2
   int err = errnum;
   if(err==0)
   {
     //inverting the hessian to compute variance/covariance matrix.
     err = gsl_linalg_LU_decomp (&h.matrix, P, &s); //V.1.2
     if(err==0)
     {
        if(LOGISTIC_DEBUG_LVL>=2)
            printf("Successful LU decomposition %s\n",snp->name);
        err = gsl_linalg_LU_invert (&h.matrix, P, &hinv.matrix);
        if(err==0)
        {
          if(LOGISTIC_DEBUG_LVL>=2)
             printf("Successful LU inversion \n");
          if(LOGISTIC_DEBUG_LVL>=2 && !perm)
          {
    	    printf("h inv = \n");
    	    print2(&hinv.matrix);
          }
          int k = 0;
          for(k=0;k<p+1;k++) 
          {
             LG->se[k] = sqrt(gsl_matrix_get(&hinv.matrix,k,k));
             double beta = gsl_vector_get(&W.vector,k);
             if(LG->se[k]>0)
             {
               LG->wald[k] = beta/LG->se[k];
               LG->wald[k] *= LG->wald[k];
             }
             else
               LG->wald[k] = GSL_NAN;
             if(LOGISTIC_DEBUG_LVL>=2)
             {
               if(k==0)
                 printf("  get_wald_statistic: (%d) %s se=%f beta=%f wald=%f\n",k,"Intercept",LG->se[k],beta,LG->wald[k]);
               else
                 printf("  get_wald_statistic: (%d) %s se=%f beta=%f wald=%f\n",k,LG->curr_model[k-1]->name,LG->se[k],beta,LG->wald[k]);
               //printf("  get_wald_statistic: (%d) %s se=%f beta=%f wald=%f\n",k,snp->name,LG->se[k],beta,LG->wald[k]);
             }
          }
       }
     }
   }

   //V.1.2
   if(err>0 && !perm)
   {
      printf("-real trait : error in inverting hessian for SNP = %s with maf = %lg, error code = %d\n",snp->name,snp->MAF,err);
      int k = 0;
      for(k=0;k<p+1;k++) //k=0 for snp1, k=1 for snp2, .... Intercept not stored.
      {
         LG->se[k] = GSL_NAN;
         LG->wald[k] = GSL_NAN;
      }
   }
   gsl_permutation_free (P);

   //V.1.2
   if(!perm)
   {
      snp->wald = LG->wald[LG->n_covariates+1];
      if(gsl_isnan(snp->wald)==0)
         snp->beta_logistic = gsl_vector_get(&W.vector,LG->n_covariates+1);
      else
         snp->beta_logistic = GSL_NAN;
      snp->se_logistic = LG->se[LG->n_covariates+1];
      snp->loglik_logistic = LL;
      if(gsl_isnan(snp->wald)==0)
         snp->pval_logistic =  gsl_cdf_chisq_Q(snp->wald,1);
      else
         snp->pval_logistic = 1.0;
   }
   else //for permutations
   {
      if(err>0) //V.1.2
      {
         if(VERBOSE)
         {
           printf("-permutations : error in inverting hessian : error code = %d\n",err);
           printf("-permutations : setting chisq to nan for snp %s\n",snp->name);
         }
         LZ->wald_perm = GSL_NAN; //FIX V.1.2
         LZ->loglik_logistic_perm = LL; //still use the half baked LL for future model search ?
      }
      else
      {
         LZ->wald_perm = LG->wald[LG->n_covariates+1];
         LZ->loglik_logistic_perm = LL;
      }
   }
}

void remove_missing_samples(LOGISTIC_SCRATCH* LG)
{  
   if(MISSING_DATA==false)
   {  
      if(VERBOSE)
        printf("No missing data for logistic regression\n");
      return; //no missing data present in the entire data set. Either user specified or determined from missingness of the snps in genes.
   }
   if(VERBOSE)
        printf("missing data present for logistic regression\n");
   //remove missing data if present.
   //populate a OMIT vector for each sample. OMIT[i] = 1 means to omit the sample.
   //In this way you need not change the original genotypes and phenotypes.
   int i = 0;
   int p = LG->curr_model_size; //exclude the intercept.
   int N = LG->phenotype->N_sample;
   gsl_vector_set_zero(LG->OMIT);

   //Check if any snp in the model has missing data.
   int j = 0;
   bool missing = false;
   for(j=LG->n_covariates;j<p;j++)
   {  
      SNP* snp_j = LG->curr_model[j];
      if(snp_j->missingness > 0)
      {  
         missing = true;
         break;
      }
   }

   if(!missing) //no snp in the model has missing data.
       return;

   for(i=0;i<N;i++)
   {
        j = 0;
        missing = false;
        for(j=LG->n_covariates;j<p;j++)
        {   
            SNP* snp_j = LG->curr_model[j];
            double x_j_i = snp_j->geno[i];
            if(x_j_i==MISSING_VAL)
            {  
               missing = true;
               gsl_vector_set(LG->OMIT,i,1);
               break;
            }
        }
   }
}

void runLogisticSNP(SNP* snp, LOGISTIC_SCRATCH* LG,bool perm,LOGISTIC_Z * LZ,bool need_hessian)
{
        //printf("I am in runLogistic\n");fflush(stdout);
        gsl_error_handler_t* old_handler = gsl_set_error_handler (&my_error_handler);
        int N = LG->phenotype->N_sample;
	LG->curr_model[LG->n_covariates] = snp;
	LG->curr_model_size = LG->n_covariates+1;
        if(snp->missingness>0)
           remove_missing_samples(LG);
        int err = 0;//V.1.2    
        double LL = gradient_descent(LG, LG->phenotype->pheno_array_log,N,perm,&err); //V.1.2 log likelihood after model fit.
        if(perm)
        {
           LZ->loglik_logistic_perm = LL; 
        }
        gsl_set_error_handler(&my_error_handler_1); //V.1.2
        if(need_hessian)
        {
           get_wald_statistic(LG,snp,LL,LZ,perm,err); //V.1.2
        }
        gsl_set_error_handler (old_handler);
}

BIC_STATE* runLogistic(C_QUEUE* snp_queue, GENE* gene, BIC_STATE* bic_state, LOGISTIC_SCRATCH* LG, FILE* fp_result,LOGISTIC_Z * LZ_PERM,bool PERM)
{
        gsl_error_handler_t* old_handler = gsl_set_error_handler (&my_error_handler);

        static double logistic_intercept_LL = GSL_POSINF;

        //no debug for permutations.
        int N = LG->phenotype->N_sample;
        int n_covariates = LG->n_covariates;
        LG->curr_model_size = n_covariates;

    	if(LOGISTIC_DEBUG_LVL>=1)
        {
            if(!PERM)
              printf("\n\n ################ Logistic Model search for gene = %s ###########\n",gene->name);
            else if(VERBOSE)
              printf("\n\n ################ PERM : Logistic Model search for gene = %s ###########\n",gene->name); 
        }

	int k = 0;

	//Initialize BIC state.
	initBIC_State(bic_state, 0);

	//k = 0, model size = 0 is intercept only model.
        double LL_old = 0;
        int err = 0;//V.1.2
        if(!PERM || LG->n_covariates==0)
        {
          //if no covariates, the intercept remains same for real trait and permutations for all genes. So compute it just once.
          //if covariates are present, the intercept needs to be computed once for real trait for all permutations.
          if(logistic_intercept_LL==GSL_POSINF)
          {
             //When covariates are present, you cannot use loglik_0().
             if(LG->n_covariates==0)
             {
	       LL_old = loglik_0(); //log likelihood for intercept only model.
             }
             else 
	       LL_old = gradient_descent(LG, LG->phenotype->pheno_array_log,N,PERM,&err); //log likelihood for intercept only model + covariates. V.1.2
             logistic_intercept_LL = LL_old;
             if(LOGISTIC_DEBUG_LVL>=1 && VERBOSE)
                printf("Computed logistic intercept only, LL=%g\n",LL_old);
          }
          else
          {
             LL_old = logistic_intercept_LL;
             if(LOGISTIC_DEBUG_LVL>=1)
             {
                if(!PERM)
                  printf("-Using pre-computed logistic intercept = %f\n",logistic_intercept_LL);
                else if(VERBOSE)
                  printf("-PERM : Using pre-computed logistic intercept, no covariates = %f\n",logistic_intercept_LL);
             }
          }
        }
        else //perm==true and n_covariates > 0, need to compute intercept for each shuffled trait.
        {
           //With permutations, need to compute intercept for each shuffled trait if covariates are present.
	   LL_old = gradient_descent(LG, LG->phenotype->pheno_array_log,N,PERM,&err); //log likelihood for intercept only model + covariates. V.1.2
           if(VERBOSE && LOGISTIC_DEBUG_LVL>=1)
              printf("-PERM : Recomputed logistic intercept as covariates are present = %f\n",LL_old);
        } 

	//update BIC state with intercept only model information.
	//Calculate BIC relative to BIC[0].
	bic_state->BIC[0] = 0;//LL_old - log(gene->eSNP+1); //LL - log(T+1)
	bic_state->bestSNP[0] = NULL; //no snps.
	bic_state->iSNP = 0;
        bic_state->LL[0] = logistic_intercept_LL;

	if(LOGISTIC_DEBUG_LVL>=1 && !PERM)
	{
		printf("Intercept LL = %f\n",LL_old);
		printf("Starting model search...\n");
	}

	if(fp_result != NULL)
	   fprintf(fp_result, "%d\t%s\t%s\t%d\t%d\t%d\t%d\t%g\t%s\t%s\t%s\t%s\t%d\t%g\t%g\n", 
                              gene->chr, 
                              gene->ccds, 
                              gene->name, 
                              gene->bp_start, 
                              gene->bp_end, 
                              gene->bp_end-gene->bp_start+1, 
                              gene->nSNP, 
                              gene->eSNP,
                              "NONE", //SNP.name 
                              "-", 
                              "-", 
                              "-",//R2 (imputation quality) 
                              0, //K
                              0.0, //BIC
                              0.0);//lrt chi2
                              //"-"); //R2, multiple R2


	for(k=1; k <= MAX_INCLUDED_SNP;k++) //for each model size, should equal LG->curr_model_size.
	{
    	        if(LOGISTIC_DEBUG_LVL>=1)
                {
                   if(!PERM)
                     printf("\n************* Model search for k = %d ************          \n",k); 
                   else if(VERBOSE)
                     printf("\nPERM:     ************* Model search for k = %d ************          \n",k); 
                }

		double max_increment = GSL_NEGINF;
		double LL_best = 0;
		SNP* best_SNP = NULL;
		int i = 0;
                int curr_model_size = LG->curr_model_size; //current size of the model.

                if(VERBOSE)
                   printf("CURRENT MODEL SIZE = %d\n",curr_model_size);
		SNP* snp = (SNP*) cq_getItem(gene->snp_start, snp_queue);



		for(i = gene->snp_start; i <= gene->snp_end; i++) //for each snp in the gene, add the snp to the model if not already in it.
		{
			if(!isSNPCorrelated_logistic(snp,LG->curr_model,LG->curr_model_size,gene->LD,n_covariates)&& gsl_isnan(snp->wald)==false)//FIX V.1.2 LATEST
			{
				//add snp to current model.
				if(LOGISTIC_DEBUG_LVL>=1)
				{
                                   if(!PERM)
                                   {
				      printf("Adding %d %s\n",k-1,snp->name);
                                      print_model_1(LG->curr_model, LG->curr_model_size);
                                   }
                                   else if(VERBOSE)
				      printf("PERM : Adding %d %s\n",k-1,snp->name);
				}
				LG->curr_model[k-1 + n_covariates] = snp;
				LG->curr_model_size = k + n_covariates;
                                double LL = 0;

                                if(k==1)
                                {
                                   if(!PERM)
                                      LL = snp->loglik_logistic; //already computed, no need to recompute.
                                   else
                                   {
                                      //LL = snp->loglik_logistic_perm_sh; //already computed, no need to recompute.
                                      LL = LZ_PERM[snp->gene_id].loglik_logistic_perm; 
                                   }
                                   if(LOGISTIC_DEBUG_LVL>=2)
                                   {  
                                      printf("Using pre-computed LL = %g\n",LL);
                                   }
                                } 
				else
                                {
                                   remove_missing_samples(LG);
                                   LL = gradient_descent(LG, LG->phenotype->pheno_array_log,N,PERM,&err); //log likelihood after model fit.V.1.2
                                }
				double increment_BIC = get_Increment(LL, LL_old, gene->eSNP, k, N, PERM);
                                //printf("incrmt = %f\n",increment_BIC);

				if(LOGISTIC_DEBUG_LVL>=1)
				{
                                    if(!PERM)
                                    {
					print_model(LG->curr_model,LG->curr_model_size,LL,increment_BIC);
                                        printf("\n");
                                    }
                                    else if(VERBOSE)
                                    {
                                        printf("PERM -> \n");
					print_model(LG->curr_model,LG->curr_model_size,LL,increment_BIC);
                                        printf("\n");
                                    }
				}
				if(increment_BIC > max_increment)
				{
					max_increment = increment_BIC;
					LL_best = LL;
					best_SNP = snp;
				}
			}
		        else if(LOGISTIC_DEBUG_LVL>=1 && VERBOSE)
                        {
                           if(!PERM)
			      printf("Skipping for corr %s\n",snp->name);
                           else
			      printf("PERM : Skipping for corr %s\n",snp->name);
                        }
			snp = (SNP*) cq_getNext(snp, snp_queue);
		}

                if(LOGISTIC_DEBUG_LVL>=1)
                   printf("max_increment = %f\n",max_increment);

		if(max_increment>0 || (k==1 && max_increment > GSL_NEGINF))
		{
                        if(max_increment>0)
                        {
			   LG->curr_model[k-1 + n_covariates] = best_SNP;
			   LL_old = LL_best;
			   LG->curr_model_size = k + n_covariates;

			   //update BIC state.
			   bic_state->BIC[k] = max_increment + bic_state->BIC[k-1];
                           bic_state->LL[k] = LL_best;
 
                           if(LOGISTIC_DEBUG_LVL>=2)
                              printf("Updating %f = %f + %f for gene = %s\n",bic_state->BIC[k],max_increment,bic_state->BIC[k-1],gene->name);
			   bic_state->bestSNP[k] = best_SNP; //best snp.
			   bic_state->iSNP = k;
                        }
                        else //for case (k==1 && max_increment > GSL_NEGINF)
                        {
                           //retain this just for printing the SUMMARY.
                           bic_state->BIC[k] = max_increment;
                           bic_state->LL[k] = LL_best;
                           break;
                        }

			if(LOGISTIC_DEBUG_LVL>=1)
			{
                             if(!PERM)
                             { 
                                printf("Model size = %d\n",LG->curr_model_size);
				printf("Best model for k = %d with BIC = %g\n",k,bic_state->BIC[k]);
				printf("Obtained by adding %s  (inc=%g and old_bic=%g) \n",best_SNP->name,max_increment,bic_state->BIC[k-1]);
				print_model_1(LG->curr_model,LG->curr_model_size);
                             }
                             else if(VERBOSE)
                             {                      
                                printf("PERM : Model size = %d\n",LG->curr_model_size);
				printf("PERM : Best model for k = %d with BIC = %g\n",k,bic_state->BIC[k]);
				printf("PERM : Obtained by adding %s  (inc=%g and old_bic=%g) \n",best_SNP->name,max_increment,bic_state->BIC[k-1]);
				print_model_1(LG->curr_model,LG->curr_model_size);
                             }
			}
		}
		else //no positive increment in BIC, stop iterating.
		{
                        //go back to old model size.
			LG->curr_model_size = curr_model_size;
			if(LOGISTIC_DEBUG_LVL>=1)
			{
                           if(!PERM)
				printf("No %d snp model found\n",k);
                           else if(VERBOSE)
				printf("PERM : No %d snp model found\n",k);
			}
			break;
		}
	}

        fflush(stdout);
	if(fp_result != NULL)
	{
		for(k=1; k<=bic_state->iSNP; k++)
                {
		    fprintf(fp_result, "%d\t%s\t%s\t%d\t%d\t%d\t%d\t%g\t%s\t%d\t%g\t%g\t%d\t%g\t%g\t-\t-\t-\n", 
                                  gene->chr, 
                                  gene->ccds, 
                                  gene->name, 
                                  gene->bp_start, 
                                  gene->bp_end, 
                                  gene->bp_end-gene->bp_start+1, 
                                  gene->nSNP, 
                                  gene->eSNP, 
                                  bic_state->bestSNP[k]->name,
                                  bic_state->bestSNP[k]->bp, 
                                  bic_state->bestSNP[k]->MAF,
                                  bic_state->bestSNP[k]->R2, //imputation quality of the snp
                                  k, 
                                  bic_state->BIC[k], 
                                  2*(bic_state->LL[k] - logistic_intercept_LL));//lrt
                }

		fprintf(fp_result, "%d\t%s\t%s\t%d\t%d\t%d\t%d\t%g\t%s\t%s\t%s\t%s\t%d\t%g\t%g\t-\t-\t-\n", 
                                   gene->chr, 
                                   gene->ccds, 
                                   gene->name, 
                                   gene->bp_start, 
                                   gene->bp_end, 
                                   gene->bp_end-gene->bp_start+1, 
                                   gene->nSNP, 
                                   gene->eSNP,
                                   "SUMMARY", 
                                   "-", //bp
                                   "-", //maf
                                   "-", //R2
                                   bic_state->iSNP, //k
                                   bic_state->iSNP>0 ? bic_state->BIC[bic_state->iSNP]:bic_state->BIC[1],//bic
                                   bic_state->iSNP>0 ? 2*(bic_state->LL[bic_state->iSNP] - logistic_intercept_LL) : 2*(bic_state->LL[1] - logistic_intercept_LL) ); //lrt chi2.                   
	}

        //restore default error handler.
        gsl_set_error_handler (old_handler);
  
	return bic_state;
}

