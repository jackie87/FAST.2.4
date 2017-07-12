/*
 *  * BFLogistic.c
 *   *
 *    *  Created on: Jul 8, 2010
 *     *  Updated on July 26, 2010
 *      *      Author: pchanda
 *       */

#include "GWiS.h"
//static clock_t time_sum = 0;
//static int cnt = 0;

double objfun_and_grad(BG_SCRATCH * BG, const gsl_vector * W)
{
     //compute the objective function and gradient simultanelously
     //setting p = 0 will compute the objective function under the null.
     //s1 = 1./(1+exp(w'*x)).
     int p = BG->curr_model_size; //exclude the intercept.
     int N = BG->phenotype->N_sample;
     double g_0 = 0;
     double LL = 0;
     int i,j;
     for(i=0;i<N;i++)
     {
       double o = gsl_vector_get(BG->OMIT,i);
       if(o==0) //no missing data in this sample.
       {
         double pheno = gsl_vector_get(BG->phenotype->pheno_array_log,i);
         double w_x = gsl_vector_get(W,0);
         for(j=0;j<p;j++)
         {
             double w_j = gsl_vector_get(W,j+1);
             SNP* snp_j = BG->curr_model[j];
             double x_j_i = snp_j->geno[i];
             w_x += w_j*x_j_i;
         }
         if(pheno<0)
         {
             BG->s1[i] = 1/(1 + exp(-w_x));
             g_0 -= BG->s1[i];
             LL += log(1 + exp(w_x));
         }
         else
         {
             BG->s1[i] = 1/(1 + exp(w_x));
             g_0 += BG->s1[i];
             LL += log(1 + exp(-w_x));
         }
       }
     }
     g_0 = -g_0;

     g_0 += gsl_vector_get(W,0)/gsl_vector_get(BG->nu,0); 
 
     //g = x*s1'.
     //g_0 is already calculated above as \Sigma_i (intercept*s[i]*y[i]).

     double W_0 = gsl_vector_get(W,0);
     double Sigma_sqr_0 = gsl_vector_get(BG->nu,0);
     double sum1 = ((p+1)/2.0)*log(2*M_PI) + 0.5*log(Sigma_sqr_0);
     double sum2 = pow(W_0,2)/Sigma_sqr_0;

     gsl_vector_set(BG->g,0,g_0);
     for(j=0;j<p;j++)
     {
        double g_j = 0;
        double W_j = gsl_vector_get(W,j+1);
        double Sigma_sqr_j = gsl_vector_get(BG->nu,j+1);
        sum1 += 0.5*log(Sigma_sqr_j);
        sum2 += pow(W_j,2)/Sigma_sqr_j;

        for(i=0;i<N;i++)
        {
          double o = gsl_vector_get(BG->OMIT,i);
          if(o==0) //no missing data in this sample.
          {
            double pheno = gsl_vector_get(BG->phenotype->pheno_array_log,i);
            SNP* snp_j = BG->curr_model[j];
            double x_j_i = snp_j->geno[i];
            if(pheno<0)
               g_j -= x_j_i*BG->s1[i];
            else
               g_j += x_j_i*BG->s1[i];
          }
        }
        g_j = -g_j;
        g_j += gsl_vector_get(W,j+1)/gsl_vector_get(BG->nu,j+1);
        gsl_vector_set(BG->g,j+1,g_j);
     }
     sum2 *= 0.5;
     //Now BG->g holds the new gradient of the objective function.
     return (LL + sum1 + sum2);
}
     
double my_f_BF (const gsl_vector *W, void *params)
{
       BG_SCRATCH *BG = (BG_SCRATCH *)params;
       int p = BG->curr_model_size+1;
       int i = 0;
       //clock_t st = clock();
       for(i=0;i<p;i++) gsl_vector_set(BG->W,i,gsl_vector_get(W,i));
       //double ofn = objfun(BG,W);
       //compute both objfun and grad.
       double ofn = objfun_and_grad(BG,W);
       //cnt++;
       //time_sum += clock() - st;
       //printf("f(%d) : %f\n",cnt,((double)(time_sum))/CLOCKS_PER_SEC);
       return ofn; //neg of log like is convex to be minimized.
}
     
/* The gradient of f*/
void my_df_BF (const gsl_vector *W, void *params, gsl_vector *df)
{
       BG_SCRATCH *BG = (BG_SCRATCH *)params;
       int p = BG->curr_model_size+1;
       int i = 0;
       //grad(BG,W);
       //grad is already computed in my_f_BF, so just copy grad to df.
       for(i=0;i<p;i++) gsl_vector_set(df,i,gsl_vector_get(BG->g,i));
}
     
/* Compute both f and df together. */
void my_fdf_BF (const gsl_vector *x, void *params, double *f, gsl_vector *df) 
{
       BG_SCRATCH *BG = (BG_SCRATCH *)params;
       int p = BG->curr_model_size+1;
       int i = 0;
       for(i=0;i<p;i++) gsl_vector_set(BG->W,i,gsl_vector_get(x,i));
       *f = objfun_and_grad(BG,x);//will update BG->g.
       for(i=0;i<p;i++) gsl_vector_set(df,i,gsl_vector_get(BG->g,i));
} 

void compute_hessian_BF(BG_SCRATCH * BG)
{
   //computes the hessian matrix with the MAP solution.
   //use the s1.
   int N = BG->phenotype->N_sample;
   double a[N];
   int i = 0;
   int p = BG->curr_model_size;
   //a = s1.*(1-s1).
   for(i=0;i<N;i++)
   {
      double o = gsl_vector_get(BG->OMIT,i);
      if(o==0) //no missing data in this sample.
        a[i] = BG->s1[i]*(1-BG->s1[i]);
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
         if(j1==0 && j2==0) //diagonal element
         {
             for(i=0;i<N;i++)
             {
               double o = gsl_vector_get(BG->OMIT,i);
               if(o==0) //no missing data in this sample.
                 val += a[i];//FIX HESSIAN LATEST V.1.2
             }
             val += 1.0/gsl_vector_get(BG->nu,0);//FIX HESSIAN LATEST V.1.2
         }
         else if(j1==0 && j2>0) //off dianola element
         {
              for(i=0;i<N;i++)
              {
                 double o = gsl_vector_get(BG->OMIT,i);
                 if(o==0) //no missing data in this sample.
                 {
                    SNP* snp_j2 = BG->curr_model[j2-1];
                    double x_j2_i = snp_j2->geno[i];
                    val += a[i]*x_j2_i;
                 }
              }
         }
         else if(j1>0 && j2==0) //off diagonal element
         {
             for(i=0;i<N;i++)
             {
                 double o = gsl_vector_get(BG->OMIT,i);
                 if(o==0) //no missing data in this sample.
                 {
                   SNP* snp_j1 = BG->curr_model[j1-1];
                   double x_j1_i = snp_j1->geno[i];
                   val += a[i]*x_j1_i;
                 }
             }
             gsl_matrix_set(BG->h,j1,j2,val);
         }
         else //j1==j2 and both > 0, diagonal element
         {
             for(i=0;i<N;i++)
             {
                 double o = gsl_vector_get(BG->OMIT,i);
                 if(o==0) //no missing data in this sample.
                 {
                   SNP* snp_j1 = BG->curr_model[j1-1];
                   double x_j1_i = snp_j1->geno[i];
                   SNP* snp_j2 = BG->curr_model[j2-1];
                   double x_j2_i = snp_j2->geno[i];
                   val += a[i]*x_j1_i*x_j2_i;//FIX HESSIAN LATEST V.1.2
                 }
             }
             val += 1.0/gsl_vector_get(BG->nu,j1);//FIX HESSIAN LATEST V.1.2
         }
         gsl_matrix_set(BG->h,j1,j2,val);
      }
   }
}

double matrix_determinant(gsl_matrix* h)
{  
   int s = 0;
   int p = h->size1;
   gsl_permutation * P = gsl_permutation_alloc (p);
   gsl_linalg_LU_decomp (h, P, &s); //will destroy h.
   double det = gsl_linalg_LU_det(h,s);
   gsl_permutation_free (P);
   return det;
}

double matrix_determinant_2by2(gsl_matrix* h)
{
   int p1 = h->size1;
   int p2 = h->size2;
   if(p1!=2 && p2!=2) {printf("Error h is not 2 x 2\n");abort();}
   double h00 = gsl_matrix_get(h,0,0);
   double h01 = gsl_matrix_get(h,0,1);
   double h10 = gsl_matrix_get(h,1,0);
   double h11 = gsl_matrix_get(h,1,1);
   return (h00*h11 - h01*h10);
}

void minimize_BF(BG_SCRATCH * BG, gsl_multimin_function_fdf * my_func)
{
    //time_sum = 0;
    //cnt = 0;
    //clock_t st = clock();
    gsl_vector_set_zero (BG->x);//Initialize all weights to 0 as starting point.
    my_func->n = BG->curr_model_size + 1; //intercept + single snp.
    gsl_vector_view x = gsl_vector_subvector(BG->x, 0, my_func->n);
    gsl_multimin_fdfminimizer_set (BG->s, my_func, &x.vector, 0.01, 1e-1);
    //start iterations.
    int iter = 0;
    int status = 0;
    do
    {
        iter++;
        status = gsl_multimin_fdfminimizer_iterate (BG->s);
        //printf("%d status = %d\n",iter,status); 
        if (status)
            break;
        status = gsl_multimin_test_gradient (BG->s->gradient, 1e-1);
        if (status == GSL_SUCCESS)
        {
          //printf ("%d : Minimum found at: (%g %g)",iter,gsl_vector_get(BG->W,0),gsl_vector_get(BG->W,1)); 
          //printf ("Minimum found at iter %d\n",iter); 
        }
        //printf("Iter = %d : fr Obj = %g\n",iter,BG->s->f);
    } while (status == GSL_CONTINUE && iter < MAX_BF_LOGISTIC_ITER);

    if(status != GSL_SUCCESS && VERBOSE)
    {
      printf ("%d Warning : BF logistic minimization did not converge %d\n",iter,status); 
      printf ("%d W=(%g %g)\n",iter,gsl_vector_get(BG->W,0),gsl_vector_get(BG->W,1)); 
    }
    //clock_t fi = clock();
    //printf("Total per snp = %g\n",((double)(fi-st))/CLOCKS_PER_SEC);
}

void remove_missing_samples_bf(BG_SCRATCH* BG)
{
   //remove missing data if present.
   //populate a OMIT vector for each sample. OMIT[i] = 1 means to omit the sample.
   //In this way you need not change the original genotypes and phenotypes.
   int i = 0;
   int p = BG->curr_model_size; //exclude the intercept.
   int N = BG->phenotype->N_sample;
   gsl_vector_set_zero(BG->OMIT);

   //Check if any snp in the model has missing data.
   int j = 0;
   bool missing = false;
   for(j=0;j<p;j++)
   {
      SNP* snp_j = BG->curr_model[j];
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
        for(j=0;j<p;j++)
        {
            SNP* snp_j = BG->curr_model[j];
            double x_j_i = snp_j->geno[i];
            if(x_j_i==MISSING_VAL)
            {
               missing = true;
               gsl_vector_set(BG->OMIT,i,1);
               break;
            }
        }
   }
}

double runBFLogisticSNP_fr(SNP* snp, BG_SCRATCH* BG, bool perm)
{
     //printf("-Using runBFLogisticSNP_fr\n");
     //printf("-------------------------------\n");

     gsl_vector_set_zero(BG->nu);
     int i = 0;
     for(i=0;i<BG->n_covariates+1;i++) //for intercept + covariates.
     {
       gsl_vector_set(BG->nu,i,SIGMA_U*SIGMA_U);
     }     
     gsl_vector_set(BG->nu,i,SIGMA_A*SIGMA_A); //for the snp

     double log_numerator = 0;
     //first alternative model
     {
       if(VERBOSE)
         printf("Computing alt model for %s\n",snp->name);
       BG->curr_model[BG->n_covariates] = snp; 
       BG->curr_model_size = BG->n_covariates+1;

       gsl_multimin_function_fdf my_func;
       my_func.f = my_f_BF;
       my_func.df = my_df_BF;
       my_func.fdf = my_fdf_BF;
       my_func.params = BG;

       BG->s = gsl_multimin_fdfminimizer_alloc(BG->T, BG->n_covariates+2);
  
       if(snp->missingness > 0) 
          remove_missing_samples_bf(BG);
       minimize_BF(BG, &my_func);
       double objval = BG->s->f; //objective function after model fit at MAP.
       compute_hessian_BF(BG); //compute the hessian at MAP.
       gsl_matrix_view h = gsl_matrix_submatrix(BG->h,0,0,BG->n_covariates+2,BG->n_covariates+2);
       double det = 0;
       if(BG->n_covariates==0)
          det = fabs(matrix_determinant_2by2(&h.matrix));
       else
          det = fabs(matrix_determinant(&h.matrix));
       log_numerator = ((BG->n_covariates+2)/2.0)*log(2*M_PI) - 0.5*log(det) - objval; //FIX HESSIAN LATEST V.1.2 
       gsl_multimin_fdfminimizer_free(BG->s);
       BG->s = NULL;
       if(VERBOSE)
         printf("FR : %s log numerator = %g\n",snp->name,log_numerator);
     }

     static double bf_intercept = GSL_POSINF;
     double log_denominator = 0;
 
     //next null model
     {
        bool compute_intercept = true;
        if(!perm || BG->n_covariates==0)
        {
          //if no covariates, the intercept remains same for real trait and permutations for all genes. So compute it just once.
          //if covariates are present, the intercept needs to be computed once for real trait for all genes.
          if(bf_intercept==GSL_POSINF)
          {
             compute_intercept = true;
          }
          else
          {
             compute_intercept = false;
          }
        }
        else //perm==true and n_covariates > 0, need to compute intercept for each shuffled trait.
        {
           //With permutations, need to compute intercept for each shuffled trait if covariates are present.
           //printf("-PERM : Recomputing BF logistic intercept as covariates are present\n");
           compute_intercept = true;
        }
        
        if(compute_intercept==true)
        {
           //printf("-Computing BF logistic intercept\n");
           //printf("Computing null model for %s\n",snp->name);
           BG->curr_model_size = BG->n_covariates;

           gsl_multimin_function_fdf my_func;
           my_func.f = my_f_BF;
           my_func.df = my_df_BF;
           my_func.fdf = my_fdf_BF;
           my_func.params = BG;

           BG->s = gsl_multimin_fdfminimizer_alloc(BG->T, BG->n_covariates+1);
           minimize_BF(BG, &my_func);
           double objval = BG->s->f; //objective function after model fit at MAP.
           compute_hessian_BF(BG); //compute the hessian at MAP.
           double det = 0;
           if(BG->n_covariates==0)
             det = fabs(gsl_matrix_get(BG->h,0,0));
           else
           {
             gsl_matrix_view h = gsl_matrix_submatrix(BG->h,0,0,BG->n_covariates+1,BG->n_covariates+1);
             det = fabs(matrix_determinant(&h.matrix));
           }
           //log_denominator = objval + M_LNPI - 0.5*log(det);
           log_denominator = ((BG->n_covariates+1)/2.0)*log(2*M_PI) - 0.5*log(det) - objval;//FIX HESSIAN LATEST V.1.2
           if(!perm || BG->n_covariates==0)
           {
              bf_intercept = log_denominator;//store.
           } 
           gsl_multimin_fdfminimizer_free(BG->s);
           BG->s = NULL;
           //printf("FR : %s log denominator = %g\n",snp->name,log_denominator);
        }
        else
        {
           //printf("Using Precomputed logistic BF intercept = %g\n",bf_intercept);
           log_denominator = bf_intercept;
        }
     }
     //printf("-------------------------------\n");
     double bayes = (log_numerator - log_denominator)/log(10);//FIX HESSIAN LATEST V.1.2
     //printf("Got BAYES = %lg\n\n",bayes); 
     if(VERBOSE)
     {
       if(!perm)
          printf("Got BAYES(log10) = %lg\n",bayes); 
       //else 
       //   printf("Perm : Got BAYES = %lg\n",bayes); 
     }
     return bayes; //ln bayes factor.
}
