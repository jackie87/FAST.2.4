/*
 *  * BFLogistic_newton.c
 *   *
 *    *  Created on: Jul 8, 2010
 *     *  Updated on July 26, 2010
 *      *      Author: pchanda
 *       */

#include "GWiS.h"
static bool SINGULAR_BF = false;

double loglikelihood_bf_nw(BG_SCRATCH* BG, const gsl_vector* Beta)
{
   //objective function of the alternaitve model to be maximized.
   //loglik + P(Beta) = -LL + P(Beta)
   //compute the -LL = loglik.
   //setting p = n_covariates will compute the objective function under the null.
   double LL = 0;
   int p = BG->curr_model_size;
   //printf("objfun : p = %d\n",p); 
   int i = 0;
   int N = BG->phenotype->N_sample;
   for(i=0;i<N;i++)
   {
     double o = gsl_vector_get(BG->OMIT,i);
     if(o==0) //no missing data in this sample.
     {
        double w_x = gsl_vector_get(Beta,0); //intercept * Beta[0] = Beta[0].
        int j = 0;
        double pheno = gsl_vector_get(BG->phenotype->pheno_array_log,i);
        for(j=0;j<p;j++)
        {
            double w_j = gsl_vector_get(Beta,j+1);
            SNP* snp_j = BG->curr_model[j];
            double x_j_i = snp_j->geno[i];
            w_x += w_j*x_j_i;
        }
        if(pheno<0)
                LL += log(1 + exp(w_x));
        else
                LL += log(1 + exp(-w_x));
     }
   }
   //compute P(Beta)
   int j = 0;
   double Beta_0 = gsl_vector_get(Beta,0);
   double Sigma_sqr_0 = gsl_vector_get(BG->nu,0);
   double sum1 = ((p+1)/2.0)*log(2*M_PI) + 0.5*log(Sigma_sqr_0);
   double sum2 = pow(Beta_0,2)/Sigma_sqr_0;

   for(j=0;j<p;j++)
   {
       double Beta_j = gsl_vector_get(Beta,j+1);
       double Sigma_sqr_j = gsl_vector_get(BG->nu,j+1);
       sum1 += 0.5*log(Sigma_sqr_j);
       sum2 += pow(Beta_j,2)/Sigma_sqr_j;
   }
   sum2 *= 0.5;    
   //printf("objfun = %g\n",(LL + sum1 + sum2));    
   //return -(LL + sum1 + sum2); 
   return (LL + sum1 + sum2); 
}

double matrix_determinant_bf_nw(gsl_matrix* h)
{ 
   int s = 0;
   int p = h->size1;
   gsl_permutation * P = gsl_permutation_alloc (p);
   gsl_linalg_LU_decomp (h, P, &s); //will destroy h.
   double det = gsl_linalg_LU_det(h,s);
   gsl_permutation_free (P);
   return det;
}

double matrix_determinant_2by2_bf_nw(gsl_matrix* h)
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

bool stop_iter_bf_nw(gsl_vector* W, gsl_vector* old_W)
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

int solve_bf_nw(gsl_matrix* h, gsl_vector* g, gsl_vector* x)
{
   // x = h\g is the solution to the equation hx = g.
   int s;
   int p = h->size1;
   int err = 0;
   if(p==1)
   {
     printf("Logistic BF : Error in solve, size of matrix should be > 1\n");
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

void error_handler_bf_nw(const char * reason,
                        const char * file,
                        int line,
                        int gsl_errno)
{
  //printf("-FAST error handler invoked for Bayes Factor with Logistic Regression\n");
  //printf("-reason = %s\n",reason);
  //printf("-file = %s\n",file);
  //printf("-line = %d\n",line);
  //printf("-errno = %d\n",gsl_errno);
  SINGULAR_BF = true;
}

double gradient_descent_bf_nw(BG_SCRATCH* BG)
{
        int N = BG->phenotype->N_sample;
        SINGULAR_BF = false;
	//phenotype is +/- 1.
	int p = BG->curr_model_size; // count of snps in the model, intercept is an extra dimension.

	//Initialize scratch space.
	gsl_vector_view g = gsl_vector_subvector(BG->g, 0, p+1);
	gsl_vector_view W = gsl_vector_subvector(BG->W, 0, p+1);
	gsl_vector_view old_W = gsl_vector_subvector(BG->old_W, 0, p+1);
	gsl_vector_view W_delta = gsl_vector_subvector(BG->W_delta, 0, p+1);
	gsl_matrix_view h = gsl_matrix_submatrix(BG->h,0,0,p+1,p+1);
	gsl_matrix_view h_temp = gsl_matrix_submatrix(BG->h_temp,0,0,p+1,p+1);

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

        bool flag = false;
	int iter = 0;
	for(iter=0;iter<MAX_BF_LOGISTIC_ITER;iter++)
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
              double o = gsl_vector_get(BG->OMIT,i);
              if(o==0) //no missing data in this sample.
              { 
		  double pheno = gsl_vector_get(BG->phenotype->pheno_array_log,i);
		  double w_x = gsl_vector_get(&W.vector,0);
		  for(j=0;j<p;j++)
		  {
			 double w_j = gsl_vector_get(&W.vector,j+1);
			 SNP* snp_j = BG->curr_model[j];
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
           //g_0 -= gsl_vector_get(&W.vector,0)/gsl_vector_get(BG->nu,0);
          
           g_0 = -g_0; 
           g_0 += gsl_vector_get(&W.vector,0)/gsl_vector_get(BG->nu,0);

           //g = x*s1'.
	   //g_0 is already calculated above as \Sigma_i (intercept*s[i]*y[i]).
	   gsl_vector_set(&g.vector,0,g_0);
           for(j=0;j<p;j++)
           {
             double g_j = 0;
             for(i=0;i<N;i++)
             {
               double o = gsl_vector_get(BG->OMIT,i);
               if(o==0) //no missing data in this sample.
               { 
        	  double pheno = gsl_vector_get(BG->phenotype->pheno_array_log,i);
 		  SNP* snp_j = BG->curr_model[j];
 		  double x_j_i = snp_j->geno[i];
 		  if(pheno<0)
 			  g_j -= x_j_i*s1[i];
 		  else
 			 g_j += x_j_i*s1[i];
               }
             }
             //g_j -= gsl_vector_get(&W.vector,j+1)/gsl_vector_get(BG->nu,j+1);

             g_j = -g_j;
             g_j += gsl_vector_get(&W.vector,j+1)/gsl_vector_get(BG->nu,j+1);
             gsl_vector_set(&g.vector,j+1,g_j);
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
                  double o = gsl_vector_get(BG->OMIT,i);
                  if(o==0)
		     val += a[i];//FIX HESSIAN LATEST V.1.2
		}
		val += 1.0/gsl_vector_get(BG->nu,0);//FIX HESSIAN LATEST V.1.2
              }
              else if(j1==0 && j2>0)
              {
		 for(i=0;i<N;i++)
		 {
                    double o = gsl_vector_get(BG->OMIT,i);
                    if(o==0)
                    {
			 SNP* snp_j2 = BG->curr_model[j2-1];
			 double x_j2_i = snp_j2->geno[i];
			 val += a[i]*x_j2_i;
                    }
		 }
              }
              else if(j1>0 && j2==0)
              {
		 for(i=0;i<N;i++)
		 {
                    double o = gsl_vector_get(BG->OMIT,i);
                    if(o==0)
                    {
			 SNP* snp_j1 = BG->curr_model[j1-1];
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
                    double o = gsl_vector_get(BG->OMIT,i);
                    if(o==0)
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
              gsl_matrix_set(&h.matrix,j1,j2,val);
           }
         }

         //to do h\g.
         // X = A\B is the solution to the equation AX = B.
         //So solve : h W_delta = g.
         //So W_delta = inv(h)g.
         int err = 0;
         if(p>0)
         {
           gsl_matrix_memcpy (&h_temp.matrix, &h.matrix); //create a copy of h for solve to overwrite.
           err = solve_bf_nw(&h_temp.matrix,&g.vector,&W_delta.vector);
         }
         else //p=0
         {
           gsl_vector_set(&W_delta.vector,p,g_0/gsl_matrix_get(&h.matrix,0,0));
         }
    	 //printf(" del W = "); gsl_vector_fprintf (stdout, &W_delta.vector, "%g");
         //w = w + h\g.
         gsl_vector_sub(&W.vector,&W_delta.vector);
  	 //printf("\n W = "); gsl_vector_fprintf (stdout, &W.vector, "%g");

    	 //compute negative loglikelihood.
         //double LL = loglikelihood_bf_nw(BG,&W.vector);
    	 //printf("Iter = %d  : nw  LL = %.5f\n",iter,LL);
    	 //printf("==========================================\n");

         //if max(abs(w - old_w)) < 1e-8, break.
         flag = stop_iter_bf_nw(&W.vector,&old_W.vector);

         if(flag || err>0 || SINGULAR_BF)
         {
           SINGULAR_BF = false;
           break;
         }
       }//for each gradient descent iteration.

       //if(!flag) printf("Did not converge\n");
       
       double LL = loglikelihood_bf_nw(BG,&W.vector);

       //printf("h = \n");
       //print2(&h.matrix);
       return LL;
}

void remove_missing_samples_bf_nw(BG_SCRATCH* BG)
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

double runBFLogisticSNP_newton(SNP* snp, BG_SCRATCH* BG, bool perm)
{
     //printf("Using runBFLogisticSNP_newton\n");
     gsl_error_handler_t* old_handler = gsl_set_error_handler (&error_handler_bf_nw);
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
       //printf("Computing alt model for %s\n",snp->name);
       BG->curr_model[BG->n_covariates] = snp; 
       BG->curr_model_size = BG->n_covariates+1;
       if(snp->missingness > 0) 
          remove_missing_samples_bf_nw(BG);
       double objval = gradient_descent_bf_nw(BG); //objective function after model fit at MAP.
       //printf("objval = %g\n",LL);
       gsl_matrix_view h = gsl_matrix_submatrix(BG->h,0,0,BG->n_covariates+2,BG->n_covariates+2);
       double det = 0;
       if(BG->n_covariates==0)
          det = fabs(matrix_determinant_2by2_bf_nw(&h.matrix));
       else
          det = fabs(matrix_determinant_bf_nw(&h.matrix));
       log_numerator = ((BG->n_covariates+2)/2.0)*log(2*M_PI) - 0.5*log(det) - objval; //V.1.2
       //printf("NW : %s log numerator = %g\n",snp->name,log_numerator);
     }

     double log_denominator = 0;
     static double bf_intercept_nw = GSL_POSINF;
     //next null model
     {
        bool compute_intercept = true;
        if(!perm || BG->n_covariates==0)
        {
          //if no covariates, the intercept remains same for real trait and permutations for all genes. So compute it just once.
          //if covariates are present, the intercept needs to be computed once for real trait for all genes.
          if(bf_intercept_nw==GSL_POSINF)
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
          //printf("Computing null model for %s\n",snp->name);
          BG->curr_model_size = BG->n_covariates;
          double objval = gradient_descent_bf_nw(BG); //objective function after model fit at MAP.
          double det = 0;
          if(BG->n_covariates==0)
            det = fabs(gsl_matrix_get(BG->h,0,0));
          else
          {
            gsl_matrix_view h = gsl_matrix_submatrix(BG->h,0,0,BG->n_covariates+1,BG->n_covariates+1);
            det = fabs(matrix_determinant_bf_nw(&h.matrix));
          }
          log_denominator = ((BG->n_covariates+1)/2.0)*log(2*M_PI) - 0.5*log(det) - objval; //V.1.2
          if(!perm || BG->n_covariates==0)
          {
             bf_intercept_nw = log_denominator;
          }
          //printf("NW : %s log denominator = %g\n",snp->name,log_denominator);
       }
       else
       {
          //printf("Using Precomputed logistic BF intercept = %g\n",bf_intercept_nw);
          log_denominator = bf_intercept_nw;
       }
     }
     //printf("-------------------------------\n");
     double bayes = (log_numerator - log_denominator)/log(10); //V.1.2
     if(VERBOSE)
     {
       if(!perm)
          printf("Got BAYES = %lg\n\n",bayes); 
       //else 
       //   printf("Perm : Got BAYES = %lg\n",bayes); 
     }
     gsl_set_error_handler (old_handler);
     return bayes; //ln bayes factor.
}
