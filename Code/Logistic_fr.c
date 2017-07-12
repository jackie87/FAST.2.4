/*
 *  * Logistic.c
 *   *
 *    *  Created on: Jul 8, 2010
 *     *  Updated on July 26, 2010
 *      *      Author: pchanda
 *       */

#include "GWiS.h"
bool PERM = false;

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

double loglikelihood(LOGISTIC_SCRATCH* LG, const gsl_vector* W)
{
   //compute the log likelihood.
   double LL = 0;
   int p = LG->curr_model_size; //exclude the intercept.
   int i = 0;
   int N = LG->phenotype->N_sample;
   for(i=0;i<N;i++)
   {
      double o = gsl_vector_get(LG->OMIT,i);
      if(o==0) //no missing data in this sample.
      {
        double w_x = gsl_vector_get(W,0); //intercept * w[0] = w[0].
        int j = 0;
        double pheno = gsl_vector_get(LG->phenotype->pheno_array_log,i);
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
   return -LL; //return the log likehood, a negative number.
}

void gradient(LOGISTIC_SCRATCH * LG, const gsl_vector * W)
{
     //s1 = 1./(1+exp(w'*x)).
     int p = LG->curr_model_size; //exclude the intercept.
     int N = LG->phenotype->N_sample;
     double g_0 = 0;
     int i,j;
     for(i=0;i<N;i++)
     {
       double o = gsl_vector_get(LG->OMIT,i);
       if(o==0) //no missing data in this sample.
       {
         double pheno = gsl_vector_get(LG->phenotype->pheno_array_log,i);
         double w_x = gsl_vector_get(W,0);
         for(j=0;j<p;j++)
         {
             double w_j = gsl_vector_get(W,j+1);
             SNP* snp_j = LG->curr_model[j];
             double x_j_i = snp_j->geno[i];
             w_x += w_j*x_j_i;
         }
         if(pheno<0)
         {
             LG->s1[i] = 1/(1 + exp(-w_x));
             g_0 -= LG->s1[i];
         }
         else
         {
             LG->s1[i] = 1/(1 + exp(w_x));
             g_0 += LG->s1[i];
         }
       }
     }
     //g = x*s1'.
     //g_0 is already calculated above as \Sigma_i (intercept*s[i]*y[i]).
     gsl_vector_set(LG->g,0,-g_0); //made -ve as we are minimizing.
     for(j=0;j<p;j++)
     {
        double g_j = 0;
        for(i=0;i<N;i++)
        {
          double o = gsl_vector_get(LG->OMIT,i);
          if(o==0) //no missing data in this sample.
          {
            double pheno = gsl_vector_get(LG->phenotype->pheno_array_log,i);
            SNP* snp_j = LG->curr_model[j];
            double x_j_i = snp_j->geno[i];
            if(pheno<0)
               g_j -= x_j_i*LG->s1[i];
            else
               g_j += x_j_i*LG->s1[i];
          }
        }
        gsl_vector_set(LG->g,j+1,-g_j);//made -ve as we are minimizing.
     }
     //Now LG->g holds the new gradient.
}

double ll_and_gradient(LOGISTIC_SCRATCH * LG, const gsl_vector * W)
{
     //s1 = 1./(1+exp(w'*x)).
     int p = LG->curr_model_size; //exclude the intercept.
     int N = LG->phenotype->N_sample;
     double g_0 = 0;
     double LL = 0;
     int i,j;
     for(i=0;i<N;i++)
     {
       double o = gsl_vector_get(LG->OMIT,i);
       if(o==0) //no missing data in this sample.
       {
         double pheno = gsl_vector_get(LG->phenotype->pheno_array_log,i);
         double w_x = gsl_vector_get(W,0);
         for(j=0;j<p;j++)
         {
             double w_j = gsl_vector_get(W,j+1);
             SNP* snp_j = LG->curr_model[j];
             double x_j_i = snp_j->geno[i];
             w_x += w_j*x_j_i;
         }
         if(pheno<0)
         {
             LL += log(1 + exp(w_x));
             LG->s1[i] = 1/(1 + exp(-w_x));
             g_0 -= LG->s1[i];
         }
         else
         {
             LL += log(1 + exp(-w_x));
             LG->s1[i] = 1/(1 + exp(w_x));
             g_0 += LG->s1[i];
         }
       }
     }
     //g = x*s1'.
     //g_0 is already calculated above as \Sigma_i (intercept*s[i]*y[i]).
     gsl_vector_set(LG->g,0,-g_0); //made -ve as we are minimizing.
     for(j=0;j<p;j++)
     {
        double g_j = 0;
        for(i=0;i<N;i++)
        {
          double o = gsl_vector_get(LG->OMIT,i);
          if(o==0) //no missing data in this sample.
          {
            double pheno = gsl_vector_get(LG->phenotype->pheno_array_log,i);
            SNP* snp_j = LG->curr_model[j];
            double x_j_i = snp_j->geno[i];
            if(pheno<0)
               g_j -= x_j_i*LG->s1[i];
            else
               g_j += x_j_i*LG->s1[i];
          }
        }
        gsl_vector_set(LG->g,j+1,-g_j);//made -ve as we are minimizing.
     }
     //Now LG->g holds the new gradient.
     return -LL; //return the log likehood, a negative number.
}
     
double my_f (const gsl_vector *W, void *params)
{
       LOGISTIC_SCRATCH *LG = (LOGISTIC_SCRATCH *)params;

       int p = LG->curr_model_size+1;
       int i = 0;
       //LG->W = W;
       for(i=0;i<p;i++) gsl_vector_set(LG->W,i,gsl_vector_get(W,i));
       //printf("f : W = %f %f %f\n",gsl_vector_get(W,0),gsl_vector_get(W,1),gsl_vector_get(W,2));
       double LL = loglikelihood(LG,W);
       //printf("f = %f\n-------\n",-LL);
       return -LL; //neg of log like is convex to be minimized.
}
     
/* The gradient of f*/
void my_df (const gsl_vector *W, void *params, gsl_vector *df)
{
       LOGISTIC_SCRATCH *LG = (LOGISTIC_SCRATCH *)params;
       int p = LG->curr_model_size+1;
       int i = 0;
       for(i=0;i<p;i++) gsl_vector_set(LG->W,i,gsl_vector_get(W,i));
       //printf("df : W = %f %f %f\n",gsl_vector_get(W,0),gsl_vector_get(W,1),gsl_vector_get(W,2));
       gradient(LG,W);
       //df = LG->g;
       for(i=0;i<p;i++) gsl_vector_set(df,i,gsl_vector_get(LG->g,i));
       //printf("df : g = %f %f %f\n-------\n",gsl_vector_get(df,0),gsl_vector_get(df,1),gsl_vector_get(df,2));
}
     
/* Compute both f and df together. */
void my_fdf (const gsl_vector *x, void *params, double *f, gsl_vector *df) 
{
       //old.
       //*f = my_f(x, params); 
       //my_df(x, params, df);

       //new
       LOGISTIC_SCRATCH *LG = (LOGISTIC_SCRATCH *)params;
       int p = LG->curr_model_size+1;
       int i = 0;
       for(i=0;i<p;i++) gsl_vector_set(LG->W,i,gsl_vector_get(x,i));
       //printf("fdf : W = %f %f %f\n",gsl_vector_get(x,0),gsl_vector_get(x,1),gsl_vector_get(x,2));
       *f = -ll_and_gradient(LG,x);//will update LG->g.
       for(i=0;i<p;i++) gsl_vector_set(df,i,gsl_vector_get(LG->g,i));
       //printf("fdf -LL = %f\n-------\n",*f);
       //printf("fdf : g = %f %f %f\n-------\n",gsl_vector_get(df,0),gsl_vector_get(df,1),gsl_vector_get(df,2));
} 

//get the change of GWiS statistics when adding a SNP into the model
double get_Increment(double LL_new, double LL_old, double eSNP, int k, int nsample)
{
        //the change in logProb from k-1 to k = BIC(k-1) - BIC(k).
        double increment = LL_old - LL_new;
        //printf("Get inc : %f %f\n",LL_old,LL_new);

        increment += log(eSNP-k+1) - log(k);
        increment += log(nsample)/2;

        if(LOGISTIC_DEBUG_LVL>=1)
        {
                //commented because gsl_sf_lnbeta sometimes throws error!!!
                //double BIC_abs = LL_new - (k/2.0)*log(nsample) + gsl_sf_lnbeta (k+1,eSNP-k+1);
                //printf("BIC for (k=%d, eSNP=%g, L = %g) = %g\n",k,eSNP,LL_new,BIC_abs);
                if(!PERM)
                    printf("Increment = %g\n",-increment);
                else if(VERBOSE_L)
                    printf("PERM : Increment = %g\n",-increment);
        }
        return -increment;
}

double loglik_0()
{
  double N = count_1+count_2;
  printf("Log lik for model 0 : counts = %d %d %g\n",count_1,count_2,N);
  return count_1*log(count_1/N) + count_2*log(count_2/N); //log likelihood, a negative number.
}

void compute_hessian(LOGISTIC_SCRATCH * LG)
{
   //computes the hessian matrix with the MLE solution.
   //use the s1.
   int N = LG->phenotype->N_sample;
   double a[N];
   int i = 0;
   int p = LG->curr_model_size;
   //a = s1.*(1-s1).
   for(i=0;i<N;i++)
   {
     double o = gsl_vector_get(LG->OMIT,i);
     if(o==0) //no missing data in this sample.
       a[i] = LG->s1[i]*(1-LG->s1[i]);
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
                if(o==0) //no missing data in this sample.
                  val += a[i];
             }
         }
         else if(j1==0 && j2>0)
         {
              for(i=0;i<N;i++)
              {
                 double o = gsl_vector_get(LG->OMIT,i);
                 if(o==0) //no missing data in this sample.
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
                 if(o==0) //no missing data in this sample.
                 {
                   SNP* snp_j1 = LG->curr_model[j1-1];
                   double x_j1_i = snp_j1->geno[i];
                   val += a[i]*x_j1_i;
                 }
             }
             gsl_matrix_set(LG->h,j1,j2,val);
         }
         else
         {
             for(i=0;i<N;i++)
             {
                 double o = gsl_vector_get(LG->OMIT,i);
                 if(o==0) //no missing data in this sample.
                 {
                   SNP* snp_j1 = LG->curr_model[j1-1];
                   double x_j1_i = snp_j1->geno[i];
                   SNP* snp_j2 = LG->curr_model[j2-1];
                   double x_j2_i = snp_j2->geno[i];
                   val += a[i]*x_j1_i*x_j2_i;
                 }
             }
         }
         gsl_matrix_set(LG->h,j1,j2,val);
      }
   }
}

void get_wald_statistic(LOGISTIC_SCRATCH* LG,SNP* snp, double LL, bool perm)
{
   if(LOGISTIC_DEBUG_LVL>=3 && !perm)
      printf("Computing wald \n");
   int s;
   int p = LG->curr_model_size; // count of covariates + 1 snp in the model, intercept is an extra dimension.
   gsl_matrix_view h = gsl_matrix_submatrix(LG->h,0,0,p+1,p+1);
   gsl_matrix_view hinv = gsl_matrix_submatrix(LG->h_inv,0,0,p+1,p+1);
   gsl_permutation * P = gsl_permutation_alloc (p+1);
   gsl_vector_view W = gsl_vector_subvector(LG->W, 0, p+1);

   //gsl_set_error_handler_off(); //set error handler off
   if(LOGISTIC_DEBUG_LVL>=3)
   {
      printf("h = \n");
   }

   //inverting the hessian to compute variance/covariance matrix.
   int err = gsl_linalg_LU_decomp (&h.matrix, P, &s);
   if(err==0)
   {
      if(LOGISTIC_DEBUG_LVL>=3 && !perm)
            printf("Successful LU decomposition \n");
      err = gsl_linalg_LU_invert (&h.matrix, P, &hinv.matrix);
      if(err==0)
      {
         if(LOGISTIC_DEBUG_LVL>=3 && !perm)
            printf("Successful LU inversion \n");
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
            if(LOGISTIC_DEBUG_LVL>=3 && !perm)
               printf("(%d) %s se=%f beta=%f wald=%f\n",k,snp->name,LG->se[k],beta,LG->wald[k]);
         }
      }
   }
   if(err>0)
   {
      printf("Error in invering hessian for SNP = %s\n",snp->name);
      int k = 0;
      for(k=0;k<p+1;k++) //k=0 for snp1, k=1 for snp2, .... Intercept not stored.
      {
         LG->se[k] = GSL_NAN;
         LG->wald[k] = GSL_NAN;
      }
   }
   //gsl_set_error_handler(NULL); //restore error handler
   gsl_permutation_free (P);
   if(!perm)
   {
      snp->beta_logistic = gsl_vector_get(&W.vector,LG->n_covariates+1);
      snp->se_logistic = LG->se[LG->n_covariates+1];
      snp->wald = LG->wald[LG->n_covariates+1];
      snp->loglik_logistic = LL;
      if(gsl_isnan(snp->wald)==0)
         snp->pval_logistic =  gsl_cdf_chisq_Q(snp->wald,1);
      else
         snp->pval_logistic = 1.0;
   }
   else
   {
      snp->wald_perm = LG->wald[LG->n_covariates+1];
      snp->loglik_logistic_perm = LL;
   }
}

void minimize(LOGISTIC_SCRATCH * LG, gsl_multimin_function_fdf * my_func)
{
    gsl_vector_set_zero (LG->x);//Initialize all weights to 0 as starting point.
    my_func->n = LG->curr_model_size + 1; //intercept + single snp.
    gsl_vector_view x = gsl_vector_subvector(LG->x, 0, my_func->n);
    
    gsl_multimin_fdfminimizer_set (LG->s, my_func, &x.vector, 0.01, 1e-4);

    //start iterations.
    int iter = 0;
    int status = 0;
    do
    {
        iter++;
        status = gsl_multimin_fdfminimizer_iterate (LG->s);
        //printf("%d status = %d\n",iter,status); 
        if (status)
            break;
        status = gsl_multimin_test_gradient (LG->s->gradient, 1e-3);
        //if (status == GSL_SUCCESS)
        //  printf ("Minimum found at:\n");

    }while (status == GSL_CONTINUE && iter < MAX_LOGISTIC_ITER);
}

void remove_missing_samples(LOGISTIC_SCRATCH* LG)
{
   if(MISSING_DATA==false)
   {
      if(VERBOSE_L)
        printf("No missing data for logistic regression\n");
      return; //no missing data present in the entire data set. Either user specified or determined from missingness of the snps in genes.
   }
   if(VERBOSE_L)
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

void runLogisticSNP(SNP* snp, LOGISTIC_SCRATCH* LG, bool perm, bool need_hessian)
{
     //if(!perm)
     //   printf("runLogistic for %s\n",snp->name);fflush(stdout);
     //clock_t s1,e1,d;
     
     //s1 = clock();
     LG->curr_model[LG->n_covariates] = snp; //covariates are already added to the model if present.
     LG->curr_model_size = LG->n_covariates+1;

     gsl_multimin_function_fdf my_func;
     my_func.f = my_f;
     my_func.df = my_df;
     my_func.fdf = my_fdf;
     my_func.params = LG;

     LG->s = gsl_multimin_fdfminimizer_alloc(LG->T, LG->n_covariates + 2);
     if(snp->missingness>0)
        remove_missing_samples(LG);
     //e1 = clock();
     //d = e1 - s1;

     //clock_t s,e;
     //s = clock();
     minimize(LG, &my_func);
     //e = clock();
     //printf(" %s minimize took %g seconds\n",snp->name,((double)(e-s))/CLOCKS_PER_SEC);

     //s1 = clock();
     double LL = -LG->s->f; //log likelihood after model fit.
     if(perm)
        snp->loglik_logistic_perm = LL;
     if(need_hessian)
     {
       compute_hessian(LG); //compute the hessian.
       get_wald_statistic(LG,snp,LL,perm); //will compute the inverse of hessian and wald.
     }
     gsl_multimin_fdfminimizer_free(LG->s);
     LG->s = NULL;
     //e1 = clock();
     //d += e1 - s1;
     //printf(" %s others took %g seconds\n",snp->name,((double)(d))/CLOCKS_PER_SEC);
}

BIC_STATE* runLogistic(C_QUEUE* snp_queue, GENE* gene, BIC_STATE* bic_state, LOGISTIC_SCRATCH* LG, FILE* fp_result, bool perm)
{
        //no debug for permutations.
        PERM = perm;
        static double logistic_intercept_LL = GSL_POSINF;

        if(LOGISTIC_DEBUG_LVL>=1)
        {
            if(!PERM)
              printf("\n\n ################ Logistic Model search for gene = %s ########### \n",gene->name);
            else if(VERBOSE_L)
              printf("\n\n ################ PERM : Logistic Model search for gene = %s ########### \n",gene->name);
        }

        int k = 0;
        int n_covariates = LG->n_covariates;
        int N = LG->phenotype->N_sample;
        LG->curr_model_size = n_covariates;

        //Initialize BIC state.
        initBIC_State(bic_state, 0);

        gsl_multimin_function_fdf my_func;
        my_func.f = my_f;
        my_func.df = my_df;
        my_func.fdf = my_fdf;
        my_func.params = LG;

        //k = 0, model size = 0 is intercept only model.
        double LL_old = 0;
        //if(logistic_intercept_LL_computed==false)
        if(logistic_intercept_LL==GSL_POSINF)
        {
            printf("Computing logistic intercept only LL\n");
            //When covariates are present, you cannot use loglik_0().
            if(n_covariates==0)
            {
               LL_old = loglik_0(); //log likelihood for intercept only model.
            }
            else
            {
                //LL_old = gradient_descent(LG, phenotype,N); //log likelihood for intercept only model.
                LG->s = gsl_multimin_fdfminimizer_alloc(LG->T, n_covariates + 1);
                minimize(LG, &my_func);
                LL_old = -LG->s->f; //log likelihood after model fit.
                gsl_multimin_fdfminimizer_free(LG->s);
                LG->s = NULL; 
            }
            logistic_intercept_LL = LL_old;
            if(LOGISTIC_DEBUG_LVL>=2)
            {
                printf("Logistic regression : computed intercept LL = %lg\n",logistic_intercept_LL);
            }
        }
        else
        {
            if(LOGISTIC_DEBUG_LVL>=2)
            { 
              if(!PERM)
                 printf("Using pre-computed logistic intercept = %f\n",logistic_intercept_LL);
              else if(VERBOSE_L)
                 printf("PERM : Using pre-computed logistic intercept = %f\n",logistic_intercept_LL);
            }
            LL_old = logistic_intercept_LL;
        }

        //update BIC state with intercept only model information.
        //Calculate BIC relative to BIC[0].
        bic_state->BIC[0] = 0;//LL_old - log(gene->eSNP+1); //LL - log(T+1)
        bic_state->bestSNP[0] = NULL; //no snps.
        bic_state->iSNP = 0;
        bic_state->LL[0] = logistic_intercept_LL;

        if(LOGISTIC_DEBUG_LVL>=2 && !PERM)
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
                   else if(VERBOSE_L)
                     printf("\nPERM:     ************* Model search for k = %d ************          \n",k);
                }

                double max_increment = GSL_NEGINF;
                double LL_best = 0;
                SNP* best_SNP = NULL;
                int i = 0;
                int curr_model_size = LG->curr_model_size; //current size of the model.

                if(VERBOSE_L && LOGISTIC_DEBUG_LVL>=2)
                   printf("Current Model Size = %d\n",curr_model_size);
                SNP* snp = (SNP*) cq_getItem(gene->snp_start, snp_queue);

                LG->s = gsl_multimin_fdfminimizer_alloc(LG->T, k + n_covariates + 1);

                for(i = gene->snp_start; i <= gene->snp_end; i++) //for each snp in the gene, add the snp to the model if not already in it.
                {
                        if(!isSNPCorrelated_logistic(snp,LG->curr_model,LG->curr_model_size,gene->LD,n_covariates))
                        {
                                //add snp to current model.
                                if(LOGISTIC_DEBUG_LVL>=1)
                                {
                                   if(!PERM)
                                   {
                                      printf("Adding %d %s\n",k-1,snp->name);
                                      print_model_1(LG->curr_model, LG->curr_model_size);
                                   }
                                   else if(VERBOSE_L)
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
                                      LL = snp->loglik_logistic_perm; //already computed, no need to recompute.
                                   if(LOGISTIC_DEBUG_LVL>=2)
                                   {
                                      printf("Using pre-computed LL = %g\n",LL);
                                   }
                                }
                                else
                                {
                                   //LL = gradient_descent(LG, phenotype,N); //log likelihood after model fit.
                                   remove_missing_samples(LG);
                                   minimize(LG, &my_func); 
                                   LL = -LG->s->f; //log likelihood after model fit.
                                }

                                double increment_BIC = get_Increment(LL, LL_old, gene->eSNP, k, N);
                                //printf("incrmt = %f\n",increment_BIC);

                                if(LOGISTIC_DEBUG_LVL>=2)
                                {
                                    if(!PERM)
                                    {
                                        print_model(LG->curr_model,LG->curr_model_size,LL,increment_BIC);
                                        printf("\n");
                                    }
                                    else if(VERBOSE_L)
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
                        else if(LOGISTIC_DEBUG_LVL>=2 && VERBOSE_L)
                        {
                           if(!PERM)
                              printf("Skipping for corr %s\n",snp->name);
                           else
                              printf("PERM : Skipping for corr %s\n",snp->name);
                        }
                        snp = (SNP*) cq_getNext(snp, snp_queue);
                }

                gsl_multimin_fdfminimizer_free(LG->s);
                LG->s = NULL; 

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
                             else if(VERBOSE_L)
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
                           else if(VERBOSE_L)
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
                    fprintf(fp_result, "%d\t%s\t%s\t%d\t%d\t%d\t%d\t%g\t%s\t%d\t%g\t%g\t%d\t%g\t%g\n",
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

                fprintf(fp_result, "%d\t%s\t%s\t%d\t%d\t%d\t%d\t%g\t%s\t%s\t%s\t%s\t%d\t%g\t%g\n",
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

        return bic_state;
}

/*
int main (void)
{
       size_t iter = 0;
       int status;
     
       const gsl_multimin_fdfminimizer_type *T;
       gsl_multimin_fdfminimizer *s;

       LOGISTIC_SCRATCH LG;
       LG.W = gsl_vector_alloc(20);
       LG.g = gsl_vector_alloc(20);

       SNP s1;
       strcpy(s1.name,"snp1");
       s1.geno = (double*)malloc(sizeof(double)*50);
       s1.geno[0]=2.0;s1.geno[1]=0.0;s1.geno[2]=2.0;s1.geno[3]=2.0;s1.geno[4]=1.0;
       s1.geno[5]=0.0;s1.geno[6]=1.0;s1.geno[7]=2.0;s1.geno[8]=1.0;s1.geno[9]=0.0;
       s1.geno[10]=2.0;s1.geno[11]=1.0;s1.geno[12]=0.0;s1.geno[13]=1.0;s1.geno[14]=0.0;
       s1.geno[15]=0.0;s1.geno[16]=0.0;s1.geno[17]=1.0;s1.geno[18]=0.0;s1.geno[19]=1.0;
   
       SNP s2;
       strcpy(s2.name,"snp2");
       s2.geno = (double*)malloc(sizeof(double)*50);
       s2.geno[0]=0.0;s2.geno[1]=1.0;s2.geno[2]=0.0;s2.geno[3]=2.0;s2.geno[4]=1.0;
       s2.geno[5]=1.0;s2.geno[6]=1.0;s2.geno[7]=2.0;s2.geno[8]=1.0;s2.geno[9]=1.0;
       s2.geno[10]=1.0;s2.geno[11]=1.0;s2.geno[12]=2.0;s2.geno[13]=1.0;s2.geno[14]=0.0;
       s2.geno[15]=0.0;s2.geno[16]=0.0;s2.geno[17]=1.0;s2.geno[18]=2.0;s2.geno[19]=0.0;

       / *
       SNP s1;
       strcpy(s1.name,"snp1");
       s1.geno = (double*)malloc(sizeof(double)*50);
       s1.geno[0]=1.0;s1.geno[1]=2.0;s1.geno[2]=1.0;s1.geno[3]=1.0;s1.geno[4]=1.0;
       s1.geno[5]=1.0;s1.geno[6]=1.0;s1.geno[7]=1.0;s1.geno[8]=1.0;s1.geno[9]=1.0;
       s1.geno[10]=1.0;s1.geno[11]=0.0;s1.geno[12]=1.0;s1.geno[13]=1.0;s1.geno[14]=1.0;
       s1.geno[15]=1.0;s1.geno[16]=1.0;s1.geno[17]=1.0;s1.geno[18]=1.0;s1.geno[19]=1.0;
   
       SNP s2;
       strcpy(s2.name,"snp2");
       s2.geno = (double*)malloc(sizeof(double)*50);
       s2.geno[0]=2.0;s2.geno[1]=2.0;s2.geno[2]=2.0;s2.geno[3]=2.0;s2.geno[4]=2.0;
       s2.geno[5]=1.0;s2.geno[6]=2.0;s2.geno[7]=2.0;s2.geno[8]=2.0;s2.geno[9]=2.0;
       s2.geno[10]=2.0;s2.geno[11]=0.0;s2.geno[12]=2.0;s2.geno[13]=2.0;s2.geno[14]=2.0;
       s2.geno[15]=2.0;s2.geno[16]=2.0;s2.geno[17]=2.0;s2.geno[18]=2.0;s2.geno[19]=2.0;
       * /

       PHENOTYPE P;
       P.pheno_array = (gsl_vector*)gsl_vector_alloc (50);
       int i = 0;
       for(i=0;i<10;i++) gsl_vector_set(P.pheno_array,i,1);
       for(i=10;i<20;i++) gsl_vector_set(P.pheno_array,i,-1);
       P.N_sample = 20;
 
       LG.curr_model[0] = &s1;
       LG.curr_model[1] = &s2;
       LG.curr_model_size = 2;
       LG.phenotype = &P;

       gsl_vector *x; //starting point.
       gsl_multimin_function_fdf my_func;
     
       my_func.n = 3;
       my_func.f = my_f;
       my_func.df = my_df;
       my_func.fdf = my_fdf;
       my_func.params = &LG;
     
       T = gsl_multimin_fdfminimizer_conjugate_fr;
       //T = gsl_multimin_fdfminimizer_vector_bfgs2;
       s = gsl_multimin_fdfminimizer_alloc (T, 3);

       x = gsl_vector_alloc (3);

       clock_t start, end, tot;
       start = clock();
    for(i=0;i<100000;i++) 
    { 
       gsl_vector_set (x, 0, 0.0);
       gsl_vector_set (x, 1, 0.0);
       gsl_vector_set (x, 2, 0.0);
     
       gsl_multimin_fdfminimizer_set (s, &my_func, x, 0.01, 1e-4);
     
       do
         {
           iter++;
           status = gsl_multimin_fdfminimizer_iterate (s);
           //printf("status = %d\n",status); 
           if (status)
             break;
     
           status = gsl_multimin_test_gradient (s->gradient, 1e-3);
     
           //if (status == GSL_SUCCESS)
           //  printf ("Minimum found at:\n");
     
           //printf ("%5d %.5f %.5f %.5f -LL = %10.5f\n", iter,
                   //gsl_vector_get (s->x, 0), 
                   //gsl_vector_get (s->x, 1), 
                   //gsl_vector_get (s->x, 2), 
                   //s->f);
     
       } while (status == GSL_CONTINUE && iter < 1000);
   }
       end = clock();
       printf("Time = %ld\n",end-start);
  
       gsl_multimin_fdfminimizer_free (s);
       gsl_vector_free (x);
       return 0;
}
*/
