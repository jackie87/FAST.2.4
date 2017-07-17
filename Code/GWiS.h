//++Pritam
#ifndef GWIS_H_
#define GWIS_H_
//--Pritam

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <pthread.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_fit.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_multifit.h> //Pritam to regress out covariates on the phenotype
#include <gsl/gsl_multimin.h> //Pritam for faster logistic regression
#include <getopt.h> //for future use to parse the command line parameters
#include <gsl/gsl_sf_gamma.h>
#include "hashtable.h"

#include <unistd.h> //V.1.2 LINUX NUMCPU

#define VERBOSE_PERM_TIME false //Time for each permutation loop.
#define CQ_DEBUG false
#define DEBUG false

#define SMART_STOP true

#define USE_BG_FR false

//minimum model size.  set to 0 in most cases
#define K_MIN 0

#define PI 3.1415926535
#define EPS 0.0000001
#define P0 0
//the corresponding R (not R2) to VIF= 5.  Set to prevent colinearity  
#define VIF_R 0.89442719
#define VIF_R2 0.8
//#define VIF_R 0.547722
//#define VIF_R2 0.3
//set N_HAT (number of heterozygous) and R2_CUTOFF to ensure the SNP quality.
#define N_HAT_CUTOFF 5
#define R2_CUTOFF 0.3

#define MAX_FILENAME_LEN 200
#define MAX_INCLUDED_SNP 200 //
#define MAX_PL_INCLUDED_SNP 40000 // Jianan added for PL, to differiate between the previous constant; PL usually envolves more SNPs
#define MAX_SNP_FOR_COR 100000 //Jianan added, to calculate phenotype correlations
#define MAX_N_INDIV 20000
//#define MAX_N_INDIV_NO_PHENO 2000
#define MAX_LINE_WIDTH 120000
#define MAX_LINE_WIDTH_LONG 8200000
//it is safe to estimate a smaller number.  The program will auto resize when necessary
#define MAX_SNP_PER_GENE 256
#define MAX_SNP_PER_GENE_FIXED 5000
#define MAX_COR_SNP 1024
#define MAX_COX_FACTOR_LEN 200

#define PHENOTYPE_VALUE_LEN 200
#define INDIV_ID_LEN 200
#define SNP_ID_LEN 20
#define SNP_ALLELE_LEN 1000000
#define MAX_N_COVARIATES 20 //Pritam
#define GENE_ID_LEN 2000
#define MAX_N_PHENOTYPES 20//Jianan
#define PHENO_NAME_LEN 300

#define MAF_CUTOFF 0.01 //V.1.4.mc
#define COX_STRATA_NUM 0
#define COX_COV_NUM 0


#define min(x,y) ((x)<(y)?(x):(y))
#define max(x,y) ((x)>(y)?(x):(y))

#define SNP_INFO_SUMMARY_INCLUDE 0
#define SNP_INFO_SUMMARY_EXCLUDE_MULTIPOS 1
#define SNP_INFO_SUMMARY_EXCLUDE_REDUNDENT 2
#define SNP_INFO_SUMMARY_EXCLUDE_A1MISS 3
#define SNP_INFO_SUMMARY_EXCLUDE_A1NOMATCH 4

#define SIGMA_U 1.0 //should be configurable.

//Pritam
#define MAX_LOGISTIC_ITER 140  //
#define MAX_BF_LOGISTIC_ITER 1000
#define LOGISTIC_DEBUG_LVL 0 //set values : 0 (no debuf info), 1 (lvl 1), 2 (more debug info), 3 (max debug info).
#define NSHUFFLEBUF 1000
#define MAX_COX_ITER 20
#define COX_err 1e-9
#define COX_toler 1.818989e-12

#define ADJUST_LD true

bool OMIT_ST;
bool SIMPLE_SUMMARY; //V.1.7
bool SIMPLE_PL;
bool VERBOSE;
bool NO_SIGN_FLIP;
bool BOUND_BETA;
double maf_cutoff; //V.1.4.mac
int cox_strata_num;
int cox_cov_num;

int n_hat_cutoff;
double r2_cutoff;
double SIGMA_A;  //should be configurable. LATEST V.1.2

bool SKIPPERM;
bool FULLSEARCH;

//V.1.5.mc
bool IMPUTE2_input;

//+V.1.4.mc
bool COMPUTE_LD;

bool PLEIOTROPY_APPROX;//Jianan added, switcher to turn on iteration ; default  should be false; when turned on (to true), will only do one iteration.

bool ESTIMATE_PHENO_VAR;
bool ESTIMATE_N;

bool USE_WEIGHTED_LD;
double * haplotype_weights;
//-V.1.4.mc

double MAX_MISSINGNESS;
bool MISSING_DATA;
double MISSING_VAL; //value for missing genotypes

int N_PERM_MIN;
int N_CUTOFF;
int N_PERM;
 
int FLANK;
bool INTERCEPT;
bool DIFFNCBI;

FILE *fp_log; //V.1.5.mc, global.

//single snp
bool REGULAR_PHENOTYPE;//Jianan added. Regular phenotype is used when we are not working with cox or pleiotropy
bool DO_COX;
bool GET_SINGLE_SNP_COX;
bool GET_GENE_COX;
bool GET_SINGLE_SNP_LINEAR; //compute single snp logistic wald + parametric pval
bool GET_SINGLE_SNP_LOGISTIC; //compute single snp logistic wald + parametric pval
bool GET_PLEIOTROPY;
bool GET_SAPPHOC;
bool GET_SAPPHOI;
bool GET_SAPPHO_MIN_PVAL;
bool GET_SAPPHO_ALPHA;
//min snp
bool GET_MINSNP_LINEAR; //compute best single snp logistic wald + parametric pval for gene
bool GET_MINSNP_PVAL_LINEAR; //compute best single snp logistic wald + permutation pval for gene

bool GET_MINSNP_LOGISTIC; //compute best single snp logistic wald + parametric pval for gene
bool GET_MINSNP_PVAL_LOGISTIC; //compute best single snp logistic wald + permutation pval for gene

//min-snp-p (permutations are must)
bool GET_MINSNP_P_PVAL_LINEAR; //compute best single snp logistic wald + permutation pval for gene
bool GET_MINSNP_P_PVAL_LOGISTIC; //compute best single snp logistic wald + permutation pval for gene

//BF
bool GET_BF_LINEAR; //just BF score
bool GET_BF_PVAL_LINEAR; //BF score + permutation pvalue
bool GET_BF_LOGISTIC; //just BF score
bool GET_BF_PVAL_LOGISTIC; //BF score + permutation pvalue

//VEGAS
bool GET_VEGAS_LINEAR; //just vegas score
bool GET_VEGAS_PVAL_LINEAR; //vegas score + permutation pvalue
bool GET_VEGAS_LOGISTIC; //just vegas score
bool GET_VEGAS_PVAL_LOGISTIC; //vegas score + permutation pvalue

//BIC
bool GET_GENE_BIC_LINEAR; //just BIC score
bool GET_GENE_BIC_PVAL_LINEAR; //BIC score + permutation pvalue
bool GET_GENE_BIC_LOGISTIC; //just BIC score
bool GET_GENE_BIC_PVAL_LOGISTIC; //BIC score + permutation pvalue
//bool GET_GENE_READ_ESNPS;
//bool GET_GENE_WRITE_ESNPS;


//no permutations needed, we only do the parametric pvalues.
bool GET_GATES_LINEAR;
bool GET_GATES_LOGISTIC;

//Summary data
bool SUMMARY;

bool BINARY_PHENO;//Pritam
bool SHUFFLEBUF;

bool NEED_SNP_LINEAR; //Need to run single snp linear regression ?
bool NEED_SNP_LOGISTIC; //Need to run single snp logistic regression ?

bool NEED_SNP_LINEAR_SUMMARY; 
bool NEED_SNP_LOGISTIC_SUMMARY; 

bool work_on_genes;

int n_threads;

gsl_vector* pheno_buff[NSHUFFLEBUF];

#define Test_increment 0;

pthread_cond_t nthr_completed_cond;
pthread_mutex_t nthr_completed_mutex;
int nthr_completed;

int perm_loop_counter;
pthread_mutex_t perm_loop_counter_mutex;

//mutexes for threads.
pthread_mutex_t mutex_minsnp_linear;
pthread_mutex_t mutex_minsnp_p_linear;
pthread_mutex_t mutex_bf_linear;
pthread_mutex_t mutex_vegas_linear;
pthread_mutex_t mutex_bic_linear;

pthread_mutex_t mutex_minsnp_logistic;
pthread_mutex_t mutex_minsnp_p_logistic;
pthread_mutex_t mutex_bf_logistic;
pthread_mutex_t mutex_vegas_logistic;
pthread_mutex_t mutex_bic_logistic;

pthread_mutex_t mutex_other;

typedef struct OUTFILE_TYPE{
  //FILE *fp_PL_linear;
  FILE * fp_sapphoC_linear;
  FILE * fp_sapphoI_linear;
  FILE *fp_allSNP_cox;
  FILE *fp_gene_cox;
  FILE *fp_allSNP_linear;
  FILE *fp_allSNP_logistic;
  FILE *fp_snp_pval_linear;
  FILE *fp_snp_pval_logistic;
  FILE *fp_snp_perm_pval_linear;
  FILE *fp_snp_perm_pval_logistic;
  FILE *fp_gene_snp;
  //FILE *fp_bic_logistic_result;
  FILE *fp_bic_logistic_perm_result;
  //FILE *fp_bic_linear_result;
  FILE *fp_bic_linear_perm_result;
  FILE *fp_bf_pval_linear;
  FILE *fp_bf_pval_logistic;
  FILE *fp_vegas_pval_linear;
  FILE *fp_vegas_pval_logistic;
  FILE *fp_gates_pval_linear;
  FILE *fp_gates_pval_logistic;
  FILE *fp_bic_linear_back_result;
  FILE *fp_bic_linear_all_result;
} OUTFILE;

typedef struct PAR_TYPE
{
  char *impute2_geno_file;//V.1.5.mc
  char *impute2_info_file;//V.1.5.mc

  char *pheno_var_file;//Jianan added

  char *tped_file;//V.1.4.mc
  char *snpinfo_file;//V.1.4.mc
  char *mlinfo_file;//V.1.4.mc
  char *summary_file;//V.1.4.mc
  char *multipos_file;//V.1.4.mc
  char *indivfile;
  int chr;
  char* trait_file;
  char* output;
  char* ldfile;
  char* allelefile;
  char* posfile; //V.1.4.mc
  char* hapfile; //V.1.4.mc
  char* hap_wt_file; //V.1.4.mc, haplotype weights if present.
  //  char* association_file;//Jianan added for Pleiotropy

  //double pheno_mean;//V.1.4.mc
  double pheno_var;
  bool estimate_pheno_var; //V.1.4.mc
  int n_sample;
  double maf_cutoff;
  int cox_strata_num;
  int cox_cov_num;
  char* gene_set;
  int random_seed;
  int flank;
  int max_perm;
  int n_perm_min; //V.1.7.mc
  char *mode;
  bool skip_perm;
  int n_hat_cutoff; //V.1.5.mc
  double r2_cutoff; //V.1.5.mc
  double missingness;
  double missing_data;
  bool scalepheno;
  bool quantilepheno;
  bool compute_ld; //V.1.4.mc
  bool use_weighted_ld; //V.1.4.mc



  double sigma_a;  //V.1.2

  int n_threads;

  bool pleiotropy;//Jianan added
  bool sapphoC;//Jianan added
  bool sapphoI;//Jianan added
  bool single_snp_cox;          //single snp cox PH model
  bool single_snp_linear;       //single snp linear parametric pvalue
  bool single_snp_linear_gene;  //single snp linear parametric pvalue for only snps mapped to the genes.
  bool minsnp_linear;           //minsnp linear parametric pvalue
  bool minsnp_pval_linear;      //minsnp linear permutation pvalue, if skipperm, will become minsnp_linear
  bool minsnp_p_pval_linear;    //minsnp p linear permutation pvalue, if skipperm, will not run.
  bool gene_cox;
  bool gene_bic_linear;         //bic per gene, no pvalue 
  bool gene_bic_pval_linear;    //bic per gene, pvalue by permutations
  bool gene_bf_linear;
  bool gene_bf_pval_linear;
  bool gene_vegas_linear;
  bool gene_vegas_pval_linear;
  bool gene_gates_linear;
  bool gene_gates_logistic;
  bool gene_bic_linear_back;
  bool gene_bic_linear_all;

  bool single_snp_logistic;       //single snp logistic parametric pvalue
  bool single_snp_logistic_gene;  //single snp logistic parametric pvalue for only snps mapped to the genes.
  bool minsnp_logistic;           //minsnp logistic parametric pvalue
  bool minsnp_pval_logistic;      //minsnp logistic permutation pvalue, if skipperm, will report parametric pvalue
  bool minsnp_p_pval_logistic;    //minsnp p logistic permutation pvalue, if skipperm, will report parametric pvalue
  bool gene_bic_logistic;         //bic per gene, no pvalue 
  bool gene_bic_pval_logistic;    //bic per gene, pvalue by permutations
  bool gene_bf_logistic;
  bool gene_bf_pval_logistic;
  bool gene_vegas_logistic;
  bool gene_vegas_pval_logistic;
  //  bool gene_read_eSNPs;
  //bool gene_write_eSNPs;

  int ncov; //Pritam covariates
  int npheno;
  double sappho_min_pval;
  double sappho_alpha;//Jianan added

  bool impute2_geno_file_parsed;//V.1.5.mc
  bool impute2_info_file_parsed;//V.1.5.mc


  bool cox_surv_parsed;//Jianan added, parse the suvival time
  bool cox_status_parsed;//Jianan added, parse whether the sample was censored or died

  bool pheno_var_file_parsed;

  bool tped_file_parsed;
  bool snpinfo_file_parsed;
  bool mlinfo_file_parsed;
  bool summary_file_parsed;
  bool multipos_file_parsed;
  bool indivfile_parsed;
  bool chr_parsed;
  bool trait_file_parsed;
  bool output_parsed;
  bool ldfile_parsed;
  bool posfile_parsed; //V.1.4.mc
  bool hapfile_parsed; //V.1.4.mc
  bool hap_wt_file_parsed; //V.1.4.mc
  bool allelefile_parsed;
  bool associationfile_parsed;//Jianan added
  //bool pheno_mean_parsed; //V.1.4.mc
  bool pheno_var_parsed;
  bool estimate_pheno_var_parsed; //V.1.4.mc
  bool n_sample_parsed;
  bool maf_cutoff_parsed;
  bool gene_set_parsed;
  bool random_seed_parsed;
  bool flank_parsed;
  bool max_perm_parsed;
  int n_perm_min_parsed; //V.1.7.mc
  bool mode_parsed;
  bool skip_perm_parsed;
  bool n_hat_cutoff_parsed; //V.1.5.mc
  bool r2_cutoff_parsed; //V.1.5.mc
  bool missingness_parsed;
  bool missing_data_parsed;
  bool scalepheno_parsed;
  bool quantilepheno_parsed;
  bool compute_ld_parsed; //V.1.4.mc
  bool use_weighted_ld_parsed; //V.1.4.mc
  bool cox_strata_parsed;
  bool cox_cov_parsed;


  bool pleiotropy_parsed; //Jianan added
  bool sapphoC_parsed;//Jianan added
  bool sapphoI_parsed;//Jianan added
  bool single_snp_cox_parsed; //Jianan added, cox model parsed
  bool gene_cox_parsed;//Jianan added, gene based cox
  bool single_snp_logistic_parsed;       //single snp logistic parametric pvalue
  bool single_snp_logistic_gene_parsed;  //single snp logistic parametric pvalue for only snps mapped to the genes.
  bool minsnp_logistic_parsed;      //minsnp logistic permutation pvalue
  bool minsnp_pval_logistic_parsed;      //minsnp logistic permutation pvalue
  bool minsnp_p_pval_logistic_parsed;    //minsnp p logistic permutation pvalue
  bool gene_bic_logistic_parsed;         //bic per gene, no pvalue 
  bool gene_bic_pval_logistic_parsed;    //bic per gene, pvalue by permutations
  bool gene_bf_logistic_parsed;
  bool gene_bf_pval_logistic_parsed;
  bool gene_vegas_logistic_parsed;
  bool gene_vegas_pval_logistic_parsed;

  bool single_snp_linear_parsed;       //single snp linear parametric pvalue
  bool single_snp_linear_gene_parsed;  //single snp linear parametric pvalue for only snps mapped to the genes.
  bool minsnp_linear_parsed;      //minsnp linear permutation pvalue
  bool minsnp_pval_linear_parsed;      //minsnp linear permutation pvalue
  bool minsnp_p_pval_linear_parsed;    //minsnp p linear permutation pvalue
  bool gene_bic_linear_parsed;         //bic per gene, no pvalue 
  bool gene_bic_pval_linear_parsed;    //bic per gene, pvalue by permutations
  bool gene_bf_linear_parsed;
  bool gene_bf_pval_linear_parsed;
  bool gene_vegas_linear_parsed;
  bool gene_vegas_pval_linear_parsed;
  bool gene_gates_linear_parsed;
  bool gene_gates_logistic_parsed;

  //  bool gene_write_eSNPs_parsed;
  //bool gene_read_eSNPs_parsed;

  bool ncov_parsed; //Pritam covariates
  bool npheno_parsed;//Jianan added
  bool sappho_min_pval_parsed;//Jianan addd
  bool sappho_alpha_parsed;//Jianan added
  bool n_threads_parsed;
  double sigma_a_parsed; //V.1.2
} PAR;

//+V.1.4.mc
typedef struct SNP_SNP_CORR_TYPE
{
  double ref_ld; //ld from reference panel.
  double ref_cov; //cov from reference panel.
} SNP_SNP_CORR;
//-V.1.4.mc

typedef struct SNP_TYPE 
{
  int id; //V.1.4.mc
  char name[SNP_ID_LEN];
  int gene_id;
  int chr;
  double file_pos; //V.1.4.mc
  double pos;
  double MAF;
  double AF1; //allele freq of coded allele
  double R2;
  char* A1;
  char* A2;
  int bp;
  int nGene;
  double f_stat;

  double BF_linear;
  double BF_logistic;

  //for summary data
  double beta;
  bool sign_kept; //V.1.4.mc

  double se;
  double metaP;
  char coding_allele;
  int nmiss; //only for summary data

  double * ref_geno; //genotype from reference samples. //V.1.4.mc
  int ref_nsample; //no. of samples in the reference data. //V.1.4.mc

  double sum_pheno_geno; //vector product (dot product) of the phenotype and genotype vectors for the SNP

  //for permutations.
  //shared and updated by all threads, need mutex for each ?
  int nHit_linear_sh; //for minSNP, number of hits for the SNP, no need to permuate again in case of overlapping genes
  int iPerm_linear_sh; //for minSNP, number of perms for the SNP, no need to permuate again in case of overlapping genes
  int nHit_bonf_linear_sh; //for minSNP-P, number of hits for the SNP,  no need to permuate again in case of overlapping genes
  int iPerm_bonf_linear_sh;//for minSNP-P, number of perms for the SNP, no need to permuate again in case of overlapping genes

  int nHit_logistic_sh; //for minSNP, number of hits for the SNP, no need to permuate again in case of overlapping genes
  int iPerm_logistic_sh; //for minSNP, number of perms for the SNP, no need to permuate again in case of overlapping genes
  int nHit_bonf_logistic_sh; //for minSNP-P, number of hits for the SNP,  no need to permuate again in case of overlapping genes
  int iPerm_bonf_logistic_sh;//for minSNP-P, number of perms for the SNP, no need to permuate again in case of overlapping genes

  double ref_geno_tss; //covariance from refrence samples, for SUMMARY only. //V.1.4.mc
  double ref_AF1; //AF of coded allele from reference samples, for SUMMARY only.//V.1.4.mc
  double ref_MAF; //MAF from reference samples, for SUMMARY only.//V.1.4.mc

  double geno_tss; //tss of geno, covariance
  double *geno; //genotype
  double eSampleSize; //effective number of sample size, (imputation quality)*MAF*nsample

  double *r; //correlation with other SNPs
  char* r_names; 
  int n_correlated_snp;
  int n_correlated_snp_max;

  struct hashtable* correlated_snps; //hashtable of correlated snps for this snp.//V.1.4.mc
  double c1; //single snp linear regression coefficient computed from genotype data, Pritam added.
  double c1_se; //and the corresponding standard error.

  double pval_linear; //single snp pvalue from linear regression
  double beta_logistic; //Pritam
  double se_logistic; //Pritam
  double loglik_logistic; //Pritam
  double wald; //Pritam
  double pval_logistic; //single snp pvalue from logistic regression

  double missingness;

} SNP;

typedef struct BIC_STATE_TYPE // data for GWiS
{
  double BIC[MAX_INCLUDED_SNP+1]; // GWiS test statistics for each SNP added to the model
  double RSS[MAX_INCLUDED_SNP+1]; // Redisual sum of squares for each SNP
  SNP* bestSNP[MAX_INCLUDED_SNP+1]; // best SNPs
  int iSNP; //model size, "k"
  double LL[MAX_INCLUDED_SNP+1];//logistic loglikelihood added by Pritam
}BIC_STATE;

typedef struct SAPPHOC_STATE_TYPE{
  double BIC[MAX_PL_INCLUDED_SNP+1]; // Plioetropy GWiS score, Jianan
  double RSS[MAX_PL_INCLUDED_SNP+1];
  SNP* bestSNP[MAX_PL_INCLUDED_SNP+1]; // best SNPs, Jianan
  int iSNP; //model size, "k"

  int pheno[MAX_PL_INCLUDED_SNP+1];//best associated phenotype index for each step
  int geno[MAX_PL_INCLUDED_SNP+1];//best associated genotype index for each step
  
  //double * patterns;
  gsl_matrix * associations;
  int * nSNP; //number of SNPs added for each phenotype
  int ** pheno_geno_indices;
  int ** pheno_geno_beta_indices;
  
  int added_SNPs[MAX_PL_INCLUDED_SNP+1];//which SNPs were added, to calculated pattern;
  int num_added_SNPs;//number of SNPs added, to calculate pattern
  int * patterns;
  double T;// THe tuning parameter
  double alpha;//weight for two prior terms
  double TP; //totol number of possible associations
  int * correlated_SNPs;// vector to exclude highly correlated SNPs

}SAPPHOC_STATE;

typedef struct SAPPHOI_STATE_TYPE
{
  double BIC[MAX_PL_INCLUDED_SNP+1]; // Plioetropy GWiS score, Jianan
  double RSS[MAX_PL_INCLUDED_SNP+1];
  SNP* bestSNP[MAX_PL_INCLUDED_SNP+1]; // best SNPs, Jianan           
  int iSNP; //model size, "k"
  int pheno[MAX_PL_INCLUDED_SNP+1];
  int geno[MAX_PL_INCLUDED_SNP+1];

  //double * patterns;
  gsl_matrix * associations;
  int * nSNP; //number of SNPs added for each phenotype 
  int added_SNPs[MAX_PL_INCLUDED_SNP+1];//which SNPs were added, to calculated pattern;                              
  int num_added_SNPs;//number of SNPs added, to calculate pattern      
  int * patterns;
  double T;// THe tuning parameter
  double alpha;//weight for two prior terms
  double TP; //totol number of possible associations
  int * correlated_SNPs;// vector to exclude highly correlated SNPs
}SAPPHOI_STATE;

typedef struct SAPPHOI_WORKSPACE_TYPE{
  double * pheno_var;
  double * pheno_var_orig;
  double ** geno_cov_orig;
  double ** pheno_geno_cov_orig;

  double *** geno_cov;
  double ** pheno_geno_cov;
}SAPPHOI_WORKSPACE;

typedef struct SAPPHOC_WORKSPACE_TYPE{
  gsl_matrix * pheno_var_cov;//pheno_var_cov for the null model

  gsl_matrix * last_best_var_cov;// best var_cov before adding the current association
  gsl_matrix * last_best_inverse_var_cov;//best inverse of var_cov before adding the current association

  gsl_matrix * cur_best_var_cov;//  best var_cov for cur k_model
  gsl_matrix * cur_best_inverse_var_cov; //best inverse of var_cov for cur k_model

  gsl_matrix * temp_var_cov;//working var_cov
  gsl_matrix * temp_inverse_var_cov;//working inverse var_cov
  gsl_permutation * p_inverse;//working permutation vector

  gsl_matrix * LU;//for LU decomposition, so that the input matrix wouldn't change; for both inverse and calDet
  gsl_permutation * p_calDet;//for calculating determinant

  //Sufficient statistics
  double ** geno_cov;
  double ** pheno_geno_cov;
}SAPPHOC_WORKSPACE;



typedef struct HIT_COUNTER_TYPE // data structure to save how many permutations / hits we have done 
{
  int k_pick_linear[MAX_INCLUDED_SNP+1];
  int k_hits_linear[MAX_INCLUDED_SNP+1];
  int maxK_linear;

  int k_pick_logistic[MAX_INCLUDED_SNP+1];
  int k_hits_logistic[MAX_INCLUDED_SNP+1];
  int maxK_logistic;

  int hit_bf_linear;
  int perm_bf_linear;

  int hit_bf_logistic;
  int perm_bf_logistic;

  int hit_vegas_linear;
  int perm_vegas_linear;

  int hit_vegas_logistic;
  int perm_vegas_logistic;

  int * hit_snp_linear;
  int * perm_snp_linear;

  int * hit_snp_logistic;
  int * perm_snp_logistic;

  int * hit_snp_bonf_linear;
  int * perm_snp_bonf_linear;

  int * hit_snp_bonf_logistic;
  int * perm_snp_bonf_logistic;

}HIT_COUNTER;

typedef struct GENE_TYPE 
{
  char name[GENE_ID_LEN];
  char ccds[GENE_ID_LEN];
  int chr;
  int bp_start;
  int bp_end;

  int nSNP; // total number of SNPs in a gene
  double eSNP; // effective number of SNPs in a gene

  int snp_start; //coordinate for the first SNP in this gene in the circular queue
  int snp_end; //coordinate for the last SNP in this gene in the circular queue
 
  BIC_STATE bic_state_logistic; //Logistic regression bic and models.
  BIC_STATE bic_state_linear; // GWiS test statistics, RSS, and SNPs chosen   

  //for permutations.
  //shared and updated by all threads, need mutex.
  HIT_COUNTER hits_sh; // GWiS permutation results

  double BF_sum_linear;
  double BF_sum_logistic;

  double vegas_linear;
  double vegas_logistic;

  double gates_linear;
  double gates_logistic;

  //shared and updated by all threads, need mutex.
  SNP* maxSSM_SNP_linear_sh; // for minSNP, best SNP picked up by minSNP, for this gene
  SNP* maxSSM_SNP_logistic_sh; // for minSNP, best SNP picked up by minSNP, for this gene

  SNP* maxBonf_SNP_linear_sh;// for minSNP-P, best SNP picked up by minSNP-P, for this gene
  SNP* maxBonf_SNP_logistic_sh;// for minSNP-P, best SNP picked up by minSNP-P, for this gene

  //fields for performance
  gsl_matrix* LD; // Correlation matrix
  gsl_matrix* pvalCorr; // pvalue Correlation matrix computed for GATES

  gsl_matrix* Cov; // covariance matrix

  bool skip; // a indicator to skip this gene
  //fields for linked list

  struct GENE_TYPE * next;

} GENE;

typedef struct ORTHNORM_TYPE
{
  double norm; //norm of the genotype, will be adjusted in the ortho-nornmalization procedures
  double norm_original; //norm of the genotype, stays the same.  Compare norm and norm_original to look into VIF
  double projP; // projection of trait vector to the genotype vector, adjusted in each of the ortho-nornmalization procedures 
  int k; // step of the ortho-nornmalization procedures
  SNP * snp; // SNP corresponding to this data sructure
  double sum_X_bestZ[MAX_SNP_PER_GENE_FIXED]; //intermediate calculations, saved for future ortho-nornmalization procedures
}OrthNorm;

typedef struct SNP_INFO_SUMMARY_TYPE 
{
  long long file_pos; //position in hap file to facilitate computing LD. //V.1.4.mc
  int pos;
  int a1;
  int a2;
  int exclude;
  double ref_maf;
  double ref_af1;
} SNP_INFO_SUMMARY;

typedef struct SNP_CNT_TYPE
{

  int nSNPAssigned;
  int nSNPRedundent;
  int nSNPMulti;
  int nSNPnoLD;
  int nSNPambig;
  int nSNPsmallSample;
  int nSNPnoA1;
  int nSNPA1MissMatch;
  int nSignKept;
  int nSignFlipped;
  int nInvSE; //V.1.2 VEGAS FIX, invalid standard error.

  //++FIX MONOMORPHIC SNPS V.1.2
  int nSNP_missingness;
  int nSNP_esamplesize;
  int nSNP_quality;
  int nSNP_mono;
  //--FIX MONOMORPHIC SNPS V.1.2

  int nSNP_small_maf; //V.1.4.mc

} SNP_CNT;


typedef struct C_QUEUE_TYPE 
{
  void * dat; // where the data is saved
  void * last; // pointer for the memory boundary
  size_t nsize; //size of each variable
  size_t chunkSize;
  size_t start; //start of the logic position
  size_t end; // end of the logic position

} C_QUEUE;

typedef struct PHENOTYPE_TYPE //phenotype vector
{
  gsl_vector * pheno_array_org; //phenotype array original and scaled.
  gsl_vector * pheno_array_reg; //phenotype array after regressing out the covariates.
  gsl_vector * pheno_array_log; //phenotype array for logistic regression. 
  //double mean; //mean of the phenotype //V.1.4.mc
  double tss_per_n; // tss/n of the phenotype
  int N_na; // number of NAs in the phenotype file
  int N_indiv; // number of individuals, including NA
  int N_sample; //number of samples, nsample=N_indiv - N_na
  bool NA[MAX_N_INDIV]; // which indiv we are missing
  int SEX[MAX_N_INDIV]; //sex for each indiv.

  gsl_vector* covariates[MAX_N_COVARIATES]; //covariates.
  int n_covariates;
} PHENOTYPE;

typedef struct pleiophenotype{
  gsl_vector ** pheno_vectors_org;
  gsl_vector ** pheno_vectors_reg;
  gsl_vector ** pheno_vectors_log;

  double * tss_per_n;
  int N_na;
  int N_indiv;
  int N_sample;
  bool NA[MAX_N_INDIV];
  int SEX[MAX_N_INDIV];
  char Pheno_names[MAX_N_PHENOTYPES][PHENO_NAME_LEN];
  
  gsl_vector * covariates[MAX_N_COVARIATES];
  int n_covariates;
  int n_pheno;

}PLEIOPHENOTYPE;


typedef struct COXRESULT_TYPE
{
  double ** imat;
  double ** cmat;
  double ** cmat2;
  double * beta;
  double ** snp;
  double flag[1];
  double *scale;
  double loglikratio[1];
}COXRESULT;


typedef struct COXRESULTGENE_TYPE
{
  SNP* bestSNP[MAX_INCLUDED_SNP+1];
  double loglik[MAX_INCLUDED_SNP+1];
  double BIC[MAX_INCLUDED_SNP+1];
  double* offset;
  int iSNP;
}COXRESULTGENE;

typedef struct COXPHENOTYPE_TYPE
{
  double ** cov;// load both the covariates and the snp data
  int * new_strata;// load the organized strata to be passed to coxfit6
  double* surv;// the survival time for each individual
  int* status;//the final status for each individual
  int N_na;//number of NAs in the phenotype file
  int N_indiv;//number of individuals, includ NAs
  int N_sample;//number of samples, nsample=N_indiv - N_na 
  double *offset;
  double* cov_beta;
  bool NA[MAX_N_INDIV];
  int SEX[MAX_N_INDIV];
  int * map;

}COXPHENOTYPE;

int count_1;
int count_2;

//scratch space for logistic regression.
typedef struct LOGISTIC_SCRATCH_TYPE
{
        gsl_matrix* h;//hessian.
        gsl_vector* W;
        gsl_vector* g; //gradient.
        double s1[MAX_N_INDIV]; //will be used to compute the hessian after minimization of neg LL.

        gsl_matrix* h_temp;//hessian to be used for holding the result of LU decomposition.
        gsl_matrix* h_inv;//inverse of the hessian.
        gsl_vector* old_W;
        gsl_vector* W_delta;

        SNP* curr_model[MAX_INCLUDED_SNP + MAX_N_COVARIATES]; //array of pointers to snp object which reside in c_queue.
        int curr_model_size;

        //snps mimicking covariates.
        SNP* cov_snps[MAX_N_COVARIATES];

        //for single snp analysis (plus covariates).
        //standard errors
        double se[MAX_INCLUDED_SNP + MAX_N_COVARIATES + 1]; //intercept included.
        //wald statistics
        double wald[MAX_INCLUDED_SNP + MAX_N_COVARIATES + 1]; //intercept included.
        int n_covariates;
        PHENOTYPE * phenotype;

        //const gsl_multimin_fdfminimizer_type *T;
        //gsl_multimin_fdfminimizer *s;
        //gsl_vector * x; //to initialize weights.

        gsl_vector * OMIT; //to indicate if any snp has missing data.
        bool allocated;

} LOGISTIC_SCRATCH;

//scratch space for Bayes factor using logistic regression.
typedef struct BG_SCRATCH_TYPE
{
        gsl_matrix* h;//hessian.
        gsl_vector* nu;//priors for beta.
        gsl_vector* W;
        gsl_vector* g; //gradient.
        double s1[MAX_N_INDIV]; //will be used to compute the hessian after minimization of neg LL.
        gsl_matrix* h_temp;//hessian to be used for holding the result of LU decomposition.

        gsl_matrix* h_inv;//inverse of the hessian.
        gsl_vector* old_W;
        gsl_vector* W_delta;

        SNP* curr_model[MAX_INCLUDED_SNP + MAX_N_COVARIATES]; //array of pointers to snp object which reside in c_queue.
        int curr_model_size;

        //snps mimicking covariates.
        SNP* cov_snps[MAX_N_COVARIATES];
        int n_covariates;

        PHENOTYPE * phenotype;
        const gsl_multimin_fdfminimizer_type *T;
        gsl_multimin_fdfminimizer *s;
        gsl_vector * x; //to initialize weights.

        gsl_vector * OMIT; //to indicate if any snp has missing data.
        bool allocated;

} BG_SCRATCH;

//to hold per snp information computed during each permutation.
typedef struct LOGISTIC_Z_TYPE
{
  SNP * snp;
  double BF_logistic_perm; //BF logistic computed during permutations.
  double wald_perm; //wald computed during permutations.
  double loglik_logistic_perm; //logistic loglikelihood computed during permutations.
}LOGISTIC_Z;

typedef struct BIC_THREAD_DATA_TYPE
{
  int thread_id;
  PHENOTYPE * phenotype; //Pritam added to regress out covaraites during permutations.
  gsl_rng *r;
  GENE* gene;
  C_QUEUE * snp_queue;
  //double pheno_mean; //V.1.4.mc
  double pheno_tss_per_n;
  int nsample;

  double* z_dat;
  OrthNorm *Z;
  LOGISTIC_SCRATCH * LG;
  BG_SCRATCH * BG;
  
  LOGISTIC_Z * LZ;
 
  gsl_matrix * L; //V.1.4.mc
  
}BIC_THREAD_DATA;

gsl_permutation * P2; //size = 2
gsl_permutation * P3; //size = 3
gsl_permutation * P4; //size = 4
gsl_permutation * P5; //size = 5
gsl_permutation * P6; //size = 6

typedef struct LOGISTIC_ERR_REP
{
        LOGISTIC_SCRATCH *LG;
        gsl_vector* phenotype;
        int N;
} LOGISTIC_ERR_REP;

extern void alloc_P();
extern void free_P();

extern C_QUEUE* cq_init(size_t nsize, size_t chunkSize);
extern bool cq_isInQ(size_t curItem, C_QUEUE* c_queue);
extern void* cq_getItem(size_t i, C_QUEUE* c_queue);
extern void* cq_getNext(void* ptr, C_QUEUE* c_queue);
extern bool cq_isEmpty(C_QUEUE* c_queue);
extern bool cq_isFull(C_QUEUE* c_queue);
extern void cq_resize(C_QUEUE * c_queue);
extern void* cq_push(C_QUEUE* c_queue);
extern void* cq_shift(C_QUEUE* c_queue);
extern void* cq_pop(C_QUEUE* c_queue, bool remove);
extern void initBIC_State(BIC_STATE* bic_state, double pheno_tss_per_n);
extern double runTest_pritam(double *geno, double *pheno, int nsample);
extern BIC_STATE* runLogistic(C_QUEUE* snp_queue, GENE* gene, BIC_STATE* bic_state, LOGISTIC_SCRATCH* LG, FILE* fp_result,LOGISTIC_Z * LZ,bool perm);
//extern void runLogisticSNP(SNP* snp, LOGISTIC_SCRATCH* LG,bool,bool);
extern void runLogisticSNP( SNP * snp, LOGISTIC_SCRATCH * LG, bool perm, LOGISTIC_Z * LZ, bool need_hessian );
extern double runBFLogisticSNP_fr(SNP* snp, BG_SCRATCH* BG, bool perm);
extern double runBFLogisticSNP_newton(SNP* snp, BG_SCRATCH* BG, bool perm);

extern int partition_linear(SNP** y, int f, int l);
void quicksort_linear(SNP** x, int first, int last);
void quicksort_logistic(SNP** x, int first, int last);
void quicksort_summary(SNP** x, int first, int last);

extern int solve(gsl_matrix* h, gsl_vector* g, gsl_vector* x);
#endif /* GWIS_H_ */
//--Pritam
