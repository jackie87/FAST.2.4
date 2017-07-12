#include "GWiS.h"

int numCPU_linux() //V.1.2 LINUX NUMCPU
{
   return sysconf( _SC_NPROCESSORS_ONLN );
}

int count_tokens(char * line)
{
  int count = 0;
  //char * ptr = line;
  char * pch = strtok(line," \t");
  while(pch!=NULL)
  {
    pch = strtok(NULL," \t");
    count++;
  }
  return count;
}

//V.1.5.mc
//open a gzipped file using zcat and pipes.
FILE * gzopen(char * fname)
{
  const char prefix[] = "zcat ";
  char* cmd = malloc(sizeof(prefix) + strlen(fname) + 1);
  if (!cmd)
  {
     fprintf(stderr, "Cannot open gzipped file %s: malloc: %s\n", fname, strerror(errno));
     exit(1);
  }
  sprintf(cmd,"%s%s",prefix,fname);
  //printf("%s\n",cmd);
  FILE * in = popen(cmd, "r");
  if (!in)
  {
    fprintf(stderr, "Cannot open gzipped file %s: popen: %s\n", fname, strerror(errno));
    exit(1);
  }
  free(cmd);
  return in;
}

//V.1.5.mc
// check if this is a gzipped file.
bool checkgz(char* fname)
{
  //printf("Reading file %s\n",fname);
  FILE* fp = fopen ( fname , "rb" );
  char buffer[4]="";
  fread(buffer,1,2,fp);
  //printf("%X\n",(unsigned int)buffer[0]);
  //printf("%X\n",(unsigned char)buffer[1]);
  //printf("%d\n",(unsigned char)buffer[0]==0x1f);
  //printf("%d\n",(unsigned char)buffer[1]==0x8b);
  fclose(fp);
  return ((unsigned char)buffer[0]==0x1f) && ((unsigned char)buffer[1]==0x8b);
}

//++urgent
char* decompress(char* outpath,int chr, char *fname, char* type)
{
   if(checkgz(fname))
   {
      srand(time(NULL));
      int r = rand() % 100 + 1;
      static char newf[MAX_FILENAME_LEN] = "";
      sprintf(newf,"%s.%d.%s.%d.tmp",outpath,chr,type,r);

      if(VERBOSE)
         printf("-Decompressing %s to %s\n",fname,newf);
      const char prefix[] = "gunzip -f -c ";
      char* cmd = malloc(sizeof(prefix) + strlen(fname) + MAX_FILENAME_LEN);
      if (!cmd)
      {
        fprintf(stderr, "Cannot open gzipped file %s: malloc: %s\n", fname, strerror(errno));
        exit(1);
      }
      sprintf(cmd,"%s %s > %s",prefix,fname,newf);
      if(VERBOSE) printf("%s\n",cmd);
      system(cmd);
      free(cmd);
      if(VERBOSE)
         printf("Done decompression, new file name = %s\n",newf);
      return newf;
   }
   else { printf("-Error: to be here you must have a gzipped file\n");exit(0);}
   return NULL;
}
//--urgent

C_QUEUE* cq_init(size_t nsize, size_t chunkSize){

  C_QUEUE* c_queue = (C_QUEUE*) malloc(sizeof(C_QUEUE));

  c_queue->dat = malloc(chunkSize * nsize);
  c_queue->last = c_queue->dat + chunkSize * (nsize-1);
  c_queue->nsize=nsize;
  c_queue->chunkSize=chunkSize;
  c_queue->start = 0;
  c_queue->end = 0;
  
  return c_queue;
}

bool cq_isInQ(size_t curItem, C_QUEUE* c_queue){
  
  bool ret = false;
  
  if(curItem >= c_queue->start && curItem < c_queue->end)
    ret = true;
  
  return ret;
}

void* cq_getItem(size_t i, C_QUEUE* c_queue){
  
  size_t pos  = i%c_queue->nsize * c_queue->chunkSize;
  return c_queue->dat+pos;

}

void* cq_getNext(void* ptr, C_QUEUE* c_queue){
  
  if(ptr != c_queue->last)
    ptr +=  c_queue->chunkSize;
  else
    ptr = c_queue->dat;

  return ptr;

}

bool cq_isEmpty(C_QUEUE* c_queue){

  if(c_queue->start == c_queue->end )
    return true;
  else
    return false;
}

bool cq_isFull(C_QUEUE* c_queue){

  if( c_queue->end - c_queue->start >= c_queue->nsize)
    return true;

  return false;
}

void cq_resize(C_QUEUE * c_queue){

  size_t newSize = 2* c_queue->nsize;
  void * newDat = calloc(newSize, c_queue->chunkSize);
  if(newDat==NULL){
    printf("-Failed to allocate %lux%lu=%lu memory.\n",c_queue->chunkSize,newSize, c_queue->chunkSize*newSize);
    abort();
  }else{
    printf("-Memory usage increased to %g M\n",(double)c_queue->chunkSize*newSize/1024/1024 );
  }

  size_t startTrans = c_queue->start%c_queue->nsize;
  size_t endTrans = c_queue->end%c_queue->nsize;
  size_t len = c_queue->end - c_queue->start;

  if(len > 0){
    if(endTrans > startTrans) 
      memcpy(newDat+c_queue->start%newSize*c_queue->chunkSize, cq_getItem(c_queue->start, c_queue), len*c_queue->chunkSize);
    else if(endTrans <= startTrans){
      memcpy(newDat+c_queue->start%newSize*c_queue->chunkSize, cq_getItem(c_queue->start, c_queue), (c_queue->nsize - c_queue->start%c_queue->nsize)*c_queue->chunkSize);
      memcpy(newDat+c_queue->end/c_queue->nsize%2*c_queue->nsize*c_queue->chunkSize, c_queue->dat, (c_queue->end%c_queue->nsize + 1)*c_queue->chunkSize);
    }
   
  }else if(len < 0){
    printf("-The queue length is negative!\n");
    abort();
  }

  free(c_queue->dat);

  c_queue->dat = newDat;
  c_queue->nsize = newSize;
  c_queue->last = newDat + c_queue->chunkSize*(newSize-1);
  
}

void* cq_push(C_QUEUE* c_queue){

  void * ret;

  if(cq_isFull(c_queue)){
    printf("-Resize the queue from %lu to %lu\n", c_queue->nsize, c_queue->nsize*2);
    cq_resize(c_queue);
  }  
  ret = cq_getItem(c_queue->end, c_queue);
  
  c_queue->end ++;

  return ret;
}

//function to get one element from end 
void* cq_shift(C_QUEUE* c_queue){

  if(cq_isEmpty(c_queue)){
    printf("-Queue is empty.  Can't shift\n");
    abort();
  }

  c_queue->end--;
 
  return cq_getItem(c_queue->end, c_queue);
}

//function to get one element from start 
void* cq_pop(C_QUEUE* c_queue, bool remove){

  void * ret;

  if(cq_isEmpty(c_queue)){
    printf("-Queue is empty.  Can't pop\n");
    abort();
  }

  ret = cq_getItem(c_queue->start, c_queue);

  if(remove)
    c_queue->start ++;

  return ret;
}

char* getTime(){
  time_t raw_time;
  time(&raw_time);
  return ctime(&raw_time);
}

gsl_rng* initRand(int seed){
	
  const gsl_rng_type * T;
  gsl_rng * r;
  gsl_rng_env_setup();
	
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
	
  gsl_rng_set(r, seed);
	
  return r;
}

/*
void print_pheno(gsl_vector * array)
{
   int i = 0;
   //for(i=0;i<20;i++) printf("%.2g ",gsl_vector_get(array,i));
   //printf("\n");
   double s = 0;
   for(i=0;i<array->size;i++) s+= gsl_vector_get(array,i);
   printf("sum = %.2g\n",s);
}
*/

//vector operations
void shuffle(gsl_vector * array, int nSize, gsl_rng *r, int thr)
{
  int i, j;
  for(i=nSize; i>1; i--){
    j = gsl_rng_uniform_int(r, i);
    if(j != i ){
      gsl_vector_swap_elements(array, i-1, j);
    }
  }
  //printf("thr = %d loop = %d   ",thr,count); print_pheno(array);
}

void zeroArray_int(int* a, int nsample){
  int i;
  for(i=0; i<nsample; i++)
    a[i]=0;
}

double mean(double *v, int n, bool isGeno){
 
  if(isGeno) //check for missingness
  { 
    int i;
    double sum;
    sum=0;
    int k = 0;
    for(i=0; i<n; i++)
    {
       if(v[i]!=MISSING_VAL)
       {
         sum+=v[i];
         k++;
       }
    }
    return sum/k;
  }
  else
  {
    int i;
    double sum;
    sum=0;
    for(i=0; i<n; i++) sum+=v[i];
    return sum/n;
  }
}


double tss(double* v, int n,bool isGeno){
 
  if(isGeno) //check for missing data.
  { 
    int i;
    double sum;
    sum=0;
    double m = mean(v,n,isGeno);
    int k = 0;
    for(i=0; i<n; i++)
    {
      if(v[i]!=MISSING_VAL)
      { 
        sum+= v[i]*v[i];
        k++;
      }
    }
    if(INTERCEPT){
      return sum-k*m*m;
    }else{
      return sum;
    }
  }
  else
  {
    int i;
    double sum;
    sum=0;
    double m = mean(v, n,isGeno);
    for(i=0; i<n; i++)
    { 
       sum+= v[i]*v[i];
    }
    if(INTERCEPT){
      return sum-n*m*m;
    }else{
      return sum;
    }
  }
}

double variance(double* v, int n, bool isGeno){
 
  if(isGeno)
  { 
    int i;
    double sum;
    sum=0;
    double m = mean(v, n,isGeno);
    int k = 0;
    for(i=0; i<n; i++)
    {
      if(v[i]!=MISSING_VAL)
      {
        sum+= v[i]*v[i];
        k++;
      }
    }
    if(INTERCEPT){
      return sum/k-m*m;
    }else{
      return sum/k;
    }
  }
  else
  {
    int i;
    double sum;
    sum=0;
    double m = mean(v, n, isGeno);
    for(i=0; i<n; i++) sum+= v[i]*v[i];
    if(INTERCEPT){
      return sum/n-m*m;
    }else{
      return sum/n;
    }
  }
}

double covariance(double* v1, double* v2, int n, bool isGeno1, bool isGeno2){
 
  if(isGeno1 || isGeno2)
  { 
    int i;
    double sum;
    double s1;
    double s2;
    sum=0;
    s1=0;
    s2 =0;
    int k = 0;
    for(i=0; i<n; i++)
    {
      if(isGeno1 && isGeno2)
      {
        if(v1[i]!=MISSING_VAL && v2[i]!=MISSING_VAL)
        {
          sum += v1[i]*v2[i];
          s1 +=v1[i];
          s2 +=v2[i];
          k++;
        }
      }
      else if(isGeno1 && !isGeno2)
      {
        if(v1[i]!=MISSING_VAL)
        {
          sum += v1[i]*v2[i];
          s1 +=v1[i];
          s2 +=v2[i];
          k++;
        }
      }
      else if(!isGeno1 && isGeno2)
      {
        if(v2[i]!=MISSING_VAL)
        {
          sum += v1[i]*v2[i];
          s1 +=v1[i];
          s2 +=v2[i];
          k++;
        }
      }
    }
    if(INTERCEPT){
      return sum/k - s1/k*s2/k;
    }else{
      return sum/k;
    }
  }
  else
  {
    int i;
    double sum;
    double s1;
    double s2;
    sum=0;
    s1=0;
    s2 =0;
    for(i=0; i<n; i++)
    {
      sum += v1[i]*v2[i];
      s1 +=v1[i];
      s2 +=v2[i];
    }
    if(INTERCEPT){
      return sum/n - s1/n*s2/n;
    }else{
      return sum/n;
    }
  }
}

double correlation(double* v1, double* v2, int n, bool isGeno1, bool isGeno2){

  if(isGeno1 || isGeno2)
  {
    int i;
    double sum12;
    double sum1;
    double sum2;
    double sum22;
    double sum11;
  
    sum12=0;
    sum1=0;
    sum2 =0;
    sum11=0;
    sum22=0;
    int k = 0;

    for(i=0; i<n; i++)
    {
      if(isGeno1 && isGeno2)
      {
        if(v1[i]!=MISSING_VAL && v2[i]!=MISSING_VAL)
        {
          sum12 += v1[i]*v2[i];
          sum1 +=v1[i];
          sum2 +=v2[i];
          sum11 +=v1[i]*v1[i];
          sum22 +=v2[i]*v2[i];
          k++;
        }
      }
      else if(isGeno1 && !isGeno2)
      {
        if(v1[i]!=MISSING_VAL)
        {
          sum12 += v1[i]*v2[i];
          sum1 +=v1[i];
          sum2 +=v2[i];
          sum11 +=v1[i]*v1[i];
          sum22 +=v2[i]*v2[i];
          k++;
        }
      }
      else if(!isGeno1 && isGeno2)
      {
        if(v2[i]!=MISSING_VAL)
        {
          sum12 += v1[i]*v2[i];
          sum1 +=v1[i];
          sum2 +=v2[i];
          sum11 +=v1[i]*v1[i];
          sum22 +=v2[i]*v2[i];
          k++;
        }
      }
    }
    return (sum12/k - sum1/k*sum2/k)/sqrt((sum11/k-sum1/k*sum1/k)*(sum22/k-sum2/k*sum2/k));
  }
  else
  {
    int i;
    double sum12;
    double sum1;
    double sum2;
    double sum22;
    double sum11;
  
    sum12=0;
    sum1=0;
    sum2 =0;
    sum11=0;
    sum22=0;

    for(i=0; i<n; i++)
    {
        if(v1[i]!=MISSING_VAL && v2[i]!=MISSING_VAL)
        {
          sum12 += v1[i]*v2[i];
          sum1 +=v1[i];
          sum2 +=v2[i];
          sum11 +=v1[i]*v1[i];
          sum22 +=v2[i]*v2[i];
        }
    }
    return (sum12/n - sum1/n*sum2/n)/sqrt((sum11/n-sum1/n*sum1/n)*(sum22/n-sum2/n*sum2/n));
  }
}
//+V.1.4.mc
double correlation_and_covariance(double* v1, double* v2, int n, double * cov)
{
    int i;
    double sum12;
    double sum1;
    double sum2;
    double sum22;
    double sum11;
  
    sum12=0;
    sum1=0;
    sum2 =0;
    sum11=0;
    sum22=0;
    int k = 0;

    for(i=0; i<n; i++)
    {
        sum12 += v1[i]*v2[i];
        sum1 +=v1[i];
        sum2 +=v2[i];
        sum11 +=v1[i]*v1[i];
        sum22 +=v2[i]*v2[i];
        k++;
    }

    if(INTERCEPT)
      *cov = sum12/k - sum1/k*sum2/k;
    else
      *cov = sum12/k;
   
    return (sum12/k - sum1/k*sum2/k)/sqrt((sum11/k-sum1/k*sum1/k)*(sum22/k-sum2/k*sum2/k));
}
//-V.1.4.mc
//FIX HESSIAN LATEST V.1.2
double safe_log10_sum(double log_bf1,double log_bf2)
{
   //compute log(bf1 + bf2) from log(bf1) and log(bf2).
   if(gsl_finite(log_bf1)==0 && gsl_finite(log_bf2)==0)
      return GSL_NAN;
   else if(gsl_finite(log_bf1)==0 && gsl_finite(log_bf2)!=0)
      return log_bf2;
   else if(gsl_finite(log_bf1)!=0 && gsl_finite(log_bf2)==0)
      return log_bf1;

   if(log_bf1 > log_bf2)
      return log_bf1 + log10(1 + pow(10,log_bf2 - log_bf1));
   else
      return log_bf2 + log10(1 + pow(10,log_bf1 - log_bf2));
}

void spec_decomp_makePositive(gsl_matrix* X, gsl_matrix *Y){

  static int workspace_size = MAX_SNP_PER_GENE;
  static gsl_eigen_symmv_workspace  *w = NULL; 

  static double* eigen_value_data = NULL;
  static gsl_vector_view eigen_value;

  static double* t_data = NULL;
  
  static double* eigen_vector_data=NULL;
  static gsl_matrix_view eigen_vector;

  static int i, j, k, n;

   n = X->size1;

  if( eigen_value_data == NULL){
    if(w==NULL) w = gsl_eigen_symmv_alloc(MAX_SNP_PER_GENE); else abort();
    if(eigen_value_data==NULL) eigen_value_data = (double*) malloc(sizeof(double)*MAX_SNP_PER_GENE);else abort();
    if(t_data==NULL) t_data = (double*) malloc(sizeof(double)*MAX_SNP_PER_GENE);else abort();
    if(eigen_vector_data==NULL) eigen_vector_data = (double*) malloc(sizeof(double)*MAX_SNP_PER_GENE*MAX_SNP_PER_GENE); else abort();
  }
  if(n >  workspace_size){
    workspace_size=n;
    gsl_eigen_symmv_free(w); w = NULL;
    if(w==NULL) w = gsl_eigen_symmv_alloc(n); else abort();
    free(eigen_value_data); eigen_value_data=NULL;
    if(eigen_value_data==NULL) eigen_value_data = (double*) malloc(sizeof(double) * n);else abort();
    free(t_data);t_data=NULL;
    if(t_data==NULL)t_data = (double*) malloc(sizeof(double) * n);else abort();
    free(eigen_vector_data);eigen_vector_data=NULL;
    if(eigen_vector_data==NULL) eigen_vector_data = (double*) malloc(sizeof(double) * n*n);else abort();
  }

  //V.1.7.mc
  //printf("LD matrix size = %d \n",n);fflush(stdout); //REMOVE
  if(w==NULL || eigen_value_data==NULL || t_data==NULL || eigen_vector_data==NULL) {printf("-Failed to allocate memory\n");exit(1);}

  eigen_value = gsl_vector_view_array(eigen_value_data, n);
  eigen_vector = gsl_matrix_view_array(eigen_vector_data, n, n);

  gsl_eigen_symmv(X, &eigen_value.vector, &eigen_vector.matrix, w);

  //set negative eigen values to 0
  for (i=0; i <n; i++)
    if(gsl_vector_get(&eigen_value.vector, i)<0)
    gsl_vector_set(&eigen_value.vector, i, 0);
  //calculate the scaling vector
  for (i=0; i <n; i++){
    t_data[i] = 0;
    for (j=0; j <n; j++)
      t_data[i] += pow(gsl_matrix_get(&eigen_vector.matrix, i, j), 2) *gsl_vector_get(&eigen_value.vector, j);
    t_data[i] = 1/t_data[i];
  }
  
  //
  for (i=0; i <n; i++)
    for (j=0; j <=i; j++){
      double sum=0;
      for (k=0; k<n; k++)
	sum+=gsl_matrix_get(&eigen_vector.matrix, i, k)*gsl_matrix_get(&eigen_vector.matrix, j,k)*gsl_vector_get(&eigen_value.vector, k);
      gsl_matrix_set(Y, i, j, sum*sqrt( t_data[i])*sqrt( t_data[j]));
    }
  for (i=0; i <n; i++)
    for (j=i+1; j <n; j++)
      gsl_matrix_set(Y, i, j,  gsl_matrix_get(Y, j, i));

  //restore the LD matrix
  for (i=0; i< n; i++){
    gsl_matrix_set(X, i, i, 1);
    for (j=0; j<i; j++){
      gsl_matrix_set(X, i, j, gsl_matrix_get(X, j, i));
    }
  }
}

void Projection_S(gsl_matrix* X, gsl_matrix * R){

  static int workspace_size = MAX_SNP_PER_GENE;
  static gsl_eigen_symmv_workspace  *w = NULL; 

  static double* eigen_value_data = NULL;
  static gsl_vector_view eigen_value;
  
  static double* eigen_vector_data=NULL;
  static gsl_matrix_view eigen_vector;

  int n = R->size1;

  if( eigen_value_data == NULL){
    if(w==NULL) w = gsl_eigen_symmv_alloc(MAX_SNP_PER_GENE);else abort();
    if(eigen_value_data==NULL) eigen_value_data = (double*) malloc(sizeof(double)*MAX_SNP_PER_GENE);else abort();
    if(eigen_vector_data==NULL) eigen_vector_data = (double*) malloc(sizeof(double)*MAX_SNP_PER_GENE*MAX_SNP_PER_GENE);else abort();
  }
  if(n >  workspace_size){
    workspace_size=n;
    gsl_eigen_symmv_free(w);w=NULL;
    if(w==NULL)w = gsl_eigen_symmv_alloc(n);else abort();
    free(eigen_value_data);eigen_value_data=NULL;
    if(eigen_value_data==NULL) eigen_value_data = (double*) malloc(sizeof(double) * n);else abort();
    free(eigen_vector_data);eigen_vector_data=NULL;
    if(eigen_vector_data==NULL) eigen_vector_data = (double*) malloc(sizeof(double) * n*n);else abort();
  }

  eigen_value = gsl_vector_view_array(eigen_value_data, n);
  eigen_vector = gsl_matrix_view_array(eigen_vector_data, n, n);

  gsl_eigen_symmv(R, &eigen_value.vector, &eigen_vector.matrix, w);

  int i, j, k;
  for (i=0; i<n; i++){
    for(j=0; j<n; j++){
      double v=0;
      for(k=0; k<n; k++){
	v +=   gsl_matrix_get(&eigen_vector.matrix, i, k) *  gsl_matrix_get(&eigen_vector.matrix, j, k) * (gsl_vector_get(&eigen_value.vector, k) < 0? 0:gsl_vector_get(&eigen_value.vector, k));
      }
      gsl_matrix_set(X, i, j,  v);
    }
  }
  
}

void Projection_U(gsl_matrix * m){
  int i;
  for (i=0; i< m->size1; i++)
    gsl_matrix_set(m,i, i, 1);
}

void LDLt_decomp(gsl_matrix * m, gsl_vector *D, gsl_matrix *L){
  int i, j, k, n;

  if(m->size1 != m->size2){
    printf("-Correlation matrix has to be square: Dim 1: %Zud, Dim 2: %Zud\n", m->size1, m->size2);
    exit(1);
  }else{
    n = m->size1;
  }

  for(j=0; j<n; j++){
    double cumsum = 0;
    for(k=0; k<j; k++)
      cumsum += gsl_matrix_get(L, j, k)* gsl_matrix_get(L, j, k)* gsl_vector_get(D, k);
    gsl_vector_set(D, j, gsl_matrix_get(m, j, j) - cumsum);
    gsl_matrix_set(L, j, j, 1);
    for (i=0; i<n; i++){
      if(i<j){
	gsl_matrix_set(L, i, j,0 );
	continue;
      }
      double cumsum=0;
      for(k=0; k<j; k++)
	cumsum += gsl_matrix_get(L, i, k)* gsl_matrix_get(L, j, k)* gsl_vector_get(D, k);
      if(fabs(gsl_vector_get(D, j)) < EPS){
      	gsl_matrix_set(L, i, j,0 );
      }
      else
	gsl_matrix_set(L, i, j, (gsl_matrix_get(m, i, j) - cumsum)/gsl_vector_get(D, j) );
    }
  }
  
}

void makePositiveDef(gsl_matrix * m ){
  
  static double * R_dat=NULL;
  static gsl_matrix_view R;
  static double * S_dat=NULL;
  static gsl_matrix_view S;
  static double * X_dat=NULL;
  static gsl_matrix_view X;
  static double * Y_dat=NULL;
  static gsl_matrix_view Y;
  static double * Y0_dat=NULL;
  static gsl_matrix_view Y0;

  static int worksize = MAX_SNP_PER_GENE;

  int n;
  if(m->size1 != m->size2){
    printf("-Correlation matrix has to be square: Dim 1: %Zud, Dim 2: %Zud\n", m->size1, m->size2);
    exit(1);
  }else{
    n = m->size1;
  }

  if(R_dat==NULL){
    if(R_dat==NULL) R_dat=(double*)malloc(worksize*worksize*sizeof(double));else abort();
    if(S_dat==NULL) S_dat=(double*)malloc(worksize*worksize*sizeof(double));else abort();
    if(X_dat==NULL) X_dat=(double*)malloc(worksize*worksize*sizeof(double));else abort();
    if(Y_dat==NULL) Y_dat=(double*)malloc(worksize*worksize*sizeof(double));else abort();
    if(Y0_dat==NULL) Y0_dat=(double*)malloc(worksize*worksize*sizeof(double));else abort();
  }
  if(n >worksize ) {
    free(R_dat);R_dat=NULL;
    free(S_dat);S_dat=NULL;
    free(X_dat);X_dat=NULL;
    free(Y_dat);Y_dat=NULL;
    free(Y0_dat);Y0_dat=NULL;
    worksize=n;
    if(R_dat==NULL)R_dat=(double*)malloc(worksize*worksize*sizeof(double));else abort();
    if(S_dat==NULL)S_dat=(double*)malloc(worksize*worksize*sizeof(double));else abort();
    if(X_dat==NULL)X_dat=(double*)malloc(worksize*worksize*sizeof(double));else abort();
    if(Y_dat==NULL)Y_dat=(double*)malloc(worksize*worksize*sizeof(double));else abort();
    if(Y0_dat==NULL)Y0_dat=(double*)malloc(worksize*worksize*sizeof(double));else abort();
  }

    if(R_dat == NULL)
      abort();
    if(S_dat == NULL)
      abort();
    if(X_dat == NULL)
      abort();
    if(Y_dat == NULL)
      abort();
    if(Y0_dat == NULL)
      abort();


  R = gsl_matrix_view_array(R_dat, n, n);
  S = gsl_matrix_view_array(S_dat, n, n);
  X = gsl_matrix_view_array(X_dat, n, n);
  Y = gsl_matrix_view_array(Y_dat, n, n);
  Y0 = gsl_matrix_view_array(Y0_dat, n, n);

  gsl_matrix_memcpy(&Y.matrix, m);
  gsl_matrix_memcpy(&Y0.matrix, m);
  double delta;
  int nItr=0;
  do{
    delta=0;
    gsl_matrix_sub(&Y.matrix, &S.matrix);
    gsl_matrix_memcpy(&R.matrix, &Y.matrix);
    Projection_S(&X.matrix, &Y.matrix);
    int i, j;

    gsl_matrix_memcpy(&S.matrix, &X.matrix);
    gsl_matrix_sub(&S.matrix, &R.matrix);
    Projection_U(&X.matrix);
    gsl_matrix_memcpy(&Y.matrix, &X.matrix);

    for (i=0; i<n; i++){
      for(j=0; j<n; j++){
	//	printf("%d, %d, %g\n", i, j, gsl_matrix_get(&Y.matrix, i, j));
	delta+= fabs(gsl_matrix_get(&Y.matrix, i, j)- gsl_matrix_get(&Y0.matrix, i, j));
      }
    }
    nItr++;
    gsl_matrix_memcpy(&Y0.matrix, &Y.matrix);
    //printf("delta=%g, nItr=%d\n", delta, nItr);
  }while(delta > 1E-30 && nItr <5);
  // printf("delta=%g, nItr=%d\n", delta, nItr);
  gsl_matrix_memcpy(m, &Y.matrix);

}

unsigned int hash_key(void *p){
  unsigned char * str = (unsigned char*) p;
  unsigned int hash = 5381;
  int c;

  while ((c = *str++))
    hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

  return hash;
}

static int  keys_equal_fn ( void *key1, void *key2 ){

  return (0==strcmp((char*) key1, (char*) key2));
}

void scalePheno(PHENOTYPE* pheno){

  int i;

  for (i = 0; i < pheno->N_sample; i++){
    gsl_vector_set(pheno->pheno_array_org, i,  gsl_vector_get(pheno->pheno_array_org, i)/sqrt(pheno->tss_per_n));
    gsl_vector_set(pheno->pheno_array_reg, i,  gsl_vector_get(pheno->pheno_array_reg, i)/sqrt(pheno->tss_per_n)); 
    //Pritam, pheno_array_org and pheno_array_reg should be the same here.
  }

  //V.1.4.mc
  //pheno->mean=mean(pheno->pheno_array_org->data, pheno->N_sample,false);
  pheno->tss_per_n=variance(pheno->pheno_array_org->data, pheno->N_sample,false);
 
  //printf("-After scaling : phenotype mean = %g, phenotype variance = %g\n",  pheno->mean, pheno->tss_per_n);
  printf("-After scaling : phenotype variance = %g\n",  pheno->tss_per_n);
  
  //  fclose(fp);
}

void standardizePheno(PHENOTYPE* pheno){
  //rank the phenotypes and give the corresponding normal quantile to each individual
  gsl_permutation * perm = gsl_permutation_alloc(pheno->N_sample);
  gsl_permutation * rank = gsl_permutation_alloc(pheno->N_sample);
  int i=0;

  //for pheno_array_org
  gsl_sort_vector_index (perm, pheno->pheno_array_org);
  gsl_permutation_inverse (rank, perm);
  for (i = 0; i < pheno->N_sample; i++){
    double q = gsl_cdf_ugaussian_Qinv(((double)rank->data[i]+0.5)/(pheno->N_sample));
    gsl_vector_set(pheno->pheno_array_org, i, q);
  }

  //for pheno_array_reg
  gsl_sort_vector_index (perm, pheno->pheno_array_reg);
  gsl_permutation_inverse (rank, perm);
  for (i = 0; i < pheno->N_sample; i++){
    double q = gsl_cdf_ugaussian_Qinv(((double)rank->data[i]+0.5)/(pheno->N_sample));
    gsl_vector_set(pheno->pheno_array_reg, i, q);
  }

  //V.1.4.mc
  //pheno->mean=mean(pheno->pheno_array_org->data, pheno->N_sample,false);
  pheno->tss_per_n=variance(pheno->pheno_array_org->data, pheno->N_sample,false);
 
  gsl_permutation_free (perm);
  gsl_permutation_free (rank);
  printf("-After standardizing : phenotype variance = %g\n",  pheno->tss_per_n);
}

double Proj_X_Z(OrthNorm* Z, OrthNorm** bestZ, gsl_matrix *Cov){

  double ret = 0;
  double cor=  gsl_matrix_get(Cov, Z->snp->gene_id, bestZ[Z->k-2]->snp->gene_id);
  int k;

  for(k=2; k<Z->k; k++)
    ret += Z->sum_X_bestZ[k-2]*bestZ[Z->k-2]->sum_X_bestZ[k-2]/bestZ[k-2]->norm;
  if(NO_SIGN_FLIP && ((cor-ret)*cor < 0 || fabs((cor-ret)*cor)<EPS)){
    return 0;
  }else{
    return cor-ret;
  }
}

double getNormZ(OrthNorm * Z, OrthNorm** bestZ){

  if(pow(Z->sum_X_bestZ[Z->k-2], 2)/ bestZ[Z->k-2]->norm <0){
    printf("-Alert: %s, %g, %g\n", Z->snp->name, Z->sum_X_bestZ[Z->k-2],  bestZ[Z->k-2]->norm);
  }
  return Z->norm - pow(Z->sum_X_bestZ[Z->k-2], 2)/ bestZ[Z->k-2]->norm;
  
}


double Proj_P_Z(OrthNorm* Z, OrthNorm**  bestZ){

  return Z->projP - bestZ[Z->k-2]->projP*Z->sum_X_bestZ[Z->k-2]/bestZ[Z->k-2]->norm;

}

void calculateZ(OrthNorm* Z, OrthNorm** bestZ,  int k,  gsl_matrix* Cov){

  Z->k = k;
  double projP =0;
  double norm =0;
  //k=1, 2, ....
  //start at k=2
  //k=1 skipped
  
  if(k>1){
    Z->sum_X_bestZ[k-2] = Proj_X_Z(Z, bestZ, Cov);
    norm = getNormZ(Z, bestZ);
    projP= Proj_P_Z(Z, bestZ);

    //printf("calculateZ : %lg %lg\n",norm,projP); REMOVE

    if(NO_SIGN_FLIP && (projP * Z->projP <0 || fabs(projP * Z->projP)<EPS )){
      Z->projP=0;
      //printf("Set to %lg\n",Z->projP);REMOVE
    }else{
      if(BOUND_BETA && projP / Z->projP >= norm/Z->norm ){
	projP = Z->projP* (norm / Z->norm);
        //printf("Set to %lg\n",Z->projP);REMOVE
      }
      Z->projP=projP;
    }

    Z->norm = norm;
  }
}

double getSSM(OrthNorm*Z){

  return Z->projP/Z->norm * Z->projP;
}

//function for SNP chisq tests
double runTest(double *Geno, double *Pheno, int nsample,  double tss_per_n, double* c0, double* c1, double* rss, double* std_err,double missingness,int n_covariates){

  if(missingness > 0)
  {
    double cov[2][2];
 
    double geno[nsample];
    double pheno[nsample];
    int i = 0;
    int k = 0;
    for(i=0;i<nsample;i++)
    {
      if(Geno[i]!=MISSING_VAL)
      {
        geno[k] = Geno[i];
        pheno[k] = Pheno[i];
        k++;
      }
    }

    int df=k - n_covariates;

    if(INTERCEPT){
      gsl_fit_linear(geno, 1, pheno, 1, k, c0, c1, &cov[0][0], &cov[0][1], &cov[1][1], rss);
      df-=2;
    }else{
      gsl_fit_mul(geno, 1, pheno, 1, k, c1,  &cov[1][1], rss);    
      cov[0][0]=-1;
      cov[0][1]=-1;
      *c0=-1;
      df--;
    }

    *std_err = sqrt(cov[1][1]);
	
    if(gsl_isnan(*rss)){
      *c0=-1;
      *c1=-1;
      *rss = -1;
      return -1;
    }else{
      return (tss_per_n*nsample-(*rss))/(*rss) * df;
    }
  }
  else
  {
    double cov[2][2];
    int df=nsample - n_covariates;

    if(INTERCEPT){
      gsl_fit_linear(Geno, 1, Pheno, 1, nsample, c0, c1, &cov[0][0], &cov[0][1], &cov[1][1], rss);
      df-=2;
    }else{
      gsl_fit_mul(Geno, 1, Pheno, 1, nsample, c1,  &cov[1][1], rss);    
      cov[0][0]=-1;
      cov[0][1]=-1;
      *c0=-1;
      df--;
    }

    *std_err = sqrt(cov[1][1]);
	
    if(gsl_isnan(*rss)){
      *c0=-1;
      *c1=-1;
      *rss = -1;
      return -1;
    }else{
      return (tss_per_n*nsample-(*rss))/(*rss) * df;
    }
  }
}

//Pritam++
//function for regressing out covariates for linear regression 
void regCov(PHENOTYPE * pheno)
{
   //pheno_array_org  ---regressed---> pheno_array_reg
   if(pheno->n_covariates==0)
      return;
   if(VERBOSE)
     printf("-Regressing out the covariates for Linear Regression............\n");
   int p = pheno->n_covariates+1;
   int n = pheno->N_sample;
   gsl_matrix * X = gsl_matrix_alloc(n,p);
   gsl_matrix * cov = gsl_matrix_alloc(p,p);

   int i = 0;
   int j = 0;
   for(i=0;i<n;i++)
   {
      gsl_matrix_set(X, i, 0, 1);
      for(j=1;j<p;j++)
      {
         gsl_matrix_set(X, i, j, gsl_vector_get(pheno->covariates[j-1],i));
      }
   }
   gsl_vector * c = gsl_vector_alloc (p);
   gsl_multifit_linear_workspace * W = gsl_multifit_linear_alloc (n, p);
   double chisq = 0;
   gsl_multifit_linear(X, pheno->pheno_array_org, c, cov, &chisq, W);

   for(i=0;i<n;i++)
   {
      double v = 0;
      int j = 0;
      for(j=0;j<p;j++)
      {
         v += gsl_matrix_get(X,i,j)*gsl_vector_get(c,j);
      }
      //printf("r = %g\n",gsl_vector_get(pheno->pheno_array_org,i) - v);
      gsl_vector_set(pheno->pheno_array_reg,i,gsl_vector_get(pheno->pheno_array_org,i) - v); //pheno_array_org  ---regressed---> pheno_array_reg
   }
   //gsl_vector_fprintf(stdout,pheno->pheno_array_reg,"%g");
   gsl_vector_free(c);
   gsl_matrix_free(X);
   gsl_matrix_free(cov);
   if(W!=NULL) gsl_multifit_linear_free(W); //V.1.8

   //recompute these as the phenotype has now changed.
   //V.1.4.mc
   //pheno->mean = mean(pheno->pheno_array_reg->data, pheno->N_sample,false);
   pheno->tss_per_n = tss(pheno->pheno_array_reg->data, pheno->N_sample,false)/pheno->N_sample;
}

//to be called from Logistic.c
double runTest_logistic_fstat(double *geno, double *pheno, int nsample)
{
  double cov[2][2];
  int df=nsample;
  double rss = 0;
  double c0,c1;

  double tss_per_n = variance(pheno, nsample,false);
  if(INTERCEPT){
    gsl_fit_linear(geno, 1, pheno, 1, nsample, &c0, &c1, &cov[0][0], &cov[0][1], &cov[1][1], &rss);
    df-=2;
  }/*else{
    gsl_fit_mul(geno, 1, pheno, 1, nsample, c1,  &cov[1][1], rss);    
    cov[0][0]=-1;
    cov[0][1]=-1;
    *c0=-1;
    df--;
  }
  */   
  if(gsl_isnan(rss)){
    //*c0=-1;
    //*c1=-1;
    rss = -1;
    return -1;
  }else{
    return (tss_per_n*nsample-(rss))/(rss) * df;
  }
}
//--Pritam

int allele1234(char a){

  if(a == 'A' || a== 'a' || a == '1')
    return 1;
  if(a == 'C' || a== 'c' || a == '2')
    return 2;
  if(a == 'G' || a== 'g' || a == '3')
    return 3;
  if(a == 'T' || a== 't' || a == '4')
    return 4;
  
  return -1;
}

int alleleACGT(int a){

  if(a == 1)
    return 'A';
  if(a == 2)
    return 'C';
  if(a == 3)
    return 'G';
  if(a == 4)
    return 'T';
 
  printf("Error %d\n",a); exit(0); 
  return -1;
}

int flipAllele(char a){

  if(a == 'A' || a== 'a' || a == '1')
    return 4;
  if(a == 'C' || a== 'c' || a == '2')
    return 3;
  if(a == 'G' || a== 'g' || a == '3')
    return 2;
  if(a == 'T' || a== 't' || a == '4')
    return 1;
  
  return -1;
}

double compute_ref_freq(char * line_geno,int ncols)
{
  char name[SNP_ID_LEN];
  long bp;
  char a1,a2;
  double freq = 0;
  
  char * nextGeno;
  int k = 0;//indexes genotypes read.
  int i = 0;
  double last_geno = -1;
  double this_geno = -1;
  int chr = 0;
  for (i=0; i<ncols; i++)
  {
    nextGeno = strchr(line_geno, '\t');
    //truncate the string so that sscanf runs much faster
    if(nextGeno!=NULL) *nextGeno='\0';

    if(i==0)
       sscanf(line_geno,"%d",&chr);
    else if(i==1)
       sscanf(line_geno,"%s",name);
    else if(i==2)
       sscanf(line_geno,"%ld",&bp);
    else if(i==3)
       sscanf(line_geno,"%c",&a1);
    else if(i==4)
       sscanf(line_geno,"%c",&a2);
    else
       sscanf(line_geno, "%lg", &this_geno);
    if(last_geno<0)
       last_geno = this_geno;
    else
    {
        //got a pair of genotypes.
        double g = last_geno + this_geno; //convert to genotype.
        //printf("%d:(%lg %lg) %lg %lg\n",i,last_geno,this_geno,g,freq);
        freq += g;
        k++;
        last_geno = -1;
    }
    if(nextGeno!=NULL)
      line_geno = nextGeno+1;
  }
  freq = (freq)/(ncols-5);
  if(freq > 1.0) {printf("-compute_ref_freq: Error in snp %s freq= %lg\n",name,freq);} //urgent
  //printf("Pritam : compute_ref %lg %d %d\n",freq,i,ncols);
  return freq;
}

void Load_POS_TABLE_simple(FILE* fp, FILE* fp_exclude, FILE * fp_pos, FILE * fp_hap, struct hashtable * snp_pos_table, int ncols)
{
  if(VERBOSE) 
     printf("-Summary mode : In Load_POS_TABLE simple ...%d \n",ncols);
  char sline_frq[MAX_LINE_WIDTH];
  //char sline_hap[MAX_LINE_WIDTH];
  fpos_t prevPos;
  void * info;
  int pos;
  char a1, a2;
  char snp_id [SNP_ID_LEN];

  //Pritam : first load all the snps in hash table.
  fgetpos(fp, &prevPos);
  while(feof(fp) == 0 && fgets(sline_frq, MAX_LINE_WIDTH, fp) !=NULL)
  {
    char* key = (char*) malloc(sizeof(char)* SNP_ID_LEN);
    SNP_INFO_SUMMARY * info = (SNP_INFO_SUMMARY*) malloc(sizeof(SNP_INFO_SUMMARY));
    int status= sscanf(sline_frq, "%*d %s %d %*g",key, &pos); //chr snpid pos pvalue

    if(status != 2){
      printf("-Error in summary file format, found %d values\n", status);
      printf("-Error in line : %s \n", sline_frq);
      exit(1);
    }
    info->file_pos = -1; 
    info->pos = pos;
    info->a1 = -1;
    info->a2 = -1;
    info->exclude = SNP_INFO_SUMMARY_EXCLUDE_A1MISS;
    info->ref_maf = -1;
    info->ref_af1 = -1;
    hashtable_insert(snp_pos_table, key, info);
    //printf("Load_POS_Table : inserted %s\n",key);
  }
  fsetpos(fp, &prevPos); //reset file pointer to file start ?

  if(fp_exclude!=NULL)
  {
    //then remove snps mapped to multiple positions.
    while(feof(fp_exclude) == 0 && fgets(sline_frq, MAX_LINE_WIDTH, fp_exclude) !=NULL){
      int status= sscanf(sline_frq, "%s",snp_id);
      if(status != 1){
        printf("-Error in multipos file format, found %d values\n", status);
        printf("-Error in line : %s \n", sline_frq);
        exit(1);
      }
      info = hashtable_search(snp_pos_table, snp_id);
      if(info == NULL){
        continue;
      }else{
        ((SNP_INFO_SUMMARY* ) info)->exclude = SNP_INFO_SUMMARY_EXCLUDE_MULTIPOS;
        printf("-snp %s will be ignored because it is mapped to multiple loci\n", snp_id);
      }
    }
  }

  if(COMPUTE_LD)
  {
    long long file_pos = 0;
    int chr = 0;
    double af1 = 0; //af for allele 1 (among allele 0 and allele 1, the second is always the coded allele)
    while(feof(fp_pos) == 0 && fgets(sline_frq, MAX_LINE_WIDTH, fp_pos) !=NULL)
    {
      sscanf(sline_frq, "%d %s %lld %c %c %lg",&chr,snp_id, &file_pos, &a1, &a2,&af1);
      info = hashtable_search(snp_pos_table, snp_id);
      if(info == NULL)
      {
        continue;
      }
      else
      {
        //fseek(fp_hap,file_pos,SEEK_SET);
        //fgets(sline_hap,MAX_LINE_WIDTH,fp_hap);

        //double freq1 = compute_ref_freq(sline_hap,ncols);
        double freq1 = af1;
        if(freq1<0 || freq1>1) continue; //ignore snp, possible error in haplotype file.
        if(a1 != '-')
          ((SNP_INFO_SUMMARY* ) info)->a1 = allele1234(a1);
        if(a2 != '-')
          ((SNP_INFO_SUMMARY* ) info)->a2=allele1234(a2);
        if(((SNP_INFO_SUMMARY* ) info)->exclude ==SNP_INFO_SUMMARY_EXCLUDE_A1MISS)
          ((SNP_INFO_SUMMARY* ) info)->exclude = SNP_INFO_SUMMARY_INCLUDE;
        ((SNP_INFO_SUMMARY* ) info)->file_pos = file_pos;
        double MAF = (freq1>0.5)?(1.0-freq1):freq1;
        //printf("%s f=%lg maf=%lg\n",snp_id,freq1,MAF);
        ((SNP_INFO_SUMMARY* ) info)->ref_maf = MAF;
        ((SNP_INFO_SUMMARY* ) info)->ref_af1 = freq1;
        //printf("%s %lg\n",snp_id,MAF);
        //printf("%s %d %d\n",snp_id,((SNP_INFO_SUMMARY* ) info)->a1,((SNP_INFO_SUMMARY* ) info)->a2);
      }
    }//end while.
  }
  else { printf("COMPUTE_LD has to be true here\n"); exit(0); }
  printf("-Done Load simple\n");
}

//V.1.4.mc
void Load_POS_TABLE(FILE* fp, FILE* fp_exclude, FILE *fp_allele_info, FILE * fp_pos,struct hashtable * snp_pos_table)
{
  if(VERBOSE) 
     printf("-Summary mode : In Load_POS_TABLE ... \n");
  char sline_frq[MAX_LINE_WIDTH];
  fpos_t prevPos;
  void * info;
  int pos;
  char a1, a2;
  char snp_id [SNP_ID_LEN];

  //Pritam : first load all the snps in hash table.
  fgetpos(fp, &prevPos);
  while(feof(fp) == 0 && fgets(sline_frq, MAX_LINE_WIDTH, fp) !=NULL)
  {
    char* key = (char*) malloc(sizeof(char)* SNP_ID_LEN);
    SNP_INFO_SUMMARY * info = (SNP_INFO_SUMMARY*) malloc(sizeof(SNP_INFO_SUMMARY));
    int status= sscanf(sline_frq, "%*d %s %*s %*s %*g %*g %d %*g %*g %*g",key, &pos);

    if(status != 2){
      printf("-Error in summary file format, found %d values\n", status);
      printf("-Error in line : %s \n", sline_frq);
      exit(1);
    }
    info->file_pos = -1; //V.1.4.mc
    info->pos = pos;
    info->a1 = -1;
    info->a2 = -1;
    info->exclude = SNP_INFO_SUMMARY_EXCLUDE_A1MISS;
    hashtable_insert(snp_pos_table, key, info);
    //printf("Load_POS_Table : inserted %s\n",key);
  }
  fsetpos(fp, &prevPos); //reset file pointer to file start ?

  if(fp_exclude!=NULL)
  {
    //then remove snps mapped to multiple positions.
    while(feof(fp_exclude) == 0 && fgets(sline_frq, MAX_LINE_WIDTH, fp_exclude) !=NULL){
      int status= sscanf(sline_frq, "%s",snp_id);
      if(status != 1){
        printf("-Error in multipos file format, found %d values\n", status);
        printf("-Error in line : %s \n", sline_frq);
        exit(1);
      }
      info = hashtable_search(snp_pos_table, snp_id);
      if(info == NULL){
        continue;
      }else{
        ((SNP_INFO_SUMMARY* ) info)->exclude = SNP_INFO_SUMMARY_EXCLUDE_MULTIPOS;
        printf("-snp %s will be ignored because it is mapped to multiple loci\n", snp_id);
      }
    }
  }

  //+V.1.4.mc
  if(fp_allele_info!=NULL)
  {
    while(feof(fp_allele_info) == 0 && fgets(sline_frq, MAX_LINE_WIDTH, fp_allele_info) !=NULL){
      int status= sscanf(sline_frq, "%s %c %c",snp_id, &a1, &a2);
      //printf("%s => %c %c\n",snp_id,a1,a2);
      if(status != 3){
        printf("-Error in allele file format, found %d values\n", status);
        printf("-Error in line : %s \n", sline_frq);
        exit(1);
      }
      info = hashtable_search(snp_pos_table, snp_id);
      if(info == NULL){
        continue;
      }else{
        if(a1 != '-')
  	  ((SNP_INFO_SUMMARY* ) info)->a1 = allele1234(a1);
        if(a2 != '-')
	  ((SNP_INFO_SUMMARY* ) info)->a2=allele1234(a2);
        if(((SNP_INFO_SUMMARY* ) info)->exclude ==SNP_INFO_SUMMARY_EXCLUDE_A1MISS)
	  ((SNP_INFO_SUMMARY* ) info)->exclude = SNP_INFO_SUMMARY_INCLUDE;
      }
    }
  }
  else if(COMPUTE_LD)
  {
    long long file_pos = 0;
    int chr = 0;
    double af1 = 0; //af for allele 1 (among allele 0 and allele 1, the second is always the coded allele)
    while(feof(fp_pos) == 0 && fgets(sline_frq, MAX_LINE_WIDTH, fp_pos) !=NULL){
      sscanf(sline_frq, "%d %s %lld %c %c %lg",&chr,snp_id, &file_pos, &a1, &a2,&af1);
      info = hashtable_search(snp_pos_table, snp_id);
      if(info == NULL){
        continue;
      }else{

        double freq1 = af1;
        if(freq1<0 || freq1>1) continue; //ignore snp, possible error in haplotype file.

        if(a1 != '-')
          ((SNP_INFO_SUMMARY* ) info)->a1 = allele1234(a1);
        if(a2 != '-')
          ((SNP_INFO_SUMMARY* ) info)->a2=allele1234(a2);
        if(((SNP_INFO_SUMMARY* ) info)->exclude ==SNP_INFO_SUMMARY_EXCLUDE_A1MISS)
          ((SNP_INFO_SUMMARY* ) info)->exclude = SNP_INFO_SUMMARY_INCLUDE;
        ((SNP_INFO_SUMMARY* ) info)->file_pos = file_pos;
        double MAF = (freq1>0.5)?(1.0-freq1):freq1;
        //printf("%s f=%lg maf=%lg\n",snp_id,freq1,MAF);
        ((SNP_INFO_SUMMARY* ) info)->ref_maf = MAF;
        ((SNP_INFO_SUMMARY* ) info)->ref_af1 = freq1;
        //printf("%s %lg\n",snp_id,MAF);
      }
    }//end while.
  }
  //-V.1.4.mc
}

//Pritam++ to parse covariates
int parse(char* input, char output[][PHENOTYPE_VALUE_LEN],int nfields)
{
   int f = 0;
   int n = 0;
   char b[MAX_LINE_WIDTH];
   for (f=0; f<nfields; f++)
   {
      strcpy(output[f], "");
   }
   char *ptr = input;
   for (f=0; f<nfields; f++)
   {
      sscanf(ptr, "%[^\t\n]%n", b, &n);
      sscanf(b,"%s",output[f]);
      //printf("%d %s\n",f,output[f]);
      ptr += n; //advance the pointer by the number of chars read
      if (*ptr != '\t' || *ptr == '\n'){ return f+1;}
      ptr += 1; //skip the delimiter
      while (*ptr == '\t')
      {
         ptr += 1;
      }
   }
   return f;
}
//--Pritam

//function for reading phenotype file
int readPhenotype(char (*indiv_id)[INDIV_ID_LEN], double* indiv_val, int* nas, char* filename,
                  double indiv_cov[][MAX_N_COVARIATES],int n_covariates, int* SEX) //Pritam added covariates.
{
	
  FILE *fp;
  char s_phenotype[PHENOTYPE_VALUE_LEN];
  char sline[MAX_LINE_WIDTH];
  char s_sex[PHENOTYPE_VALUE_LEN];
  int nsample;
  
  bool isgz = false; //BIG BUG
  //V.1.5.mc
  if(!checkgz(filename))
     fp = fopen(filename, "r");
  else
  {
     isgz = true;
     fp = gzopen(filename);
  }

  nsample=0;
  (*nas) = 0;

  fgets(sline, MAX_LINE_WIDTH, fp); //skip header line.
	
  while(!feof(fp)){
    strcpy(sline, "");
    fgets(sline, MAX_LINE_WIDTH, fp);
		
    //skip header line
    if(sline[0]=='#') continue;
    //skip blank line
    if(strlen(sline) == 0 ) continue;
		
    //int nstatus = sscanf(sline,"%*s %s %*s %*s %*s %s\n", *indiv_id, s_phenotype); //Pritam
    //++Pritam	
    char output[6+n_covariates][PHENOTYPE_VALUE_LEN];
    int nstatus = parse(sline,output,6+n_covariates);
    strcpy(*indiv_id,output[1]);
    strcpy(s_sex,output[4]);
    strcpy(s_phenotype,output[5]);

    if(sscanf(s_sex, "%d", SEX) != 1){
	  printf("-Wrong format in phenotype line : %s \n Sex should be 1/2, quitting...\n",sline);
	  exit(1);
    }
    else
      SEX++;
    //--Pritam

    //if(nstatus == 2){//Pritam
    if(nstatus == 6+n_covariates)//Pritam
    {//Pritam
   
      //check for missing in cov/pheno.
      if(s_phenotype[0] == 'N'){
	(*nas)++;
	continue;
      }
      //++Pritam
      int k = 0;
      bool missing_cov = false;
      for(k=0;k<n_covariates;k++)
      {
         if(output[6+k][0] == 'N')
         {
            (*nas)++;
            //printf("missing c\n");
            missing_cov = true;
            break;
         }
         else
         {
            sscanf(output[6+k],"%lf",&indiv_cov[nsample][k]);
         }
      }
      if(missing_cov)
         continue;
      //else{
      //--Pritam
	if(sscanf(s_phenotype, "%lf", indiv_val) != 1){
	  printf("-Wrong format in phenotype line (%d) : %s\n quitting...\n", nsample, sline);
	  exit(1);
	}
      //}//Pritam
      //--Pritam
      indiv_id++;
      indiv_val++;
      nsample++;
    }else{
      printf("-Wrong format in phenotype line (%d) : %s", nsample, sline);
      printf("-possibly not tab delimited or extra covariates ?\n");
      printf("-Expected %d values, got %d values instead\n quitting...\n",(6+n_covariates),nstatus);
      exit(1);
    }
  }
  if(!isgz)
    fclose(fp);
  else
    pclose(fp);
  return(nsample);
}

//search through an array for an item
double* array_search(char id[INDIV_ID_LEN], int nsample, char (*indiv_id)[INDIV_ID_LEN],  double* indiv_val){

  int i;

  for(i=0; i<nsample; i++)
    if(strcmp(id, *indiv_id)==0)
      return indiv_val;
    else{
      indiv_val++;
      indiv_id++;
    }

  return NULL;

}

//++Pritam
//search through an array for an item, search for id in array indiv_id
int array_search_new(char id[INDIV_ID_LEN], int nsample, char (*indiv_id)[INDIV_ID_LEN],  double* indiv_val)
{
  int i;
  for(i=0; i<nsample; i++)
  {
    //printf("Comparing [%s] [%s]\n",id,*indiv_id);
    if(strcmp(id, *indiv_id)==0)
    {
      //printf("Got %d\n",i);
      return i; //return the index
    }
    else
    {
      indiv_val++;
      indiv_id++;
    }
  }
  return -1;
}
//--Pritam

//matching the individuals in phenotype and genotype files
//int sortPhenotype(char (*indiv_id)[INDIV_ID_LEN], double* indiv_val, int nsample, double * sorted_phenotype, bool *NA, int * nindiv, char* indiv_id_file){ //Pritam 
//++Pritam
int sortPhenotype(char (*indiv_id)[INDIV_ID_LEN],
                  double* indiv_val,
                  int nsample,
                  double * sorted_phenotype,
                  bool *NA,
                  int * nindiv,
                  char* indiv_id_file,
                  double ** cov_dat,
                  double (* indiv_cov)[MAX_N_COVARIATES],
                  int n_covariates
                  )
{
//--Pritam
  FILE * fp;
  bool isgz = false; //BIG BUG
  if(!checkgz(indiv_id_file))
     fp = fopen(indiv_id_file, "r");
  else
  {
     isgz = true;
     fp = gzopen(indiv_id_file);
  }

  //double* found; //Pritam
  int i=0;
  char id[INDIV_ID_LEN];
  *nindiv = 0;
  while(!feof(fp)){
    if(fscanf(fp, "%s", id) != 1) break;
    if(strlen(id) == 0)  break;
    int index = array_search_new(id, nsample, indiv_id, indiv_val);
    //printf("Searching for %s (nsample : %d) , got index = %d\n",id,nsample,index);
    if(index >= 0)
    {
      sorted_phenotype[i] = indiv_val[index];
      //printf(" -> %s\t%lg\t",id,indiv_val[index]);
      int k = 0;
      for(k=0;k<n_covariates;k++)
      {
          cov_dat[k][i] = indiv_cov[index][k];
      }
      i++;
      *NA = false;
    }else{
      *NA = true;
    }
    (*nindiv)++;
    NA++;
  }
  if(!isgz)
    fclose(fp);
  else
    pclose(fp);

  printf("-%d out of %d individuals have matching genotype. There are %d total individuals having avaliable genotype.\n", i, nsample, *nindiv);
  if(i<=0 || *nindiv <=0)
  {
     printf("-Error, no of indviduals with available genotype cannot be <= 0, please check the data\n");
     exit(1);
  }
  return i;
}

//get the phenotype data structure from the file
PHENOTYPE * getPhenotype(char* tfam_file, char* indiv_id_file,int n_covariates) //Pritam covariates
{
  PHENOTYPE * phenotype = (PHENOTYPE*) malloc(sizeof(PHENOTYPE));
  phenotype->pheno_array_org = NULL;
  phenotype->pheno_array_reg = NULL;
  phenotype->pheno_array_log = NULL;

  int k = 0;
  for(k=0;k<MAX_N_COVARIATES;k++)
  {
     phenotype->covariates[k] = NULL;
  }

  phenotype->n_covariates = n_covariates; //Pritam covariates 
  //read phenotype from tfam
  //printf("-Time: %s", getTime());
  printf("-Start to load phenotypes from %s\n", tfam_file);
  
  char indiv_id[MAX_N_INDIV][INDIV_ID_LEN];
  double indiv_val[MAX_N_INDIV];
  double indiv_cov[MAX_N_INDIV][MAX_N_COVARIATES]; //Pritam covariates
  int nsample, nindiv, nna;
  
  nsample = readPhenotype(indiv_id, indiv_val, &nna, tfam_file
                         ,indiv_cov, n_covariates,phenotype->SEX);//Pritam covariates

  printf("-Done loading phenotypes and covariates\n");//Pritam


  nindiv= nsample + nna;

  printf("-%d total individuals, %d have avaliable phenotype and covariates\n", nindiv, nsample);//Pritam

  double * pheno_dat = (double*) malloc (sizeof(double) * nsample);

  //++Pritam
  double * cov_dat[MAX_N_COVARIATES];
  for(k=0;k<n_covariates;k++)
  {
    cov_dat[k] = (double*) malloc (sizeof(double) * nsample); //one array per covariate
  }
  //--Pritam

  printf("-Start to load individual IDs from %s\n", indiv_id_file);

  phenotype->N_sample = sortPhenotype(indiv_id, indiv_val, nsample, pheno_dat, phenotype->NA, &phenotype->N_indiv, indiv_id_file
                                      , cov_dat, indiv_cov, n_covariates); //Pritam covariates

  printf("-Done sorting individuals\n");
  printf("-Total samples = %d, available samples = %d\n",nindiv,phenotype->N_sample);//Pritam

  phenotype->pheno_array_org = gsl_vector_alloc(phenotype->N_sample);
  phenotype->pheno_array_reg = gsl_vector_alloc(phenotype->N_sample);

  //++Pritam
  phenotype->pheno_array_log = gsl_vector_alloc(phenotype->N_sample);
  for(k=0;k<n_covariates;k++)
  {
     phenotype->covariates[k] = gsl_vector_alloc(phenotype->N_sample);
  }
  //--Pritam

  int i;
  double v1 = GSL_NEGINF;
  double v2 = GSL_NEGINF;
  bool BINARY_PHENO = true;//Pritam
  for(i=0; i<phenotype->N_sample; i++)
  {
    gsl_vector_set(phenotype->pheno_array_org, i, pheno_dat[i]);
    gsl_vector_set(phenotype->pheno_array_reg, i, pheno_dat[i]);
    gsl_vector_set(phenotype->pheno_array_log, i, pheno_dat[i]);
    for(k=0;k<n_covariates;k++)
    {
       gsl_vector_set(phenotype->covariates[k], i , cov_dat[k][i]);
    }
    if(v1==GSL_NEGINF)
       v1 = pheno_dat[i];
    else if(v1!=pheno_dat[i])
    {
       if(v2==GSL_NEGINF)
          v2=pheno_dat[i];
       else if(v2!=pheno_dat[i])
          BINARY_PHENO = false;
    }
  }
  if(v2==GSL_NEGINF)
     v2 = v1;
  
  if(BINARY_PHENO)
  {
    if((v1+v2)==0 && v1*v2==-1)
    {
      printf("-Found binary phenotype with values %f and %f\n",v1,v2);
      for(i=0; i<phenotype->N_sample; i++)
      {
        if(pheno_dat[i]==v1)
        {
           count_1++;
        }
        else if(pheno_dat[i]==v2)
        {
           count_2++;
        }
      }
    }
    else
    {
      printf("-Found binary phenotype with values %f and %f, converting to +1/-1\n",v1,v2);
      if(v1>v2)
      {
        double temp = v1;
        v1 = v2;
        v2 = temp; 
      }
 
      for(i=0; i<phenotype->N_sample; i++)
      {
        if(pheno_dat[i]!=v1 && pheno_dat[i]!=v2)
        {
           BINARY_PHENO = false;
           break;
        }
        if(pheno_dat[i]==v1)
        {
           count_1++;
           gsl_vector_set(phenotype->pheno_array_log, i, -1.0);
        }
        else if(pheno_dat[i]==v2)
        {
           count_2++;
           gsl_vector_set(phenotype->pheno_array_log, i, 1.0);
        }
      }
    }
  }

  free(pheno_dat); pheno_dat=NULL;

  for(k=0;k<n_covariates;k++)
  {
     free(cov_dat[k]);cov_dat[k]=NULL;
  }

  if(NEED_SNP_LOGISTIC) //Pritam
  {
     if(BINARY_PHENO==false)
     {
        //printf("*************** Warning : GET_GENE_PVAL_LOGISTIC/GET_SINGLE_SNP_LOGISTIC set to true\n");
        printf("\n-WARNING : Phenotype not binary, not doing logistic regression\n\n");
        GET_MINSNP_LOGISTIC = GET_MINSNP_PVAL_LOGISTIC = false;
        GET_MINSNP_P_PVAL_LOGISTIC = false;
        GET_GENE_BIC_LOGISTIC = GET_GENE_BIC_PVAL_LOGISTIC = false;
        GET_BF_LOGISTIC = GET_BF_PVAL_LOGISTIC = false;
        GET_VEGAS_LOGISTIC = GET_VEGAS_PVAL_LOGISTIC = false;
        GET_GATES_LOGISTIC = false;
        NEED_SNP_LOGISTIC = false;
        //gsl_vector_free(phenotype->pheno_array_log); //V.1.5.mc
        //phenotype->pheno_array_log = NULL; //V.1.5.mc
     }
  }
  //--Pritam

   //V.1.4.mc
  //phenotype->mean = mean(phenotype->pheno_array_org->data, phenotype->N_sample,false);
  phenotype->tss_per_n = tss(phenotype->pheno_array_org->data, phenotype->N_sample,false)/phenotype->N_sample;

  //printf("-Phenotype mean: %f, variance: %f\n", phenotype->mean, phenotype->tss_per_n);
  printf("-Phenotype variance: %f\n", phenotype->tss_per_n);

  return phenotype;
}

//read a line from the genotype file, format the data and save it in the SNP data structure
int readGeno( char* name, int * chr, double * pos, int * bp, double * AF1, double *R2, double * geno, int nsample, char * sline_geno, char* sline_snp, char * sline_mlinfo,  bool * NA, double * miss, char * a0, char * a1){
 
  static char temp_id[SNP_ID_LEN + 10]; 
  int status = sscanf(sline_snp, "%s %d %lg %d", temp_id, chr, pos, bp);
  if(strlen(temp_id)<SNP_ID_LEN)
  {
     //printf("copied %d\n",strlen(temp_id));   
     strcpy(name,temp_id);
  }
  else
  {
     //printf("copied %d\n",SNP_ID_LEN-2);   
     strncpy(name,temp_id,SNP_ID_LEN-2);
     name[SNP_ID_LEN-2] = '\0';
  }

  int i;
  char * nextGeno;

  if(status != 4){
    printf("-Skipping snp: %s", sline_snp);
    return 0;
  }
  
  if(strlen(sline_mlinfo) == 0) {
    *AF1 = 0.5;
    *R2 = 1;
  }else
    sscanf(sline_mlinfo, "%*s %c %c %*g %lg %lg", a0, a1, AF1, R2); //FIX a0 a1

  //++FIX MONOMORPHIC SNPS V.1.2
  int k = 0;//indexes genotypes read.
  double first_geno;
  bool monomorphic = true;
  double maf = 0.0;
  //--FIX MONOMORPHIC SNPS V.1.2
    
  for (i=0; i<nsample; i++){
    
    nextGeno = strchr(sline_geno, '\t');
    
    //truncate the string so that sscanf runs much faster
    if(nextGeno!=NULL) *nextGeno='\0';

    if(sscanf(sline_geno, "%lg", geno) != 1){
      printf("-Error in tped file for mode=genotype\n");
      exit(1);
    }

    if(*geno==MISSING_VAL)
      (*miss)++; 
    
    if(nextGeno!=NULL)
      sline_geno = nextGeno+1;
	
    if(NA[i]) continue;

    //++FIX MONOMORPHIC SNPS V.1.2
    if(k==0)
      first_geno = *geno;
    if(monomorphic)
    {
       if(*geno!=first_geno)
          monomorphic = false;
    }
    k++;
    maf += *geno;
    //--FIX MONOMORPHIC SNPS V.1.2

    geno++;
  }

  //++FIX MONOMORPHIC SNPS V.1.2
  maf /= (2*k);
  //if(maf>0.5) maf = 1.0 - maf;
  *AF1 = maf; //V.1.5.mc

  //printf("snp=%s %f\n",name,maf);
  if(monomorphic)
    return -1;
  else
    return 1;
  //--FIX MONOMORPHIC SNPS V.1.2
}

//read a line from the genotype file, format the data and save it in the SNP data structure
int readGeno_impute2( char* name, int * bp, double * AF1, double *R2, double * geno, int nsample, char * sline_geno, char* sline_info, bool * NA, double * miss, char * a0, char * a1, int * sex_dat, int chr)
{
  static char temp_id[SNP_ID_LEN + 10];
                                  //id rs bp f1 info crt type t0 t1  r2 
  int status = sscanf(sline_info, "%*s %s %d %lg %lg %*g %*g %*g %*g %*g", temp_id,bp,AF1,R2);
  if(strlen(temp_id)<SNP_ID_LEN)
  {
     //printf("copied %d\n",strlen(temp_id));
     strcpy(name,temp_id);
  }
  else
  {
     strncpy(name,temp_id,SNP_ID_LEN-2);
     name[SNP_ID_LEN-2] = '\0';
  }

  int i;

  if(status != 4){
    printf("-Incorrect impute2 info file format, Skipping snp: %s\n", sline_info);
    return 0;
  }

  //no longer the case. if(*AF1 <= 0.000001) return -1; //monomorphic snp 
  *a0 = 'w';
  *a1 = 'z';
  //snp id 
  char * pch = strtok(sline_geno," \t"); if(pch==NULL) {printf("-Error in impute2 genotype file for mode=genotype at snp id %s\n",name); return(-1);} 
  //snp name
  pch = strtok(NULL," \t");if(pch==NULL) {printf("-Error in impute2 genotype file for mode=genotype at snp name %s\n",name); return(-1);}
  static char XX[50] = "";
  sscanf(pch, "%s", XX);
  //bp
  pch = strtok(NULL," \t");if(pch==NULL) {printf("-Error in impute2 genotype file for mode=genotype at snp position for %s\n",name); return(-1);}
  //a0
  pch = strtok(NULL," \t");if(pch==NULL) {printf("-Error in impute2 genotype file for mode=genotype at snp allele0 for %s\n",name); return(-1);}
  status = sscanf(pch, "%c", a0);if(status!=1) {printf("-Error in impute2 genotype file for mode=genotype at snp allele0 for %s\n",name); return(-1);}
  //a1
  pch = strtok(NULL," \t");if(pch==NULL) {printf("-Error in impute2 genotype file for mode=genotype at snp allele1 for %s\n",name); return(-1);}
  status = sscanf(pch, "%c", a1); if(status!=1) {printf("-Error in impute2 genotype file for mode=genotype at snp allele1 for %s\n",name); return(-1);}
 
  //printf("Read %s %s %c %c\n",XX,name,*a0,*a1);

  char * nextProb = NULL; 
  //V.1.5.mc minor allele count (MAC) to detect monomorphic snps.
  double dosage_sum = 0;
  double valid_samples = 0;

  for (i=0; i<nsample; i++)
  {
     double dosage = 0;
     double prob = 0;
     bool missing = false;
     nextProb = strtok(NULL," \t");if(nextProb==NULL) {printf("-Error in impute2 genotype file for mode=genotype at prob 0 for %s\n",name); return(-1);}
     status = sscanf(nextProb, "%lg", &prob);if(prob==MISSING_VAL) missing = true;
     if(status!=1) {printf("-Error in impute2 genotype file for mode=genotype at prob 0 for %s\n",name); return(-1);}

     //printf("%d prob0 = %g\n",i,prob);
     nextProb = strtok(NULL," \t");if(nextProb==NULL) {printf("-Error in impute2 genotype file for mode=genotype at prob 1 for %s\n",name); return(-1);}
     status = sscanf(nextProb, "%lg", &prob);
     if(status!=1) {printf("-Error in impute2 genotype file for mode=genotype at prob 1 for %s\n",name); return(-1);}

     //printf("%d prob1 = %g\n",i,prob);
     if(prob==MISSING_VAL)
        missing = true;
     else
        dosage += prob;
     nextProb = strtok(NULL," \t");if(nextProb==NULL) {printf("-Error in impute2 genotype file for mode=genotype at prob 2 for %s\n",name); return(-1);}
     status = sscanf(nextProb, "%lg", &prob);
     if(status!=1) {printf("-Error in impute2 genotype file for mode=genotype at prob 2 for %s\n",name); return(-1);}
     //printf("%d prob2 = %g\n\n",i,prob);

     if(NA[i]) continue; //ignore this genotype.
     if(chr==23) //X chromosome
     {
        //check for sex correctness.
        if(sex_dat[i]!=1 && sex_dat[i]!=2)
        {
          printf("-Error for X chromosome analysis : sex should be 1/2 in phenotype file\n"); 
          exit(1);
        }
        if(sex_dat[i]==1)//male hemizygote
           dosage = 0; //ignore prob1 as dosage should be 0 or 2.
     }
     if(prob==MISSING_VAL)
        missing = true;
     else
        dosage += 2*prob;
     if(missing)
     {
       dosage = MISSING_VAL;
       (*miss)++; 
     }
     *geno = dosage;
     dosage_sum += dosage;
     valid_samples++;
     geno++;
  }

  //V.1.5.mc MAC
  double allele_count1 = dosage_sum;
  double allele_count2 = 2*valid_samples - dosage_sum;
  double MAC = allele_count1;
  if(MAC > allele_count2)
     MAC = allele_count2;
  *AF1 = dosage_sum/(2*valid_samples);
  //if(*MAF > 0.5) *MAF = 1.0 - *MAF;
  if(MAC<3) //monomorphic snp, less than 3 rare variants.
     return -1;
  return 1;
}

// assign SNPs to gene
int assignSNP2Gene(GENE** startGene, SNP* snp, int curSNP_id, GENE** readyGene, FILE * fp,SNP_CNT * snp_cnt)//FIX MONOMORPHIC SNPS V.1.2
{
  //FIX NEW MONO V.1.2 : just update counts
  if(snp!=NULL)
  {
    //printf("assignSNP2Gene : %s eSampleSize = %g\n",snp->name,snp->eSampleSize);
    if(snp->eSampleSize <= n_hat_cutoff || snp->R2 <= r2_cutoff || snp->missingness > MAX_MISSINGNESS || snp->MAF < maf_cutoff) //V.1.4.mc
    {
      //++FIX MONOMORPHIC SNPS V.1.2
      if(snp->missingness > MAX_MISSINGNESS)
      {
        if(VERBOSE)
            printf("-Omitting snp %s with missingness = %f > %f\n",snp->name,snp->missingness,MAX_MISSINGNESS);
        fprintf(fp_log,"Omitting snp %s with missingness = %f > %f\n",snp->name,snp->missingness,MAX_MISSINGNESS);
        snp_cnt->nSNP_missingness++;
      }
      if(0 < snp->eSampleSize && snp->eSampleSize <= n_hat_cutoff)
      {
         if(VERBOSE)
            printf("-Omitting snp %s with snp->eSampleSize=%f <= N_HAT_CUTOFF=%d\n",snp->name,snp->eSampleSize,n_hat_cutoff);
         fprintf(fp_log,"Omitting snp %s with snp->eSampleSize=%f <= N_HAT_CUTOFF=%d\n",snp->name,snp->eSampleSize,n_hat_cutoff);
         snp_cnt->nSNP_esamplesize++;
      }
      if(snp->R2 <= r2_cutoff)
      {
        if(VERBOSE)
           printf("-Omitting snp %s with snp->R2=%f <= R2_CUTOFF=%f\n",snp->name,snp->R2,r2_cutoff);
        fprintf(fp_log,"Omitting snp %s with snp->R2=%f <= R2_CUTOFF=%f\n",snp->name,snp->R2,r2_cutoff);
        snp_cnt->nSNP_quality++;
      }
      if(snp->MAF <= maf_cutoff) //V.1.4.mc
      {
        if(VERBOSE)
           printf("-Omitting snp %s with maf=%g <= maf_cutoff=%g\n",snp->name,snp->MAF,maf_cutoff);
        fprintf(fp_log,"Omitting snp %s with maf=%g <= maf_cutoff=%g\n",snp->name,snp->MAF,maf_cutoff);
        snp_cnt->nSNP_small_maf++;
      }
      //--FIX MONOMORPHIC SNPS V.1.2
    }
  }
  
  int nReadyGene = 0;
  bool firstRun = true;
  GENE* gene=*startGene;
  for(; gene != NULL; gene = gene->next){
    
    if(gene->skip) continue;
    
    int downstream = gene->bp_end + FLANK;
    int upstream = gene->bp_start - FLANK;
    if(snp == NULL){
      *readyGene= gene;
      readyGene++;
      nReadyGene++;
      gene->skip =true;
      continue;
    }
    
    if( snp->chr > gene->chr){
      printf("-SNP %s is on chr %d!!\n", snp->name, snp->chr);
      exit(1);
    }
    
    //upstream is not decreasing
    if(snp->bp <upstream){
      break;
    }
    else if(snp->bp > downstream){
      *readyGene= gene;
      readyGene++;
      nReadyGene++;
      gene->skip =true;
    }else{
      if(snp->eSampleSize<0 || (snp->eSampleSize > n_hat_cutoff && snp->R2 > r2_cutoff && snp->missingness <= MAX_MISSINGNESS && snp->MAF >= maf_cutoff)) //V.1.4.mc
      {
	//R2 here is the imputation quality
	if(firstRun){
	  firstRun = false;
	  *startGene = gene;
	}
	if(gene->snp_start == -1){
	  gene->snp_start = curSNP_id;
	  gene->nSNP = 0;
	}
	gene->snp_end = curSNP_id;
	gene->nSNP++;
	snp->nGene++;
        //printf("Assigned %s to %s\n",snp->name,gene->name);
      }
      fprintf(fp, "%s\t%d\t%d\t%s\t%s\t%d\t%d\t%g\t%g\t%g\n", snp->name, snp->chr, snp->bp, gene->ccds, gene->name, gene->bp_start, gene->bp_end, snp->MAF, snp->R2, snp->eSampleSize);
    }
  }
  return nReadyGene;
}

//remove 1 from the nGene counter (tells how many genes this SNP is assigned to).
void cleanSNPinGene(GENE* gene, C_QUEUE* snp_queue){

  int i;
  SNP* snp;
  
  for(i = gene->snp_start; i <= gene->snp_end; i++){
    snp = (SNP*) cq_getItem(i, snp_queue);
    snp->nGene--;
    //printf("SNP %s now has %d genes\n",snp->name,snp->nGene);
  }
}

// remove SNPS that have nGene = 0
void cleanSNPQ(C_QUEUE* snp_queue){

  SNP* snp;

  //keep at least the most recent SNP
  while(snp_queue->end-snp_queue->start > 1){
    snp = (SNP*)cq_pop(snp_queue,  false);
    
    if(snp->nGene ==0){
      cq_pop(snp_queue,  true);
      if(snp->geno != NULL){
	free(snp->geno);
	snp->geno=NULL;
      }
      //+V.1.4.mc
      if(snp->ref_geno!=NULL){
         free(snp->ref_geno);
         //if(VERBOSE)
         //  printf("Freeing ref geno of %s\n",snp->name);
         snp->ref_geno = NULL;
      }
  
      if(snp->correlated_snps!=NULL)
      {
        //printf("Pritam freeing : %s %p size = %d\n",snp->name,snp->correlated_snps,hashtable_count(snp->correlated_snps));
        hashtable_destroy(snp->correlated_snps, 1);
        snp->correlated_snps = NULL;
      }     
      //-V.1.4.mc
      if(snp->r != NULL){
        //printf("Pritam freeing r : %p\n",snp->r);
	free(snp->r);
	snp->r=NULL;
      }
      if(CQ_DEBUG)
	printf("popped, (start, end) = (%lu, %lu)\n", snp_queue->start, snp_queue->end);
      
    }
    else  break;
  }
}

//create a SNP data structure
void createSNP(SNP* snp, char* name, double pos, int bp, int chr, double AF1, double R2, double eSampleSize, double miss, char a0, char a1){
	
  strcpy(snp->name, name);
  snp->gene_id = -1;
	
  snp->pos=pos;
  snp->bp=bp;
  snp->chr=chr;
  snp->AF1=AF1;
  snp->MAF=(AF1<0.5)?AF1:(1-AF1);
  snp->coding_allele=0;

  snp->R2=R2;
  snp->nGene = 0;
  snp->f_stat=-1;
 
  snp->A1 = a0;
  snp->A2 = a1;

  snp->BF_linear = 0;
  snp->BF_logistic = 0;

  //shared between the threads. 
  snp->nHit_linear_sh = 0;
  snp->iPerm_linear_sh = 0;
  snp->nHit_bonf_linear_sh = 0;
  snp->iPerm_bonf_linear_sh = 0;

  snp->nHit_logistic_sh = 0;
  snp->iPerm_logistic_sh = 0;
  snp->nHit_bonf_logistic_sh = 0;
  snp->iPerm_bonf_logistic_sh = 0;

  snp->ref_geno = NULL; //V.1.4.mc
  snp->ref_nsample = 0; //V.1.4.mc
  //snp->BF_logistic_perm_sh = 0;;
  //snp->wald_perm_sh = 0;
  //snp->loglik_logistic_perm_sh = 0;

  snp->eSampleSize = eSampleSize;

  snp->r = NULL;
  snp->r_names = NULL;
  
  snp->c1 = 0.0;
  snp->c1_se = 0.0;

  snp->pval_linear = 1.0;
  snp->beta_logistic = 0;
  snp->se_logistic = 0;
  snp->loglik_logistic = 0;
  snp->wald = 0;
  snp->pval_logistic = 1.0;

  snp->missingness = miss;
  snp->nmiss = -1;

  snp->sign_kept = true; //V.1.4.mc.

  snp->correlated_snps = NULL;//V.1.5.mc
}

void estimate_phenotype_variance_and_N(char * META_file,double * est_var, int * est_N)
{
  static char sline[MAX_LINE_WIDTH];
  static int chr;
  static char snp_id[SNP_ID_LEN];
  static double MAF;
  static double N;
  static int pos;
  static char a1, a2;
  static int status;
  static double beta;
  static double se;
  static double metaP;
  static double array[1000000]; //a million snps at most for a choromosome ?
  static int array1[1000000]; //a million snps at most for a choromosome ?

  FILE * fp;
  //printf("Estimating phenotype variance...\n");
  bool isgz = false;
  if(!checkgz(META_file))
   fp = fopen(META_file,"r");
  else
  {
    isgz = true;
    fp = gzopen(META_file);
  }
 
  int i = 0;
  int n = 0;

  while(feof(fp) == 0 && fgets(sline, MAX_LINE_WIDTH, fp) !=NULL){
    if(sline[0]=='#') continue;
    status= sscanf(sline, "%d %s %c %c %lg %lg %d %lg %lg %lg", &chr, snp_id, &a1, &a2, &MAF, &N, &pos, &beta,&se, &metaP);
    if(status != 10){
      printf("-Error in summary file format: %d\n", status);
      printf("-[%s]", sline);
      exit(1);
    }
    if( N<5 || se<EPS )
      continue;
    double geno_tss = 2*MAF*(1.0-MAF)*N; //2p(1-p)n
    double yy = geno_tss*( (N-1)*se*se + beta*beta)/N;
    array[n] = yy;
    array1[n] = N;
    n++;
  }
  if(!isgz)
    fclose(fp);
  else
    pclose(fp);

  gsl_vector * v = gsl_vector_alloc(n);
  gsl_vector * v1 = gsl_vector_alloc(n);
  for(i=0;i<n;i++)
  {
    gsl_vector_set(v,i,array[i]);
    gsl_vector_set(v1,i,array1[i]);
  }
  gsl_sort_vector(v);
  gsl_sort_vector(v1);

  double median_yy = gsl_stats_median_from_sorted_data (v->data,1,v->size); 
  int median_N = gsl_stats_median_from_sorted_data (v1->data,1,v1->size); 
  //printf("-Estimated phenotype variance = %g\n",median_yy);
  printf("-Estimated sample size = %d\n",median_N);
  gsl_vector_free(v);
  gsl_vector_free(v1);
  *est_var = median_yy; //actually gwis, bimbam does not depend on phenotype variance
  *est_N = median_N;
}

void get_haplotype_weights(char * hap_wt_file, int nsamples)
{
  haplotype_weights = (double*)malloc(nsamples*sizeof(double));  
  FILE* fp;
  bool isgz = false; //BIG BUG
  if(!checkgz(hap_wt_file))
    fp = fopen(hap_wt_file, "r");
  else
  {
    isgz = true;
    fp = gzopen(hap_wt_file);
  }

  int i=0;
  double weight =-1;
  while(!feof(fp))
  {
    if(fscanf(fp, "%lg", &weight) != 1) break;
    haplotype_weights[i] = weight;
    i++;
  }
  printf("-Read %d hapltoype weights\n",i);
  if(!isgz)
    fclose(fp);
  else
    pclose(fp);
}
//-V.1.4.mc

//V.1.7 : read just chr snp bp pvalue
SNP* readSNP_SUMMARY_simple(C_QUEUE * snp_queue, struct hashtable * snp_pos_table, FILE * fp_frq, SNP_CNT* snp_cnt,int N, double pheno_var)
{
  static char sline_frq[MAX_LINE_WIDTH];
  static int chr;
  static char snp_id[SNP_ID_LEN];
  static int pos;
  static SNP* snp = NULL;
  static int status;
  static double metaP;
  static void* snp1_info;

  strcpy(sline_frq, "");

  while(feof(fp_frq) == 0 && fgets(sline_frq, MAX_LINE_WIDTH, fp_frq) !=NULL)
  {
    status= sscanf(sline_frq, "%d %s %d %lg", &chr, snp_id, &pos, &metaP);
    //printf("READ : %d %s %d %lg\n",chr,snp_id,pos,metaP);
    if(status != 4){
      printf("-Error in summary file format: %d\n", status);
      printf("-Error in line : %s \n", sline_frq);
      exit(1);
    }
    snp1_info = hashtable_search(snp_pos_table, snp_id);
    if(snp1_info == NULL)
    {
      printf("-Error in reading summary file, %s not in hashtable\n", snp_id);
      exit(1);
    }
    else if( ((SNP_INFO_SUMMARY *) snp1_info)->exclude == SNP_INFO_SUMMARY_EXCLUDE_MULTIPOS ){
      fprintf(fp_log,"%s skipped because it is mapped to multiple positions\n", snp_id);
      printf("%s skipped because it is mapped to multiple positions\n", snp_id);
      snp_cnt->nSNPMulti++;
      continue;
    }else if(((SNP_INFO_SUMMARY *) snp1_info)->exclude == SNP_INFO_SUMMARY_EXCLUDE_REDUNDENT ){
      fprintf(fp_log,"%s skipped because it is highly correlated to another SNP\n", snp_id);
      printf("%s skipped because it is highly correlated to another SNP\n", snp_id);
      snp_cnt->nSNPRedundent++;
      continue;
    }
    else if(((SNP_INFO_SUMMARY *) snp1_info)->exclude == SNP_INFO_SUMMARY_EXCLUDE_A1MISS )
    {
      snp_cnt->nSNPnoA1++;
      fprintf(fp_log,"%s is absent in reference population, skipped\n",snp_id);
      printf("%s is absent in reference population, skipped\n",snp_id);
      continue;
    }

    double beta = GSL_NEGINF;
    double se = GSL_NEGINF; 
    double MAF = ((SNP_INFO_SUMMARY *) snp1_info)->ref_maf;
    double AF1 = ((SNP_INFO_SUMMARY *) snp1_info)->ref_af1;

    if(pheno_var>0 && N>0)
    { 
      if(VERBOSE) 
         printf("Estimating beta for %s with MAF = %lg and N = %d....\n",snp_id,MAF,N);
      double sigma_snp = sqrt(2*MAF*(1-MAF));
      double chisqr = gsl_cdf_chisq_Qinv (metaP,1);
      double t = sqrt(chisqr);
      se = sqrt(pheno_var)/(sigma_snp*sqrt(N));
      beta = t*se;
      if(VERBOSE) 
         printf("Estimated beta,se for %s = %lg %lg....\n",snp_id,beta,se);
    }
    //else printf("Cannot estimate beta as phenotype variance is %lg\n",pheno_var);

    snp = (SNP*) cq_push(snp_queue);
    strcpy(snp->name, snp_id);
    snp->pos=pos/1000000;
    snp->file_pos = ((SNP_INFO_SUMMARY* ) snp1_info)->file_pos; //V.1.4.mc
    snp->bp=pos;
    snp->chr=chr;
    snp->MAF = MAF; //get from reference population
    snp->AF1 = AF1; //get from reference population
    snp->R2=1;
    snp->nGene = 0;
    snp->f_stat=-1;
    snp->missingness = 0;

    //Pritam added.
    snp->A1 = alleleACGT(((SNP_INFO_SUMMARY *) snp1_info)->a1); //get from reference population
    snp->A2 = alleleACGT(((SNP_INFO_SUMMARY *) snp1_info)->a2); //get from reference population
    snp->nmiss = 0; //user should input
    //printf("XXXXXXXX %s %c %c\n",snp->name,snp->A1,snp->A2);
  
    snp->ref_geno_tss = GSL_NEGINF;
    snp->ref_MAF = MAF;
    snp->sign_kept = true;

    snp->nHit_linear_sh=0;
    snp->iPerm_linear_sh=0;

    snp->nHit_bonf_linear_sh=0;
    snp->iPerm_bonf_linear_sh=0;

    snp->nHit_logistic_sh=0;
    snp->iPerm_logistic_sh=0;

    snp->nHit_bonf_logistic_sh=0;
    snp->iPerm_bonf_logistic_sh=0;

    if(N>0)
        snp->eSampleSize = N*2*MAF*(1-MAF);
    else
        snp->eSampleSize = -1;

    snp->beta=beta;
    snp->se=se;
    snp->metaP=metaP;
   
    snp->ref_geno = NULL; //V.1.4.mc
    snp->ref_nsample = 0; //V.1.4.mc
    snp->id++; //V.1.4.mc

    snp->geno=NULL;
    snp->n_correlated_snp_max = MAX_COR_SNP;
    snp->n_correlated_snp = 0;

    snp->correlated_snps = create_hashtable(16, hash_key, keys_equal_fn);//V.1.4.mc

    if(VERBOSE)
       printf("Read snp summary simple : %s maf = %lg N=%d\n",snp->name,snp->MAF,N);

    return snp; //NO LD BUG
  }//end while
  return NULL;
}

//read SNP from freq file
SNP* readSNP_SUMMARY(C_QUEUE * snp_queue, struct hashtable * snp_pos_table, FILE * fp_frq, FILE * fp_ld, FILE * fp_allele_info, SNP_CNT* snp_cnt){

  static char sline_frq[MAX_LINE_WIDTH];
  static char sline_ld[MAX_LINE_WIDTH]="";
  static int chr;
  static char snp_id[SNP_ID_LEN];
  static double AF1;
  static double N;
  static int pos;
  static char a1, a2; //noncoded coded alleles.
  static SNP* snp = NULL;
  static int status;
  static double beta;
  static double se;
  static double metaP;
  static char snp1[SNP_ID_LEN], snp2[SNP_ID_LEN];
  static void* snp1_info;
  static double r;
  bool found = false;
  bool sign_kept = true;//V.1.4.mc

  strcpy(sline_frq, "");

  while(!found && feof(fp_frq) == 0 && fgets(sline_frq, MAX_LINE_WIDTH, fp_frq) !=NULL){
    status= sscanf(sline_frq, "%d %s %c %c %lg %lg %d %lg %lg %lg", &chr, snp_id, &a1, &a2, &AF1, &N, &pos, &beta,&se, &metaP);
    if(status != 10){
      printf("-Error in summary file format: %d\n", status);
      printf("-Error in line : %s \n", sline_frq);
      exit(1);
    }
    //printf("Read %s\n",snp_id);
    snp1_info = hashtable_search(snp_pos_table, snp_id);
    if(snp1_info == NULL)
    {
      printf("-Error in reading summary file, %s not in hashtable\n", snp_id);
      exit(1);
    }
    else if( se<EPS){ //V.1.2 VEGAS FIX, invalid standard error.
      snp_cnt->nInvSE++;
      continue;
    }else if( ((SNP_INFO_SUMMARY *) snp1_info)->exclude == SNP_INFO_SUMMARY_EXCLUDE_MULTIPOS ){
      fprintf(fp_log,"%s skipped because it is mapped to multiple positions\n", snp_id);
      snp_cnt->nSNPMulti++;
      continue;
    }else if(((SNP_INFO_SUMMARY *) snp1_info)->exclude == SNP_INFO_SUMMARY_EXCLUDE_REDUNDENT ){
      fprintf(fp_log,"%s skipped because it is highly correlated to another SNP\n", snp_id);
      snp_cnt->nSNPRedundent++;
    continue;
    }
    else if(((SNP_INFO_SUMMARY *) snp1_info)->exclude == SNP_INFO_SUMMARY_EXCLUDE_A1MISS )
    {
      //this snp does not occur in reference panel.
      //if LD and allele info files are provided (and this snp does not occur in them), we throw this snp out.
      //if(COMPUTE_LD), then lets keep it, we will just assume it has LD=0 with other snps.
      snp_cnt->nSNPnoA1++;
      if(fp_allele_info!=NULL) //V.1.4.mc
      {
         fprintf(fp_log,"%s is skipped because it doesn't have allele coding / LD profile\n", snp_id);
         if(VERBOSE)
            printf("-%s skipped because it doesn't have allele coding / LD profile\n", snp_id);
         continue;
      }
      else
      {
         fprintf(fp_log,"%s is absent in reference population\n",snp_id);//but we keep the snp.
      }
    }
    else
    {
      //check for strand-ambiguity.
      int z1 = allele1234(a1);
      int z2 = allele1234(a2);
      double ref_af1 = ((SNP_INFO_SUMMARY *) snp1_info)->ref_af1;
      double diff = fabs(AF1 - ref_af1);
      bool processed = false;
      if ( (z1==1 && z2==4) || (z1==4 && z2==1) || (z1==2 && z2==3) || (z1==3 && z2==2) )
      {
           if(diff > 0.15 && ((AF1 > 0.5 && ref_af1 < 0.5) || (AF1 < 0.5 && ref_af1 > 0.5)) )
           {
              if(VERBOSE)
                 printf("check_ambig: %s %d %d : af1 %lg ref_af1 %lg diff %lg\n",snp_id,z1,z2,AF1,ref_af1,diff);
              //ambiguous snp, possible strand flipped.
              processed = true;
              if(OMIT_ST)
              {
                //drop
                snp_cnt->nSNPambig++; //V.1.8.mc
                fprintf(fp_log,"%s is skipped because it is possibly strand flipped and ambiguous w.r.t reference panel (AF1 = %lg, ref_af1=%lg)\n", snp_id,AF1,ref_af1);
                if(VERBOSE)
                  printf("%s is skipped because it is possibly strand flipped and ambiguous w.r.t reference panel (AF1 = %lg, ref_af1=%lg)\n", snp_id,AF1,ref_af1);
                continue;
              }
              else //flip strand
              {
                 if(VERBOSE)
                    printf("%s is strand flipped (AF1 = %lg, ref_af1=%lg)\n", snp_id,AF1,ref_af1);
                 fprintf(fp_log,"%s is strand flipped (AF1 = %lg, ref_af1=%lg)\n", snp_id,AF1,ref_af1);
                 int a1_f = flipAllele(a1);   
                 int a2_f = flipAllele(a2);   
                 if (a1_f == ((SNP_INFO_SUMMARY *) snp1_info)->a1 && a2_f == ((SNP_INFO_SUMMARY *) snp1_info)->a2) 
                 {
	            snp_cnt->nSignKept++;
                    sign_kept = true;
                 }
                 else if (a1_f == ((SNP_INFO_SUMMARY *) snp1_info)->a2 && a2_f == ((SNP_INFO_SUMMARY *) snp1_info)->a1) 
                 {
	            beta = -beta;
	            snp_cnt->nSignFlipped++;
                    sign_kept = false;
                 }
                 else
                 {
                    //alleles do not match with reference panel. Discard this snp.
                    if(VERBOSE)	
	               printf("-%s skipped because allele mismatch after flipping strand\n", snp_id);
	            fprintf(fp_log,"%s is skipped because allele mismatch with reference population / LD profile after flipping strand\n", snp_id);
	            ((SNP_INFO_SUMMARY *) snp1_info)->exclude =  SNP_INFO_SUMMARY_EXCLUDE_A1NOMATCH;
	            snp_cnt->nSNPA1MissMatch++;
	            continue;
                 }
              }
           }
      }
      
      if(processed==false)
      {
         //check if the snp alleles match with that of the background population (e.g 1000G).
         if( (allele1234(a1) == ((SNP_INFO_SUMMARY *) snp1_info)->a1 &&allele1234(a2) == ((SNP_INFO_SUMMARY *) snp1_info)->a2) 
              || (flipAllele(a1)== ((SNP_INFO_SUMMARY *) snp1_info)->a1 &&flipAllele(a2) == ((SNP_INFO_SUMMARY *) snp1_info)->a2))
         {
	   snp_cnt->nSignKept++;
           sign_kept = true; //V.1.4.mc
	   //printf("%s sign kept\n", snp_id);
         }
         else if((allele1234(a1) == ((SNP_INFO_SUMMARY *) snp1_info)->a2 &&allele1234(a2) == ((SNP_INFO_SUMMARY *) snp1_info)->a1) 
                   ||(flipAllele(a1)== ((SNP_INFO_SUMMARY *) snp1_info)->a2 &&flipAllele(a2) == ((SNP_INFO_SUMMARY *) snp1_info)->a1) )
         {
	    beta = -beta;
	    snp_cnt->nSignFlipped++;
            sign_kept = false; //V.1.4.mc
	    //printf("%s sign flipped\n", snp_id);
         }
         else
         {
           //alleles do not match with reference panel. Discard this snp.
           if(VERBOSE)	
	      printf("-%s skipped because allele mismatch\n", snp_id);
	   fprintf(fp_log,"%s is skipped because allele mismatch with reference population / LD profile\n", snp_id);
	   ((SNP_INFO_SUMMARY *) snp1_info)->exclude =  SNP_INFO_SUMMARY_EXCLUDE_A1NOMATCH;
	   snp_cnt->nSNPA1MissMatch++;
	   continue;
         }
      }
   }

  if(fp_ld!=NULL) //NO LD BUG
  { 
    //seek to snp1 in the ld file.
    while(!feof(fp_ld)){
      fpos_t prevPos;
      fgetpos(fp_ld, &prevPos);
      if(fgets(sline_ld, MAX_LINE_WIDTH, fp_ld) ==NULL){
	found = false;
	break;
      }
      status= sscanf(sline_ld, "%*d %*d %s %*d %*d %s %lg", snp1, snp2, &r);
      if(status != 3) //V.1.5.mc
      {
        printf("-Error in LD file format: %d fields present\n", status);
        printf("-Error in line : %s \n", sline_ld);
        exit(1);
      }
      snp1_info = hashtable_search(snp_pos_table, snp1);
      if(snp1_info == NULL || ((SNP_INFO_SUMMARY*) snp1_info)->exclude > 0){ 
	//    	printf("looking for %s at %d, skip line: %s",snp_id, pos, sline_ld);
	continue;
      }
      if(strcmp(snp1, snp_id)==0){
	fsetpos(fp_ld, &prevPos);
	found = true;
	//printf("found, %s, %s\n",snp1,snp2);
	break;
      }else if( ((SNP_INFO_SUMMARY*) snp1_info)->pos >pos) //position exceeded.
       {
	fsetpos(fp_ld, &prevPos);
	//printf("%s not found at bp %d vs %s at %d\n", snp_id, pos, snp1,  * (int *)snp1_pos );
	found = false;
	break;
       }
       else
       {
	//	printf("looking for %s at %d, skip line: %s",snp_id, pos, sline_ld);
       }
    }
  }

  if(VERBOSE)
      if(!found && !COMPUTE_LD) //V.1.4.mc
          printf("No LD information for %s\n",snp_id); //NO LD BUG

    //read the LD for SNP1
    snp = (SNP*) cq_push(snp_queue);
    strcpy(snp->name, snp_id);
    snp->pos=pos/1000000;
    snp->file_pos = ((SNP_INFO_SUMMARY* ) snp1_info)->file_pos; //V.1.4.mc
    snp->bp=pos;
    snp->chr=chr;
    snp->AF1=AF1;
    snp->MAF=(AF1<0.5)?AF1:(1-AF1);
    snp->R2=1;
    snp->nGene = 0;
    snp->f_stat=-1;
    snp->missingness = 0;

    //Pritam added.
    snp->A1 = a1;
    snp->A2 = a2;
    snp->nmiss = N;
  
    snp->ref_geno_tss = GSL_NEGINF; //V.1.4.mc
    snp->ref_MAF = 0; //V.1.4.mc
    snp->sign_kept = sign_kept;//V.1.4.mc

    snp->nHit_linear_sh=0;
    snp->iPerm_linear_sh=0;

    snp->nHit_bonf_linear_sh=0;
    snp->iPerm_bonf_linear_sh=0;

    //++V.1.2 FIX META
    snp->nHit_logistic_sh=0;
    snp->iPerm_logistic_sh=0;

    snp->nHit_bonf_logistic_sh=0;
    snp->iPerm_bonf_logistic_sh=0;
    //--V.1.2 FIX META

    //snp->eSampleSize = N/2;
    snp->eSampleSize = N*2*AF1*(1-AF1); //V.1.4.mc
    snp->beta=beta;
    snp->se=se;
    snp->metaP=metaP;
   
    snp->ref_geno = NULL; //V.1.4.mc
    snp->ref_nsample = 0; //V.1.4.mc
    snp->id++; //V.1.4.mc

    //snp->GpermDat=NULL; //Pritam
    snp->geno=NULL;
    snp->n_correlated_snp_max = MAX_COR_SNP;
    snp->n_correlated_snp = 0;

    if(fp_ld!=NULL) //V.1.4.mc
    {
      snp->r = (double*)malloc(sizeof(double)* snp->n_correlated_snp_max);
      snp->r_names = (char*)malloc(sizeof(char)* SNP_ID_LEN*snp->n_correlated_snp_max);
    }
    else
    {
      snp->r = NULL;
      snp->r_names= NULL;
    }

    snp->correlated_snps = create_hashtable(16, hash_key, keys_equal_fn);//V.1.4.mc
    //printf("Pritam allocating : %s %p\n",snp->name,snp->correlated_snps);
    //last snp fix.
    static char snp2_last[SNP_ID_LEN];             

    if(!found)
    {
      if(strcmp(snp_id,snp2_last)==0)
      {
         //ok to keep this snp.
         //printf("%s is the last snp, so keep it\n",snp_id);
         return snp;
      }
      else
      {
        if(!COMPUTE_LD) //V.1.4.mc
        {
            snp_cnt->nSNPnoLD++; //V.1.5.mc
            if(VERBOSE) //NO LD BUG
                 printf("-%s does not have LD info as primary SNP\n", snp_id); //NO LD BUG
            fprintf(fp_log,"%s does not have LD info as primary SNP\n", snp_id); //NO LD BUG
        }
        if(fp_ld!=NULL) //NO LD BUG
           cq_shift(snp_queue); //NO LD BUG 
        else //NO LD BUG
           return snp; //NO LD BUG
      }
    }else{
      while(!feof(fp_ld)){
	fpos_t prevPos;
	fgetpos(fp_ld, &prevPos);
	if(fgets(sline_ld, MAX_LINE_WIDTH, fp_ld) ==NULL)  break;
	status= sscanf(sline_ld, "%*d %*d %s %*d %*d %s %lg", snp1, snp2, &r);
	if(isnan(r))  r=0;
	if(status == 3){
	  if(strcmp(snp1, snp_id)!=0){
	    fsetpos(fp_ld, &prevPos);
	    break;
	  }else{
	    if(snp->n_correlated_snp>= snp->n_correlated_snp_max-5) //LD BUG NEW
            {
               printf("-increasing correlated snp memory to %d\n",2*snp->n_correlated_snp_max); //LD BUG NEW
	       double* newR =  (double* ) malloc(sizeof(double)*snp->n_correlated_snp_max*2);
	       char* newR_names =  (char* ) malloc(sizeof(char)*SNP_ID_LEN*snp->n_correlated_snp_max*2);
	       if(newR== NULL|| newR_names==NULL )
               {
		printf("-Error allocating memory.\n");
		abort();
	       }else{
		//	    printf("-Increased memory to %d for correlation array for %s\n",snp->n_correlated_snp_max*2, snp->name);
	       }
	       memcpy(newR, snp->r, sizeof(double) * snp->n_correlated_snp_max);
	       memcpy(newR_names, snp->r_names, sizeof(char) * SNP_ID_LEN* snp->n_correlated_snp_max);
	
	       snp->n_correlated_snp_max *= 2;
	       free(snp->r);
	       free(snp->r_names);
	       snp->r = newR;
	       snp->r_names = newR_names;
               printf("-done increasing correlated snp memory to %d\n",snp->n_correlated_snp_max);

	    } 

            //last snp fix, remember the last snp of the last LD block.
            strcpy(snp2_last,snp2);

	    snp->r[snp->n_correlated_snp] = r;
	    strcpy(&snp->r_names[snp->n_correlated_snp*SNP_ID_LEN], snp2);
	    if(fabs(fabs(r)-1) < EPS){
	      void * info = hashtable_search(snp_pos_table, snp2);
	      if(info != NULL){
		if( ((SNP_INFO_SUMMARY*) info)->exclude == SNP_INFO_SUMMARY_INCLUDE )
		  ((SNP_INFO_SUMMARY*) info)->exclude =  SNP_INFO_SUMMARY_EXCLUDE_REDUNDENT;
		printf("-%s will be skipped because it is in high LD with %s r: %g \n", snp2, snp1, r);
		fprintf(fp_log,"%s will be skipped because it is in high LD with %s r: %g \n", snp2, snp1, r);
	      }
	    }
	    snp->n_correlated_snp++;
	    strcpy(sline_ld,"");
	    //	   printf("%s, %s, %g\n", snp1, snp2, r);
	  }
	}else{
	  printf("-Error in file format in LD file\n");
	  exit(1);
	}
      }
      //printf("Pritam : %s has %d correlated snps\n",snp->name,snp->n_correlated_snp);
      return snp;
    }
  }
  return NULL;
}

//V.1.5.mc
//read SNP from file in impute2 format
SNP* readSNP_impute2(C_QUEUE * snp_queue,  FILE * fp_impute2_geno, FILE * fp_impute2_info, PHENOTYPE* phenotype, SNP_CNT* snp_cnt, int chr)
{
  static char sline_geno[3*MAX_LINE_WIDTH_LONG];
  static char sline_info[MAX_LINE_WIDTH];
  static char snp_id[SNP_ID_LEN];
  static int bp;
  static char a0 = '-';
  static char a1 = '-';
  static double AF1;
  static double R2;
  static SNP* snp = NULL;
  snp = (SNP*) cq_push(snp_queue);
  snp->geno = NULL;

  if(CQ_DEBUG)
    printf("pushed, (start, end) = (%lu, %lu)\n", snp_queue->start, snp_queue->end);

  //to skip blank lines
  while(!feof(fp_impute2_geno)){
    strcpy(sline_geno, "");
    fgets(sline_geno, 3*MAX_LINE_WIDTH_LONG, fp_impute2_geno);
	  
    strcpy(sline_info, "");
    fgets(sline_info, MAX_LINE_WIDTH, fp_impute2_info);
	  
    if(strlen(sline_geno) > 0){
      snp->geno=malloc(sizeof(double)*MAX_N_INDIV);
      double miss = 0;
      int status =  readGeno_impute2(snp_id, &bp, &AF1, &R2, snp->geno, phenotype->N_indiv, sline_geno, sline_info, phenotype->NA, &miss, &a0, &a1, phenotype->SEX, chr);
      miss /= phenotype->N_sample;	    
      if(status >0 ) {
	createSNP(snp, snp_id, 0, bp, chr, AF1, R2, phenotype->N_sample*R2*2*AF1*(1-AF1),miss, a0,a1);
	return snp;
      }else
      {
        if(status==-1)
        {
          snp_cnt->nSNP_mono++;
          if(VERBOSE)
             printf("-Omitting snp %s as monomorphic\n",snp_id);
          fprintf(fp_log,"-Omitting snp %s as monomorphic\n",snp_id);
        }
	free(snp->geno);
        snp->geno=NULL;
      }
    }
  }
  cq_shift(snp_queue);
  if(CQ_DEBUG)
    printf("shifted, (start, end) = (%lu, %lu)\n", snp_queue->start, snp_queue->end);
  return NULL;
}

//read SNP from file
SNP* readSNP(C_QUEUE * snp_queue,  FILE * fp_tped, FILE * fp_snp_info, FILE* fp_mlinfo,  PHENOTYPE* phenotype, SNP_CNT* snp_cnt)//FIX MONOMORPHIC SNPS V.1.2
{
  static char sline_geno[MAX_LINE_WIDTH];
  static char sline_snp[MAX_LINE_WIDTH];
  static char sline_mlinfo[MAX_LINE_WIDTH];
  static char snp_id[SNP_ID_LEN];
  static int chr;
  static int bp;
  static char a0;
  static char a1;
  static double pos;
  static double AF1;
  static double R2;
  static SNP* snp = NULL;
  snp = (SNP*) cq_push(snp_queue);
  snp->geno = NULL;

  if(CQ_DEBUG)
    printf("pushed, (start, end) = (%lu, %lu)\n", snp_queue->start, snp_queue->end);

  // to skip blank lines
  while(!feof(fp_tped)){
    strcpy(sline_geno, "");
    fgets(sline_geno, MAX_LINE_WIDTH, fp_tped);
	  
    strcpy(sline_snp, "");
    fgets(sline_snp, MAX_LINE_WIDTH, fp_snp_info);
	  
    strcpy(sline_mlinfo, "");
    if(fp_mlinfo != NULL)
      fgets(sline_mlinfo, MAX_LINE_WIDTH, fp_mlinfo);
	  
    if(strlen(sline_geno) > 0){
      snp->geno=malloc(sizeof(double)*MAX_N_INDIV);
      double miss = 0;
      int status =  readGeno(snp_id, &chr, &pos, &bp, &AF1, &R2, snp->geno, phenotype->N_indiv, sline_geno,sline_snp, sline_mlinfo, phenotype->NA, &miss, &a0, &a1);
      miss /= phenotype->N_sample;	    
      if(status >0 ) {
        //if(VERBOSE) 
        //  printf("-Creating snp %s with %d samples, qual=%f , maf=%f  , miss=%f\n",snp_id,phenotype->N_sample,R2,AF1,miss);
	createSNP(snp, snp_id, pos, bp, chr, AF1, R2, phenotype->N_sample*R2*2*AF1*(1-AF1),miss, a0,a1);
	return snp;
      }else
      {
        //++FIX MONOMORPHIC SNPS V.1.2
        if(status==-1)
        {
          snp_cnt->nSNP_mono++;
          printf("-Omitting snp %s as monomorphic\n",snp_id);
          fprintf(fp_log,"-Omitting snp %s as monomorphic\n",snp_id);
        }
        //--FIX MONOMORPHIC SNPS V.1.2
	free(snp->geno);
        snp->geno=NULL;
      }
    }
  }
	 
  cq_shift(snp_queue);
  if(CQ_DEBUG)
    printf("shifted, (start, end) = (%lu, %lu)\n", snp_queue->start, snp_queue->end);
  return NULL;
}

//create data structure for a gene
GENE* createGene(char* name, char* ccds, int chr, int bp_start, int bp_end, int iPerm){

  GENE* gene = (GENE*) malloc(sizeof(GENE));
  		
  strcpy(gene->name, name);
  strcpy(gene->ccds, ccds);
  gene->bp_start = bp_start;
  gene->bp_end = bp_end;
  gene->chr = chr;
  //gene->iPerm = iPerm; //comment korlam just to see if it is really used.
  gene->nSNP = 0;

  gene->next=NULL;
  gene->LD = NULL;
  gene->Cov = NULL;
  gene->pvalCorr = NULL;
  gene->snp_start = -1;
  gene->snp_end = -1;

  gene->eSNP = -1;
  gene->skip = false;

  gene->maxSSM_SNP_linear_sh = NULL;
  gene->maxSSM_SNP_logistic_sh = NULL;
  gene->maxBonf_SNP_linear_sh = NULL;
  gene->maxBonf_SNP_logistic_sh = NULL;

  gene->BF_sum_linear = GSL_NEGINF;
  gene->BF_sum_logistic = GSL_NEGINF;

  gene->vegas_linear = 0;
  gene->vegas_logistic = 0;

  gene->gates_linear = 0;
  gene->gates_logistic = 0;

  return gene;
}
//read genes from file
int readGene(char* refseq_file, GENE** gene, int SNPchr){

  FILE *fp;
  bool isgz = false; //BIG BUG
  if(!checkgz(refseq_file))
     fp = fopen(refseq_file, "r");
  else
  {
     fp = gzopen(refseq_file);
     isgz = true;
  }

  int chr;
  int bp_start;
  int bp_end;
  int iPerm;
  int nGene = 0;
  int prevchr = -1;
  int prevbp = -1;

  char sline[MAX_LINE_WIDTH];
  char ccds[GENE_ID_LEN];
  char name[GENE_ID_LEN];

  GENE* curGene=NULL;
  GENE* prevGene = NULL;

  *gene = createGene("HEAD", "HEAD", -1, -1, -1, -1);
  prevGene = *gene;

  //check header line.
  {
    fgets(sline, MAX_LINE_WIDTH, fp); 
    if(sline[0] != '#')
    {
      printf("-Header line missing in gene-set file %s, please add one starting with '#'\n",refseq_file);
      exit(1);
    }
  }

  //printf("-Time: %s", getTime());
  printf("-Start to load genes from %s\n", refseq_file);

  while(!feof(fp)){

    strcpy(sline, "");
    fgets(sline, MAX_LINE_WIDTH, fp); 
    if(strlen(sline)==0)
      continue;

    if(sline[0] == '#'){
      printf("-Skipping header: %s", sline);
      continue;
    }
    int nstatus;
    if(DIFFNCBI)
      nstatus= sscanf(sline, "%s %s %d %d %d %d",ccds, name, &chr, &bp_start, &bp_end, &iPerm);
    else{
      nstatus= sscanf(sline, "%s %s %d %d %d",ccds, name, &chr, &bp_start, &bp_end);
      iPerm =0;
    }
    if(nstatus == 5 ||nstatus == 6 ){

      if(prevchr < chr)
	prevchr = chr;
      else if(prevchr > chr){
	printf("-%s not sorted in chr\n", refseq_file);
	exit(1);
      }

      if(SNPchr == chr){

	if(prevbp < bp_start)
	  prevbp = bp_start;
	else if(prevbp > bp_start){
	  printf("-genes in file %s not sorted in bp for chromosome %d !\n", refseq_file, SNPchr);
          printf("-check near gene %s\n",name);
	  exit(1);
	}
		
	curGene= createGene(name, ccds, chr, bp_start, bp_end, iPerm);
	prevGene->next = curGene;
	prevGene = curGene;

	nGene++;
      }
    }else{
      printf("-Skipping %s", sline);
      continue;
    }
  }
  printf("-Done loading genes\n");

  if(isgz)
    pclose(fp);
  else
    fclose(fp);
  return nGene;
}

int count_columns(char * line_geno)
{
  char * nextGeno;
  int i = 1;
  while(true)
  {
    nextGeno = strchr(line_geno, '\t');
    if(nextGeno!=NULL)
       *nextGeno='\0';
    else
       break;
    if(nextGeno!=NULL)
       line_geno = nextGeno+1;
    else
       break;
    i++;
  }
  //printf("count_columns = %d\n",i);
  return i;
}

int readHaps(char* name, long * bp, char * a1, char * a2, double * geno, int ncols, char * line_geno, double * freq)
{
  char * nextGeno;
  int k = 0;//indexes genotypes read.
  int i = 0;
  *freq = 0;
  double last_geno = -1;
  double this_geno = -1;
  int chr = 0;
  double af1 = 0;
  for (i=0; i<ncols; i++)
  {
    nextGeno = strchr(line_geno, '\t');
    //truncate the string so that sscanf runs much faster
    if(nextGeno!=NULL) *nextGeno='\0';

    if(i==0)
       sscanf(line_geno,"%d",&chr);
    else if(i==1)
       sscanf(line_geno,"%s",name);
    else if(i==2)
       sscanf(line_geno,"%ld",bp);
    else if(i==3)
       sscanf(line_geno,"%lg",&af1);
    else if(i==4)
       sscanf(line_geno,"%c",a1);
    else if(i==5)
       sscanf(line_geno,"%c",a2);
    else
       sscanf(line_geno, "%lg", &this_geno);
    if(last_geno<0)
       last_geno = this_geno;
    else
    {
        //got a pair of genotypes.
        geno[k] = last_geno + this_geno; //convert to genotype.
        *freq += geno[k];
        k++;
        last_geno = -1;
    }
    if(nextGeno!=NULL)
      line_geno = nextGeno+1;
  }
  *freq = (*freq)/(ncols-5);
  if(abs(*freq - af1) > 0.0001) {printf("readHaps: Error in snp %s freq= %lg != %lg\n",name,*freq,af1);exit(0);} //urgent
  if(*freq > 1.0) {printf("readHaps: Error in snp %s freq= %lg\n",name,*freq);exit(0);} //urgent
  return k;
}

int search_ld(SNP * snp1, SNP* snp2, double *ld, double *cov)
{
  if(snp1->ref_geno==NULL || snp2->ref_geno==NULL)
    return -1;

  SNP_SNP_CORR * x= NULL;
  if(snp1->id < snp2->id)
     x = (SNP_SNP_CORR*)hashtable_search(snp1->correlated_snps, snp2->name);
  else
     x = (SNP_SNP_CORR*)hashtable_search(snp2->correlated_snps, snp1->name);

  if(x!=NULL)
  {
    *ld = x->ref_ld;
    *cov = x->ref_cov;
    //printf("Pre-Computed  %s %s :  ld=%g cov=%g\n",snp1->name,snp2->name,*ld,*cov);
    //if(VERBOSE)
    //  printf("Got 1 %g %g\n",*ld,*cov);
    return 1;
  }
  else
  {
    *ld = 0;
    *cov = 0;
    //if(VERBOSE)
    //  printf("Got nothing %g %g\n",*ld,*cov);
    return -1;
  }
}

void check_alleles(SNP * snp, char a1, char a2)
{
  //check if alleles match, sort of redundant as beta should be correcly inverted already in readSNP.
  if((allele1234(snp->A1) == a1 && allele1234(snp->A2) == a2) || (flipAllele(snp->A1) == a1 && flipAllele(snp->A2) == a2))
  {
     if(snp->sign_kept != true)
     {
       printf("--- ERROR : beta should NOT be inverted ? check %s (%c %c) vs. haplotype alleles (%c %c)\n",snp->name,snp->A1,snp->A2,a1,a2);
       exit(1);
     }
  }
  else if((allele1234(snp->A1) == a2 && allele1234(snp->A2) == a1) || (flipAllele(snp->A1) == a2 && flipAllele(snp->A2) == a1))
  {
     if(snp->sign_kept != false)
     {
       printf("--- ERROR : beta should BE inverted ? check %s (%c %c) vs. haplotype alleles (%c %c)\n",snp->name,snp->A1,snp->A2,a1,a2);
       exit(1);
     }
  }
}

//+V.1.4.mc
void get_pairwise_ld(FILE * fp_hap, SNP * snp1, SNP* snp2, int ncols, double* ref_LD, double* ref_Cov)
{
  static char line1[MAX_LINE_WIDTH];
  static char line2[MAX_LINE_WIDTH];
  static char name1[SNP_ID_LEN];
  static char name2[SNP_ID_LEN];
  char a1;
  char a2;
  long bp1;
  long bp2;
  double freq1;
  double freq2;
  double ld = 0;
  double cov = 0;
  //if(VERBOSE)
  //   printf("In get_pairwise_ld with %s and %s\n",snp1->name,snp2->name);
  
  if(search_ld(snp1, snp2, ref_LD, ref_Cov)>0)
     return;

  if(snp1->file_pos >= 0 && snp2->file_pos >= 0)
  {
      int n1 = 0;
      int n2 = 0;
      if(snp1->ref_geno==NULL)
      {
        fseek(fp_hap,snp1->file_pos,SEEK_SET);
        fgets(line1,MAX_LINE_WIDTH,fp_hap);
        //printf("-> snp1 = %s\n",line1);
        snp1->ref_geno = (double*)malloc(sizeof(double)*MAX_N_INDIV);
        n1 = readHaps(name1, &bp1, &a1, &a2, snp1->ref_geno, ncols, line1, &freq1);
        snp1->ref_MAF = (freq1>0.5)?(1.0-freq1):freq1;
        snp1->ref_geno_tss = 2*n1*freq1*(1.0-freq1);
        snp1->ref_nsample = n1;
        //printf("snp1 = %d\n",n1);
        check_alleles(snp1,a1,a2);
        if(strcmp (snp1->name, name1 ) != 0) {printf("readHaps: Error in snp1 name %s != %s\n",snp1->name,name1);exit(0);}
      }
      else
        n1 = snp1->ref_nsample;

      if(snp2->ref_geno==NULL)
      {
        fseek(fp_hap,snp2->file_pos,SEEK_SET);
        fgets(line2,MAX_LINE_WIDTH,fp_hap);
        //printf("-> snp2 = %s\n",line2);
        snp2->ref_geno = (double*)malloc(sizeof(double)*MAX_N_INDIV);
        n2 = readHaps(name2, &bp2, &a1, &a2, snp2->ref_geno, ncols, line2, &freq2);
        snp2->ref_MAF = (freq2>0.5)?(1.0-freq2):freq2;
        snp2->ref_geno_tss = 2*n2*freq2*(1.0-freq2);
        snp2->ref_nsample = n2;
        //printf("snp2 = %d\n",n2);
        check_alleles(snp2,a1,a2);
        if(strcmp (snp2->name, name2 ) != 0) {printf("readHaps: Error in snp2 name %s != %s\n",snp2->name,name2);exit(0);}
      }
      else
        n2 = snp2->ref_nsample;

      if(n1!=n2)
      {
         printf("Error in pairwise LD : n1=%d n2=%d\n",n1,n2);
      }
      ld = correlation_and_covariance(snp1->ref_geno, snp2->ref_geno, n1, &cov);
      cov = cov*n1;
      //if(VERBOSE)
      //printf("Computed from reference %s %s :  ld=%g cov=%g\n",snp1->name,snp2->name,ld,cov);
  }

  *ref_LD = ld;
  *ref_Cov = cov;
  //insert ld into hash for this snp.
  SNP_SNP_CORR * S = (SNP_SNP_CORR*)malloc(sizeof(SNP_SNP_CORR));
  S->ref_ld = ld;
  S->ref_cov = cov;
  if(snp1->id < snp2->id)
  {
    char * key = (char*)malloc(sizeof(char)*SNP_ID_LEN);
    //printf("allocated %p\n",(void*)key);
    strcpy(key,snp2->name);
    //printf("Inserting into hashtable for %s (%p) : key = %s(%p) value = %p\n",snp1->name,snp1->correlated_snps,key,(void*)key,S);
    hashtable_insert(snp1->correlated_snps,key,S);
  }
  else
  {
    char * key = (char*)malloc(sizeof(char)*SNP_ID_LEN);
    //printf("allocated %p\n",(void*)key);
    strcpy(key,snp1->name);
    //printf("Inserting into hashtable for %s (%p) : key = %s(%p) value = %p\n",snp2->name,snp2->correlated_snps,key,(void*)key,S);
    hashtable_insert(snp2->correlated_snps,key,S);
  }
}
//-V.1.4.mc

void getGeneLD_SUMMARY(GENE* gene, gsl_matrix* geneLD, gsl_matrix* geneCov, gsl_matrix * genePvalCorr, C_QUEUE * snp_queue, int ncols_hap, FILE * fp_hap) //V.1.4.mc
{
  //printf("getGeneLD_SUMMARY called for %s\n",gene->name);
  int i=0, j, k;
  double ld, cov;
  SNP* snp1, * snp2;
  int snp1_id, snp2_id;
  gene->LD =  geneLD; 
  gene->Cov =  geneCov;
  gene->pvalCorr = genePvalCorr;//FIX GATES V.1.2

  snp1 = (SNP*)cq_getItem(gene->snp_start, snp_queue);
  for(snp1_id = gene->snp_start; snp1_id <= gene->snp_end; snp1_id++){
    snp1->gene_id = i;
    gsl_matrix_set(gene->LD, i, i, 1);
    gsl_matrix_set(gene->Cov, i, i, snp1->geno_tss);

    //+GATES BUG, V.1.3
    if(gene->pvalCorr!=NULL)
       gsl_matrix_set(gene->pvalCorr,i,i,1.0); 
    //-GATES BUG, V.1.3

    j=i+1;
    snp2 = (SNP*)cq_getNext(snp1, snp_queue);
    k=0; //snp2_id-snp1_id-1
    for(snp2_id = snp1_id+1; snp2_id <= gene->snp_end; snp2_id++){


      if(fp_hap==NULL) //+V.1.4.mc
      {
        for (;k < snp1->n_correlated_snp; k++){
	  if(strcmp (&snp1->r_names[k*SNP_ID_LEN], snp2->name ) == 0){
	    ld=snp1->r[k];
	    break;
	  }
        }
        if(k>=snp1->n_correlated_snp){
	  ld=0;
	  k++;
        }
        cov = ld*sqrt(snp1->geno_tss * snp2->geno_tss);
      }
      else
      {
         double ref_ld = 0;
         double ref_cov = 0;
         get_pairwise_ld(fp_hap, snp1, snp2, ncols_hap, &ref_ld, &ref_cov);
         ld = ref_ld;
         if(ADJUST_LD)
            cov = sqrt( (snp1->geno_tss * snp2->geno_tss) / (snp1->ref_geno_tss * snp2->ref_geno_tss) ) * ref_cov;
         else
            cov = ref_cov;
      }//-V.1.4.mc
      if(gene->pvalCorr!=NULL)
      {
         double x = ld*ld;//GATES BUG, V.1.3
         x = (((((0.7723 * x - 1.5659) * x + 1.201) * x - 0.2355) * x + 0.2184) * x + 0.6086) * x;//GATES BUG, V.1.3
         //double x = 0.2982*pow(ld,6) - 0.0127*pow(ld,5) + 0.0588*pow(ld,4) + 0.0099*pow(ld,3) + 0.6281*pow(ld,2) - 0.0009*ld;
         gsl_matrix_set(gene->pvalCorr,i,j,x);
         gsl_matrix_set(gene->pvalCorr,j,i,x);
      }
      //printf("LD %s %s = %g\n",snp1->name,snp2->name,ld);
      gsl_matrix_set(gene->LD, i, j, ld);
      gsl_matrix_set(gene->LD, j, i, ld);
      gsl_matrix_set(gene->Cov, i, j, cov);
      gsl_matrix_set(gene->Cov, j, i, cov);

      //printf("Computed [%d,%d] = %lg %lg\n",i,j,ld,cov);

      j++;
      snp2 = cq_getNext(snp2, snp_queue);
    }
    i++;
    snp1 = cq_getNext(snp1, snp_queue);
  }
}

//calculate correlation and covariance matrix for SNPs in a gene
void getGeneLD(GENE* gene, gsl_matrix* geneLD, gsl_matrix* geneCov,gsl_matrix * genePvalCorr,C_QUEUE * snp_queue, int nsample)//FIX GATES V.1.2
{
  if(gene!=NULL)
  {
     printf("-Computing LD for gene %s with %d snps\n",gene->name,gene->nSNP);
     fflush(stdout);
  }
  int i=0, j;
  double ld, cov;
  SNP* snp1, * snp2;
  int snp1_id, snp2_id;
  gene->LD =  geneLD; 
  gene->Cov =  geneCov;
  gene->pvalCorr = genePvalCorr;//FIX GATES V.1.2  

  snp1 = (SNP*)cq_getItem(gene->snp_start, snp_queue);
  for(snp1_id = gene->snp_start; snp1_id <= gene->snp_end; snp1_id++){
    snp1->gene_id = i;
    cov = variance(snp1->geno, nsample,true)*nsample;
    gsl_matrix_set(gene->LD, i, i, 1);
    gsl_matrix_set(gene->Cov, i, i, cov);

    //+GATES BUG, V.1.3
    if(gene->pvalCorr!=NULL)
       gsl_matrix_set(gene->pvalCorr,i,i,1.0); 
    //-GATES BUG, V.1.3

    j=i+1;
    snp2 = (SNP*)cq_getNext(snp1, snp_queue);
    for(snp2_id = snp1_id+1; snp2_id <= gene->snp_end; snp2_id++){
      ld = correlation(snp1->geno, snp2->geno, nsample,true,true);

      cov = covariance(snp1->geno, snp2->geno, nsample,true,true)*nsample;
      gsl_matrix_set(gene->LD, i, j, ld);
      gsl_matrix_set(gene->LD, j, i, ld);


      //printf("LD %s %s = %g\n",snp1->name,snp2->name,ld);

      if(gene->pvalCorr!=NULL)
      {
        double x = ld*ld;//GATES BUG, V.1.3
        x = (((((0.7723 * x - 1.5659) * x + 1.201) * x - 0.2355) * x + 0.2184) * x + 0.6086) * x;//GATES BUG, V.1.3
        //double x = 0.2982*pow(ld,6) - 0.0127*pow(ld,5) + 0.0588*pow(ld,4) + 0.0099*pow(ld,3) + 0.6281*pow(ld,2) - 0.0009*ld;
        gsl_matrix_set(gene->pvalCorr,i,j,x);
        gsl_matrix_set(gene->pvalCorr,j,i,x);
      }
      gsl_matrix_set(gene->Cov, i, j, cov);
      gsl_matrix_set(gene->Cov, j, i, cov);

      j++;
      snp2 = cq_getNext(snp2, snp_queue);
    }
    i++;
    snp1 = cq_getNext(snp1, snp_queue);
  }
}

//"partial" forward regression, to be used in Dan's method to calculate the effective number of tests
double forwardRegression(gsl_matrix * LD, gsl_vector * Y, int nSNP, gsl_vector* workspace ){

  double sum=0, weight;
  int i, j, bestSNP;
  
  for(i=0; i<nSNP; i++){
    bestSNP = gsl_vector_max_index(Y);
    weight= gsl_vector_get(Y, bestSNP);
    if(weight <=0) break;
    sum += weight;
    for(j=0; j<nSNP; j++)
      gsl_vector_set(workspace, j, pow(gsl_matrix_get(LD, bestSNP, j), 2));
    gsl_vector_scale(workspace, weight);
    gsl_vector_sub(Y, workspace);
  }
  return sum;
}
//calculate the effective numeber of SNPs using Dan's way
double getGeneESNP_dan(gsl_matrix * LD){

  static double* Y_data = NULL;
  static double* U_data = NULL;
  static gsl_vector_view Y;
  static gsl_vector_view U;
  static int max_dim=MAX_SNP_PER_GENE;
  
  int nSNP = LD->size1;
  
  if(U_data == NULL){
    if(U_data==NULL) U_data=(double*) malloc(sizeof(double)*MAX_SNP_PER_GENE);else abort();
    if(Y_data==NULL) Y_data=(double*) malloc(sizeof(double)*MAX_SNP_PER_GENE);else abort();
  }

  if(nSNP > max_dim){
    max_dim = nSNP;
    free(U_data);U_data=NULL;
    free(Y_data);Y_data=NULL;
    if(U_data==NULL) U_data=(double*) malloc(sizeof(double)*nSNP);else abort();
    if(Y_data==NULL) Y_data=(double*) malloc(sizeof(double)*nSNP);else abort();
  }

  Y = gsl_vector_view_array(Y_data, nSNP);
  U = gsl_vector_view_array(U_data, nSNP);

  gsl_vector_set_all(&Y.vector, 1);

  return forwardRegression(LD, &Y.vector, nSNP, &U.vector);
}

/*
//calculate the effective number of tests using matrix eigen values
double getGeneESNP_matrix(gsl_matrix * LD, int nSNP){

  static int workspace_size = MAX_SNP_PER_GENE;
  static gsl_eigen_symm_workspace *w = NULL; 
  static double* eigenv_data = NULL;
  static gsl_vector_view eigenv;

  if( w == NULL){
    w = gsl_eigen_symm_alloc(MAX_SNP_PER_GENE);
    eigenv_data = (double*) malloc(sizeof(double)*MAX_SNP_PER_GENE);
  }
  if(nSNP >  workspace_size){
    gsl_eigen_symm_free(w);
    w = gsl_eigen_symm_alloc(nSNP);
    free(eigenv_data);
    eigenv_data = (double*) malloc(sizeof(double) * nSNP);
  }
  eigenv = gsl_vector_view_array(eigenv_data, nSNP);

  gsl_eigen_symm (LD, &eigenv.vector, w);
  int i;
  for(i=0; i<nSNP; i++){
    printf("%g\n", gsl_vector_get(&eigenv.vector, i));
  }
  double lambda = variance(eigenv.vector.data, nSNP,false);

  return 1+ (nSNP-1)*(1-lambda/nSNP);
}
*/

//printe the eigen values of the covariance matrix for debug purpose
void printCovEigen(gsl_matrix * Cov, int nSNP, int chr, char*gene, FILE* fp_cor_eigen_vector){

  static int workspace_size = MAX_SNP_PER_GENE;
  static gsl_eigen_symm_workspace *w = NULL; 
  static double* eigenv_data = NULL;
  static gsl_vector_view eigenv;

  if( w == NULL){
    if(w==NULL) w = gsl_eigen_symm_alloc(MAX_SNP_PER_GENE); else abort();
    if(eigenv_data==NULL) eigenv_data = (double*) malloc(sizeof(double)*MAX_SNP_PER_GENE); else abort();
  }
  if(nSNP >  workspace_size){
    gsl_eigen_symm_free(w);w=NULL;
    if(w==NULL) w = gsl_eigen_symm_alloc(nSNP);else abort();
    free(eigenv_data);eigenv_data=NULL;
    if(eigenv_data==NULL) eigenv_data = (double*) malloc(sizeof(double) * nSNP); else abort();
  }

  eigenv = gsl_vector_view_array(eigenv_data, nSNP);

  gsl_eigen_symm (Cov, &eigenv.vector, w);
	
  int i;
  for (i =0; i< nSNP; i++){
    fprintf(fp_cor_eigen_vector, "%d\t%s\t%g\n", chr, gene, gsl_vector_get(&eigenv.vector, i));
  }
  
}

double GATES_me(gsl_vector * eigv, int n)
{
  double me = 0;
  int i = 0;
  for(i=0;i<n;i++)
  {
     double v = gsl_vector_get(eigv,i);
     if(v>1)
     {
        me += (v - 1); 
     }
  }
  me = n - me;
  return me; 
}

void get_GATES_statistic(GENE* gene, C_QUEUE * snp_queue)
{
  //compute the eigenvalues of the pairwise snp-pvalue-correlation matrix
  gsl_matrix * Corr = gene->pvalCorr;//FIX GATES V.1.2  

  static int workspace_size = MAX_SNP_PER_GENE;
  static gsl_eigen_symm_workspace *w = NULL;
  static double * eigenv_data = NULL;
  static gsl_vector_view eigenv;
  static double * Corr_copy_data = NULL;
  static double * M_data = NULL;
  static double * M_copy_data = NULL;
  static gsl_matrix_view Corr_copy;
  static gsl_matrix_view M;
  static gsl_matrix_view M_copy;

  int nSNP = gene->nSNP;

  if( w == NULL){
    if(w==NULL) w = gsl_eigen_symm_alloc(MAX_SNP_PER_GENE); else abort();
    if(eigenv_data==NULL) eigenv_data = (double*) malloc(sizeof(double)*MAX_SNP_PER_GENE); else abort();
    if(Corr_copy_data==NULL) Corr_copy_data = malloc(MAX_SNP_PER_GENE*MAX_SNP_PER_GENE*sizeof(double)); else abort();
    if(M_data==NULL) M_data = malloc(MAX_SNP_PER_GENE*MAX_SNP_PER_GENE*sizeof(double)); else abort();
    if(M_copy_data==NULL) M_copy_data = malloc(MAX_SNP_PER_GENE*MAX_SNP_PER_GENE*sizeof(double)); else abort();
  }
  if(nSNP >  workspace_size){
    gsl_eigen_symm_free(w);w=NULL;
    free(eigenv_data);eigenv_data=NULL;
    free(Corr_copy_data);Corr_copy_data=NULL;
    free(M_data);M_data=NULL;
    free(M_copy_data);M_copy_data=NULL;
    
    if(w==NULL)w = gsl_eigen_symm_alloc(nSNP); else abort();
    if(eigenv_data==NULL) eigenv_data = (double*) malloc(sizeof(double) * nSNP); else abort();
    if(Corr_copy_data==NULL) Corr_copy_data = malloc(nSNP*nSNP*sizeof(double)); else abort();
    if(M_data==NULL) M_data = malloc(nSNP*nSNP*sizeof(double)); else abort();
    if(M_copy_data==NULL) M_copy_data = malloc(nSNP*nSNP*sizeof(double)); else abort();
  }

  //make a copy of Corr so that it is not destroyed by gsl_eigen_symm
  Corr_copy = gsl_matrix_view_array(Corr_copy_data,gene->nSNP,gene->nSNP); 
  gsl_matrix_memcpy (&Corr_copy.matrix, Corr);

  eigenv = gsl_vector_view_array(eigenv_data, nSNP);
  gsl_eigen_symm(&Corr_copy.matrix, &eigenv.vector, w); //eigen values for the entire gene.

  //now sort the snps by pvalue.
  //first create a copy.
  SNP* array[nSNP];
  int i = 0;
  int j = 0;
  SNP* snp = cq_getItem(gene->snp_start, snp_queue);
  for(i= gene->snp_start; i <= gene->snp_end; i++)
  {
     array[j++] = snp;
     snp = cq_getNext(snp, snp_queue);
  }

  gsl_matrix_view M_sub;
  M = gsl_matrix_view_array(M_data,gene->nSNP,gene->nSNP); 
  M_copy = gsl_matrix_view_array(M_copy_data,gene->nSNP,gene->nSNP); 

  int a = 0;
  int b = 0;

  if(GET_GATES_LINEAR)
  {
    if(VERBOSE)
       printf("-GATES LINEAR : starting to sort pvalues\n"); 
    quicksort_linear(array,0,nSNP-1);
    //now array is sorted by ascending pvalues.

    //GATES BUG V.1.3.mc
    //store sorted Corr matrix in M.
    for(i=0;i<nSNP;i++)
    {  
       int id1 = array[i]->gene_id;
       gsl_matrix_set(&M.matrix,i,i,gsl_matrix_get(Corr,id1,id1));
       for(j=i+1;j<nSNP;j++)
       {
         int id2 = array[j]->gene_id;
         double corr = gsl_matrix_get(Corr,id1,id2);
         gsl_matrix_set(&M.matrix,i,j,corr);
         gsl_matrix_set(&M.matrix,j,i,corr);
       }
    }

    if(VERBOSE)
      printf("-GATES LINEAR : pvalues sorted\n"); 
    //for(j=0;j<nSNP;j++) printf("%s pval[%d] = %g\n",array[j]->name,j,array[j]->pval_linear); 
    double PG = GSL_POSINF;
    double me = GATES_me(&eigenv.vector, nSNP);
    if(VERBOSE)
      printf("me for gene = %g\n",me);

    //GATES BUG V.1.3.mc
    bool flag = false; //V.1.4.mc
    PG = me*array[0]->pval_linear;
    for(j=1;j<nSNP-1;j++)
    { 
      //copy submatrix to M_copy.
      for(a=0;a<=j;a++)
      {   
          gsl_matrix_set(&M_copy.matrix,a,a,gsl_matrix_get(&M.matrix,a,a));
          for(b=a+1;b<=j;b++)
          {   
              gsl_matrix_set(&M_copy.matrix,a,b,gsl_matrix_get(&M.matrix,a,b));
              gsl_matrix_set(&M_copy.matrix,b,a,gsl_matrix_get(&M.matrix,b,a));
          }
      }
      M_sub = gsl_matrix_submatrix(&M_copy.matrix,0,0,j+1,j+1); //create a submatix view of M_copy.

      eigenv = gsl_vector_view_array(eigenv_data, j+1);
      gsl_eigen_symm(&M_sub.matrix, &eigenv.vector, w); //eigen values for the matrix upto j snps. 

      double me_j = GATES_me(&eigenv.vector, j+1);
      double g = me*array[j]->pval_linear/me_j;
      if(g<PG)
        PG = g;
      if(array[j]->pval_linear>=0.01){ //V.1.4.mc
        printf("-GATES : early termination\n");fflush(stdout);
        flag = true;
        break;
      }

      //printf("j=%d  me_j=%g  pvalue=%g PG=%g\n****************************************\n",j,me_j,array[j]->pval_linear,PG);
    }
    //+GATES BUG NEW
    if(nSNP>1 && !flag) //V.1.4.mc
    {
      double g = array[j]->pval_linear;
      if(g<PG)
         PG = g;
    }
    //-GATES BUG NEW
    if(PG>1) PG=1.0; //V.1.7.mc
    gene->gates_linear = PG;
    if(VERBOSE)
       printf("-Gene GATES statistic linear = %g\n",gene->gates_linear);
    if(PG<0)
    {
       printf("\nWARNING : Gates Linear PG<0\n\n");
    }
  }
  else if(GET_GATES_LOGISTIC)
  {
    if(VERBOSE)
       printf("-GATES LOGISTIC : starting to sort pvalues\n"); 
    quicksort_logistic(array,0,nSNP-1);
    //now array is sorted by ascending pvalues.

    //GATES BUG V.1.3.mc
    //store sorted Corr matrix in M.
    for(i=0;i<nSNP;i++)
    { 
       int id1 = array[i]->gene_id;
       gsl_matrix_set(&M.matrix,i,i,gsl_matrix_get(Corr,id1,id1));
       for(j=i+1;j<nSNP;j++)
       {
         int id2 = array[j]->gene_id;
         double corr = gsl_matrix_get(Corr,id1,id2);
         gsl_matrix_set(&M.matrix,i,j,corr);
         gsl_matrix_set(&M.matrix,j,i,corr);
       }
    }

    if(VERBOSE)
       printf("-GATES LOGISTIC : pvalues sorted\n"); 
    //for(j=0;j<nSNP;j++) printf("%s pval[%d] = %g\n",array[j]->name,j,array[j]->pval_logistic); 
    double PG = GSL_POSINF;
    double me = GATES_me(&eigenv.vector, nSNP);
    if(VERBOSE)
      printf("me for gene = %g\n",me);

    //GATES BUG V.1.3.mc
    bool flag = false; //V.1.4.mc
    PG = me*array[0]->pval_logistic;
    for(j=1;j<nSNP-1;j++)
    {
      //copy submatrix to M_copy.
      for(a=0;a<=j;a++)
      {
         gsl_matrix_set(&M_copy.matrix,a,a,gsl_matrix_get(&M.matrix,a,a));
         for(b=a+1;b<=j;b++)
         {
            gsl_matrix_set(&M_copy.matrix,a,b,gsl_matrix_get(&M.matrix,a,b));
            gsl_matrix_set(&M_copy.matrix,b,a,gsl_matrix_get(&M.matrix,b,a));
         }
      }
      M_sub = gsl_matrix_submatrix(&M_copy.matrix,0,0,j+1,j+1); //create a submatix view of M_copy.
      eigenv = gsl_vector_view_array(eigenv_data, j+1);
      gsl_eigen_symm(&M_sub.matrix, &eigenv.vector, w); //eigen values for the matrix upto j snps. 

      double me_j = GATES_me(&eigenv.vector, j+1);
      double g = me*array[j]->pval_logistic/me_j;
      if(g<PG)
        PG = g;
      if(array[j]->pval_logistic>=0.01){ //V.1.4.mc
         flag = true;
         printf("-GATES : early termination\n");fflush(stdout);
         break;
      }

      //printf("j=%d  me_j=%g  pvalue=%g PG=%g\n****************************************\n",j,me_j,array[j]->pval_logistic,PG);
    }
    //+GATES BUG NEW
    if(nSNP>1 && !flag) //V.1.4.mc
    {
      double g = array[j]->pval_logistic;
      if(g<PG)
         PG = g;
      //printf("j=%d  me_j=%g  pvalue=%g PG=%g\n****************************************\n",j,me,array[j]->pval_logistic,PG);
    }
    //-GATES BUG NEW
    if(PG>1) PG=1.0; //V.1.7.mc
    gene->gates_logistic = PG;
    if(VERBOSE)
       printf("-Gene GATES statistic logistic = %g\n",gene->gates_logistic);
    if(PG<0)
    {
       printf("\nWARNING Gates Logistic PG<0\n\n");
    }
  }
}

void get_GATES_statistic_SUMMARY(GENE* gene, C_QUEUE * snp_queue)
{
  if(VERBOSE)
     printf("-Performing GATES summary\n");
  //compute the eigenvalues of the pairwise snp-pvalue-correlation matrix
  gsl_matrix * Corr = gene->pvalCorr;//FIX GATES V.1.2  

  static int workspace_size = MAX_SNP_PER_GENE;
  static gsl_eigen_symm_workspace *w = NULL;
  static double * eigenv_data = NULL;
  static gsl_vector_view eigenv;
  static double * Corr_copy_data = NULL;
  static double * M_data = NULL;
  static double * M_copy_data = NULL;
  static gsl_matrix_view Corr_copy;
  static gsl_matrix_view M;
  static gsl_matrix_view M_copy;

  int nSNP = gene->nSNP;

  if( w == NULL){
    if(w==NULL) w = gsl_eigen_symm_alloc(MAX_SNP_PER_GENE);else abort();
    if(eigenv_data==NULL) eigenv_data = (double*) malloc(sizeof(double)*MAX_SNP_PER_GENE);else abort();
    if(Corr_copy_data==NULL) Corr_copy_data = malloc(MAX_SNP_PER_GENE*MAX_SNP_PER_GENE*sizeof(double));else abort();
    if(M_data==NULL) M_data = malloc(MAX_SNP_PER_GENE*MAX_SNP_PER_GENE*sizeof(double));else abort();
    if(M_copy_data==NULL) M_copy_data = malloc(MAX_SNP_PER_GENE*MAX_SNP_PER_GENE*sizeof(double));else abort();
  }
  if(nSNP >  workspace_size){
    gsl_eigen_symm_free(w);w=NULL;
    free(eigenv_data);eigenv_data=NULL;
    free(Corr_copy_data);Corr_copy_data=NULL;
    free(M_data);M_data=NULL;
    free(M_copy_data);M_copy_data=NULL;
    if(w==NULL) w = gsl_eigen_symm_alloc(nSNP);else abort();
    if(eigenv_data==NULL) eigenv_data = (double*) malloc(sizeof(double) * nSNP);else abort();
    if(Corr_copy_data==NULL) Corr_copy_data = malloc(nSNP*nSNP*sizeof(double));else abort();
    if(M_data==NULL) M_data = malloc(nSNP*nSNP*sizeof(double));else abort();
    if(M_copy_data==NULL) M_copy_data = malloc(nSNP*nSNP*sizeof(double));else abort();
  }

  //make a copy of Corr so that it is not destroyed by gsl_eigen_symm
  Corr_copy = gsl_matrix_view_array(Corr_copy_data,gene->nSNP,gene->nSNP); 
  gsl_matrix_memcpy (&Corr_copy.matrix, Corr);

  eigenv = gsl_vector_view_array(eigenv_data, nSNP);
  gsl_eigen_symm(&Corr_copy.matrix, &eigenv.vector, w); //eigen values for the entire gene.

  //now sort the snps by pvalue.
  //first create a copy.
  SNP* array[nSNP];
  int i = 0;
  int j = 0;
  SNP* snp = cq_getItem(gene->snp_start, snp_queue);
  for(i= gene->snp_start; i <= gene->snp_end; i++)
  {
     //snp->eig_val = gsl_vector_get(&eigenv.vector,j);
     array[j++] = snp;
     snp = cq_getNext(snp, snp_queue);
  }

  gsl_matrix_view M_sub;
  M = gsl_matrix_view_array(M_data,gene->nSNP,gene->nSNP); 
  M_copy = gsl_matrix_view_array(M_copy_data,gene->nSNP,gene->nSNP); 

  if(VERBOSE)
     printf("-GATES SUMMARY : starting to sort pvalues\n"); 
  quicksort_summary(array,0,nSNP-1);
  //now array is sorted by ascending pvalues.
  if(VERBOSE)
    printf("-GATES SUMMARY : pvalues sorted\n"); 
  //for(j=0;j<nSNP;j++) printf("%s pval[%d] = %g\n",array[j]->name,j,array[j]->metaP); 
  //GATES BUG V.1.3.mc
  int a = 0;
  int b = 0;
  //store sorted Corr matrix in M.
  for(i=0;i<nSNP;i++)
  {  
     int id1 = array[i]->gene_id;
     gsl_matrix_set(&M.matrix,i,i,gsl_matrix_get(Corr,id1,id1));
     for(j=i+1;j<nSNP;j++)
     {
         int id2 = array[j]->gene_id;
         double corr = gsl_matrix_get(Corr,id1,id2);
         gsl_matrix_set(&M.matrix,i,j,corr);
         gsl_matrix_set(&M.matrix,j,i,corr);
     }
  }
 
  double PG = GSL_POSINF;
  double me = GATES_me(&eigenv.vector, nSNP);
  if(VERBOSE)
    printf("me for gene = %g\n",me);

  bool flag = false; //V.1.4.mc
  //GATES BUG V.1.3.mc
  PG = me*array[0]->metaP;
  if(VERBOSE)
     printf("%s pvalue=%g PG=%g\n****************************************\n",array[0]->name,array[0]->metaP,PG);
  for(j=1;j<nSNP-1;j++)
  {
      //copy submatrix to M_copy.
      for(a=0;a<=j;a++)
      {
          gsl_matrix_set(&M_copy.matrix,a,a,gsl_matrix_get(&M.matrix,a,a));
          for(b=a+1;b<=j;b++)
          {
              gsl_matrix_set(&M_copy.matrix,a,b,gsl_matrix_get(&M.matrix,a,b));
              gsl_matrix_set(&M_copy.matrix,b,a,gsl_matrix_get(&M.matrix,b,a));
          }
      }
      M_sub = gsl_matrix_submatrix(&M_copy.matrix,0,0,j+1,j+1); //create a submatix view of M_copy.

      eigenv = gsl_vector_view_array(eigenv_data, j+1);
      gsl_eigen_symm(&M_sub.matrix, &eigenv.vector, w); //eigen values for the matrix upto j snps. 

      double me_j = GATES_me(&eigenv.vector, j+1);
      double g = me*array[j]->metaP/me_j;
      if(g<PG)
        PG = g;
      if(array[j]->metaP>=0.01){
        flag = true;
        printf("-GATES : early termination\n");fflush(stdout);
        break;
      }

      if(VERBOSE)
        printf("j=%d  me_j=%g  pvalue=%g PG=%g\n****************************************\n",j,me_j,array[j]->metaP,PG);
  }
  //+GATES BUG NEW V.1.3.mc
  if(nSNP>1 && !flag) //V.1.4.mc
  {
    double g = array[j]->metaP;
    if(g<PG)
       PG = g;
    //printf("j=%d  me_j=%g  pvalue=%g PG=%g\n****************************************\n",j,me,array[j]->metaP,PG);
  }
  //-GATES BUG NEW V.1.3.mc
  if(PG>1) PG=1.0; //V.1.7.mc 
  gene->gates_linear = PG;
  gene->gates_logistic = PG;
  if(VERBOSE)
     printf("-Gene GATES statistic summary = %g\n",gene->gates_linear);
  if(PG<0)
  {
     printf("\nWARNING : Gates meta PG<0\n\n");
  }
}

//functions for BIC tests
//check whether the stop condition for permutations is satisfied
bool ifStop(int nHit, int iPerm){
 
  //printf("ifStop : nHit=%d iPerm=%d N_PERM_MIN=%d N_PERM=%d\n",nHit,iPerm,N_PERM_MIN,N_PERM);
  if(SMART_STOP)
  {
    if(iPerm>=N_PERM_MIN)
    {
      if(nHit<=1 && iPerm <N_PERM)
	return false;
      double p = ((double)nHit)/iPerm;
      double upper= p+1.65*sqrt(nHit*(1-p))/iPerm;
      double lower= p-1.65*sqrt(nHit*(1-p))/iPerm;
      if(upper>1) upper=1;
      if(lower<0) lower=0;
      if(upper<2E-6 || lower>2E-6 || (iPerm >=  N_PERM))
	return true;
      else
	return false;
    }else
      return false;
  }else{
    return ((nHit >= N_CUTOFF) && (iPerm>=N_PERM_MIN)) || (iPerm >=  N_PERM);
  }
}
//print GWiS output
void printBICPermResult(FILE * fp, int i, int nsample, GENE * gene, OrthNorm *Z,  int k_pick, int k_hits, double pval){

  double SSM;
  double f_stat;
  bool summary;
  BIC_STATE* ptr = &gene->bic_state_linear;

  summary = pval >= -EPS;

  SSM =  ptr->RSS[0] - ptr->RSS[i];
  
  if(i > 0 && i <= ptr->iSNP){
    if(summary)
      f_stat = SSM/ptr->RSS[i] * (nsample-i-1)/i;
    else
      f_stat = ptr->bestSNP[i]->f_stat;
  }else
    f_stat = 0;

  //print basic info about the gene
  fprintf(fp, "%d\t%s\t%s\t%d\t%d\t%d\t%d\t%g\t", gene->chr, gene->ccds, gene->name, gene->bp_start, gene->bp_end, gene->bp_end - gene->bp_start+1, gene->nSNP, gene->eSNP);
    
  //print the info about the last snp
  if(summary) 
    fprintf(fp, "SUMMARY\t-\t-\t-\t");
  else
    if(i == 0) 
      fprintf(fp, "NONE\t-\t-\t-\t");
    else if(i <= ptr->iSNP) 
      fprintf(fp, "%s\t%d\t%g\t%g\t", ptr->bestSNP[i]->name, ptr->bestSNP[i]->bp, ptr->bestSNP[i]->MAF,  ptr->bestSNP[i]->R2);
    else fprintf(fp, "-\t-\t-\t-\t");
  
  //print the model info 
  if(i>ptr->iSNP) fprintf(fp, "%d\t-\t-\t-\t-\t", i);
  else{
    if(!summary || i >0)
      fprintf(fp, "%d\t%g\t%g\t%g\t", i, SSM,  ptr->BIC[i] - ptr->BIC[0], f_stat);
    else
    {
      fprintf(fp, "%d\t%g\t%g\t%g\t", i, ptr->RSS[0] - ptr->RSS[1],  ptr->BIC[1] - ptr->BIC[0], ptr->bestSNP[1]->f_stat);
    }

    if(i> 0 && !summary){
      fprintf(fp, "%g\t", 1-Z[ptr->bestSNP[i]->gene_id].norm/Z[ptr->bestSNP[i]->gene_id].norm_original);
      //      printf( "%d, %g, %g\n", ptr->bestSNP[i]->gene_id, Z[ptr->bestSNP[i]->gene_id].norm, Z[ptr->bestSNP[i]->gene_id].norm_original);
    }
    else
      fprintf(fp, "-\t");
  }
  
  //print the model selection info
  fprintf(fp, "%d\t%d\t", k_pick, k_hits);
  if(summary)
    fprintf(fp, "%g\n", pval);
  else
    fprintf(fp, "-\n");
}
//++Pritam : new function for writing permutation output of Logistic regression and negative binomial regression.
void printBIC_Logistic_PermResult(FILE * fp, int i, int nsample, GENE * gene, int k_pick, int k_hits, double pval)
{
  bool summary;
  BIC_STATE* ptr = &gene->bic_state_logistic;
  summary = pval >= -EPS;
  //printf("printBIC_Logistic_PermResult : %d %d %s %d\n",i,ptr->iSNP,ptr->bestSNP[i]->name,summary);

  //print basic info about the gene
  fprintf(fp, "%d\t%s\t%s\t%d\t%d\t%d\t%d\t%g\t", gene->chr, gene->ccds, gene->name, gene->bp_start, gene->bp_end, gene->bp_end - gene->bp_start+1, gene->nSNP, gene->eSNP);

  //print the info about the last snp
  if(summary)
        fprintf(fp, "SUMMARY\t-\t-\t-\t");
  else
  {
        if(i == 0)
          fprintf(fp, "NONE\t-\t-\t-\t");
        else if(i <= ptr->iSNP)
        {
          fprintf(fp, "%s\t%d\t%g\t%g\t", ptr->bestSNP[i]->name, ptr->bestSNP[i]->bp, ptr->bestSNP[i]->MAF,  ptr->bestSNP[i]->R2);
        }
        else fprintf(fp, "-\t-\t-\t");
  }
  //print the model info
  if(i>ptr->iSNP) fprintf(fp, "%d\t-\t-\t-\t", i);
  else
  {
      if(i> 0 || !summary)
      {
         fprintf(fp, "%d\t%g\t%g\t", i, ptr->BIC[i] - ptr->BIC[0], 2*(ptr->LL[i] - ptr->LL[0]) ); //Pritam
         //printf("=>%d\t%g\t%g\t", i, ptr->BIC[i] - ptr->BIC[0], 2*(ptr->LL[i] - ptr->LL[0]) ); //Pritam
      }
      else
      {
         fprintf(fp, "%d\t%g\t%g\t", i, ptr->BIC[1] - ptr->BIC[0], 2*(ptr->LL[1] - ptr->LL[0]) ); //Pritam
         //printf("summary=>%d\t%g\t%g\t", i, ptr->BIC[1] - ptr->BIC[0], 2*(ptr->LL[1] - ptr->LL[0]) ); //Pritam
      }
      //if(i> 0 && !summary)
      //  fprintf(fp, "%g\t", 0.0);//multiple R2
      //else
      //  fprintf(fp, "-\t");
  }
  //print the model selection info
  fprintf(fp, "%d\t%d\t", k_pick, k_hits);
  if(summary)
        fprintf(fp, "%g\n", pval);
  else
        fprintf(fp, "-\n");
}

//--Pritam

//set the counter after each permutation
//1) increase iPerm by 1
//2) incrase Hit by 1 if necessary
//3) enough permutation
bool countBIChit_linear(GENE* gene, int nsample, BIC_STATE* bic_state_new,  FILE* fp_diag)
{
  SNP *curSNP;
  HIT_COUNTER *hits = &gene->hits_sh;
  BIC_STATE * ptr = &gene->bic_state_linear;
  int i, nHit, iPerm;
  bool stop;

  nHit = 0;
  iPerm = 0;

  if(VERBOSE)
  { 
    printf("-countBIChit_linear: start to count\n");
    printf("-Lin iSNP =  %d\n",bic_state_new->iSNP);
    printf("-Linear : start to count BIC org = %f, BIC perm = %f \n",ptr->BIC[ptr->iSNP],bic_state_new->BIC[bic_state_new->iSNP]);
  }

  if(n_threads>1)
     pthread_mutex_lock(&mutex_bic_linear); //<--lock mutex for bic linear
  hits->k_pick_linear[bic_state_new->iSNP]++;

  if(bic_state_new->BIC[bic_state_new->iSNP] >= ptr->BIC[ptr->iSNP]-EPS)
    hits->k_hits_linear[bic_state_new->iSNP]++;
  if(bic_state_new->iSNP > hits->maxK_linear)
    hits->maxK_linear = bic_state_new->iSNP;

  for(i=0; i<=hits->maxK_linear; i++)
  {
    nHit += hits->k_hits_linear[i];
    iPerm += hits->k_pick_linear[i];
  }
  if(n_threads>1)
     pthread_mutex_unlock(&mutex_bic_linear); //<--unlock mutex for bic linear

  stop = ifStop(nHit, iPerm);
  if(VERBOSE)
    if(!stop)
      printf("-Bic hits = %d iPerm = %d\n", nHit,iPerm);

  if(fp_diag != NULL)
  {
    for(i=0; i<=min(bic_state_new->iSNP+1, min(gene->eSNP, MAX_INCLUDED_SNP)) ; i++)
    {
      curSNP = bic_state_new->bestSNP[i];
      fprintf(fp_diag, "%d\t%s\t%s\tperm%d\t%d\t%g\t%d\t%d\t%s\t%g\t%g\n", gene->chr, gene->ccds, gene->name, iPerm, gene->nSNP, gene->eSNP,bic_state_new->iSNP, i, (curSNP==NULL)? "NONE": curSNP->name, bic_state_new->BIC[i], bic_state_new->RSS[i]); 
    }
  }
  return stop;
}

//++Pritam : new function to update counts for logistic regression hits during permutation.
bool countBIChit_logistic(GENE* gene, int nsample, BIC_STATE* bic_state_new)
{
  //SNP *curSNP;
  HIT_COUNTER *hits = &gene->hits_sh;
  BIC_STATE * ptr = &gene->bic_state_logistic;
  int i, nHit, iPerm;
  bool stop;

  nHit = 0;
  iPerm = 0;
  if(VERBOSE) printf("-countBIChit_logistic : start to count BIC org = %f, BIC perm = %f \n",ptr->BIC[ptr->iSNP],bic_state_new->BIC[bic_state_new->iSNP]);

  if(n_threads>1)
     pthread_mutex_lock(&mutex_bic_logistic); //<--lock mutex for bic logistic
  hits->k_pick_logistic[bic_state_new->iSNP]++;

  if(bic_state_new->BIC[bic_state_new->iSNP] >= ptr->BIC[ptr->iSNP]-EPS)
        hits->k_hits_logistic[bic_state_new->iSNP]++;
  if(bic_state_new->iSNP > hits->maxK_logistic)
        hits->maxK_logistic = bic_state_new->iSNP;

  for(i=0; i<=hits->maxK_logistic; i++)
  {
        nHit += hits->k_hits_logistic[i];
        iPerm += hits->k_pick_logistic[i];
  }
  if(n_threads>1)
     pthread_mutex_unlock(&mutex_bic_logistic); //<--unlock mutex for bic logistic

  stop = ifStop(nHit, iPerm);
  if(VERBOSE)
        if(!stop)
          printf("-Logistic Bic hits = %d iPerm = %d \n", nHit, iPerm);


  //for(i=0; i<=min(bic_state_new->iSNP+1, min(gene->eSNP, MAX_INCLUDED_SNP)) ; i++)
  //{
        //curSNP = bic_state_new->bestSNP[i];
        //if(fp_diag != NULL)
        //  fprintf(fp_diag, "%d\t%s\t%s\tperm%d\t%d\t%g\t%d\t%d\t%s\t%g\t%g\n", gene->chr, gene->ccds, gene->name, iPerm, gene->nSNP, gene->eSNP,bic_state_new->iSNP, i, (curSNP==NULL)? "NONE": curSNP->name, bic_state_new->BIC[i], bic_state_new->RSS[i]);
  //}

  if(VERBOSE) printf("-Logistic stop = %d\n",stop);

  return stop;
}


// check if a SNP has been selected into the model
bool isSNPUsed(OrthNorm* Z, OrthNorm** bestSNP, int k){
  int i;
  for(i=0; i<k-1; i++)
    if(Z == bestSNP[i])
      return true;
  return false;
}
//check of a SNP has high correlation w/ SNP in the model
bool isSNPCorrelated(OrthNorm* Z, OrthNorm** bestSNP, int k,  gsl_matrix *LD){
  int i;
  for(i=0; i<k-1; i++)
    if(fabs(gsl_matrix_get(LD, Z->snp->gene_id, bestSNP[i]->snp->gene_id)) >VIF_R)
      return true;
  return false;
}
//calculate SSM from ortho-norm 
double calculateMfromZ(OrthNorm*Z, OrthNorm**bestZ, int k, gsl_matrix* LD, gsl_matrix* Cov){

  double SSM;

  if(isSNPUsed(Z, bestZ, k))
    return -1;
  if(isSNPCorrelated(Z, bestZ, k, LD))
    return -1;

  //if(VERBOSE)
  //  printf("Stop1\n");
 
 if(VERBOSE)
     printf("here 1: %d, %s,%g, %g \n", k, Z->snp->name, Z->norm, Z->projP);
  
  calculateZ(Z, bestZ, k, Cov);
  //if(VERBOSE) 
  //  printf("Stop2\n");
  if(1-Z->norm/Z->norm_original>VIF_R2)
    return -1;
  if(fabs(Z->norm)<EPS)
    SSM = -1;
  else
    SSM = getSSM(Z);

  if(VERBOSE)
     printf("here 2: %d, %s,%g, %g, %g, \n", k, Z->snp->name, Z->norm, Z->projP, SSM);

  return SSM;
}

//get the change of GWiS statistics when adding a SNP into the model
//positive: add
//negative: don't add
double getIncrement(double SSM, double eSNP, int k, int nsample, double RSS){

  if(SSM >= RSS) return GSL_NEGINF; //V.1.3.mc nan BIC FIX

  //the change in logProb from k-1 to k
  double increment;

  //printf("--getIncrement: %lg %lg %d %d %lg\n",SSM,eSNP,k,nsample,RSS); //REMOVE
  
  increment = log(eSNP-k+1) - log(k);
#ifdef AIC
  increment ++;
#else
  increment += log(nsample)/2;
#endif

  //increment += log(nsample)/2;
  
  increment += log(RSS - SSM)/2 * nsample;
  increment -= log(RSS)/2 *nsample;
  
  if(k==1)
    increment +=log(eSNP+1) -log((1-P0)/(P0+ (1-P0)/(1+eSNP)));

  return -increment + Test_increment;
}

//FIX HESSIAN LATEST V.1.2
double getlog10BF(double pheno_var, double geno_var, int N, double cov_pheno_geno ){
  double bf=0;
  //  printf("%g, %g, %d, %g\n", pheno_var, geno_var, N, cov_pheno_geno );
  bf=-log10(N)/2.0-log10(geno_var+1.0/(N*SIGMA_A*SIGMA_A))/2.0-log10(SIGMA_A);
  //  printf("%g\n", bf);
  bf+= -((double)N)/2.0 * log10(1.0-cov_pheno_geno*cov_pheno_geno/(geno_var*pheno_var+pheno_var/N/SIGMA_A/SIGMA_A));
  return bf;
}

//find the best fit snp to add to the GWiS model for the gene
SNP* bestFitSNP(OrthNorm **bestZ, OrthNorm *Z, int k, GENE* gene, double *SSM, int nsample, double RSS, double BIC_prev, double SSM_prev, double*increment, FILE* fp_diag){
 
  SNP* ret=NULL;
  int i, j;
  ret = NULL;
  double maxSSM=-1; //FIX V.1.3.mc
  double maxIncrease = -1;

  for(i=0; i<gene->nSNP; i++, Z++){

    //printf("%d, %s,%g, %g, \n", k, Z->snp->name, Z->norm, Z->projP);//REMOVE
	
    *SSM= calculateMfromZ(Z, bestZ, k, gene->LD, gene->Cov)/nsample;
    if(VERBOSE) printf("-%d bestFitSNP: working on %s, got SSM %g\n",i,Z->snp->name, *SSM);
    //	double beta =  Z->projP/Z->norm;
    //	double se2 = (225.265-*SSM-SSM_prev)/Z->norm;
    //	printf("SE2 = %g, z=%g, SSM=%g\n", se2, beta/sqrt(se2),*SSM+SSM_prev);
    //	printf("%s\t%g\n", Z->snp->name, *SSM);
    if(*SSM<0) continue;
    *increment = getIncrement(*SSM, gene->eSNP, k, nsample, RSS);
    if(VERBOSE) printf("-%d bestFitSNP: working on %s, got increment %g\n",i,Z->snp->name, *increment);

    //if(VERBOSE) printf("Stop3\n");
    if(fp_diag != NULL){
      fprintf(fp_diag, "%d\t%s\t%s\t%d\t%g\t%d\t", gene->chr, gene->ccds,gene->name, gene->nSNP, gene->eSNP, k); 
      for(j=1; j<k; j++)
	fprintf(fp_diag, "%s_", bestZ[j-1]->snp->name);
      fprintf(fp_diag, "%s\t", Z->snp->name);
      fprintf(fp_diag, "%g\t%g\t%g\n", *increment+BIC_prev, *SSM+SSM_prev, 1-Z->norm/Z->norm_original);
    }
    //if(VERBOSE) printf("Stop4\n");
    if(*SSM>maxSSM){
      ret=Z->snp;
      maxSSM = *SSM;
      maxIncrease = *increment;
      bestZ[k-1] = Z;
      //printf("changed to %s at %g and %g \n", ret->name, *SSM, maxSSM);REMOVE
    }
  }
  //if(VERBOSE) printf("Stop5\n");
  (*SSM) = maxSSM;
  (*increment) = maxIncrease;

  if(VERBOSE)
  {
     printf("SSM = %g\n",maxSSM);
     //printf("p=%p\n",ret);
     fflush(stdout); 
     printf("-Best snp found = %s with SSM = %g\n\n---------\n\n",ret->name,maxSSM);
  }

  return ret;
}

//to build the GWiS model by finding the best SNP and decide whether to add it or stop adding SNPs
bool calBestFitSNP( BIC_STATE * bic_state, int k, OrthNorm** bestZ, GENE* gene,  OrthNorm *Z, int nsample, FILE * fp_result, FILE * fp_diag){

  double increment, SSM, BIC;
  SNP *bestSNP;

  if(VERBOSE) printf("-Entering calBestFitSNP,k=%d looking to add snp to model of size = %d\n",k,bic_state->iSNP);

  bestSNP = bestFitSNP(bestZ, Z, k, gene, &SSM, nsample,(bic_state->RSS)[k-1],(bic_state->BIC)[k-1],(bic_state->RSS)[0]-(bic_state->RSS)[k-1], &increment, fp_diag);

  BIC = increment + (bic_state->BIC)[k-1];
 
  //printf("new BIC = %lg old BIC = %lg increment = %lg \n",BIC,(bic_state->BIC)[k-1],increment);
  
  (bic_state->bestSNP)[k] =bestSNP;
  (bic_state->RSS)[k] = (bic_state->RSS)[k-1]-SSM;
  (bic_state->BIC)[k] = BIC;

  if( increment < 0 && k>K_MIN){
    bic_state->iSNP = k-1;
    if(VERBOSE) printf("-Exiting calBestFitSNP, increment = %g, k = %d, new model size = %d\n",increment,k,bic_state->iSNP);
    return false; //stop adding SNPs
  }else if(SSM/bic_state->RSS[0] >0.99){
    bic_state->iSNP = k;
    if(VERBOSE) printf("-Exiting calBestFitSNP,  SSM/bic_state->RSS[0] = %g new model size = %d\n",SSM/bic_state->RSS[0],bic_state->iSNP);
    return false; //stop adding SNPs
  }else
    bic_state->iSNP = k;

  if(VERBOSE) printf("-Exiting calBestFitSNP, new model size = %d\n",bic_state->iSNP);

  return true;
}

//initialize the GWiS result data structure
void initBIC_State(BIC_STATE*  bic_state, double pheno_tss_per_n){

  int k;
  for (k=0; k<=MAX_INCLUDED_SNP; k++){
    bic_state->bestSNP[k]=NULL;
    bic_state->BIC[k]=-1;
    bic_state->RSS[k]=-1;
    bic_state->LL[k]=-1;//Pritam
  }
  bic_state->iSNP = 0;
  (bic_state->RSS)[0] =  pheno_tss_per_n;
  (bic_state->BIC)[0] = 0;
  (bic_state->LL)[0] = 0;
}


// make a copy of the ortho-norm array
void copyZ(OrthNorm *Z_src, OrthNorm *Z_dest, int n){

  int i, k;
  for (i = 0; i < n; i++, Z_src++, Z_dest++){ 
    Z_dest->k = Z_src->k;
    Z_dest->snp = Z_src->snp;
    Z_dest->projP = Z_src->projP;
    Z_dest->norm  = Z_src->norm;
    Z_dest->norm_original  = Z_src->norm_original;
    for (k=0; k<MAX_INCLUDED_SNP; k++)
      Z_dest->sum_X_bestZ[k]  = Z_src->sum_X_bestZ[k];
  }
}

// used for full model search
void compareModels(double* SSM, double *BIC, double *bestBIC, OrthNorm **bestZ, int k, BIC_STATE * bic_state, GENE * gene, OrthNorm *Z, FILE* fp_diag){
  
  int i;
  double totalSSM=0;

  if(BIC[k-1] > *bestBIC){
    bic_state->iSNP = k;
    *bestBIC=BIC[k-1];

    for(i=1; i<=k; i++){
      totalSSM += SSM[i-1];
      if(i==k){
	(bic_state->bestSNP)[i] = Z->snp;
      }else{
	(bic_state->bestSNP)[i] =bestZ[i-1]->snp;
      }
      (bic_state->RSS)[i] = (bic_state->RSS)[i-1]-SSM[i-1];
      (bic_state->BIC)[i] = BIC[i-1];
    }
    if(fp_diag!=NULL){
      fprintf(fp_diag, "%d\t%s\t%s\t%d\t%g\t%d\t", gene->chr, gene->ccds,gene->name, gene->nSNP, gene->eSNP, k); 
      for(i=1; i<k; i++)
	fprintf(fp_diag, "%s_", bestZ[i-1]->snp->name);
      fprintf(fp_diag, "%s\t", Z->snp->name);
      fprintf(fp_diag, "%g\t%g\t%g\n", BIC[k-1], totalSSM, 1-Z->norm/Z->norm_original);
    }
  }
}

//call GWiS core function 
BIC_STATE* runBIC(GENE* gene, OrthNorm *Z, BIC_STATE * bic_state, int nsample, /*double pheno_mean,*/ double pheno_tss_per_n, FILE* fp_result, FILE * fp_diag, bool mute) //V.1.4.mc
{
  int k;

  OrthNorm *bestZ[MAX_INCLUDED_SNP];

  initBIC_State(bic_state, pheno_tss_per_n);

  if(fp_result != NULL)
    fprintf(fp_result, "%d\t%s\t%s\t%d\t%d\t%d\t%d\t%g\t%s\t%s\t%s\t%s\t%d\t%g\t%g\t%g\t-\t-\t-\t-\n", gene->chr, gene->ccds, gene->name, gene->bp_start, gene->bp_end, gene->bp_end-gene->bp_start+1, gene->nSNP, gene->eSNP,"NONE", "-", "-", "-", 0, 0.0, 0.0,0.0); 
 
  for(k=1; k<= min(gene->eSNP, MAX_INCLUDED_SNP);k++){
    if(!calBestFitSNP(bic_state, k, bestZ, gene,  Z, nsample, fp_result, fp_diag))
      break;
  }

  if(fp_result != NULL){
    for(k=1; k<=bic_state->iSNP; k++){
      fprintf(fp_result, "%d\t%s\t%s\t%d\t%d\t%d\t%d\t%g\t%s\t%d\t%g\t%g\t%d\t%g\t%g\t%g\t%g\t-\t-\t-\n", 
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
                         bic_state->bestSNP[k]->R2, 
                         k,
                         bic_state->RSS[0] - bic_state->RSS[k], 
                         //1-bic_state->RSS[k]/bic_state->RSS[0], 
                         bic_state->BIC[k], 
                         bic_state->bestSNP[k]->f_stat, 
                         1-bestZ[k-1]->norm/bestZ[k-1]->norm_original
                         ); 
      /*
	printf("%d, %g, %g\n", k, bestZ[k-1]->norm, bestZ[k-1]->norm_original);
	if(k>1){
	printf("%d, %g\n", k, gsl_matrix_get(gene->LD, bestZ[k-1]->snp->gene_id, bestZ[k-2]->snp->gene_id));
	printf("%d, %g\n", k, gsl_matrix_get(gene->LD, bestZ[k-2]->snp->gene_id, bestZ[k-1]->snp->gene_id));
	printf("%d, %s, %s\n", k, bestZ[k-1]->snp->name, bestZ[k-2]->snp->name);
	}*/
    }
    double f_stat = bic_state->iSNP==0? (bic_state->RSS[0]-bic_state->RSS[1])/bic_state->RSS[1] * (nsample-2):(bic_state->RSS[0]-bic_state->RSS[bic_state->iSNP])/bic_state->RSS[bic_state->iSNP] * (nsample-bic_state->iSNP-1)/bic_state->iSNP;
     
    fprintf(fp_result, "%d\t%s\t%s\t%d\t%d\t%d\t%d\t%g\t%s\t%s\t%s\t%s\t%d\t%g\t%g\t%g\t-\t-\t-\t-\n", 
                       gene->chr, 
                       gene->ccds, 
                       gene->name, 
                       gene->bp_start, 
                       gene->bp_end, 
                       gene->bp_end-gene->bp_start+1, 
                       gene->nSNP, 
                       gene->eSNP,
                       "SUMMARY", 
                       "-", 
                       "-",
                       "-", 
                       bic_state->iSNP,
                       bic_state->iSNP==0? bic_state->RSS[0] - bic_state->RSS[1]:bic_state->RSS[0] - bic_state->RSS[bic_state->iSNP], 
                       //1 - (bic_state->iSNP==0? bic_state->RSS[1]:bic_state->RSS[bic_state->iSNP]) /bic_state->RSS[0], 
                       bic_state->iSNP==0?bic_state->BIC[1]:bic_state->BIC[bic_state->iSNP],
                       f_stat); 
  }

  return bic_state;
}

void checkDecompDiff(gsl_matrix * L, gsl_matrix* LD,  int nSNP){
  
  static int max_dim = MAX_SNP_PER_GENE;
  static double* result_dat = NULL;
  
  if(result_dat==NULL)
    result_dat = malloc(sizeof(double)*MAX_SNP_PER_GENE*MAX_SNP_PER_GENE);

  if(nSNP > max_dim){
    max_dim = nSNP;
    free( result_dat ); result_dat = NULL;
    if(result_dat==NULL) result_dat = malloc(sizeof(double)*max_dim*max_dim); else abort();
  }

  gsl_matrix_view result = gsl_matrix_view_array (result_dat, nSNP, nSNP);

  gsl_blas_dgemm (CblasNoTrans, CblasTrans, 1.0, L, L, 0.0, &result.matrix);

  int i, j;
  double sum = 0;
  double max = 0;
  double diff= 0;
  for (i=0; i<nSNP; i++){
    for (j=0; j <=i; j++){
      diff = gsl_matrix_get( &result.matrix, i, j) - gsl_matrix_get( LD, i, j);
      sum+=diff*diff;
      if(max < fabs(diff))
	max = fabs(diff);
    }
  }
  sum /= nSNP*(nSNP+1)/2;
  printf("-Made the correlation matrix positive. Max change=%g, Ave change=%g\n", max ,sum);
  if(max > 0.05 || sum > 0.001)
    printf("\n-WARNING: LD MATRIX CHANGED SIGNIFICANTLY\n\n");
}

gsl_matrix* getL_from_LDLt(gsl_matrix *LD, int nSNP)
{
  static double* LD_positive_dat = NULL;
  static double* D_dat = NULL;
  static double* L_dat = NULL;
  static gsl_matrix_view L;

  static int max_dim = MAX_SNP_PER_GENE;

  if(D_dat==NULL){
      if(D_dat==NULL) D_dat = malloc(sizeof(double)*MAX_SNP_PER_GENE); else abort();
      if(L_dat==NULL) L_dat = malloc(sizeof(double)*MAX_SNP_PER_GENE*MAX_SNP_PER_GENE); else abort();
      if(LD_positive_dat==NULL) LD_positive_dat = malloc(sizeof(double)*MAX_SNP_PER_GENE*MAX_SNP_PER_GENE); else abort();
  }

  if(nSNP > max_dim){
    max_dim = nSNP;
    free(D_dat); D_dat=NULL;
    free(L_dat); L_dat=NULL;
    free(LD_positive_dat);LD_positive_dat=NULL;

    if(D_dat==NULL) D_dat = malloc(sizeof(double)*max_dim); else abort();
    if(L_dat==NULL) L_dat = malloc(sizeof(double)*max_dim*max_dim); else abort();
    if(LD_positive_dat==NULL) LD_positive_dat = malloc(sizeof(double)*max_dim*max_dim); else abort();
  }

  gsl_matrix_view  LD_positive =  gsl_matrix_view_array (LD_positive_dat,nSNP, nSNP);
  L = gsl_matrix_view_array (L_dat, nSNP, nSNP);
  gsl_vector_view D = gsl_vector_view_array (D_dat, nSNP);

  //    makePositiveDef(thread_param->gene->LD);
  spec_decomp_makePositive(LD, &LD_positive.matrix);
  printf("-done spectral decomposition\n");

  //    printf("%g, %g\n", thread_param->gene->eSNP, getGeneESNP_dan(result));
  LDLt_decomp(&LD_positive.matrix, &D.vector, &L.matrix);
  printf("-done LDLt decomposition\n");

  int i, j;
  for (j=0; j < nSNP; j++){
    double d = gsl_vector_get( &D.vector, j);
    if( d < 0)
      d=0;
    for (i=j; i< nSNP; i++){
      gsl_matrix_set(&L.matrix, i, j, gsl_matrix_get(&L.matrix, i, j)*sqrt(d));
    }
  }
  printf("-done calculating L\n");
  checkDecompDiff(&L.matrix, LD, nSNP);

  return &L.matrix;
  
}

void init_one_time(LOGISTIC_SCRATCH * LG,BG_SCRATCH * BG)
{
   //V.1.7.mc
   if(LG!=NULL)
   {
     LG->h = NULL;
     LG->h_temp = NULL;
     LG->h_inv = NULL;
     LG->W = NULL;
     LG->old_W = NULL;
     LG->W_delta = NULL;
     LG->g = NULL;
     LG->OMIT = NULL;
     LG->allocated = false;
   }

   //V.1.7.mc
   if(BG!=NULL)
   {
     BG->h=NULL;
     BG->h_temp = NULL;
     BG->h_inv = NULL;
     BG->nu=NULL;
     BG->W=NULL;
     BG->old_W = NULL;
     BG->W_delta = NULL;
     BG->g=NULL;
     BG->T=NULL;
     BG->x=NULL;
     BG->OMIT=NULL;
     BG->allocated = false;
   }
}

void init_scratch(LOGISTIC_SCRATCH * LG, PHENOTYPE * phenotype)
{
       int MAX_MODEL_SZ = MAX_INCLUDED_SNP + MAX_N_COVARIATES + 1;
       LG->phenotype = phenotype;
       if(!(LG->allocated))
       {
            printf("-Allocating logistic Regression scratch space ...\n");
            if(LG->h==NULL) LG->h = gsl_matrix_alloc(MAX_MODEL_SZ,MAX_MODEL_SZ); else abort();//hessian
            if(LG->h_temp==NULL) LG->h_temp = gsl_matrix_alloc(MAX_MODEL_SZ,MAX_MODEL_SZ); else abort(); //hessian to hold the result for LU decomposition.
            if(LG->h_inv==NULL) LG->h_inv = gsl_matrix_alloc(MAX_MODEL_SZ,MAX_MODEL_SZ); else abort(); //hessian inverse
            if(LG->W==NULL) LG->W = gsl_vector_alloc(MAX_MODEL_SZ); else abort(); //weight vector, updated in gradient descent iterations.
            if(LG->old_W==NULL) LG->old_W = gsl_vector_alloc(MAX_MODEL_SZ); else abort(); //old weight vector, used in gradient descent iterations.
            if(LG->W_delta==NULL) LG->W_delta = gsl_vector_alloc(MAX_MODEL_SZ); else abort(); //delta weight vector used in gradient descent iterations.
            if(LG->g==NULL) LG->g = gsl_vector_alloc(MAX_MODEL_SZ); else abort(); //gradient vector.
            if(LG->OMIT==NULL) LG->OMIT = gsl_vector_alloc(MAX_N_INDIV);else abort();//vector of missing data.
     
            //LG->T = gsl_multimin_fdfminimizer_conjugate_fr;
            //LG->s = NULL;
            //LG->x = gsl_vector_alloc(MAX_MODEL_SZ);

            int i = 0;
            for(i=0;i<MAX_MODEL_SZ;i++)
            {
              if(i<MAX_MODEL_SZ-1)
                 LG->curr_model[i] = NULL; //null pointers.
              LG->se[i] = 0;
              LG->wald[i] = 0;
            }
            LG->curr_model_size = 0;

            //Covariates : mimic snps.
            for(i=0;i<phenotype->n_covariates;i++)
            {
               SNP* snp_temp = (SNP*)malloc(sizeof(SNP));
               snp_temp->geno = (double*)malloc(phenotype->N_sample*sizeof(double));
               sprintf(snp_temp->name,"cov%d",(i+1));
               int j = 0;
               for(j=0;j<phenotype->N_sample;j++)
                   snp_temp->geno[j] = gsl_vector_get(phenotype->covariates[i],j);
               LG->cov_snps[i] = snp_temp;
               LG->curr_model[i] = snp_temp;
               LG->curr_model_size++;
            }
            LG->n_covariates = phenotype->n_covariates;
            gsl_vector_set_zero(LG->OMIT);
            LG->allocated = true;
            printf("-Done allocating logistic Regression scratch space ...\n");
            fflush(stdout);
       }
       else
       {       
            //should be allocated.
            if(LG->h==NULL) abort();
            if(LG->h_temp==NULL) abort(); 
            if(LG->h_inv==NULL) abort();
            if(LG->W==NULL) abort();
            if(LG->old_W==NULL) abort();
            if(LG->W_delta==NULL) abort(); 
            if(LG->g==NULL) abort(); 
            if(LG->OMIT==NULL) abort(); 

            int i = 0;
            for(i=0;i<MAX_MODEL_SZ;i++)
            {
               LG->se[i] = 0;
               LG->wald[i] = 0;
            }
            //Covariates : mimic snps.
            LG->curr_model_size = phenotype->n_covariates;
            gsl_vector_set_zero(LG->OMIT);
       }
}

void init_scratch_bf(BG_SCRATCH * BG, PHENOTYPE * phenotype)
{
       int MAX_MODEL_SZ = 1 + MAX_N_COVARIATES + 1;
       BG->phenotype = phenotype;
       if(!(BG->allocated))
       {
            printf("-Allocating Bayes Factor logistic Regression scratch space ...\n");
            if(BG->h==NULL) BG->h = gsl_matrix_alloc(MAX_MODEL_SZ,MAX_MODEL_SZ); else abort();//hessian
            if(BG->h_temp==NULL) BG->h_temp = gsl_matrix_alloc(MAX_MODEL_SZ,MAX_MODEL_SZ); else abort(); //hessian to hold the result for LU decomposition.
            if(BG->nu==NULL) BG->nu = gsl_vector_alloc(MAX_MODEL_SZ); else abort();//priors on the betas.
            if(BG->W==NULL) BG->W = gsl_vector_alloc(MAX_MODEL_SZ); else abort();//weight vector, updated in gradient descent iterations.
            if(BG->g==NULL) BG->g = gsl_vector_alloc(MAX_MODEL_SZ); else abort();//gradient vector.
    
            if(BG->h_inv==NULL) BG->h_inv = gsl_matrix_alloc(MAX_MODEL_SZ,MAX_MODEL_SZ); else abort(); //hessian inverse
            if(BG->old_W==NULL) BG->old_W = gsl_vector_alloc(MAX_MODEL_SZ); else abort(); //old weight vector, used in gradient descent iterations.
            if(BG->W_delta==NULL) BG->W_delta = gsl_vector_alloc(MAX_MODEL_SZ); else abort(); //delta weight vector used in gradient descent iterations.
 
            if(BG->T==NULL) BG->T = gsl_multimin_fdfminimizer_conjugate_fr;else abort();
            BG->s = NULL;
            if(BG->x==NULL) BG->x = gsl_vector_alloc(MAX_MODEL_SZ);else abort();
            if(BG->OMIT==NULL) BG->OMIT = gsl_vector_alloc(MAX_N_INDIV);else abort();
            int i = 0;
            for(i=0;i<MAX_MODEL_SZ;i++)
            {
              if(i<MAX_MODEL_SZ-1)
                 BG->curr_model[i] = NULL; //null pointers.
            }
            BG->curr_model_size = 0;
            //Covariates : mimic snps.
            for(i=0;i<phenotype->n_covariates;i++)
            {
               SNP* snp_temp = (SNP*)malloc(sizeof(SNP));
               snp_temp->geno = (double*)malloc(phenotype->N_sample*sizeof(double));
               sprintf(snp_temp->name,"cov%d",(i+1));
               int j = 0;
               for(j=0;j<phenotype->N_sample;j++)
                   snp_temp->geno[j] = gsl_vector_get(phenotype->covariates[i],j);
               BG->cov_snps[i] = snp_temp;
               BG->curr_model[i] = snp_temp;
               BG->curr_model_size++;
            }
            BG->n_covariates = phenotype->n_covariates;
            gsl_vector_set_zero(BG->OMIT);
            BG->allocated = true;
            printf("-Done allocating Bayes Factor logistic Regression scratch space ...\n");fflush(stdout);
       }
       else
       {
            if(BG->h==NULL) abort();
            if(BG->nu==NULL) abort();
            if(BG->W==NULL) abort();
            if(BG->g==NULL) abort();
            if(BG->T==NULL) abort();
            if(BG->x==NULL) abort();
            if(BG->OMIT==NULL) abort();
            if(BG->h_temp==NULL) abort(); 
            if(BG->h_inv==NULL) abort();
            if(BG->old_W==NULL) abort();
            if(BG->W_delta==NULL) abort(); 
          
            BG->curr_model_size = 0;
            gsl_vector_set_zero(BG->OMIT);
       }
}

void free_scratch(LOGISTIC_SCRATCH * LG)
{
            printf("Freeing logistic Regression scratch space ...\n");
            gsl_matrix_free(LG->h); LG->h = NULL;
            gsl_matrix_free(LG->h_temp); LG->h_temp = NULL;
            gsl_matrix_free(LG->h_inv);  LG->h_inv = NULL;
            gsl_vector_free(LG->W);  LG->W = NULL;
            gsl_vector_free(LG->old_W); LG->old_W = NULL;
            gsl_vector_free(LG->W_delta); LG->W_delta = NULL;
            gsl_vector_free(LG->g); LG->g = NULL;
            //gsl_vector_free(LG->x); LG->x = NULL;
            gsl_vector_free(LG->OMIT); LG->OMIT = NULL;

        //Covariates : mimic snps.
        int i = 0;
        for(i=0;i<LG->n_covariates;i++)
        {
            if(LG->cov_snps[i]->geno)
            {
               free(LG->cov_snps[i]->geno);
               LG->cov_snps[i]->geno = NULL;
            }
            if(LG->cov_snps[i])
            {
               free(LG->cov_snps[i]);
               LG->cov_snps[i]=NULL;
            }
        }
}

void free_scratch_bf(BG_SCRATCH * BG)
{
            printf("Freeing BG scratch space ...\n");
            gsl_matrix_free(BG->h); BG->h = NULL;
            gsl_matrix_free(BG->h_temp); BG->h_temp = NULL;
            gsl_vector_free(BG->nu); BG->nu = NULL;
            gsl_vector_free(BG->W);  BG->W = NULL;
            BG->T = NULL;
            gsl_vector_free(BG->g); BG->g = NULL;
            gsl_vector_free(BG->x); BG->x = NULL;
            gsl_vector_free(BG->OMIT); BG->OMIT = NULL;
            gsl_matrix_free(BG->h_inv);  BG->h_inv = NULL;
            gsl_vector_free(BG->old_W); BG->old_W = NULL;
            gsl_vector_free(BG->W_delta); BG->W_delta = NULL;

        //Covariates : mimic snps.
        int i = 0;
        for(i=0;i<BG->n_covariates;i++)
        {
            if(BG->cov_snps[i]->geno)
            {
               free(BG->cov_snps[i]->geno);
               BG->cov_snps[i]->geno = NULL;
            }
            if(BG->cov_snps[i])
            {
               free(BG->cov_snps[i]);
               BG->cov_snps[i]=NULL;
            }
        }
}

//perform permutations for all methods
void * runBIC_thread_wrap(void* ptr)
{
  BIC_THREAD_DATA * thread_param = (BIC_THREAD_DATA * ) ptr;
  int thid = thread_param->thread_id;
  if(VERBOSE)
     printf("\n\n*******STARTING PERMUTATIONS for %s *******thread id = %d ***\n",thread_param->gene->name,thid);

  BIC_STATE  bic_state_linear;
  BIC_STATE  bic_state_logistic;

  SNP * snp;
  int i;
  double bestPval,bestPval_bonf, bestFstat, bestFstat_bonf, bestSSM, currentSSM, bestChi2, bestChi2_bonf,bestWald;//FIX V.1.2 LATEST
  bool ifStop_BIC_linear, ifStop_SNP_linear, ifStop_SNP_bonf_linear; 
  bool ifStop_SNP_local, ifStop_SNP_bonf_local, ifStop_BF_linear, ifStop_VEGAS_linear;

  bool ifStop_BIC_logistic, ifStop_SNP_logistic, ifStop_SNP_bonf_logistic; 
  bool ifStop_SNP_local_logistic, ifStop_SNP_bonf_local_logistic, ifStop_BF_logistic, ifStop_VEGAS_logistic;

  double * z_dat = thread_param->z_dat;
  OrthNorm * Z = thread_param->Z;

  LOGISTIC_Z * LZ_perm = thread_param->LZ;

  gsl_matrix * L = thread_param->L; //for vegas simulations ? //V.1.4.mc
  
  //don't permute for GWiS if K  (iSNP) =1
  if(GET_GENE_BIC_PVAL_LINEAR && thread_param->gene->bic_state_linear.iSNP > 0)
    ifStop_BIC_linear = false;
  else
    ifStop_BIC_linear = true;

  if(GET_GENE_BIC_PVAL_LOGISTIC && thread_param->gene->bic_state_logistic.iSNP > 0)
     ifStop_BIC_logistic = false;
  else
     ifStop_BIC_logistic = true;

  if(GET_MINSNP_PVAL_LINEAR)
    ifStop_SNP_linear = false;
  else
    ifStop_SNP_linear = true;

  if(GET_MINSNP_PVAL_LOGISTIC)
    ifStop_SNP_logistic = false;
  else
    ifStop_SNP_logistic = true;

  if(GET_MINSNP_P_PVAL_LINEAR)
    ifStop_SNP_bonf_linear = false;
  else
    ifStop_SNP_bonf_linear = true;

  if(GET_MINSNP_P_PVAL_LOGISTIC)
    ifStop_SNP_bonf_logistic = false;
  else
    ifStop_SNP_bonf_logistic = true;

  if(GET_VEGAS_PVAL_LINEAR)
    ifStop_VEGAS_linear = false;
  else
    ifStop_VEGAS_linear = true;

  if(GET_VEGAS_PVAL_LOGISTIC)
    ifStop_VEGAS_logistic = false;
  else
    ifStop_VEGAS_logistic = true;

  if(GET_BF_PVAL_LINEAR)
    ifStop_BF_linear = false;
  else
    ifStop_BF_linear = true;

  if(GET_BF_PVAL_LOGISTIC)
    ifStop_BF_logistic = false;
  else
    ifStop_BF_logistic = true;

  bool NEED_LINEAR = !ifStop_BIC_linear || !ifStop_SNP_linear || !ifStop_SNP_bonf_linear || !ifStop_BF_linear || !ifStop_VEGAS_linear;
  bool NEED_LOGISTIC = !ifStop_BIC_logistic || !ifStop_SNP_logistic || !ifStop_SNP_bonf_logistic || !ifStop_BF_logistic || !ifStop_VEGAS_logistic;

  LOGISTIC_SCRATCH * LG = thread_param->LG;
  if(NEED_LOGISTIC && !SUMMARY)
  {
     init_scratch(LG, thread_param->phenotype); 
  }

  BG_SCRATCH * BG = thread_param->BG;
  if(!ifStop_BF_logistic && !SUMMARY)
  {
     init_scratch_bf(BG, thread_param->phenotype);
  }
  //  printf("-Time: %s", getTime());
  clock_t s_time;
  clock_t e_time;

  while( !ifStop_BIC_linear || !ifStop_SNP_linear || !ifStop_SNP_bonf_linear || !ifStop_BF_linear ||!ifStop_VEGAS_linear 
         || !ifStop_BIC_logistic || !ifStop_SNP_logistic || !ifStop_SNP_bonf_logistic || !ifStop_BF_logistic ||!ifStop_VEGAS_logistic)
  {
    if(VERBOSE_PERM_TIME)
      s_time = clock();
    //1. Shuffle the phenotype.
    if(!SUMMARY)
    {
      if(thread_param->phenotype->n_covariates>0)
      {
         //shuffle pheno_array_log
         if(NEED_LOGISTIC)
         {
            shuffle(thread_param->phenotype->pheno_array_log, thread_param->nsample, thread_param->r,thid);
         }
         if(NEED_LINEAR)
         {
            shuffle(thread_param->phenotype->pheno_array_org, thread_param->nsample, thread_param->r,thid);
            //regress the covriates on pheno_array_org and store the residuals in pheno_array_reg
            regCov(thread_param->phenotype); //should recompute mean and pheno_tss_per_n after regressing out the covariates.
            //covariates residuals should be in pheno_array_reg, pheno_array_org has the original shuffled phenotype.
         }
      }
      else //no covariates.
      {
        if(NEED_LINEAR)
           shuffle(thread_param->phenotype->pheno_array_reg, thread_param->nsample, thread_param->r, thid);
         if(NEED_LOGISTIC)
           shuffle(thread_param->phenotype->pheno_array_log, thread_param->nsample, thread_param->r, thid);
      }
      //so from here onwards, use pheno_array_reg for linear and pheno_array_log for logistic.
    }
    else
    { //SUMMARY
      int i, j;
      for (i=0; i<thread_param->gene->nSNP; i++)
	z_dat[i]=gsl_ran_ugaussian (thread_param->r);
      for (i=thread_param->gene->nSNP-1; i>=0; i--){
	double sum=0;
	for (j=0; j<=i; j++)
	  sum+= gsl_matrix_get(L, i, j)*z_dat[j]; //for vegas simulations ?
	z_dat[i]=sum;
      }
    }
  
    //2. Compute the single snp statistics and and store it in Z.
    //clock_t x1,y1;
    //x1 = clock();

    snp = (SNP*) cq_getItem(thread_param->gene->snp_start, thread_param ->snp_queue);
    for(i= thread_param->gene->snp_start; i<= thread_param->gene->snp_end; i++)
    {
       //Good for linear calculations.
       bool went_here = false;
       if(snp->iPerm_linear_sh == 0 || !ifStop_SNP_bonf_linear || !ifStop_BIC_linear || !ifStop_VEGAS_linear || !ifStop_BF_linear)
       { 
	  if(!SUMMARY){
	    Z[snp->gene_id].projP = covariance(snp->geno, thread_param->phenotype->pheno_array_reg->data, thread_param->nsample,true,false) *thread_param->nsample;
	  }else{
	    double z = z_dat[snp->gene_id];
	    Z[snp->gene_id].projP = z*sqrt(snp->geno_tss/thread_param->nsample* thread_param->pheno_tss_per_n)*thread_param->nsample/sqrt(z*z+thread_param->nsample-2);
            went_here = true;
	  }
	  Z[snp->gene_id].norm = snp->geno_tss;
	  Z[snp->gene_id].norm_original = snp->geno_tss;
	  Z[snp->gene_id].snp = snp;

       }
       //Good for logistic calculations.
       //Not needed for BF logistic.
       if(!ifStop_SNP_logistic || !ifStop_SNP_bonf_logistic || !ifStop_BIC_logistic || !ifStop_VEGAS_logistic)
       {
          if(!SUMMARY)
          {
            if(snp->iPerm_logistic_sh==0  || !ifStop_SNP_bonf_logistic || !ifStop_BIC_logistic || !ifStop_VEGAS_logistic)
            {
              //single snp hessian and wald are not needed for BIC logistic.
              bool need_hessian = !ifStop_SNP_logistic || !ifStop_SNP_bonf_logistic || !ifStop_VEGAS_logistic;
              LG->phenotype = thread_param->phenotype;
              LZ_perm[snp->gene_id].snp = snp;
              runLogisticSNP(snp, LG, true, &LZ_perm[snp->gene_id] , need_hessian ); //will update wald_perm and loglik_logistic_perm for this snp for this permutation. 
            }
          }
          else
          {
             //SUMMARY valid for only Vegas, minsnp and minsnp-p = same as linear part.
             if(!went_here)
             {
	       double z = z_dat[snp->gene_id];
	       Z[snp->gene_id].projP = z*sqrt(snp->geno_tss/thread_param->nsample* thread_param->pheno_tss_per_n)*thread_param->nsample/sqrt(z*z+thread_param->nsample-2);
	       Z[snp->gene_id].norm = snp->geno_tss;
	       Z[snp->gene_id].norm_original = snp->geno_tss;
	       Z[snp->gene_id].snp = snp;
             }
          }
       }
       if(!ifStop_BF_logistic && !SUMMARY)
       {
          if(USE_BG_FR)
          {
             //snp->BF_logistic_perm_sh = runBFLogisticSNP_fr(snp, BG, true);
             LZ_perm[snp->gene_id].snp = snp;
             LZ_perm[snp->gene_id].BF_logistic_perm = runBFLogisticSNP_fr(snp, BG, true);
          }
          else
          {
             //snp->BF_logistic_perm_sh = runBFLogisticSNP_newton(snp, BG, true);
             LZ_perm[snp->gene_id].snp = snp;
             LZ_perm[snp->gene_id].BF_logistic_perm = runBFLogisticSNP_newton(snp, BG, true);
          }
       }
       snp = cq_getNext(snp,thread_param ->snp_queue);
    }
    //y1 = clock();
    //printf("     loop took %g seconds\n",((double)(y1-x1))/CLOCKS_PER_SEC);

    //3. Count hits for each method.
    if(!ifStop_SNP_bonf_linear)
    {
      if(VERBOSE)
	printf("-running Min SNP P Linear permutations\n");
      ifStop_SNP_bonf_linear = true;
      snp = (SNP*) cq_getItem(thread_param->gene->snp_start, thread_param ->snp_queue);
      bestPval_bonf =2;
      bestFstat_bonf =-1;
      bestSSM = 0;
      //get the best SSM for bonf correction
      for(i= thread_param->gene->snp_start; i<= thread_param->gene->snp_end; i++)
      {
	// has to run before GWiS, otherwise projP and norm will be changed
	currentSSM = pow(Z[snp->gene_id].projP, 2)/Z[snp->gene_id].norm; 
	if(currentSSM > bestSSM)
        {
	  bestSSM = currentSSM;
	}
	snp = cq_getNext(snp,thread_param ->snp_queue);
      }

      SNP * bestSNP = NULL;
      snp = (SNP*) cq_getItem(thread_param->gene->snp_start, thread_param ->snp_queue);
      for(i= thread_param->gene->snp_start; i<= thread_param->gene->snp_end; i++)
      {
        if(n_threads>1)
           pthread_mutex_lock(&mutex_minsnp_p_linear); //<--lock mutex for minsnp p
	if(snp->iPerm_bonf_linear_sh == 0)
        {
	  if(pow(snp->sum_pheno_geno,2)/snp->geno_tss <= bestSSM )
	     thread_param->gene->hits_sh.hit_snp_bonf_linear[snp->gene_id]++;
      
	  ifStop_SNP_bonf_local = ifStop(thread_param->gene->hits_sh.hit_snp_bonf_linear[snp->gene_id], 
                                         thread_param->gene->hits_sh.perm_snp_bonf_linear[snp->gene_id]+1);
	  if(ifStop_SNP_bonf_local){
	    snp->iPerm_bonf_linear_sh =  thread_param->gene->hits_sh.perm_snp_bonf_linear[snp->gene_id]+1;
	    snp->nHit_bonf_linear_sh =  thread_param->gene->hits_sh.hit_snp_bonf_linear[snp->gene_id];
	  }
	  ifStop_SNP_bonf_linear=ifStop_SNP_bonf_linear && ifStop_SNP_bonf_local;
	}

	thread_param->gene->hits_sh.perm_snp_bonf_linear[snp->gene_id]++;

	if(snp->iPerm_bonf_linear_sh > 0 && ((bestPval_bonf > (double)snp->nHit_bonf_linear_sh/snp->iPerm_bonf_linear_sh)|| 
           (fabs(bestPval_bonf - (double)snp->nHit_bonf_linear_sh/snp->iPerm_bonf_linear_sh) < EPS && snp->f_stat > bestFstat_bonf )) )
        {
          bestSNP = snp; 
	  bestPval_bonf = (double)snp->nHit_bonf_linear_sh/snp->iPerm_bonf_linear_sh;
	  bestFstat_bonf = snp->f_stat;
	}
        if(n_threads>1)
           pthread_mutex_unlock(&mutex_minsnp_p_linear); //<--unlock mutex for minsnp p
	snp = cq_getNext(snp,thread_param ->snp_queue);
      }

      if(n_threads>1)
         pthread_mutex_lock(&mutex_minsnp_p_linear); //<--lock mutex for minsnp p
      thread_param->gene->maxBonf_SNP_linear_sh = bestSNP;
      if(n_threads>1)
         pthread_mutex_unlock(&mutex_minsnp_p_linear); //<--unlock mutex for minsnp p

      if(VERBOSE)
	printf("-----------------------end Min SNP P linear--------------------\n");
    }

    if(!ifStop_SNP_bonf_logistic)
    {
      if(VERBOSE)
	printf("-running Min SNP P Logistic\n");
      ifStop_SNP_bonf_logistic = true;
      snp = (SNP*) cq_getItem(thread_param->gene->snp_start, thread_param ->snp_queue);
      bestPval_bonf =2;
      bestChi2_bonf =-1;
      bestWald = 0; //FIX V.1.2 LATEST
      bestSSM = 0; //FIX V.1.2 LATEST
      bestFstat_bonf =-1; //FIX V.1.2 LATEST

      //get the best SSM for bonf correction
      for(i= thread_param->gene->snp_start; i<= thread_param->gene->snp_end; i++)
      {
	// has to run before GWiS, otherwise projP and norm will be changed
	if(!SUMMARY)
        {
	   double currentChi2 = LZ_perm[snp->gene_id].wald_perm; 
           if(gsl_isnan(currentChi2)==false)//FIX V.1.2 LATEST
           {
             if(currentChi2 > bestWald) //FIX V.1.2 LATEST
             {
               bestWald = currentChi2; //FIX V.1.2 LATEST
             }
           }
        }
        else //SUMMARY, same as linear part.
        {
	   currentSSM = pow(Z[snp->gene_id].projP, 2)/Z[snp->gene_id].norm; 
	   if(currentSSM > bestSSM)
           {
	     bestSSM = currentSSM;
	   }
        }
	snp = cq_getNext(snp,thread_param ->snp_queue);
      }

      SNP * bestSNP = NULL;
      snp = (SNP*) cq_getItem(thread_param->gene->snp_start, thread_param ->snp_queue);
      for(i= thread_param->gene->snp_start; i<= thread_param->gene->snp_end; i++)
      {
        if(!SUMMARY)
           if(isnan(snp->wald))
           {
              snp = cq_getNext(snp,thread_param ->snp_queue);
              continue; //ignore snp for permutations. FIX V.1.2 NAN SNP
           }

        if(n_threads>1)
           pthread_mutex_lock(&mutex_minsnp_p_logistic); //<--lock mutex for minsnp p

	if(snp->iPerm_bonf_logistic_sh == 0)
        {
          if(!SUMMARY)
          {
             if(snp->wald <= bestWald )//FIX V.1.2 LATEST
	        thread_param->gene->hits_sh.hit_snp_bonf_logistic[snp->gene_id]++;
          }
          else //SUMMARY, same as linear part.
          {
	     if(pow(snp->sum_pheno_geno,2)/snp->geno_tss <= bestSSM )
	        thread_param->gene->hits_sh.hit_snp_bonf_logistic[snp->gene_id]++;
          }

	  ifStop_SNP_bonf_local_logistic = ifStop(thread_param->gene->hits_sh.hit_snp_bonf_logistic[snp->gene_id], 
                                         thread_param->gene->hits_sh.perm_snp_bonf_logistic[snp->gene_id]+1);
	  if(ifStop_SNP_bonf_local_logistic){
	    snp->iPerm_bonf_logistic_sh =  thread_param->gene->hits_sh.perm_snp_bonf_logistic[snp->gene_id]+1;
	    snp->nHit_bonf_logistic_sh =  thread_param->gene->hits_sh.hit_snp_bonf_logistic[snp->gene_id];
	  }
	  ifStop_SNP_bonf_logistic = ifStop_SNP_bonf_logistic && ifStop_SNP_bonf_local_logistic;
          //if(!SUMMARY)
          //{
	    //if(VERBOSE)
	    //  if(!ifStop_SNP_bonf_local_logistic)
	    //    printf("%s, %g, %d\n", snp->name, snp->wald, thread_param->gene->hits.hit_snp_bonf_logistic[snp->gene_id]);
          //}
	}

	thread_param->gene->hits_sh.perm_snp_bonf_logistic[snp->gene_id]++;

        if(!SUMMARY)
        {
	   if(snp->iPerm_bonf_logistic_sh > 0 && ((bestPval_bonf > (double)snp->nHit_bonf_logistic_sh/snp->iPerm_bonf_logistic_sh)|| 
              (fabs(bestPval_bonf - (double)snp->nHit_bonf_logistic_sh/snp->iPerm_bonf_logistic_sh) < EPS && snp->wald > bestChi2_bonf )) )//FIX V.1.2 LATEST
           {
	     bestSNP = snp;
	     bestPval_bonf =  (double)snp->nHit_bonf_logistic_sh/snp->iPerm_bonf_logistic_sh;
             bestChi2_bonf = snp->wald;//FIX V.1.2 LATEST
	   }
        }
        else
        {
	   if(snp->iPerm_bonf_logistic_sh > 0 && ((bestPval_bonf > (double)snp->nHit_bonf_logistic_sh/snp->iPerm_bonf_logistic_sh)|| 
             (fabs(bestPval_bonf - (double)snp->nHit_bonf_logistic_sh/snp->iPerm_bonf_logistic_sh) < EPS && snp->f_stat > bestFstat_bonf )) )
           {
	     bestSNP = snp;
	     bestPval_bonf =  (double)snp->nHit_bonf_logistic_sh/snp->iPerm_bonf_logistic_sh;
	     bestFstat_bonf = snp->f_stat;
	   }
        }
        if(n_threads>1)
           pthread_mutex_unlock(&mutex_minsnp_p_logistic); //<--unlock mutex for minsnp p
	snp = cq_getNext(snp,thread_param ->snp_queue);
      }

      if(n_threads>1)
         pthread_mutex_lock(&mutex_minsnp_p_logistic); //<--lock mutex for minsnp p
      thread_param->gene->maxBonf_SNP_logistic_sh = bestSNP;
      if(n_threads>1)
         pthread_mutex_unlock(&mutex_minsnp_p_logistic); //<--unlock mutex for minsnp p

      if(VERBOSE)
	printf("-end Min SNP P logistic\n");
    }

    if(!ifStop_SNP_linear)
    {
      if(VERBOSE)
	printf("-running Min SNP Linear \n");
      ifStop_SNP_linear = true;
      snp = (SNP*) cq_getItem(thread_param->gene->snp_start, thread_param ->snp_queue);
      bestPval =2;
      bestFstat =-1;
      SNP * bestSNP = NULL;
      for(i= thread_param->gene->snp_start; i<= thread_param->gene->snp_end; i++)
      { 
         if(n_threads>1)
            pthread_mutex_lock(&mutex_minsnp_linear); //<--lock mutex for minsnp
	 if(snp->iPerm_linear_sh == 0)
         {
            //Pritam: increase hit count of those snps having fstat > original fstat.
	    if(fabs(snp->sum_pheno_geno) <= fabs(Z[snp->gene_id].projP) )
	       thread_param->gene->hits_sh.hit_snp_linear[snp->gene_id]++;
	    ifStop_SNP_local = ifStop(thread_param->gene->hits_sh.hit_snp_linear[snp->gene_id], 
                                thread_param->gene->hits_sh.perm_snp_linear[snp->gene_id]+1);

	    if(ifStop_SNP_local)
            {
	       snp->iPerm_linear_sh =  thread_param->gene->hits_sh.perm_snp_linear[snp->gene_id]+1;
	       snp->nHit_linear_sh =  thread_param->gene->hits_sh.hit_snp_linear[snp->gene_id];
	    }
	    ifStop_SNP_linear = ifStop_SNP_linear && ifStop_SNP_local;
	 }
	 thread_param->gene->hits_sh.perm_snp_linear[snp->gene_id]++;

	 if(snp->iPerm_linear_sh > 0 && ((bestPval > (double)snp->nHit_linear_sh/snp->iPerm_linear_sh)|| 
            (fabs(bestPval - (double)snp->nHit_linear_sh/snp->iPerm_linear_sh) < EPS &&  snp->f_stat > bestFstat )) )
         {
            bestSNP = snp;
	    bestPval =  (double)snp->nHit_linear_sh/snp->iPerm_linear_sh;
	    bestFstat = snp->f_stat;
	 }
         if(n_threads>1)
            pthread_mutex_unlock(&mutex_minsnp_linear); //<--unlock mutex for minsnp
	 snp = cq_getNext(snp,thread_param ->snp_queue);
      }

      if(n_threads>1)
         pthread_mutex_lock(&mutex_minsnp_linear); //<--lock mutex for minsnp
      thread_param->gene->maxSSM_SNP_linear_sh = bestSNP;
      if(n_threads>1)
         pthread_mutex_unlock(&mutex_minsnp_linear); //<--unlock mutex for minsnp

      if(VERBOSE)
	printf("-end min snp linear\n");
    }

    if(!ifStop_SNP_logistic)
    {
      if(VERBOSE)
	printf("-running Min SNP Logistic \n");
      ifStop_SNP_logistic = true;
      snp = (SNP*) cq_getItem(thread_param->gene->snp_start, thread_param ->snp_queue);
      bestPval =2;
      bestChi2 =-1;
      bestFstat =-1;

      SNP * bestSNP = NULL;
      for(i= thread_param->gene->snp_start; i<= thread_param->gene->snp_end; i++)
      {
        if(!SUMMARY)
           if(isnan(snp->wald))
           {
              snp = cq_getNext(snp,thread_param ->snp_queue);
              continue; //ignore snp for permutations. FIX V.1.2 NAN SNP
           }

        if(n_threads>1)
           pthread_mutex_lock(&mutex_minsnp_logistic); //<--lock mutex for minsnp

	if(snp->iPerm_logistic_sh == 0)
        {
          if(!SUMMARY)
          {
             if(gsl_isnan(LZ_perm[snp->gene_id].wald_perm)==true)//FIX V.1.2 LATEST
                thread_param->gene->hits_sh.hit_snp_logistic[snp->gene_id]++;//FIX V.1.2 LATEST
             else if(snp->wald <= LZ_perm[snp->gene_id].wald_perm)
                thread_param->gene->hits_sh.hit_snp_logistic[snp->gene_id]++;
          }
          else //SUMMARY, same as linear part.
          {
	     if(fabs(snp->sum_pheno_geno) <= fabs(Z[snp->gene_id].projP) )
	        thread_param->gene->hits_sh.hit_snp_logistic[snp->gene_id]++;
          }

	  ifStop_SNP_local_logistic = ifStop(thread_param->gene->hits_sh.hit_snp_logistic[snp->gene_id], 
                                    thread_param->gene->hits_sh.perm_snp_logistic[snp->gene_id]+1);

	  if(ifStop_SNP_local_logistic)
          {
	    snp->iPerm_logistic_sh =  thread_param->gene->hits_sh.perm_snp_logistic[snp->gene_id]+1;
	    snp->nHit_logistic_sh =  thread_param->gene->hits_sh.hit_snp_logistic[snp->gene_id];
	  }
	  ifStop_SNP_logistic = ifStop_SNP_logistic && ifStop_SNP_local_logistic;
	}
	thread_param->gene->hits_sh.perm_snp_logistic[snp->gene_id]++;

        if(!SUMMARY)
        {
	   if(snp->iPerm_logistic_sh > 0 && ((bestPval > (double)snp->nHit_logistic_sh/snp->iPerm_logistic_sh)|| 
              (fabs(bestPval - (double)snp->nHit_logistic_sh/snp->iPerm_logistic_sh) < EPS &&  snp->wald > bestChi2 )) )
           {
              bestSNP = snp;
	      bestPval =  (double)snp->nHit_logistic_sh/snp->iPerm_logistic_sh;
	      bestChi2 = snp->wald;
	   }
        }
        else //SUMMARY, same as linear part.
        {
	   if(snp->iPerm_logistic_sh > 0 && ((bestPval > (double)snp->nHit_logistic_sh/snp->iPerm_logistic_sh)|| 
              (fabs(bestPval - (double)snp->nHit_logistic_sh/snp->iPerm_logistic_sh) < EPS &&  snp->f_stat > bestFstat )) )
           {
             bestSNP = snp;
	     bestPval =  (double)snp->nHit_logistic_sh/snp->iPerm_logistic_sh;
	     bestFstat = snp->f_stat;
	   }
        }
        if(n_threads>1)
           pthread_mutex_unlock(&mutex_minsnp_logistic); //<--unlock mutex for minsnp
	snp = cq_getNext(snp,thread_param ->snp_queue);
      }

      if(n_threads>1)
         pthread_mutex_lock(&mutex_minsnp_logistic); //<--lock mutex for minsnp
      thread_param->gene->maxSSM_SNP_logistic_sh = bestSNP;
      if(n_threads>1)
         pthread_mutex_unlock(&mutex_minsnp_logistic); //<--unlock mutex for minsnp

      if(VERBOSE)
	printf("-end min snp logistic\n");
    }

    if(!ifStop_BF_linear)
    {
      if(VERBOSE)
	printf("-running Bayes Factors linear in thread %d\n",thid);
      double bf_sum=GSL_NEGINF;
      snp = (SNP*) cq_getItem(thread_param->gene->snp_start, thread_param ->snp_queue);
      for(i= thread_param->gene->snp_start; i<= thread_param->gene->snp_end; i++)
      {
	// has to run before GWiS, otherwise projP and norm will be changed
	double v = getlog10BF(thread_param->pheno_tss_per_n, Z[snp->gene_id].norm_original/thread_param->nsample,thread_param->nsample, Z[snp->gene_id].projP/thread_param->nsample);//FIX HESSIAN LATEST V.1.2
        bf_sum = safe_log10_sum(bf_sum,v);//FIX HESSIAN LATEST V.1.2
	snp = cq_getNext(snp,thread_param ->snp_queue);
      }
      if(bf_sum >= thread_param->gene->BF_sum_linear)
      {
         if(n_threads>1)
            pthread_mutex_lock(&mutex_bf_linear); //<--lock mutex for bf linear
	 thread_param->gene->hits_sh.hit_bf_linear++;
         //printf("         (Thread %d) incremented hit to %d\n",thread_param->thread_id,thread_param->gene->hits_sh.hit_bf_linear); 
         if(n_threads>1)
            pthread_mutex_unlock(&mutex_bf_linear); //<--unlock mutex for bf linear
      }
      ifStop_BF_linear = ifStop(thread_param->gene->hits_sh.hit_bf_linear, thread_param->gene->hits_sh.perm_bf_linear+1);

      if(n_threads>1)
         pthread_mutex_lock(&mutex_bf_linear); //<--lock mutex for bf linear
      thread_param->gene->hits_sh.perm_bf_linear++;
      if(n_threads>1)
         pthread_mutex_unlock(&mutex_bf_linear); //<--unlock mutex for bf linear
    }

    if(!ifStop_BF_logistic)
    {
      if(VERBOSE)
	printf("-running Bayes Factors logistic\n");
      double bf_sum=GSL_NEGINF;
      snp = (SNP*) cq_getItem(thread_param->gene->snp_start, thread_param ->snp_queue);
      for(i= thread_param->gene->snp_start; i<= thread_param->gene->snp_end; i++)
      {
	// has to run before GWiS, otherwise projP and norm will be changed
        if(isnan(LZ_perm[snp->gene_id].BF_logistic_perm)==false) //ignore snp for permutations. FIX V.1.2 NAN SNP
           bf_sum = safe_log10_sum(bf_sum,LZ_perm[snp->gene_id].BF_logistic_perm);//FIX HESSIAN LATEST V.1.2
	snp = cq_getNext(snp,thread_param->snp_queue);
      }
      if(VERBOSE)
         printf("-Gene-Bayes-Factor (perm) = %g, org = %g\n",bf_sum,thread_param->gene->BF_sum_logistic);
      if(bf_sum >= thread_param->gene->BF_sum_logistic)
      {
        if(n_threads>1)
           pthread_mutex_lock(&mutex_bf_logistic); //<--lock mutex for bf logistic
	thread_param->gene->hits_sh.hit_bf_logistic++;
        if(n_threads>1)
           pthread_mutex_unlock(&mutex_bf_logistic); //<--unlock mutex for bf logistic
      }
      ifStop_BF_logistic = ifStop(thread_param->gene->hits_sh.hit_bf_logistic, thread_param->gene->hits_sh.perm_bf_logistic+1);

      if(n_threads>1)
         pthread_mutex_lock(&mutex_bf_logistic); //<--lock mutex for bf logistic
      thread_param->gene->hits_sh.perm_bf_logistic++;
      if(n_threads>1)
         pthread_mutex_unlock(&mutex_bf_logistic); //<--unlock mutex for bf logistic
    }

    if(!ifStop_VEGAS_linear)
    {
      if(VERBOSE)
	printf("-running VEGAS linear\n");
      double vegas=0;
      snp = (SNP*) cq_getItem(thread_param->gene->snp_start, thread_param ->snp_queue);
      for(i= thread_param->gene->snp_start; i<= thread_param->gene->snp_end; i++)
      {
	// has to run before GWiS, otherwise projP and norm will be changed
	vegas += pow(Z[snp->gene_id].projP, 2)/Z[snp->gene_id].norm/thread_param->pheno_tss_per_n/thread_param->nsample*(thread_param->nsample-2);
	snp = cq_getNext(snp,thread_param ->snp_queue);
      }
      if(vegas >= thread_param->gene->vegas_linear)
      {
        if(n_threads>1)
           pthread_mutex_lock(&mutex_vegas_linear); //<--lock mutex for vegas linear
	thread_param->gene->hits_sh.hit_vegas_linear++;
        if(n_threads>1)
           pthread_mutex_unlock(&mutex_vegas_linear); //<--unlock mutex for vegas linear
      }
      ifStop_VEGAS_linear = ifStop(thread_param->gene->hits_sh.hit_vegas_linear, thread_param->gene->hits_sh.perm_vegas_linear+1);

      if(n_threads>1)
         pthread_mutex_lock(&mutex_vegas_linear); //<--lock mutex for vegas linear
      thread_param->gene->hits_sh.perm_vegas_linear++;
      if(n_threads>1)
         pthread_mutex_unlock(&mutex_vegas_linear); //<--unlock mutex for vegas linear
    }

    if(!ifStop_VEGAS_logistic)
    {
      if(VERBOSE)
	printf("-running VEGAS logistic\n");
      double vegas=0;
      snp = (SNP*) cq_getItem(thread_param->gene->snp_start, thread_param ->snp_queue);
      for(i= thread_param->gene->snp_start; i<= thread_param->gene->snp_end; i++)
      {
        if(SUMMARY)
        {
           //same as vegas linear regression 
	   // has to run before GWiS, otherwise projP and norm will be changed
	   vegas += pow(Z[snp->gene_id].projP, 2)/Z[snp->gene_id].norm/thread_param->pheno_tss_per_n/thread_param->nsample*(thread_param->nsample-2);
        }
        else
        {
           if(gsl_isnan(LZ_perm[snp->gene_id].wald_perm)==false)//FIX V.1.2 LATEST
	      vegas += LZ_perm[snp->gene_id].wald_perm;
        }
	snp = cq_getNext(snp,thread_param ->snp_queue);
      }
      //printf("Vegas logistic perm = %g\n",vegas);
      if(vegas >= thread_param->gene->vegas_logistic)
      {
        if(n_threads>1)
           pthread_mutex_lock(&mutex_vegas_logistic); //<--lock mutex for vegas logistic
	thread_param->gene->hits_sh.hit_vegas_logistic++;
        if(n_threads>1)
           pthread_mutex_unlock(&mutex_vegas_logistic); //<--unlock mutex for vegas logistic
      }
      ifStop_VEGAS_logistic = ifStop(thread_param->gene->hits_sh.hit_vegas_logistic, thread_param->gene->hits_sh.perm_vegas_logistic+1);

      if(n_threads>1)
         pthread_mutex_lock(&mutex_vegas_logistic); //<--lock mutex for vegas logistic
      thread_param->gene->hits_sh.perm_vegas_logistic++;
      if(n_threads>1)
         pthread_mutex_unlock(&mutex_vegas_logistic); //<--unlock mutex for vegas logistic
    }

    if(!ifStop_BIC_linear){
      if(VERBOSE)
	printf("-running BIC linear permutation\n");

	runBIC(thread_param->gene, Z, &bic_state_linear, thread_param->nsample, /*thread_param->pheno_mean,*/ thread_param->pheno_tss_per_n, NULL, NULL, true);//V.1.4.mc
		
      ifStop_BIC_linear = countBIChit_linear(thread_param->gene, thread_param->nsample, &bic_state_linear, NULL);
      if(VERBOSE)
	printf("-end BIC linear permutation\n");
    }

    if(!ifStop_BIC_logistic){
       if(VERBOSE)
          printf("-running BIC logistic permutation %d\n",perm_loop_counter);
       LG->curr_model_size = thread_param->phenotype->n_covariates; //Pritam, reset model size for each permutation.
       LG->phenotype = thread_param->phenotype;
       runLogistic(thread_param->snp_queue,
                   thread_param->gene,
                   &bic_state_logistic,
                   LG,
                   NULL,
                   LZ_perm,
                   true
                  );
       ifStop_BIC_logistic = countBIChit_logistic(thread_param->gene, thread_param->nsample, &bic_state_logistic);
       if(VERBOSE)
          printf("-end BIC Logistic permutation\n");
    }

    if(VERBOSE_PERM_TIME)
    {
       e_time = clock();
       printf("thr = %d ----  perm %d took %g seconds\n",thid,perm_loop_counter,((double)(e_time-s_time))/CLOCKS_PER_SEC);
    }

    if(n_threads > 1)
      pthread_mutex_lock(&perm_loop_counter_mutex);
    perm_loop_counter++;
    if(n_threads > 1)
      pthread_mutex_unlock(&perm_loop_counter_mutex);

  }//end while

  //  printf("-Time: %s", getTime());

  //printf("*****************Done with PERMUTATIONS for gene = %s\n\n",thread_param->gene->name);

  //The main thread (thread id = n-1) should not exit upon completion. It still has to return and write out the results.
  //All other threads can simply quit. Their job is to increment counts of hits, once that is done, nothing left to do.
  
  if(thid < n_threads - 1)
  {
     printf("-Thread %d is done and exiting\n",thid);
     pthread_mutex_lock(&nthr_completed_mutex);
     nthr_completed++;
     pthread_cond_signal(&nthr_completed_cond);
     pthread_mutex_unlock(&nthr_completed_mutex);
     pthread_exit(0);
  }
  printf("-Thread %d (main) is done and exiting\n",thid);
  //the main thread should return.
  return NULL;
}

//perform permutations for only vegas, minsnp, minsnp-gene for mode=SUMMARY
//to be called when gwis or bimbam are omitted because phenotype variance or sample size is not available.
void * runBIC_thread_simple_summary(void* ptr)
{
  BIC_THREAD_DATA * thread_param = (BIC_THREAD_DATA * ) ptr;
  int thid = thread_param->thread_id;
  printf("\n\n*****STARTING SIMPLE PERMUTATIONS (minsnp,minsnp-gene and vegas) for %s *******thread id = %d ***\n",thread_param->gene->name,thid);

  SNP * snp;
  int i;
  double bestPval,bestFstat,bestPval_bonf,bestFstat_bonf; 
  bool ifStop_SNP_linear, ifStop_SNP_bonf_linear; 
  bool ifStop_VEGAS_linear;
  bool ifStop_SNP_local, ifStop_SNP_bonf_local;

  double * z_dat = thread_param->z_dat;
  gsl_matrix * L = thread_param->L; //for vegas simulations ? //V.1.4.mc
  
  if(GET_MINSNP_PVAL_LINEAR)
    ifStop_SNP_linear = false;
  else
    ifStop_SNP_linear = true;

  if(GET_MINSNP_P_PVAL_LINEAR)
    ifStop_SNP_bonf_linear = false;
  else
    ifStop_SNP_bonf_linear = true;

  if(GET_VEGAS_PVAL_LINEAR)
    ifStop_VEGAS_linear = false;
  else
    ifStop_VEGAS_linear = true;

  while( !ifStop_SNP_linear || !ifStop_SNP_bonf_linear || !ifStop_VEGAS_linear)
  {
    if(!SUMMARY)
    {
       //do nothing
    }
    else
    { //SUMMARY
      int i, j;
      for (i=0; i<thread_param->gene->nSNP; i++)
	z_dat[i]=gsl_ran_ugaussian (thread_param->r);
      for (i=thread_param->gene->nSNP-1; i>=0; i--){
	double sum=0;
	for (j=0; j<=i; j++)
	  sum+= gsl_matrix_get(L, i, j)*z_dat[j]; //for vegas simulations ?
	z_dat[i]=sum;
      }
    }
  
    //3. Count hits for each method.
    if(!ifStop_SNP_bonf_linear)
    {
      if(VERBOSE)
	printf("-running Min SNP P Linear permutations\n");
      ifStop_SNP_bonf_linear = true;

      snp = (SNP*) cq_getItem(thread_param->gene->snp_start, thread_param ->snp_queue);
      bestPval_bonf =2;
      bestFstat_bonf =-1;
      double best_z2 = 0;
      for(i= thread_param->gene->snp_start; i<= thread_param->gene->snp_end; i++)
      {
        double chi2 = z_dat[snp->gene_id]*z_dat[snp->gene_id];
	if(chi2 > best_z2)
        {
	  best_z2 = chi2;
	}
	snp = cq_getNext(snp,thread_param ->snp_queue);
      }

      SNP * bestSNP = NULL;
      snp = (SNP*) cq_getItem(thread_param->gene->snp_start, thread_param ->snp_queue);
      for(i= thread_param->gene->snp_start; i<= thread_param->gene->snp_end; i++)
      {
        if(n_threads>1)
           pthread_mutex_lock(&mutex_minsnp_p_linear); //<--lock mutex for minsnp p
	if(snp->iPerm_bonf_linear_sh == 0)
        {
	  if(snp->f_stat <= best_z2 )
	     thread_param->gene->hits_sh.hit_snp_bonf_linear[snp->gene_id]++;
      
	  ifStop_SNP_bonf_local = ifStop(thread_param->gene->hits_sh.hit_snp_bonf_linear[snp->gene_id], 
                                         thread_param->gene->hits_sh.perm_snp_bonf_linear[snp->gene_id]+1);
	  if(ifStop_SNP_bonf_local){
	    snp->iPerm_bonf_linear_sh =  thread_param->gene->hits_sh.perm_snp_bonf_linear[snp->gene_id]+1;
	    snp->nHit_bonf_linear_sh =  thread_param->gene->hits_sh.hit_snp_bonf_linear[snp->gene_id];
	  }
	  ifStop_SNP_bonf_linear=ifStop_SNP_bonf_linear && ifStop_SNP_bonf_local;
	}

	thread_param->gene->hits_sh.perm_snp_bonf_linear[snp->gene_id]++;

	if(snp->iPerm_bonf_linear_sh > 0 && ((bestPval_bonf > (double)snp->nHit_bonf_linear_sh/snp->iPerm_bonf_linear_sh)|| 
           (fabs(bestPval_bonf - (double)snp->nHit_bonf_linear_sh/snp->iPerm_bonf_linear_sh) < EPS && snp->f_stat > bestFstat_bonf )) )
        {
          bestSNP = snp; 
	  bestPval_bonf = (double)snp->nHit_bonf_linear_sh/snp->iPerm_bonf_linear_sh;
	  bestFstat_bonf = snp->f_stat;
	}
        if(n_threads>1)
           pthread_mutex_unlock(&mutex_minsnp_p_linear); //<--unlock mutex for minsnp p
	snp = cq_getNext(snp,thread_param ->snp_queue);
      }

      if(n_threads>1)
         pthread_mutex_lock(&mutex_minsnp_p_linear); //<--lock mutex for minsnp p
      thread_param->gene->maxBonf_SNP_linear_sh = bestSNP;
      if(n_threads>1)
         pthread_mutex_unlock(&mutex_minsnp_p_linear); //<--unlock mutex for minsnp p

      if(VERBOSE)
	printf("-----------------------end Min SNP P linear--------------------\n");
    }

    if(!ifStop_SNP_linear)
    {
      if(VERBOSE)
	printf("-running Min SNP Linear \n");
      ifStop_SNP_linear = true;
      snp = (SNP*) cq_getItem(thread_param->gene->snp_start, thread_param ->snp_queue);
      bestPval =2;
      bestFstat =-1;
      SNP * bestSNP = NULL;
      for(i= thread_param->gene->snp_start; i<= thread_param->gene->snp_end; i++)
      { 
         if(n_threads>1)
            pthread_mutex_lock(&mutex_minsnp_linear); //<--lock mutex for minsnp
	 if(snp->iPerm_linear_sh == 0)
         {
            //Pritam: increase hit count of those snps having fstat > original fstat.
            if(snp->f_stat <= z_dat[snp->gene_id])
	       thread_param->gene->hits_sh.hit_snp_linear[snp->gene_id]++;
	    ifStop_SNP_local = ifStop(thread_param->gene->hits_sh.hit_snp_linear[snp->gene_id], 
                                thread_param->gene->hits_sh.perm_snp_linear[snp->gene_id]+1);

	    if(ifStop_SNP_local)
            {
	       snp->iPerm_linear_sh =  thread_param->gene->hits_sh.perm_snp_linear[snp->gene_id]+1;
	       snp->nHit_linear_sh =  thread_param->gene->hits_sh.hit_snp_linear[snp->gene_id];
	    }
	    ifStop_SNP_linear = ifStop_SNP_linear && ifStop_SNP_local;
	 }
	 thread_param->gene->hits_sh.perm_snp_linear[snp->gene_id]++;

	 if(snp->iPerm_linear_sh > 0 && ((bestPval > (double)snp->nHit_linear_sh/snp->iPerm_linear_sh)|| 
            (fabs(bestPval - (double)snp->nHit_linear_sh/snp->iPerm_linear_sh) < EPS &&  snp->f_stat > bestFstat )) )
         {
            bestSNP = snp;
	    bestPval =  (double)snp->nHit_linear_sh/snp->iPerm_linear_sh;
	    bestFstat = snp->f_stat;
	 }
         if(n_threads>1)
            pthread_mutex_unlock(&mutex_minsnp_linear); //<--unlock mutex for minsnp
	 snp = cq_getNext(snp,thread_param ->snp_queue);
      }

      if(n_threads>1)
         pthread_mutex_lock(&mutex_minsnp_linear); //<--lock mutex for minsnp
      thread_param->gene->maxSSM_SNP_linear_sh = bestSNP;
      if(n_threads>1)
         pthread_mutex_unlock(&mutex_minsnp_linear); //<--unlock mutex for minsnp

      if(VERBOSE)
	printf("-end min snp linear\n");
    }

    if(!ifStop_VEGAS_linear)
    {
      if(VERBOSE)
	printf("-running VEGAS linear with score = %lg\n",thread_param->gene->vegas_linear);
      double vegas=0;
      snp = (SNP*) cq_getItem(thread_param->gene->snp_start, thread_param ->snp_queue);
      for(i= thread_param->gene->snp_start; i<= thread_param->gene->snp_end; i++)
      {
	// has to run before GWiS, otherwise projP and norm will be changed
	double chi2 = z_dat[snp->gene_id]*z_dat[snp->gene_id];
	vegas += chi2; //pow(Z[snp->gene_id].projP, 2)/Z[snp->gene_id].norm/thread_param->pheno_tss_per_n/thread_param->nsample*(thread_param->nsample-2);
	snp = cq_getNext(snp,thread_param ->snp_queue);
      }
      //printf("Got sum = %lg\n",vegas);
      if(vegas >= thread_param->gene->vegas_linear)
      {
        if(n_threads>1)
           pthread_mutex_lock(&mutex_vegas_linear); //<--lock mutex for vegas linear
	thread_param->gene->hits_sh.hit_vegas_linear++;
        if(n_threads>1)
           pthread_mutex_unlock(&mutex_vegas_linear); //<--unlock mutex for vegas linear
      }
      ifStop_VEGAS_linear = ifStop(thread_param->gene->hits_sh.hit_vegas_linear, thread_param->gene->hits_sh.perm_vegas_linear+1);

      if(n_threads>1)
         pthread_mutex_lock(&mutex_vegas_linear); //<--lock mutex for vegas linear
      thread_param->gene->hits_sh.perm_vegas_linear++;
      if(n_threads>1)
         pthread_mutex_unlock(&mutex_vegas_linear); //<--unlock mutex for vegas linear
    }

    if(n_threads > 1)
      pthread_mutex_lock(&perm_loop_counter_mutex);
    perm_loop_counter++;
    if(n_threads > 1)
      pthread_mutex_unlock(&perm_loop_counter_mutex);

  }//end while

  //  printf("-Time: %s", getTime());

  //printf("*****************Done with PERMUTATIONS for gene = %s\n\n",thread_param->gene->name);

  //The main thread (thread id = n-1) should not exit upon completion. It still has to return and write out the results.
  //All other threads can simply quit. Their job is to increment counts of hits, once that is done, nothing left to do.
  
  if(thid < n_threads - 1)
  {
     printf("-Thread %d is done and exiting\n",thid);
     pthread_mutex_lock(&nthr_completed_mutex);
     nthr_completed++;
     pthread_cond_signal(&nthr_completed_cond);
     pthread_mutex_unlock(&nthr_completed_mutex);
     pthread_exit(0);
  }
  printf("-Thread %d (main) is done and exiting\n",thid);
  //the main thread should return.
  return NULL;
}

//initialize the permutation counters
void initHit(GENE* gene,  C_QUEUE * snp_queue){

  zeroArray_int(gene->hits_sh.k_pick_linear, MAX_INCLUDED_SNP+1);
  zeroArray_int(gene->hits_sh.k_hits_linear, MAX_INCLUDED_SNP+1);
  gene->hits_sh.maxK_linear = gene->bic_state_linear.iSNP;

  gene->hits_sh.hit_bf_linear = 0;
  gene->hits_sh.hit_bf_logistic = 0;
  gene->hits_sh.perm_bf_linear = 0;
  gene->hits_sh.perm_bf_logistic = 0;
  gene->hits_sh.hit_vegas_linear = 0;
  gene->hits_sh.hit_vegas_logistic = 0;
  gene->hits_sh.perm_vegas_linear = 0;
  gene->hits_sh.perm_vegas_logistic = 0;

  //++Pritam
  zeroArray_int(gene->hits_sh.k_pick_logistic, MAX_INCLUDED_SNP+1);
  zeroArray_int(gene->hits_sh.k_hits_logistic, MAX_INCLUDED_SNP+1);
  gene->hits_sh.maxK_logistic = gene->bic_state_logistic.iSNP;
  //--Pritam

  SNP* snp = cq_getItem( gene->snp_start, snp_queue);
  int i;
  for(i= gene->snp_start; i <= gene->snp_end; i++)
  {
    if(GET_MINSNP_PVAL_LINEAR)
    {
      gene->hits_sh.hit_snp_linear[snp->gene_id] =  0;
      gene->hits_sh.perm_snp_linear[snp->gene_id] =  0;
    }
    if(GET_MINSNP_PVAL_LOGISTIC)
    {
      gene->hits_sh.hit_snp_logistic[snp->gene_id] =  0;
      gene->hits_sh.perm_snp_logistic[snp->gene_id] =  0;
    }
    if(GET_MINSNP_P_PVAL_LINEAR)
    {
      gene->hits_sh.hit_snp_bonf_linear[snp->gene_id] =  0;
      gene->hits_sh.perm_snp_bonf_linear[snp->gene_id] =  0;
    }
    if(GET_MINSNP_P_PVAL_LOGISTIC)
    {
      gene->hits_sh.hit_snp_bonf_logistic[snp->gene_id] =  0;
      gene->hits_sh.perm_snp_bonf_logistic[snp->gene_id] =  0;
    }
    snp = cq_getNext(snp, snp_queue);
  }
}

bool compute_L(GENE * gene)
{
  //don't permute for GWiS if K  (iSNP) =1
  if(GET_GENE_BIC_PVAL_LINEAR && gene->bic_state_linear.iSNP > 0)
    return true;

  if(GET_GENE_BIC_PVAL_LOGISTIC && gene->bic_state_logistic.iSNP > 0)
    return true;

  if(GET_MINSNP_PVAL_LINEAR)
    return true;

  if(GET_MINSNP_PVAL_LOGISTIC)
    return true;

  if(GET_MINSNP_P_PVAL_LINEAR)
    return true;

  if(GET_MINSNP_P_PVAL_LOGISTIC)
    return true;

  if(GET_VEGAS_PVAL_LINEAR)
    return true;

  if(GET_VEGAS_PVAL_LOGISTIC)
    return true;

  if(GET_BF_PVAL_LINEAR)
    return true;

  if(GET_BF_PVAL_LOGISTIC)
    return true;

  return false;
}


//setup the scratch space for the methods
//get the model for the original trait
//if K>1, run permutations
void getPerm(C_QUEUE * snp_queue, 
             GENE* gene, 
             PHENOTYPE * phenotype, 
             gsl_rng *r,
             OUTFILE outfile,
             LOGISTIC_SCRATCH * LG
            )
{ 
  if(VERBOSE)
     printf("-getPerm called for gene %s\n",gene->name); 

  static PHENOTYPE ** pheno_perm_pth = NULL;
  static LOGISTIC_SCRATCH * LG_perm_pth = NULL;
  static BG_SCRATCH * BG_perm_pth = NULL;
  static BIC_THREAD_DATA * thread_param_pth = NULL;

  static double ** z_dat_perm_pth = NULL;
  static OrthNorm ** Z_perm_pth = NULL;

  static LOGISTIC_Z ** LZ_perm_pth = NULL;

  static int max_dim = MAX_SNP_PER_GENE;

  bool NEED_LOGISTIC_PERM = GET_MINSNP_PVAL_LOGISTIC || GET_MINSNP_P_PVAL_LOGISTIC || GET_VEGAS_PVAL_LOGISTIC || GET_BF_PVAL_LOGISTIC || GET_GENE_BIC_PVAL_LOGISTIC;

  perm_loop_counter = 0;
  nthr_completed = 0;

  //scratch space for logistic bic.
  if(GET_BF_PVAL_LOGISTIC)
  {
     if(BG_perm_pth==NULL)
     {
        int th = 0;
        BG_perm_pth = (BG_SCRATCH*)malloc(n_threads*sizeof(BG_SCRATCH));
        if(BG_perm_pth==NULL) {printf("-Failed to allocate memory for Bayes Factor scratch space\n");exit(1);} //V.1.7.mc
        for(th=0;th<n_threads;th++)
        {
           BG_perm_pth[th].allocated = false;
           init_one_time(NULL,&(BG_perm_pth[th])); //V.1.7.mc
           init_scratch_bf(&BG_perm_pth[th], phenotype);
        }
     }
  }
  //scratch space for logistic bic.
  if(NEED_LOGISTIC_PERM)
  {
     if(LG_perm_pth==NULL)
     {
        int th = 0;
        LG_perm_pth = (LOGISTIC_SCRATCH*)malloc(n_threads*sizeof(LOGISTIC_SCRATCH));
        if(LG_perm_pth==NULL) {printf("-Failed to allocate memory for Logistic Regression scratch space\n");exit(1);} //V.1.7.mc
        for(th=0;th<n_threads;th++)
        {
          LG_perm_pth[th].allocated = false;
          init_one_time(&(LG_perm_pth[th]),NULL); //V.1.7.mc
          init_scratch(&LG_perm_pth[th],phenotype);
        }
     }
  }
  //allocate Z and LZ
  if(Z_perm_pth==NULL)
  {
    int th = 0;
    Z_perm_pth = (OrthNorm**) malloc(sizeof(OrthNorm*)*n_threads);
    if(Z_perm_pth==NULL) {printf("-Failed to allocate memory for regression scratch space\n");exit(1);} //V.1.7.mc
    if(NEED_LOGISTIC_PERM) LZ_perm_pth = (LOGISTIC_Z**) malloc(sizeof(LOGISTIC_Z*)*n_threads);
    if(SUMMARY) z_dat_perm_pth = (double**)malloc(sizeof(double*)*n_threads); 
    for(th=0;th<n_threads;th++)
    {
       Z_perm_pth[th] = (OrthNorm*) malloc(sizeof(OrthNorm)*MAX_SNP_PER_GENE);
       if(SUMMARY) z_dat_perm_pth[th] = malloc(sizeof(double)*MAX_SNP_PER_GENE);
       if(NEED_LOGISTIC_PERM) LZ_perm_pth[th] = (LOGISTIC_Z*) malloc(sizeof(LOGISTIC_Z)*MAX_SNP_PER_GENE);   
    }
  }
  if(gene->nSNP > max_dim)
  { 
    int th = 0;
    max_dim = gene->nSNP;
    if(Z_perm_pth!=NULL)
    {
       for(th=0;th<n_threads;th++)
       {
          free(Z_perm_pth[th]);
          if(SUMMARY) free(z_dat_perm_pth[th]);
          if(NEED_LOGISTIC_PERM) free(LZ_perm_pth[th]);
       }
    }
    for(th=0;th<n_threads;th++)
    {
       Z_perm_pth[th] = (OrthNorm*) malloc(sizeof(OrthNorm)*max_dim);
       if(SUMMARY)
          z_dat_perm_pth[th] = malloc(sizeof(double)*max_dim);
       if(NEED_LOGISTIC_PERM)
          LZ_perm_pth[th] = (LOGISTIC_Z*) malloc(sizeof(LOGISTIC_Z)*max_dim);
    }
  }

  if(pheno_perm_pth == NULL)
  {
    int th = 0;
    pheno_perm_pth = (PHENOTYPE**)malloc(n_threads*sizeof(PHENOTYPE*));
    if(pheno_perm_pth==NULL) {printf("-Failed to allocate memory for phenotype permutation scratch space\n");exit(1);} //V.1.7.mc
   
    for(th=0;th<n_threads;th++)
    {  
       pheno_perm_pth[th] = (PHENOTYPE*)malloc(sizeof(PHENOTYPE)); 
       pheno_perm_pth[th]->pheno_array_org = NULL;
       pheno_perm_pth[th]->pheno_array_reg = NULL;
       pheno_perm_pth[th]->pheno_array_log = NULL;
       int k = 0;
       for(k=0;k<MAX_N_COVARIATES;k++)
       {
         pheno_perm_pth[th]->covariates[k] = NULL;
       }
    }
    thread_param_pth = (BIC_THREAD_DATA*)malloc(n_threads*sizeof(BIC_THREAD_DATA));
  }

  max_dim=MAX_SNP_PER_GENE;
  static OrthNorm *Z = NULL;

  //min snp linear
  static int * hit_snp_linear = NULL;
  static int * perm_snp_linear = NULL;

  //min snp logistic
  static int * hit_snp_logistic = NULL;
  static int * perm_snp_logistic = NULL;

  //min snp p linear
  static int * hit_snp_bonf_linear = NULL;
  static int * perm_snp_bonf_linear = NULL;

  //min snp p logistic
  static int * hit_snp_bonf_logistic = NULL;
  static int* perm_snp_bonf_logistic = NULL;

  static int nHit, nPerm, i;
  static SNP* snp;

  if(Z == NULL)
  {
     if(GET_GENE_BIC_LINEAR || GET_GENE_BIC_PVAL_LINEAR)
     {
        Z = malloc(sizeof(OrthNorm)*MAX_SNP_PER_GENE);
        if(Z==NULL) {printf("-Failed to allocate memory for BIC model search\n");exit(1);} //V.1.7.mc
     }
     if(GET_MINSNP_PVAL_LINEAR)
     {
        if(hit_snp_linear==NULL) hit_snp_linear = malloc(sizeof(int)*MAX_SNP_PER_GENE);
        if(perm_snp_linear==NULL) perm_snp_linear = malloc(sizeof(int)*MAX_SNP_PER_GENE);
     }
     if(GET_MINSNP_PVAL_LOGISTIC)
     {
        if(hit_snp_logistic==NULL) hit_snp_logistic = malloc(sizeof(int)*MAX_SNP_PER_GENE);
        if(perm_snp_logistic==NULL) perm_snp_logistic = malloc(sizeof(int)*MAX_SNP_PER_GENE);
     }
     if(GET_MINSNP_P_PVAL_LINEAR)
     {
        if(hit_snp_bonf_linear==NULL) hit_snp_bonf_linear = malloc(sizeof(int)*MAX_SNP_PER_GENE);
        if(perm_snp_bonf_linear==NULL) perm_snp_bonf_linear = malloc(sizeof(int)*MAX_SNP_PER_GENE);
     }
     if(GET_MINSNP_P_PVAL_LOGISTIC)
     {
        if(hit_snp_bonf_logistic==NULL) hit_snp_bonf_logistic = malloc(sizeof(int)*MAX_SNP_PER_GENE);
        if(perm_snp_bonf_logistic==NULL) perm_snp_bonf_logistic = malloc(sizeof(int)*MAX_SNP_PER_GENE);
     }
  }
  if(gene->nSNP > max_dim)
  {
     max_dim = gene->nSNP;

     if(Z!=NULL) {free(Z);Z=NULL;}

     if(hit_snp_linear!=NULL) {free(hit_snp_linear);hit_snp_linear=NULL;}
     if(perm_snp_linear!=NULL) {free(perm_snp_linear);perm_snp_linear=NULL;}

     if(hit_snp_logistic!=NULL) {free(hit_snp_logistic);hit_snp_logistic=NULL;}
     if(perm_snp_logistic!=NULL) {free(perm_snp_logistic);perm_snp_logistic=NULL;}

     if(hit_snp_bonf_linear!=NULL) {free(hit_snp_bonf_linear);hit_snp_bonf_linear=NULL;}
     if(perm_snp_bonf_linear!=NULL) {free(perm_snp_bonf_linear);perm_snp_bonf_linear=NULL;}

     if(hit_snp_bonf_logistic!=NULL) {free(hit_snp_bonf_logistic);hit_snp_bonf_logistic=NULL;}
     if(perm_snp_bonf_logistic!=NULL) {free(perm_snp_bonf_logistic);perm_snp_bonf_logistic=NULL;}

     if(GET_GENE_BIC_LINEAR || GET_GENE_BIC_PVAL_LINEAR)
     {
        Z = malloc(sizeof(OrthNorm)*max_dim);
        if(Z==NULL) {printf("-Failed to allocate memory for BIC model search\n");exit(1);} //V.1.7.mc
     }
     if(GET_MINSNP_PVAL_LINEAR)
     {  
        hit_snp_linear = malloc(sizeof(int)*max_dim);
        perm_snp_linear = malloc(sizeof(int)*max_dim);
     }
     if(GET_MINSNP_PVAL_LOGISTIC)
     {  
        hit_snp_logistic = malloc(sizeof(int)*max_dim);
        perm_snp_logistic = malloc(sizeof(int)*max_dim);
     }
     if(GET_MINSNP_P_PVAL_LINEAR)
     {  
        hit_snp_bonf_linear = malloc(sizeof(int)*max_dim);
        perm_snp_bonf_linear = malloc(sizeof(int)*max_dim);
     }
     if(GET_MINSNP_P_PVAL_LOGISTIC)
     {  
        hit_snp_bonf_logistic = malloc(sizeof(int)*max_dim);
        perm_snp_bonf_logistic = malloc(sizeof(int)*max_dim);
     }
  }
 
  //initialized the orthnom vectors, equavalent to projecting all the snps to zo (vector of 1's)
 
  if(GET_BF_LINEAR || GET_BF_PVAL_LINEAR)
    gene->BF_sum_linear=GSL_NEGINF;

  if(GET_BF_LOGISTIC || GET_BF_PVAL_LOGISTIC)
    gene->BF_sum_logistic=GSL_NEGINF;
  
  if(GET_VEGAS_LINEAR || GET_VEGAS_PVAL_LINEAR)
    gene->vegas_linear=0;

  if(GET_VEGAS_LOGISTIC || GET_VEGAS_PVAL_LOGISTIC)
    gene->vegas_logistic=0;
  
  snp = cq_getItem(gene->snp_start, snp_queue);
  for(i= gene->snp_start; i <= gene->snp_end; i++)
  {
      if(GET_GENE_BIC_LINEAR || GET_GENE_BIC_PVAL_LINEAR)
      {
         Z[snp->gene_id].projP= snp->sum_pheno_geno;
         Z[snp->gene_id].norm = snp->geno_tss;
         Z[snp->gene_id].norm_original = snp->geno_tss;
         Z[snp->gene_id].snp = snp;

         //printf("PPPPPPPPPPPPP : %s %lg %lg\n",snp->name,snp->sum_pheno_geno,snp->geno_tss); REMOVE
      }

      if(GET_BF_LINEAR || GET_BF_PVAL_LINEAR)
      {
        gene->BF_sum_linear = safe_log10_sum(gene->BF_sum_linear,snp->BF_linear);//FIX HESSIAN LATEST V.1.2
      }

      if(GET_BF_LOGISTIC || GET_BF_PVAL_LOGISTIC)
      {
        if(gsl_isnan(snp->BF_logistic)==false)//FIX V.1.2 LATEST
           gene->BF_sum_logistic = safe_log10_sum(gene->BF_sum_logistic,snp->BF_logistic);//FIX HESSIAN LATEST V.1.2
      }
      
      if(GET_VEGAS_LINEAR || GET_VEGAS_PVAL_LINEAR)
      {
        gene->vegas_linear += snp->f_stat;
      }

      if(GET_VEGAS_LOGISTIC || GET_VEGAS_PVAL_LOGISTIC)
      {
        if(gsl_isnan(snp->wald)==false)//FIX V.1.2 LATEST
           gene->vegas_logistic += snp->wald;
      }

      snp = cq_getNext(snp, snp_queue);
   }
  
   if(VERBOSE)
   {
     printf("-Computed BF_sum_logistic = %g\n",gene->BF_sum_logistic);
     printf("-Computed BF_sum_linear = %g\n",gene->BF_sum_linear);
     printf("-getPerm : Vegas Linear = %g\n",gene->vegas_linear);
     printf("-getPerm : Vegas Logistic = %g\n",gene->vegas_logistic);
   }

   //++Pritam 
   if(GET_GENE_BIC_LOGISTIC || GET_GENE_BIC_PVAL_LOGISTIC)
   {
       init_scratch(LG, phenotype);

        printf("-Starting Logistic Regression based model search for real trait...\n");
        LG->phenotype = phenotype;
        if(GET_GENE_BIC_PVAL_LOGISTIC)
           runLogistic(snp_queue, 
                    gene, 
                    &gene->bic_state_logistic, 
                    LG, 
                    NULL,
                    NULL,
                    false
                    );
        else 
           runLogistic(snp_queue, 
                    gene, 
                    &gene->bic_state_logistic, 
                    LG, 
                    outfile.fp_bic_logistic_perm_result,
                    NULL,
                    false
                    );
        printf("-Done Logistic Regression based model search for real trait\n");
   }

  double pheno_tss_per_n = phenotype->tss_per_n; //V.1.4.mc

  if (GET_GENE_BIC_PVAL_LINEAR)
    runBIC(gene, Z, &gene->bic_state_linear, phenotype->N_sample, /*phenotype->mean,*/ pheno_tss_per_n, NULL, NULL, false); //V.1.4.mc
  else if (GET_GENE_BIC_LINEAR)
    runBIC(gene, Z, &gene->bic_state_linear, phenotype->N_sample, /*phenotype->mean,*/ pheno_tss_per_n, outfile.fp_bic_linear_perm_result, NULL, false); //V.1.4.mc

  if(VERBOSE)
    printf("%s: start=%d, end=%d, eSNP=%g, iSNP=%d\n", gene->ccds, gene->bp_start, gene->bp_end, gene->eSNP, gene->bic_state_linear.iSNP);

  //Want permutations?
  if(!SKIPPERM)
  {
    clock_t st = clock();
    printf ("-Permutations will be performed for gene %s\n",gene->name);fflush(stdout);

    //+V.1.4.mc
    gsl_matrix * L = NULL;
    if(SUMMARY)
    {
       if(compute_L(gene))
       {
          printf("-Computing L\n");
          L = getL_from_LDLt(gene->LD,gene->nSNP); //for vegas simulations ?
       }
    }
    //-V.1.4.mc

    pthread_t thread[n_threads];

    //V.1.7.mc
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

    bool simple = false;
    if(SUMMARY && SIMPLE_SUMMARY)
    {
       //simple = (phenotype->N_sample <= 0) || (phenotype->tss_per_n <= 0);
       simple = (phenotype->N_sample <= 0); //gwis,bimbam won't run
    }

    int th = 0;
    for(th=0;th<n_threads;th++)
    {
       if(!SUMMARY)
       {
         if(pheno_perm_pth[th]->pheno_array_org==NULL)
            pheno_perm_pth[th]->pheno_array_org = gsl_vector_alloc(phenotype->N_sample);
         if(pheno_perm_pth[th]->pheno_array_reg==NULL)
            pheno_perm_pth[th]->pheno_array_reg = gsl_vector_alloc(phenotype->N_sample);
         if(pheno_perm_pth[th]->pheno_array_log==NULL)
            pheno_perm_pth[th]->pheno_array_log = gsl_vector_alloc(phenotype->N_sample);

         for(i=0;i<phenotype->n_covariates;i++)
         {
           if(pheno_perm_pth[th]->covariates[i]==NULL)
              pheno_perm_pth[th]->covariates[i] = gsl_vector_alloc(phenotype->N_sample);
         }

         gsl_vector_memcpy(pheno_perm_pth[th]->pheno_array_org, phenotype->pheno_array_org); //copy of original phenotype
         gsl_vector_memcpy(pheno_perm_pth[th]->pheno_array_reg, phenotype->pheno_array_org); //copy of residuals from covariate regression.
         gsl_vector_memcpy(pheno_perm_pth[th]->pheno_array_log, phenotype->pheno_array_log); //copy of original phenotype after scaling.

         //pheno_perm_pth[th]->mean = phenotype->mean;//V.1.4.mc
         pheno_perm_pth[th]->tss_per_n = phenotype->tss_per_n;
         pheno_perm_pth[th]->N_na = phenotype->N_na;
         pheno_perm_pth[th]->N_indiv = phenotype->N_indiv;
         pheno_perm_pth[th]->N_sample = phenotype->N_sample;
         memcpy(pheno_perm_pth[th]->NA,phenotype->NA,MAX_N_INDIV*sizeof(bool)); //won't be needed during permutations.
         for(i=0;i<phenotype->n_covariates;i++)
         {
             gsl_vector_memcpy(pheno_perm_pth[th]->covariates[i],phenotype->covariates[i]);
         }
         pheno_perm_pth[th]->n_covariates = phenotype->n_covariates;
      }

      //initialize each gene
      if(GET_MINSNP_PVAL_LINEAR)
      {
        gene->hits_sh.hit_snp_linear = hit_snp_linear;
        gene->hits_sh.perm_snp_linear = perm_snp_linear;
      }
      if(GET_MINSNP_PVAL_LOGISTIC)
      {
        gene->hits_sh.hit_snp_logistic = hit_snp_logistic;
        gene->hits_sh.perm_snp_logistic = perm_snp_logistic;
      }
    
      if(GET_MINSNP_P_PVAL_LINEAR)
      {
        gene->hits_sh.hit_snp_bonf_linear = hit_snp_bonf_linear;
        gene->hits_sh.perm_snp_bonf_linear = perm_snp_bonf_linear;
      }

      if(GET_MINSNP_P_PVAL_LOGISTIC)
      {
        gene->hits_sh.hit_snp_bonf_logistic = hit_snp_bonf_logistic;
        gene->hits_sh.perm_snp_bonf_logistic = perm_snp_bonf_logistic;
      }
      initHit(gene, snp_queue);

      if(!SUMMARY)
        thread_param_pth[th].phenotype = pheno_perm_pth[th]; //Pritam : to regress our covariates during permutations.
      else
        thread_param_pth[th].phenotype = NULL; //Pritam
     
      thread_param_pth[th].thread_id = th; 
      //thread_param_pth[th].pheno_mean = phenotype->mean;//V.1.4.mc
      thread_param_pth[th].pheno_tss_per_n = pheno_tss_per_n;//V.1.4.mc
      thread_param_pth[th].nsample = phenotype->N_sample;
      thread_param_pth[th].r = r;
      thread_param_pth[th].gene = gene;
      thread_param_pth[th].snp_queue = snp_queue;
      if(SUMMARY)
        thread_param_pth[th].z_dat = z_dat_perm_pth[th];
      else
        thread_param_pth[th].z_dat = NULL;
      thread_param_pth[th].Z = Z_perm_pth[th];
      thread_param_pth[th].LG = &LG_perm_pth[th];
      if(GET_BF_PVAL_LOGISTIC)
         thread_param_pth[th].BG = &BG_perm_pth[th];
      else
         thread_param_pth[th].BG = NULL;
      if(NEED_LOGISTIC_PERM)
         thread_param_pth[th].LZ = LZ_perm_pth[th];
      else
         thread_param_pth[th].LZ = NULL;

      if(SUMMARY) //V.1.4.mc
         thread_param_pth[th].L = L;
      else
         thread_param_pth[th].L = NULL;

      printf("-Start to run permutations for gene %s\n",gene->name);fflush(stdout);
      //comment out this line if you don't want permutations.
      if(n_threads>1)
      {
        if(th < n_threads-1) //do not spawn thread n-1 which is the main thread.
        {
           //these threads should run indefinitely till enough permutations are done.
           int rc = -1;
           if(!simple)
              rc = pthread_create(&thread[th], &attr, runBIC_thread_wrap, &thread_param_pth[th]);
           else
              rc = pthread_create(&thread[th], &attr, runBIC_thread_simple_summary, &thread_param_pth[th]);
           printf("Created thread %d for permutations\n", thread_param_pth[th].thread_id);
           if (rc) 
           {
              printf("ERROR: return code from pthread_create() is %d\n", rc);
              exit(-1);
           }
        }
        else //main thread will run one set of permutations itself.
        {
          if(!simple)
             runBIC_thread_wrap(&thread_param_pth[th]);
          else
             runBIC_thread_simple_summary(&thread_param_pth[th]);
        }
      }
      else
      {
        if(!simple)
           runBIC_thread_wrap(&thread_param_pth[th]);
        else
           runBIC_thread_simple_summary(&thread_param_pth[th]);
      }
    }
    
    if(n_threads > 1)
    {
      //printf("-Main Thread waiting for other threads to complete, num completed = %d\n",nthr_completed);
      pthread_mutex_lock(&nthr_completed_mutex);
      nthr_completed++;
      while(nthr_completed < n_threads)
      {
        pthread_cond_wait(&nthr_completed_cond,&nthr_completed_mutex);
        printf("-Number of threads completed = %d\n",nthr_completed);
      }
      pthread_mutex_unlock(&nthr_completed_mutex);
      //printf("-Main Thread : Permutations finished for gene %s \n",gene->name);
    }
    clock_t ed = clock();
    printf("-Permutations finished for gene %s and took %g seconds\n",gene->name,((double)(ed-st))/CLOCKS_PER_SEC);
  }//end if(!SKIPPERM)

  if(GET_GENE_BIC_PVAL_LINEAR)
  {
      if(VERBOSE)
	printf("-starting to print linear bic perm results\n");

      nHit=1; //FIX S+1/Q+1
      nPerm=1;
      for(i=0; i<=gene->hits_sh.maxK_linear; i++)
      {
	   printBICPermResult(outfile.fp_bic_linear_perm_result, i, phenotype->N_sample, gene, Z, gene->hits_sh.k_pick_linear[i], gene->hits_sh.k_hits_linear[i], -1);
	   nHit += gene->hits_sh.k_hits_linear[i];
	   nPerm += gene->hits_sh.k_pick_linear[i];
      }
           //printf("%d %d\n",nHit,nPerm);fflush(stdout);
      printBICPermResult(outfile.fp_bic_linear_perm_result, gene->bic_state_linear.iSNP,  phenotype->N_sample, gene, Z, nPerm, nHit, nPerm==0 ? 1 : (((double)nHit) / nPerm));
	
      if(VERBOSE)
	printf("printing linear bic perm finished\n");
  }

  //++Pritam
  //for logistic regression.
  if(GET_GENE_BIC_PVAL_LOGISTIC)
  {
       nHit=1; //FIX S+1/Q+1
       nPerm=1;
       for(i=0; i<=gene->hits_sh.maxK_logistic; i++)
       {
          printBIC_Logistic_PermResult(outfile.fp_bic_logistic_perm_result, i, phenotype->N_sample, gene, gene->hits_sh.k_pick_logistic[i], gene->hits_sh.k_hits_logistic[i], -1);
          nHit += gene->hits_sh.k_hits_logistic[i];
          nPerm += gene->hits_sh.k_pick_logistic[i];
       }
       printBIC_Logistic_PermResult(outfile.fp_bic_logistic_perm_result, gene->bic_state_logistic.iSNP,  phenotype->N_sample, gene, nPerm, nHit, nPerm==0 ? 1 : (((double)nHit) / nPerm));

       if(VERBOSE)
           printf("printing logistic bic perm finished\n");
  }

  //for minSNP linear
  if(GET_MINSNP_PVAL_LINEAR)
  {
      if(VERBOSE)
        printf("reporting linear minsnp pval from permutations\n");
      snp = cq_getItem(gene->snp_start, snp_queue);
      for(i= gene->snp_start; i <= gene->snp_end; i++)
      {
	if(outfile.fp_snp_pval_linear != NULL)
        {
	  fprintf(outfile.fp_snp_pval_linear, "%d\t%s\t%s\t%d\t%d\t%d\t%d\t%g\t", gene->chr, gene->ccds, gene->name, gene->bp_start, 
                                                                                  gene->bp_end, gene->bp_end - gene->bp_start+1, 
                                                                                  gene->nSNP, gene->eSNP);
          //FIX V.1.2 PERM NAN
	  fprintf(outfile.fp_snp_pval_linear, "%s\t%d\t%g\t%g\t%g\t%d\t%d\t%g\t%d\n", snp->name,snp->bp, snp->MAF, snp->R2, snp->f_stat, 
                                                                                      snp->iPerm_linear_sh, snp->nHit_linear_sh, 
                                                                 snp->iPerm_linear_sh==0?1:((double)snp->nHit_linear_sh+1)/(snp->iPerm_linear_sh+1), 
                                                                                      gene->maxSSM_SNP_linear_sh == snp? 1: 0); //FIX S+1/Q+1
	}
	snp = cq_getNext(snp, snp_queue);
      }
      if(VERBOSE)
	printf("printing minSNP linear perm finished\n");
  }
  else if(GET_MINSNP_LINEAR) //no permutations.
  {
       if(VERBOSE)
          printf("Reporting linear minsnp parametric pval\n");
       SNP* bestSnp = NULL;
       double bestF = -1;
       snp = cq_getItem(gene->snp_start, snp_queue);
       for(i= gene->snp_start; i <= gene->snp_end; i++)
       {
          if(snp->f_stat > bestF)
          {
             bestSnp = snp;
             bestF = snp->f_stat;
          } 
	  snp = cq_getNext(snp, snp_queue);
       }

       snp = cq_getItem(gene->snp_start, snp_queue);
       for(i= gene->snp_start; i <= gene->snp_end; i++)
       {
	 if(outfile.fp_snp_pval_linear != NULL)
         {
	    fprintf(outfile.fp_snp_pval_linear, "%d\t%s\t%s\t%d\t%d\t%d\t%d\t%g\t", gene->chr, gene->ccds, gene->name, gene->bp_start, gene->bp_end, gene->bp_end - gene->bp_start+1, gene->nSNP, gene->eSNP);
	    fprintf(outfile.fp_snp_pval_linear, "%s\t%d\t%g\t%g\t%g\t-\t-\t%g\t%d\n", snp->name,snp->bp, snp->MAF, snp->R2, snp->f_stat, snp->pval_linear, snp==bestSnp? 1: 0);
	 }
	 snp = cq_getNext(snp, snp_queue);
       }
  }

  //for minSNP logistic
  if(GET_MINSNP_PVAL_LOGISTIC)
  {
      if(VERBOSE)
         printf("Reporting logistic minsnp pval from permutations\n");
      snp = cq_getItem(gene->snp_start, snp_queue);
      for(i= gene->snp_start; i <= gene->snp_end; i++)
      {
	if(outfile.fp_snp_pval_logistic != NULL)
        {
	  fprintf(outfile.fp_snp_pval_logistic, "%d\t%s\t%s\t%d\t%d\t%d\t%d\t%g\t", gene->chr, gene->ccds, gene->name, gene->bp_start, 
                                                                                    gene->bp_end, gene->bp_end - gene->bp_start+1, 
                                                                                    gene->nSNP, gene->eSNP);
          //FIX V.1.2 PERM NAN
	  fprintf(outfile.fp_snp_pval_logistic, "%s\t%d\t%g\t%g\t%g\t%d\t%d\t%g\t%d\n", snp->name,snp->bp, snp->MAF, snp->R2, snp->wald, 
                                                                                        snp->iPerm_logistic_sh, snp->nHit_logistic_sh, 
                                                               snp->iPerm_logistic_sh==0?1:((double)snp->nHit_logistic_sh+1)/(snp->iPerm_logistic_sh+1), 
                                                                                        gene->maxSSM_SNP_logistic_sh == snp? 1: 0);//FIX S+1/Q+1
	}
	snp = cq_getNext(snp, snp_queue);
      }
      if(VERBOSE)
	printf("printing minSNP logistic perm finished\n");
  }
  else if(GET_MINSNP_LOGISTIC)
  {
       if(VERBOSE)
          printf("Reporting logistic minsnp parametric pval\n");
       SNP* bestSnp = NULL;
       double bestW = -1;
       snp = cq_getItem(gene->snp_start, snp_queue);
       for(i= gene->snp_start; i <= gene->snp_end; i++)
       {
          if(snp->wald > bestW)
          {
             bestSnp = snp;
             bestW = snp->wald;
          } 
	  snp = cq_getNext(snp, snp_queue);
       }

       snp = cq_getItem(gene->snp_start, snp_queue);
       for(i= gene->snp_start; i <= gene->snp_end; i++)
       {
	 if(outfile.fp_snp_pval_logistic != NULL)
         {
	   fprintf(outfile.fp_snp_pval_logistic, "%d\t%s\t%s\t%d\t%d\t%d\t%d\t%g\t", gene->chr, gene->ccds, gene->name, gene->bp_start, gene->bp_end, gene->bp_end - gene->bp_start+1, gene->nSNP, gene->eSNP);
	   fprintf(outfile.fp_snp_pval_logistic, "%s\t%d\t%g\t%g\t%g\t-\t-\t%g\t%d\n", snp->name,snp->bp, snp->MAF, snp->R2, snp->wald,snp->pval_logistic, snp==bestSnp? 1: 0);
	 }
	 snp = cq_getNext(snp, snp_queue);
       }
  }

  //for minSNP P linear
  if(GET_MINSNP_P_PVAL_LINEAR)
  {
      snp = cq_getItem(gene->snp_start, snp_queue);
      for(i= gene->snp_start; i <= gene->snp_end; i++)
      {
	//make change
	if(outfile.fp_snp_perm_pval_linear != NULL)
        {
	  fprintf(outfile.fp_snp_perm_pval_linear, "%d\t%s\t%s\t%d\t%d\t%d\t%d\t%g\t", gene->chr, gene->ccds, gene->name, gene->bp_start, 
                                                                                       gene->bp_end, gene->bp_end - gene->bp_start+1, 
                                                                                       gene->nSNP, gene->eSNP);
          //FIX V.1.2 PERM NAN
	  fprintf(outfile.fp_snp_perm_pval_linear, "%s\t%d\t%g\t%g\t%g\t%d\t%d\t%g\t%d\n", snp->name,snp->bp, snp->MAF, snp->R2, snp->f_stat, 
                                                                                           snp->iPerm_bonf_linear_sh, snp->nHit_bonf_linear_sh, 
                                                        snp->iPerm_bonf_linear_sh==0?1:((double)snp->nHit_bonf_linear_sh+1)/(snp->iPerm_bonf_linear_sh+1), 
                                                                                           gene->maxBonf_SNP_linear_sh == snp? 1: 0);//FIX S+1/Q+1
	}
	snp->nHit_bonf_linear_sh=0;
	snp->iPerm_bonf_linear_sh=0;
	snp = cq_getNext(snp, snp_queue);
      }
      if(VERBOSE)
	printf("printing minSNP P linear perm finished\n");
  }

  //for minSNP P logistic
  if(GET_MINSNP_P_PVAL_LOGISTIC)
  {
      snp = cq_getItem(gene->snp_start, snp_queue);
      for(i= gene->snp_start; i <= gene->snp_end; i++){
	//make change
	if(outfile.fp_snp_perm_pval_logistic != NULL){
	  fprintf(outfile.fp_snp_perm_pval_logistic, "%d\t%s\t%s\t%d\t%d\t%d\t%d\t%g\t", gene->chr, gene->ccds, gene->name, gene->bp_start, gene->bp_end, gene->bp_end - gene->bp_start+1, gene->nSNP, gene->eSNP);
          //FIX V.1.2 PERM NAN
	  fprintf(outfile.fp_snp_perm_pval_logistic, "%s\t%d\t%g\t%g\t%g\t%d\t%d\t%g\t%d\n", snp->name,snp->bp, snp->MAF, snp->R2, snp->wald, 
                                                                                         snp->iPerm_bonf_logistic_sh, snp->nHit_bonf_logistic_sh, 
                                                  snp->iPerm_bonf_logistic_sh==0?1:((double)snp->nHit_bonf_logistic_sh+1)/(snp->iPerm_bonf_logistic_sh+1), 
                                                                                         gene->maxBonf_SNP_logistic_sh == snp? 1: 0);//FIX S+1/Q+1
	}
	snp->nHit_bonf_logistic_sh=0;
	snp->iPerm_bonf_logistic_sh=0;
	snp = cq_getNext(snp, snp_queue);
      }
      if(VERBOSE)
	printf("printing minSNP P logistic perm finished\n");
  }

  if(GET_BF_PVAL_LINEAR)
  {
      if(VERBOSE)
         printf("Reporting BF linear pval from permutations\n");
      if(outfile.fp_bf_pval_linear != NULL){
	fprintf(outfile.fp_bf_pval_linear, "%d\t%s\t%s\t%d\t%d\t%d\t%d\t%g\t", gene->chr, gene->ccds, gene->name, gene->bp_start, gene->bp_end, gene->bp_end - gene->bp_start+1, gene->nSNP, gene->eSNP);
        //V.1.2 sum FIX //FIX V.1.2 PERM NAN
	fprintf(outfile.fp_bf_pval_linear, "%g\t%d\t%d\t%g\n", gene->BF_sum_linear,
                                                               gene->hits_sh.perm_bf_linear, 
                                                               gene->hits_sh.hit_bf_linear, 
                                           gene->hits_sh.perm_bf_linear==0?1:((double)gene->hits_sh.hit_bf_linear+1)/(gene->hits_sh.perm_bf_linear+1));//FIX S+1/Q+1
      }
      if(VERBOSE)
	printf("printing BF linear perm finished\n");
  }
  else if(GET_BF_LINEAR)
  {
      //no pvalues reported.
      if(outfile.fp_bf_pval_linear != NULL){
	fprintf(outfile.fp_bf_pval_linear, "%d\t%s\t%s\t%d\t%d\t%d\t%d\t%g\t%g\t-\t-\t-\n", gene->chr, gene->ccds, gene->name, gene->bp_start, gene->bp_end, gene->bp_end - gene->bp_start+1, gene->nSNP, gene->eSNP, gene->BF_sum_linear);//V.1.2 sum FIX
      }
  }

  if(GET_BF_PVAL_LOGISTIC)
  {
      if(VERBOSE)
         printf("Reporting BF logistic pval from permutations\n");
      if(outfile.fp_bf_pval_logistic != NULL){
	fprintf(outfile.fp_bf_pval_logistic, "%d\t%s\t%s\t%d\t%d\t%d\t%d\t%g\t", gene->chr, gene->ccds, gene->name, gene->bp_start, gene->bp_end, gene->bp_end - gene->bp_start+1, gene->nSNP, gene->eSNP);
        //V.1.2 sum FIX //FIX V.1.2 PERM NAN
	fprintf(outfile.fp_bf_pval_logistic, "%g\t%d\t%d\t%g\n", gene->BF_sum_logistic, 
                                                                 gene->hits_sh.perm_bf_logistic, 
                                                                 gene->hits_sh.hit_bf_logistic, 
                                     gene->hits_sh.perm_bf_logistic==0?1:((double)gene->hits_sh.hit_bf_logistic+1)/(gene->hits_sh.perm_bf_logistic+1));//FIX S+1/Q+1
      }
      if(VERBOSE)
	printf("printing BF logistic perm finished\n");
  }
  else if(GET_BF_LOGISTIC)
  {
      //no pvalues reported.
      if(outfile.fp_bf_pval_logistic != NULL){
	fprintf(outfile.fp_bf_pval_logistic, "%d\t%s\t%s\t%d\t%d\t%d\t%d\t%g\t%g\t-\t-\t-\n", gene->chr, gene->ccds, gene->name, gene->bp_start, gene->bp_end, gene->bp_end - gene->bp_start+1, gene->nSNP, gene->eSNP, gene->BF_sum_logistic);//V.1.2 sum FIX
      }
  }

  if(GET_VEGAS_PVAL_LINEAR)
  {
      if(VERBOSE)
         printf("Reporting Vegas linear pval from permutations\n");
      if(outfile.fp_vegas_pval_linear != NULL){
	fprintf(outfile.fp_vegas_pval_linear, "%d\t%s\t%s\t%d\t%d\t%d\t%d\t%g\t", gene->chr, gene->ccds, gene->name, gene->bp_start, gene->bp_end, gene->bp_end - gene->bp_start+1, gene->nSNP, gene->eSNP);
        //V.1.2 sum FIX //FIX V.1.2 PERM NAN
	fprintf(outfile.fp_vegas_pval_linear, "%g\t%d\t%d\t%g\n", gene->vegas_linear, 
                                                                      gene->hits_sh.perm_vegas_linear, 
                                                                      gene->hits_sh.hit_vegas_linear, 
                                  gene->hits_sh.perm_vegas_linear==0?1:((double)gene->hits_sh.hit_vegas_linear+1)/(gene->hits_sh.perm_vegas_linear+1));//FIX S+1/Q+1
      }
      if(VERBOSE)
	printf("print Vegas linear perm finished\n");
  }
  else if(GET_VEGAS_LINEAR)
  {
       if(outfile.fp_vegas_pval_linear != NULL)
	  fprintf(outfile.fp_vegas_pval_linear, "%d\t%s\t%s\t%d\t%d\t%d\t%d\t%g\t%g\t-\t-\t-\n", gene->chr, gene->ccds, gene->name, gene->bp_start, gene->bp_end, gene->bp_end - gene->bp_start+1, gene->nSNP, gene->eSNP, gene->vegas_linear);//V.1.2 sum FIX
  }

  if(GET_VEGAS_PVAL_LOGISTIC)
  {
      if(VERBOSE)
         printf("Reporting Vegas logistic pval from permutations\n");
      if(outfile.fp_vegas_pval_logistic != NULL){
	fprintf(outfile.fp_vegas_pval_logistic, "%d\t%s\t%s\t%d\t%d\t%d\t%d\t%g\t", gene->chr, gene->ccds, gene->name, gene->bp_start, gene->bp_end, gene->bp_end - gene->bp_start+1, gene->nSNP, gene->eSNP);
        //V.1.2 sum FIX //FIX V.1.2 PERM NAN
	fprintf(outfile.fp_vegas_pval_logistic, "%g\t%d\t%d\t%g\n", gene->vegas_logistic, 
                                                                   gene->hits_sh.perm_vegas_logistic, 
                                                                   gene->hits_sh.hit_vegas_logistic, 
                        gene->hits_sh.perm_vegas_logistic==0?1:((double)gene->hits_sh.hit_vegas_logistic+1)/(gene->hits_sh.perm_vegas_logistic+1));//FIX S+1/Q+1
      }
      if(VERBOSE)
	printf("print Vegas logistic perm finished\n");
  }
  else if(GET_VEGAS_LOGISTIC)
  {
        if(outfile.fp_vegas_pval_logistic != NULL)
	   fprintf(outfile.fp_vegas_pval_logistic, "%d\t%s\t%s\t%d\t%d\t%d\t%d\t%g\t%g\t-\t-\t-\n", gene->chr, gene->ccds, gene->name, gene->bp_start, gene->bp_end, gene->bp_end - gene->bp_start+1, gene->nSNP, gene->eSNP, gene->vegas_logistic);//V.1.2 sum FIX
  }

  //DO NOT NEED TO DO MIN SNP P with parametric pvalue as the snp with best pvalue will drive the gene pvalue
  //so that minsnp is same as minsnp p with parametric pvalues.
  //printf("Main thread going to sleep....\n");
  //sleep(10);
  //printf("Main thread woke up from sleep....\n");
}


//get the additional parameters
void getEnv(){

  char* fullsearch_s = getenv("FULLSEARCH");
  if(fullsearch_s != NULL && strcmp ( fullsearch_s, "TRUE") == 0){
    FULLSEARCH = true;
    printf("-Search full model up to 3 SNPs\n");
  }else{
    FULLSEARCH=false;
  }

  char* DIFFNCBI_s = getenv("DIFFNCBI");
  if(DIFFNCBI_s != NULL && strcmp ( DIFFNCBI_s, "TRUE") == 0){
    DIFFNCBI = true;
    printf("-Calculating differentiated NCBI set for H GWiS\n");
  }else{
    DIFFNCBI=false;
  }

  if(SMART_STOP){
    printf("-will perform smart permutation for up to %d permutations if doing permutations\n", N_PERM);
  }else
    printf("-will perform up to %d permutations if doing permutations\n", N_PERM);

  char* INTERCEPT_s = getenv("INTERCEPT");
  if(INTERCEPT_s != NULL && strcmp(INTERCEPT_s, "FALSE") == 0){
    INTERCEPT=false;
    printf("-working in NO intercept mode\n");
  } else{
    INTERCEPT=true;
  }

  //printf("INTERCEPT = %d\n",INTERCEPT);

  SHUFFLEBUF = false;
  char* get_shuffle_buf_s = getenv("SHUFFLEBUF");
  if(get_shuffle_buf_s != NULL && strcmp (get_shuffle_buf_s, "TRUE") == 0){
    SHUFFLEBUF = true;
    printf("-use phenotype shuffle buffer up to %d\n", NSHUFFLEBUF);
  }else{
    SHUFFLEBUF = false;
  }
}

// check file existance
bool file_exists(const char *filename){
  FILE *file;
  if ((file = fopen(filename, "r"))) //I'm sure, you meant for READING =)
  {
     fclose(file);
     return true;
  }
  return false;
}

bool file_can_be_created(const char *filename){
  FILE *file;
  if ((file = fopen(filename, "w")))
  {
     remove(filename);
     return true;
  }
  return false;
}

void print_help()
{
   printf("****************** Help for FAST ************************\n");
   printf("--mode <genotype/summary>     : run in one of the two modes (genotype/summary)\n");
   printf("---------------Input files--------------------------------\n");
   printf("--impute2-file <fname>        : for mode=genotype, MANDATORY input genotype file in impute2 format\n");
   printf("--impute2-info-file <fname>   : for mode=genotype, MANDATORY input snp imputation info file in impute2 format\n");
   printf("                              OR\n");
   printf("--tped-file <fname>           : for mode=genotype, MANDATORY input genotype dosage file\n");
   printf("--snpinfo-file <fname>        : for mode=genotype, MANDATORY input snp base-pair information file\n");
   printf("--mlinfo-file <fname>         : for mode=genotype, MANDATORY input snp allele and imputation information file\n");
   printf("                              AND\n");
   printf("--indiv-file <fame>           : for mode=genotype, MANDATORY file containing individual ids\n");
   printf("--trait-file <trait>          : for mode=genotype, MANDATORY file having phenotype and optional covariates\n");    
   printf("--summary-file <fname>        : for mode=summary, MANDATORY input snp summary information file\n");
   printf("--multipos-file <fname>       : for mode=summary, input file for snps mapped to muliple positions\n");
   printf("--hap-file <fname>            : for mode=summary, when computing LD on the fly, MANDATORY input haplotype file\n");
   printf("--pos-file <fname>            : for mode=summary, when computing LD on the fly, MANDATORY input index file for haplotype start byte positions\n");
   printf("--ld-file <fname>             : for mode=summary, when using pre-computed LD, MANDATORY file containing pairwise LD\n");
   printf("--allele-file <fname>         : for mode=summary, when using pre-computed LD, MANDATORY file containing snp reference allele used in LD computation\n");
   printf("---------------Other input options------------------------\n");
   printf("--chr <chr no>                : MANDATORY chromosome number for either mode, 1 - 22 for autosomes and 23 for chromosome-X\n");
   printf("--out-file <outfile>          : output files are prefixed with <outfile>, default = FAST.result\n");
   //printf("--pheno-mean <value>        : for mode=summary, MANDATORY option specifying mean phenotype for all methods except gates\n"); 
   printf("--compute-ld <value>          : for mode=summary, 0/1 value to compute LD on the fly (default = 1 i.e. true)\n");
   printf("--pheno-var <value>           : for mode=summary, option specifying phenotype variance for all methods except gates\n");
   printf("--scale-pheno                 : for mode=genotype, option to scale the phenotype to have unit variance\n");
   printf("--quantile-pheno              : for mode=genotype, option to use normal quantile transformation to transform the phenotype to normal distribution\n");
   printf("--n-sample <value>            : for mode=summary,  option specifying number of samples \n");
   printf("--maf-cutoff <value>          : value to filter snps with maf < maf-cutoff for gene-based analysis (default = 0.01)\n");
   printf("--gene-set <fname>            : file containing gene names and boundaries for either mode\n");
   printf("--random-seed <value>         : seed for permutations (default = 2)\n");
   printf("--flank <value>               : gene flanking region in base-pairs (default = 20000 bp)\n");
   printf("--max-perm <value>            : max no. of permutations (default = 1000000)\n");
   printf("--n-perm-min <value>          : min no. of permutations (default = 100)\n");
   printf("--max-missingness <value>     : max allowed missingness per snp between 0 and 1.0 (default = 0.05)\n");
   printf("--missing-val <value>         : value indicating missing genotypes (default = -1)\n");
   printf("--eff-sample-size <value>     : min effective sample size per snp (default = 5)\n");
   printf("--imputation-quality <value>  : min imputation quality per snp between 0 and 1.0 (default = 0.3)\n");
   printf("--omit-strand-ambiguous       : drop strand ambiguous snps (default = no)\n");

   //Is SKIPPERM necessary ? Its useful when you want to switch running all methods with permutations to all methods without permutations.
   //So specify all methods with the -perm as suffix and simply setting --skip-perm wiill run them without permutations.
   //If you want to run with permutations,simply remove the --skip-perm flag. 
   //
   printf("--skip-perm                   : want to skip all permutations ? (default = no)\n");
   printf("--num-covariates <value>      : count of covariates (default = 0)\n");
   printf("--verbose                     : detailed output in debug mode (default = no)\n");
   printf("--n-threads <value>           : no. of threads for using multiple cores in permutations (default = 1)\n");
   printf("--sigma-a <value>             : priors for additive effects in bf computations (default = 0.2)\n");
   printf("---------------Logistic regression options----------------\n");
   printf("--logistic-snp                : single snp logistic regression for all snps\n");
   printf("--logistic-snp-gene           : single snp logistic regression for snps in gene\n");
   printf("--logistic-minsnp             : logistic regression based minsnp\n");
   printf("--logistic-minsnp-gene-perm   : logistic regression based minsnp-gene-perm\n");
   printf("--logistic-gwis               : logisitic regression based GWiS\n");
   printf("--logistic-bf                 : logistic regression based Bayes Factors\n");
   printf("--logistic-vegas              : logistic regression based Vegas\n");
   printf("--logistic-gates              : logistic regression based Gates\n");
   printf("--logistic-minsnp-perm        : logistic regression based minsnp with permutation pvalues\n");
   printf("--logistic-gwis-perm          : logistic regression based GWiS with permutation pvalues\n");
   printf("--logistic-bf-perm            : logistic regression based Bayes Factors with permutation pvalues\n");
   printf("--logistic-vegas-perm         : logistic regression based Vegas with permutation pvalues\n");
   printf("---------------Linear regression options------------------\n");
   printf("--linear-snp                  : single snp linear regression for all snps\n");
   printf("--linear-snp-gene             : single snp linear regression for snps in gene\n");
   printf("--linear-minsnp               : linear regression based minsnp\n");
   printf("--linear-minsnp-gene-perm     : linear regression based minsnp-gene-perm\n");
   printf("--linear-gwis                 : linear regression based GWiS\n");
   printf("--linear-bf                   : linear regression based Bayes Factors\n");
   printf("--linear-vegas                : linear regression based Vegas\n");
   printf("--linear-gates                : linear regression based Gates\n");
   printf("--linear-minsnp-perm          : linear regression based minsnp with permutation pvalues\n");
   printf("--linear-gwis-perm            : linear regression based GWiS with permutation pvalues\n");
   printf("--linear-bf-perm              : linear regression based Bayes Factors with permutation pvalues\n");
   printf("--linear-vegas-perm           : linear regression based Vegas with permutation pvalues\n");
   printf("******************End Help *******************************\n");
}

//parse the command line parameters
int  parseArgs(int nARG, char* ARGV[], PAR* par)
{
  int i;

  for (i=1; i<nARG; i++)
  {
    //printf("Got option [%s]\n",ARGV[i]);
    bool recog = false;

    if(ARGV[i][0]=='#') continue;

    if(strcmp(ARGV[i], "--help") == 0)
    {
       print_help();
       exit(0);
    }

    if(strcmp(ARGV[i], "--n-threads") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      if(sscanf(ARGV[++i], "%d", &par->n_threads)==0) {printf("Wrong n-threads value %s, quitting\n",ARGV[i]);exit(1);}
      par->n_threads_parsed=true;
      n_threads = par->n_threads;
      printf("\t%s: %d\n", "--n-threads",  par->n_threads );
      recog = true;
    }

    if(strcmp(ARGV[i], "--verbose") == 0)
    {
       printf("\t--verbose set, useful for debugging purpose\n");
       VERBOSE = true;
       recog = true;
    }

    if(strcmp(ARGV[i], "--omit-strand-ambiguous") == 0)
    {
       printf("\t%s: %s\n", "--omit-strand-ambiguous",  "true" );
       OMIT_ST = true;
       recog = true;
    }

    //V.1.5.mc, adding support for impute2 style input files.
    if(strcmp(ARGV[i], "--impute2-geno-file") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      par->impute2_geno_file = ARGV[++i];
      par->impute2_geno_file_parsed = true;
      printf("\t%s: %s\n", "--impute2-geno-file",  par->impute2_geno_file );
      recog = true;
      IMPUTE2_input = true; 
    }

    //V.1.5.mc, adding support for impute2 style input files.
    if(strcmp(ARGV[i], "--impute2-info-file") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      par->impute2_info_file = ARGV[++i];
      par->impute2_info_file_parsed = true;
      printf("\t%s: %s\n", "--impute2-info-file",  par->impute2_info_file );
      recog = true;
      IMPUTE2_input = true; 
    }

    if(strcmp(ARGV[i], "--tped-file") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      par->tped_file = ARGV[++i];
      par->tped_file_parsed = true;
      printf("\t%s: %s\n", "--tped-file",  par->tped_file );
      recog = true;
    }

    if(strcmp(ARGV[i], "--snpinfo-file") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      par->snpinfo_file = ARGV[++i];
      par->snpinfo_file_parsed = true;
      printf("\t%s: %s\n", "--snpinfo-file",  par->snpinfo_file );
      recog = true;
    }

    if(strcmp(ARGV[i], "--mlinfo-file") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      par->mlinfo_file = ARGV[++i];
      par->mlinfo_file_parsed = true;
      printf("\t%s: %s\n", "--mlinfo-file",  par->mlinfo_file );
      recog = true;
    }

    if(strcmp(ARGV[i], "--summary-file") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      par->summary_file = ARGV[++i];
      par->summary_file_parsed = true;
      printf("\t%s: %s\n", "--summary-file",  par->summary_file );
      recog = true;
    }

    if(strcmp(ARGV[i], "--multipos-file") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      par->multipos_file = ARGV[++i];
      par->multipos_file_parsed = true;
      printf("\t%s: %s\n", "--multipos-file",  par->multipos_file );
      recog = true;
    }

    if(strcmp(ARGV[i], "--indiv-file") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      par->indivfile = ARGV[++i];
      par->indivfile_parsed = true;
      printf("\t%s: %s\n", "--indiv-file",  par->indivfile );
      recog = true;
    }
    if(strcmp(ARGV[i], "--chr") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      if(sscanf(ARGV[++i], "%d", &par->chr)==0) {printf("Wrong chr value %s (should be between 1-22 for autosomes and 23 for chromosome-X), quitting\n",ARGV[i]);exit(1);}
      par->chr_parsed=true;
      printf("\t%s: %d\n", "--chr",  par->chr);
      recog = true;
    }
    if(strcmp(ARGV[i], "--trait-file") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      par->trait_file = ARGV[++i];
      par->trait_file_parsed = true;
      printf("\t%s: %s\n", "--trait-file",  par->trait_file );
      recog = true;
    }
    if(strcmp(ARGV[i], "--out-file") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      par->output = ARGV[++i];
      par->output_parsed = true;
      printf("\t%s: %s\n", "--out-file",  par->output );
      recog = true;
    }

    if(strcmp(ARGV[i], "--ld-file") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      par->ldfile = ARGV[++i];
      par->ldfile_parsed = true;
      printf("\t%s: %s\n", "--ld-file",  par->ldfile );
      recog = true;
    }

    if(strcmp(ARGV[i], "--allele-file") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      par->allelefile = ARGV[++i];
      par->allelefile_parsed = true;
      printf("\t%s: %s\n", "--allele-file",  par->allelefile );
      recog = true;
    }

    //+V.1.4.mc
    if(strcmp(ARGV[i], "--pos-file") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      par->posfile = ARGV[++i];
      par->posfile_parsed = true;
      printf("\t%s: %s\n", "--pos-file",  par->posfile );
      recog = true;
    }

    if(strcmp(ARGV[i], "--hap-file") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      par->hapfile = ARGV[++i];
      par->hapfile_parsed = true;
      printf("\t%s: %s\n", "--hap-file",  par->hapfile );
      recog = true;
    }

    if(strcmp(ARGV[i], "--hapW-file") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      par->hap_wt_file = ARGV[++i];
      par->hap_wt_file_parsed = true;
      printf("\t%s: %s\n", "--hapW-file",  par->hap_wt_file );
      recog = true;
    }
    //-V.1.4.mc

    /*
    if(strcmp(ARGV[i], "--pheno-mean") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      sscanf(ARGV[++i], "%lg", &par->pheno_mean);
      par->pheno_mean_parsed=true;
      printf("\t%s: %g\n", "--pheno-mean",  par->pheno_mean );
      recog = true;
    }
    */

    if(strcmp(ARGV[i], "--pheno-var") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      if(sscanf(ARGV[++i], "%lg", &par->pheno_var)==0) {printf("Wrong pheno-var value %s, quitting\n",ARGV[i]);exit(1);}
      par->pheno_var_parsed=true;
      printf("\t%s: %g\n", "--pheno-var",  par->pheno_var );
      recog = true;
    }

    if(strcmp(ARGV[i], "--n-sample") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      if(sscanf(ARGV[++i], "%d", &par->n_sample)==0) {printf("Wrong n-sample value %s, quitting\n",ARGV[i]);exit(1);}
      par->n_sample_parsed=true;
      printf("\t%s: %d\n", "--n-sample",  par->n_sample );
      recog = true;
    }

    if(strcmp(ARGV[i], "--maf-cutoff") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      if(sscanf(ARGV[++i], "%lg", &par->maf_cutoff)==0) {printf("Wrong maf-cutoff value %s, quitting\n",ARGV[i]);exit(1);}
      par->maf_cutoff_parsed=true;
      maf_cutoff = par->maf_cutoff;
      printf("\t%s: %lg\n", "--maf-cutoff",  par->maf_cutoff );
      recog = true;
    }

    if(strcmp(ARGV[i], "--gene-set") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      par->gene_set =ARGV[++i] ;
      par->gene_set_parsed = true ;
      printf("\t%s: %s\n", "--gene-set",  par->gene_set );
      recog = true;
    }
    if(strcmp(ARGV[i], "--random-seed") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      if(sscanf(ARGV[++i], "%d", &par->random_seed)==0) {printf("Wrong random-seed value %s, quitting\n",ARGV[i]);exit(1);}
      par->random_seed_parsed = true ;
      printf("\t%s: %d\n", "--random-seed",  par->random_seed );
      recog = true;
    }

    if(strcmp(ARGV[i], "--flank") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      if(sscanf(ARGV[++i], "%d", &par->flank)==0) {printf("Wrong flank value %s, quitting\n",ARGV[i]);exit(1);}
      par->flank_parsed = true ;
      printf("\t%s: %d\n", "--flank",  par->flank );
      recog = true;
    }

    if(strcmp(ARGV[i], "--max-perm") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      if(sscanf(ARGV[++i], "%d", &par->max_perm)==0) {printf("Wrong max-perm value %s, quitting\n",ARGV[i]);exit(1);}
      par->max_perm_parsed = true;
      printf("\t%s: %d\n", "--max-perm",  par->max_perm );
      recog = true;
    }

    if(strcmp(ARGV[i], "--n-perm-min") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      if(sscanf(ARGV[++i], "%d", &par->n_perm_min)==0) {printf("Wrong max-perm value %s, quitting\n",ARGV[i]);exit(1);}
      par->n_perm_min_parsed = true;
      printf("\t%s: %d\n", "--n-perm-min",  par->n_perm_min );
      recog = true;
    }

    if(strcmp(ARGV[i], "--max-missingness") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      if(sscanf(ARGV[++i], "%lg", &par->missingness)==0) {printf("Wrong max-missingness value %s, quitting\n",ARGV[i]);exit(1);}
      par->missingness_parsed = true;
      MAX_MISSINGNESS = par->missingness;
      printf("\t%s: %g\n", "--max-missingness",  par->missingness );
      recog = true;
    }

    //V.1.5.mc
    if(strcmp(ARGV[i], "--eff-sample-size") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      if(sscanf(ARGV[++i], "%d", &par->n_hat_cutoff)==0) {printf("Wrong eff-sample-size value %s, quitting\n",ARGV[i]);exit(1);}
      par->n_hat_cutoff_parsed = true;
      n_hat_cutoff = par->n_hat_cutoff;
      printf("\t%s: %d\n", "--eff-sample-size",  n_hat_cutoff);
      recog = true;
    }

    if(strcmp(ARGV[i], "--imputation-quality") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      if(sscanf(ARGV[++i], "%lg", &par->r2_cutoff)==0) {printf("Wrong imputation-quality value %s, quitting\n",ARGV[i]);exit(1);}
      par->r2_cutoff_parsed = true;
      r2_cutoff = par->r2_cutoff;
      printf("\t%s: %g\n", "--imputation-quality",  r2_cutoff);
      recog = true;
    }

    if(strcmp(ARGV[i], "--missing-val") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      if(sscanf(ARGV[++i], "%lg", &MISSING_VAL)==0) {printf("Wrong missing-val value %s, quitting\n",ARGV[i]);exit(1);}
      printf("\t%s: %g\n", "--missing-val",  MISSING_VAL );
      recog = true;
    }

    if(strcmp(ARGV[i], "--mode") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      par->mode=ARGV[++i];
      par->mode_parsed = true ;
      printf("\t%s: %s\n", "--mode",  par->mode );
      recog = true;
    }

    if(strcmp(ARGV[i], "--scale-pheno") == 0){
      par->scalepheno = true;
      par->scalepheno_parsed = true ;
      printf("\t%s: %d\n", "--scale-pheno",  par->scalepheno );
      recog = true;
    }

    if(strcmp(ARGV[i], "--quantile-pheno") == 0){
      par->quantilepheno = true;
      par->quantilepheno_parsed = true ;
      printf("\t%s: %d\n", "--quantile-pheno",  par->quantilepheno );
      recog = true;
    }

    if(strcmp(ARGV[i], "--skip-perm") == 0){
      par->skip_perm=true;
      par->skip_perm_parsed = true ;
      SKIPPERM = true;
      printf("\t%s: %s\n", "--skip_perm",  "true" );
      recog = true;
    }

    if(strcmp(ARGV[i], "--compute-ld") == 0){
      if(i+1>=nARG){printf("-Not sufficient arguments, quitting\n");exit(1);}
      int zz = 0;
      if(sscanf(ARGV[++i], "%d", &zz)==0) {printf("Wrong value %s for --compute-ld, should be 0/1, quitting\n",ARGV[i]);exit(1);}
      if(zz<=0) par->compute_ld = false;
      else par->compute_ld = true;
      par->compute_ld_parsed = true ;
      COMPUTE_LD = par->compute_ld;
      printf("\t%s: %d\n", "--compute-ld",  COMPUTE_LD );
      recog = true;
    }

    if(strcmp(ARGV[i], "--weighted-ld") == 0){
      par->use_weighted_ld=true;
      par->use_weighted_ld_parsed = true ;
      USE_WEIGHTED_LD = true;
      printf("\t%s: %s\n", "--weighted-ld",  "true" );
      recog = true;
    }
    //-V.1.4.mc

    //++Pritam 
    if(strcmp(ARGV[i], "--logistic-snp") == 0){   //parametric pvalue.
      par->single_snp_logistic=true;
      GET_SINGLE_SNP_LOGISTIC = true;
      par->single_snp_logistic_parsed = true ;
      printf("\t%s: %s\n", "--logistic-snp",  "true" );
      recog = true;
    }

    if(strcmp(ARGV[i], "--logistic-snp-gene") == 0){   //parametric pvalue.
      par->single_snp_logistic_gene=true;
      par->single_snp_logistic=true;
      GET_SINGLE_SNP_LOGISTIC = true;
      par->single_snp_logistic_gene_parsed = true ;
      par->single_snp_logistic_parsed = true ;
      printf("\t%s: %s\n", "--logistic-snp-gene",  "true" );
      recog = true;
    }

    if(strcmp(ARGV[i], "--linear-snp") == 0){   //parametric pvalue.
      par->single_snp_linear=true;
      GET_SINGLE_SNP_LINEAR = true;
      par->single_snp_linear_parsed = true ;
      printf("\t%s: %s\n", "--linear-snp",  "true" );
      recog = true;
    }

    if(strcmp(ARGV[i], "--linear-snp-gene") == 0){   //parametric pvalue.
      par->single_snp_linear_gene=true;
      par->single_snp_linear=true;
      GET_SINGLE_SNP_LINEAR = true;
      par->single_snp_linear_gene_parsed = true ;
      par->single_snp_linear_parsed = true ;
      printf("\t%s: %s\n", "--linear-snp-gene",  "true" );
      recog = true;
    }

    if(strcmp(ARGV[i], "--logistic-minsnp") == 0){ //parametric pvalue, if permall, permutation pvalue
      par->minsnp_logistic=true;
      GET_MINSNP_LOGISTIC = true;
      par->minsnp_logistic_parsed = true ;
      printf("\t%s: %s\n", "--logistic-minsnp",  "true" );
      recog = true;
    }

    if(strcmp(ARGV[i], "--linear-minsnp") == 0){ //parametric pvalue, if permall, permutation pvalue
      par->minsnp_linear=true;
      GET_MINSNP_LINEAR = true;
      par->minsnp_linear_parsed = true ;
      printf("\t%s: %s\n", "--linear-minsnp",  "true" );
      recog = true;
    }

    if(strcmp(ARGV[i], "--logistic-minsnp-perm") == 0){ //permutation pvalue, if skipperm, parametric pvalue.
      par->minsnp_pval_logistic=true;
      GET_MINSNP_PVAL_LOGISTIC = true;
      par->minsnp_pval_logistic_parsed = true ;
      printf("\t%s: %s\n", "--logistic-minsnp-perm",  "true" );
      recog = true;
    }

    if(strcmp(ARGV[i], "--linear-minsnp-perm") == 0){ //permutation pvalue, if skipperm, parametric pvalue.
      par->minsnp_pval_linear=true;
      GET_MINSNP_PVAL_LINEAR = true;
      par->minsnp_pval_linear_parsed = true ;
      printf("\t%s: %s\n", "--linear-minsnp-perm",  "true" );
      recog = true;
    }

    if(strcmp(ARGV[i], "--logistic-minsnp-gene-perm") == 0){ //permutation pvalue, if skipperm, parametric pvalue.
      par->minsnp_p_pval_logistic=true;
      GET_MINSNP_P_PVAL_LOGISTIC = true;
      par->minsnp_p_pval_logistic_parsed = true ;
      printf("\t%s: %s\n", "--logistic-minsnp-gene-perm",  "true" );
      recog = true;
    }

    if(strcmp(ARGV[i], "--linear-minsnp-gene-perm") == 0){ //permutation pvalue, if skipperm, parametric pvalue.
      par->minsnp_p_pval_linear=true;
      GET_MINSNP_P_PVAL_LINEAR = true;
      par->minsnp_p_pval_linear_parsed = true ;
      printf("\t%s: %s\n", "--linear-minsnp-gene-perm",  "true" );
      recog = true;
    }

    if(strcmp(ARGV[i], "--logistic-gwis") == 0){
      par->gene_bic_logistic=true;
      GET_GENE_BIC_LOGISTIC = true;
      par->gene_bic_logistic_parsed = true ;
      printf("\t%s: %s\n", "--logistic-gwis",  "true" );
      recog = true;
    }

    if(strcmp(ARGV[i], "--logistic-gwis-perm") == 0){
      par->gene_bic_pval_logistic=true;
      GET_GENE_BIC_PVAL_LOGISTIC = true;
      par->gene_bic_pval_logistic_parsed = true ;
      printf("\t%s: %s\n", "--logistic-gwis-perm",  "true" );
      recog = true;
    }

    if(strcmp(ARGV[i], "--linear-gwis") == 0){
      par->gene_bic_linear=true;
      GET_GENE_BIC_LINEAR = true;
      par->gene_bic_linear_parsed = true ;
      printf("\t%s: %s\n", "--linear-gwis",  "true" );
      recog = true;
    }

    if(strcmp(ARGV[i], "--linear-gwis-perm") == 0){
      par->gene_bic_pval_linear=true;
      GET_GENE_BIC_PVAL_LINEAR = true;
      par->gene_bic_pval_linear_parsed = true ;
      printf("\t%s: %s\n", "--linear-gwis-perm",  "true" );
      recog = true;
    }

    if(strcmp(ARGV[i], "--linear-bf") == 0){
      par->gene_bf_linear=true;
      GET_BF_LINEAR = true;
      par->gene_bf_linear_parsed = true ;
      printf("\t%s: %s\n", "--linear-bf",  "true" );
      recog = true;
    }

    if(strcmp(ARGV[i], "--linear-bf-perm") == 0){
      par->gene_bf_pval_linear=true;
      GET_BF_PVAL_LINEAR = true;
      par->gene_bf_pval_linear_parsed = true ;
      printf("\t%s: %s\n", "--linear-bf-perm",  "true" );
      recog = true;
    }

    if(strcmp(ARGV[i], "--logistic-bf") == 0){
      par->gene_bf_logistic=true;
      GET_BF_LOGISTIC = true;
      par->gene_bf_logistic_parsed = true ;
      printf("\t%s: %s\n", "--logistic-bf",  "true" );
      recog = true;
    }

    if(strcmp(ARGV[i], "--logistic-bf-perm") == 0){
      par->gene_bf_pval_logistic=true;
      GET_BF_PVAL_LOGISTIC = true;
      par->gene_bf_pval_logistic_parsed = true ;
      printf("\t%s: %s\n", "--logistic-bf-perm",  "true" );
      recog = true;
    }

    if(strcmp(ARGV[i], "--linear-vegas") == 0){
      par->gene_vegas_linear=true;
      GET_VEGAS_LINEAR = true;
      par->gene_vegas_linear_parsed = true ;
      printf("\t%s: %s\n", "--linear-vegas",  "true" );
      recog = true;
    }

    if(strcmp(ARGV[i], "--linear-vegas-perm") == 0){
      par->gene_vegas_pval_linear=true;
      GET_VEGAS_PVAL_LINEAR = true;
      par->gene_vegas_pval_linear_parsed = true ;
      printf("\t%s: %s\n", "--linear-vegas-pval",  "true" );
      recog = true;
    }

    if(strcmp(ARGV[i], "--logistic-vegas") == 0){
      par->gene_vegas_logistic=true;
      GET_VEGAS_LOGISTIC = true;
      par->gene_vegas_logistic_parsed = true ;
      printf("\t%s: %s\n", "--logistic-vegas",  "true" );
      recog = true;
    }

    if(strcmp(ARGV[i], "--logistic-vegas-perm") == 0){
      par->gene_vegas_pval_logistic=true;
      GET_VEGAS_PVAL_LOGISTIC = true;
      par->gene_vegas_pval_logistic_parsed = true ;
      printf("\t%s: %s\n", "--logistic-vegas-perm",  "true" );
      recog = true;
    }

    if(strcmp(ARGV[i], "--linear-gates") == 0){
      par->gene_gates_linear=true;
      GET_GATES_LINEAR = true;
      par->gene_gates_linear_parsed = true ;
      printf("\t%s: %s\n", "--linear-gates",  "true" );
      recog = true;
    }

    if(strcmp(ARGV[i], "--logistic-gates") == 0){
      par->gene_gates_logistic=true;
      GET_GATES_LOGISTIC = true;
      par->gene_gates_logistic_parsed = true ;
      printf("\t%s: %s\n", "--logistic-gates",  "true" );
      recog = true;
    }

    //covariates
    if(strcmp(ARGV[i], "--num-covariates") == 0){
      if(i+1>=nARG){printf("Not sufficient arguments, quitting\n");exit(1);}
      if(sscanf(ARGV[++i], "%d", &par->ncov)==0) {printf("Wrong num-covariates value %s, quitting\n",ARGV[i]);exit(1);}
      par->ncov_parsed = true ;
      printf("\t%s: %d\n", "--num-covariates",  par->ncov );
      recog = true;
    }
    //--Pritam covariates

    //++LATEST V.1.2 
    if(strcmp(ARGV[i], "--sigma-a") == 0){
      if(i+1>=nARG){printf("Not sufficient arguments, quitting\n");exit(1);}
      if(sscanf(ARGV[++i], "%lg", &par->sigma_a)==0) {printf("Wrong sigma-a value %s, quitting\n",ARGV[i]);exit(1);}
      par->sigma_a_parsed = true;
      SIGMA_A = par->sigma_a;
      printf("\t%s: %lg\n", "--sigma-a",  par->sigma_a );
      recog = true;
    }
    //--LATEST V.1.2
    
    if(!recog)
    {
       printf("-Unrecognized option %s\n",ARGV[i]);
       printf("-Use option --help\n");
       exit(1);
    }
  }

  if(nARG<=1)
  {
       printf("-No options specified\n");
       print_help();
       //printf("-Use option --help\n");
       exit(1);
  }
  return 1;
}

//++Pritam
void init_par(PAR* par, OUTFILE * outfile)
{
  MAX_MISSINGNESS = 0.05;
  MISSING_VAL = -1;
  MISSING_DATA = false;
  VERBOSE = false;
  SIMPLE_SUMMARY = false; //V.1.7
  OMIT_ST = false;

  N_PERM_MIN = 100;
  N_CUTOFF = 50;

  //V.1.5.mc
  n_hat_cutoff = N_HAT_CUTOFF;
  r2_cutoff = R2_CUTOFF;
  par->r2_cutoff = R2_CUTOFF;  
  par->r2_cutoff_parsed = false;  
  par->n_hat_cutoff = N_HAT_CUTOFF;  
  par->n_hat_cutoff_parsed = false;  

  IMPUTE2_input = false; //V.1.5.mc

  par->tped_file = "";
  par->snpinfo_file = "";
  par->mlinfo_file = "";
  par->summary_file = "";
  par->multipos_file = "";
  par->indivfile = "";
  par->chr = 0;
  par->trait_file = "";
  par->output = "";
  par->ldfile = "";
  par->allelefile = "";
  //par->pheno_mean = 0; //V.1.4.mc
  par->pheno_var = 0;
  par->n_sample = 0;
  par->gene_set = "";
  par->random_seed = 0;
  par->flank = 0;
  par->max_perm = 0;

  par->n_perm_min = N_PERM_MIN;

  par->mode = "";
  par->skip_perm = false;
  par->scalepheno = false;
  par->quantilepheno = false;

  par->single_snp_logistic = false;
  par->single_snp_logistic_gene = false;
  par->minsnp_logistic = false;
  par->minsnp_pval_logistic = false;
  par->minsnp_p_pval_logistic = false;
  par->gene_bic_logistic = false;
  par->gene_bic_pval_logistic = false;
  par->gene_bf_logistic = false;
  par->gene_bf_pval_logistic = false;
  par->gene_vegas_logistic = false;
  par->gene_vegas_pval_logistic = false;

  par->single_snp_linear = false;
  par->single_snp_linear_gene = false;
  par->minsnp_linear = false;
  par->minsnp_pval_linear = false;
  par->minsnp_p_pval_linear = false;
  par->gene_bic_linear = false;
  par->gene_bic_pval_linear = false;
  par->gene_bf_linear = false;
  par->gene_bf_pval_linear = false;
  par->gene_vegas_linear = false;
  par->gene_vegas_pval_linear = false;
  par->gene_gates_linear = false;
  //par->gene_gates_pval_linear = false;
  par->missingness = MAX_MISSINGNESS; //5% default

  par->ncov = 0; 

  par->tped_file_parsed = false;
  par->snpinfo_file_parsed = false;
  par->mlinfo_file_parsed = false;
  par->summary_file_parsed = false;
  par->multipos_file_parsed = false;
  par->indivfile_parsed = false;
  par->ldfile_parsed = false;
  par->chr_parsed = false;
  par->trait_file_parsed = false;
  par->output_parsed = false;
  //par->pheno_mean_parsed = false; //V.1.4.mc
  par->pheno_var_parsed = false;
  par->n_sample_parsed = false;
  par->gene_set_parsed = false;
  par->random_seed_parsed = false;
  par->flank_parsed = false;
  par->max_perm_parsed = false;
  par->mode_parsed = false;
  par->skip_perm_parsed = false;
  par->missingness_parsed = false;
  par->scalepheno_parsed = false;
  par->quantilepheno_parsed = false;

  par->single_snp_logistic_parsed = false;
  par->single_snp_logistic_gene_parsed = false;
  par->minsnp_logistic_parsed = false;
  par->minsnp_pval_logistic_parsed = false;
  par->minsnp_p_pval_logistic_parsed = false;
  par->gene_bic_logistic_parsed = false;
  par->gene_bic_pval_logistic_parsed = false;
  par->gene_bf_logistic_parsed = false;
  par->gene_bf_pval_logistic_parsed = false;
  par->gene_vegas_logistic_parsed = false;
  par->gene_vegas_pval_logistic_parsed = false;

  par->single_snp_linear_parsed = false;
  par->single_snp_linear_gene_parsed = false;
  par->minsnp_linear_parsed = false;
  par->minsnp_pval_linear_parsed = false;
  par->minsnp_p_pval_linear_parsed = false;
  par->gene_bic_linear_parsed = false;
  par->gene_bic_pval_linear_parsed = false;
  par->gene_bf_linear_parsed = false;
  par->gene_bf_pval_linear_parsed = false;
  par->gene_vegas_linear_parsed = false;
  par->gene_vegas_pval_linear_parsed = false;
  par->gene_gates_linear_parsed = false;
  //par->gene_gates_pval_linear_parsed = false;

  par->ncov_parsed = false;

  GET_SINGLE_SNP_LOGISTIC = false; //compute single snp logistic wald + parametric pval
  GET_MINSNP_PVAL_LOGISTIC = false; 
  GET_MINSNP_P_PVAL_LOGISTIC = false; 
  GET_BF_LOGISTIC = false;
  GET_BF_PVAL_LOGISTIC = false;
  GET_VEGAS_LOGISTIC = false;
  GET_VEGAS_PVAL_LOGISTIC = false;
  GET_GENE_BIC_LOGISTIC = false;
  GET_GENE_BIC_PVAL_LOGISTIC = false;
  GET_GATES_LOGISTIC = false;

  GET_SINGLE_SNP_LINEAR = false; //compute single snp fstat + parametric pval
  GET_MINSNP_PVAL_LINEAR = false; 
  GET_MINSNP_P_PVAL_LINEAR = false; 
  GET_BF_LINEAR = false;
  GET_BF_PVAL_LINEAR = false;
  GET_VEGAS_LINEAR = false;
  GET_VEGAS_PVAL_LINEAR = false;
  GET_GENE_BIC_LINEAR = false;
  GET_GENE_BIC_PVAL_LINEAR = false;
  GET_GATES_LINEAR = false;

  par->sigma_a = 0.2; //LATEST V.1.2
  par->sigma_a_parsed = false; //LATEST V.1.2
  SIGMA_A = par->sigma_a;//LATEST V.1.2

  NEED_SNP_LINEAR = false;
  NEED_SNP_LINEAR_SUMMARY = false;
  NEED_SNP_LOGISTIC = false;
  NEED_SNP_LOGISTIC_SUMMARY = false;

  COMPUTE_LD = true;

  ESTIMATE_PHENO_VAR = false;
  ESTIMATE_N = false;

  outfile->fp_allSNP_linear = NULL;
  outfile->fp_allSNP_logistic = NULL;
  outfile->fp_snp_pval_linear = NULL;
  outfile->fp_snp_pval_logistic = NULL;
  outfile->fp_snp_perm_pval_linear = NULL;
  outfile->fp_snp_perm_pval_logistic = NULL;
  outfile->fp_gene_snp = NULL;
  //outfile->fp_bic_logistic_result = NULL;
  outfile->fp_bic_logistic_perm_result = NULL;
  //outfile->fp_bic_linear_result = NULL;
  outfile->fp_bic_linear_perm_result = NULL;
  outfile->fp_bf_pval_linear = NULL;
  outfile->fp_bf_pval_logistic = NULL;
  outfile->fp_vegas_pval_linear = NULL;
  outfile->fp_vegas_pval_logistic = NULL;
  outfile->fp_gates_pval_linear = NULL;
  outfile->fp_gates_pval_logistic = NULL;

  par->n_threads = 1;
  par->n_threads_parsed = false;
  n_threads = 1;

  nthr_completed = 0;
  perm_loop_counter = 0;

  USE_WEIGHTED_LD = false; //V.1.3.mc
  haplotype_weights = NULL; //V.1.4.mc

  par->maf_cutoff = MAF_CUTOFF;
  par->maf_cutoff_parsed = false;
  maf_cutoff = MAF_CUTOFF; //V.1.4.mc
}
//--Pritam

void init_mutex()
{
  if(n_threads > 1)
  {
    pthread_mutex_init(&mutex_minsnp_linear,NULL);
    pthread_mutex_init(&mutex_minsnp_p_linear,NULL);
    pthread_mutex_init(&mutex_bf_linear,NULL);
    pthread_mutex_init(&mutex_vegas_linear,NULL);
    pthread_mutex_init(&mutex_bic_linear,NULL);

    pthread_mutex_init(&mutex_minsnp_logistic,NULL);
    pthread_mutex_init(&mutex_minsnp_p_logistic,NULL);
    pthread_mutex_init(&mutex_bf_logistic,NULL);
    pthread_mutex_init(&mutex_vegas_logistic,NULL);
    pthread_mutex_init(&mutex_bic_logistic,NULL);

    pthread_mutex_init(&mutex_other,NULL);
    pthread_mutex_init(&nthr_completed_mutex,NULL);

    pthread_cond_init(&nthr_completed_cond,NULL);

    pthread_mutex_init(&perm_loop_counter_mutex,NULL);
  }
}

void destroy_mutex()
{
  if(n_threads > 1)
  {
    pthread_mutex_destroy(&mutex_minsnp_linear);
    pthread_mutex_destroy(&mutex_minsnp_p_linear);
    pthread_mutex_destroy(&mutex_bf_linear);
    pthread_mutex_destroy(&mutex_vegas_linear);
    pthread_mutex_destroy(&mutex_bic_linear);

    pthread_mutex_destroy(&mutex_minsnp_logistic);
    pthread_mutex_destroy(&mutex_minsnp_p_logistic);
    pthread_mutex_destroy(&mutex_bf_logistic);
    pthread_mutex_destroy(&mutex_vegas_logistic);
    pthread_mutex_destroy(&mutex_bic_logistic);

    pthread_mutex_destroy(&mutex_other);
    pthread_mutex_destroy(&nthr_completed_mutex);
    pthread_cond_destroy(&nthr_completed_cond);

    pthread_mutex_destroy(&perm_loop_counter_mutex);
  }
}  

int checkArgs_1(PAR *par) //V.1.3.mc
{
  //should be called after set_precedence
  //bool f = GET_MINSNP_LINEAR || GET_BF_LINEAR || GET_VEGAS_LINEAR || GET_GENE_BIC_LINEAR || GET_MINSNP_P_PVAL_LINEAR;
  //bool f = GET_BF_LINEAR || GET_GENE_BIC_LINEAR;

  if(strcmp(par->mode, "summary") ==0)
  {
    SUMMARY=true;
    if(!par->n_sample_parsed)
       ESTIMATE_N = true;
    if(!par->pheno_var_parsed)
       ESTIMATE_PHENO_VAR = true; 
  }

    //if(f)
    //{
    //  if(!par->n_sample_parsed) //V.1.4.mc
    //  {
    //    printf("-need to specify --n-sample for summary mode if you specified any of:\n");
    //    printf("-gwis or bimbam \n");
    //    exit(1);
    //  }
    //}
    //SUMMARY=true;
 
  return 1;
}

int checkArgs(PAR *par)
{
  if(!par->mode_parsed){
    printf("need to specify --mode\n");
    exit(1);
  }

  if(strcmp(par->mode, "summary") ==0)
    SUMMARY=true;
  else if(strcmp(par->mode, "genotype") == 0){
    SUMMARY=false;

  }else{
    printf("--mode has to be \"genotype\" or \"summary\"\n");
    exit(1);
  }

  if(!SUMMARY)
  {
     if(IMPUTE2_input)
     {
        if(!par->impute2_geno_file_parsed || !par->impute2_info_file_parsed || !par->trait_file_parsed || !par->chr_parsed || !par->indivfile_parsed)
        {
          printf("-need to specify --impute2-geno-file, ---impute2-info-file, --indiv-file, --trait-file and --chr for mode = genotype\n");
          exit(1);
        }
     }
     else if(!par->tped_file_parsed || !par->snpinfo_file_parsed || !par->mlinfo_file_parsed || !par->trait_file_parsed || !par->chr_parsed || !par->indivfile_parsed)
     {
        printf("-need to specify --tped-file, ---snpinfo-file, --mlinfo-file, --indiv-file, --trait-file and --chr for mode = genotype\n");
        exit(1);
     }
  }
  else
  {
     if(!par->summary_file_parsed || !par->chr_parsed)
     {
        printf("-need to specify --summary-file and --chr for mode = summary\n");
        exit(1);
     }
  }

  //default values
  if(!par->random_seed_parsed){
    printf("-random-seed not set, use default value: 2\n");
    par->random_seed = 2;
  }
  else if(par->random_seed<=0)
  {
    printf("-WARNING : random-seed <= 0, use default value: 2\n");
    par->random_seed = 2;
  }

  if(!par->output_parsed){
    printf("-output not set, use default value: ./FAST.result\n");
    par->output="./FAST.result";
  }

  if(!par->gene_set_parsed){
    printf("-gene-set not set, no genes will be used\n");
    par->gene_set = NULL;
    work_on_genes = false;
  }

  if(!par->flank_parsed){
    printf("-flank not set, use default value: 20000\n");
    par->flank = 20000;
  } 
  else if(par->flank < 0)
  {
     printf("-WARNING : flank is < 0, setting it to default value: 20000\n"); 
     par->flank = 20000;
  }  

  if(!par->max_perm_parsed){
    printf("-max_perm not set, use default value: 1000000\n");
    par->max_perm = 1000000;
  }

  if(!par->sigma_a_parsed){
     par->sigma_a = 0.2;//default
     SIGMA_A = 0.2;
  }
  else if(par->sigma_a<=0)
  {
     printf("-WARNING : sigma-a <= 0, use default value: 0.2\n");
     par->sigma_a = 0.2;//default
     SIGMA_A = 0.2;
  }

  if(par->chr <= 0 || par->chr > 23) { printf("-Wrong chr value %d (should be between 1-22 for autosomes and 23 for chromosome-X) \nquitting",par->chr); exit(1);}

  if(!par->n_threads_parsed)
  {
    par->n_threads = 1;
    n_threads = 1;
  }
  else if(par->n_threads <= 0) { printf("-WARNING : n-threads is <= 0, setting it to 1\n"); par->n_threads = 1;n_threads = 1;}

  if(!par->skip_perm_parsed){
    par->skip_perm = false;
  }

  if(par->maf_cutoff_parsed && par->maf_cutoff < 0) 
  { 
     printf("-WARNING : maf-cutoff < 0, setting it to default %lg\n",MAF_CUTOFF); 
     par->maf_cutoff = MAF_CUTOFF;
     maf_cutoff = MAF_CUTOFF;
  }

  if(par->r2_cutoff_parsed && par->r2_cutoff < 0) 
  { 
     printf("-WARNING : imputation-quality cutoff < 0, setting it to default %lg\n",R2_CUTOFF); 
     par->r2_cutoff = R2_CUTOFF;
     r2_cutoff = R2_CUTOFF;
  }

  if(par->n_hat_cutoff_parsed && par->n_hat_cutoff < 0) 
  { 
     printf("-WARNING : eff-sample-size cutoff < 0, setting it to default %d\n",N_HAT_CUTOFF); 
     par->n_hat_cutoff = N_HAT_CUTOFF;
     n_hat_cutoff = N_HAT_CUTOFF;
  }

  if(!par->ncov_parsed){
    //printf("-num-covariates not set, use default value: 0\n");
    par->ncov = 0;
  }
  else
  {
    if(par->ncov > MAX_N_COVARIATES)
    {
       printf("-Too many covariates %d, max %d allowed\n quitting",par->ncov,MAX_N_COVARIATES);
       exit(1);
    }
    else if(par->ncov < 0)
    {
       printf("-WARNING : num-covariates is < 0, setting it to 0\n"); 
       par->ncov = 0;
    }
    else if(SUMMARY)
    {
        printf("-Covariates cannot be specified in mode=summary, covariates are ignored\n");
        par->ncov = 0;//V.1.7.mc
    }
  }

  if(par->scalepheno && par->quantilepheno)
  {
     printf("-WARNING : Both scaling and normal quantile transformations are specified for the phenotype, using scaling only\n\n");
     par->quantilepheno = false;
  }

  if(par->pheno_var_parsed && par->pheno_var <= 0) { printf("\n-Wrong pheno-var = %lg, \nquitting",par->pheno_var); exit(1);}
  if(par->n_sample_parsed && par->n_sample <= 0) { printf("\n-Wrong n-sample = %d, \nquitting",par->n_sample); exit(1);}
  if(par->missingness_parsed && par->missingness <= 0) 
  { 
    printf("-WARNING max-missingness value = %lg, setting to 0.05\n",par->missingness);
    par->missingness = 0.05;
    MAX_MISSINGNESS = 0.05;  
  }
  return 1;
}

//++V.1.2 FIX META 
void set_linear()
{
   //Basically, in summary mode, do only linear regression. If logistic options are specified, turn them off and turn on corresponding linear option.
   if(!SUMMARY)
      return;

   if(GET_MINSNP_PVAL_LOGISTIC)
   {
      GET_MINSNP_PVAL_LINEAR = true;
      GET_MINSNP_PVAL_LOGISTIC = false;
   }

   if(GET_MINSNP_P_PVAL_LOGISTIC)
   {
      GET_MINSNP_P_PVAL_LINEAR = true;
      GET_MINSNP_P_PVAL_LOGISTIC = false;
   }

   if(GET_BF_PVAL_LOGISTIC)
   {
      GET_BF_PVAL_LINEAR = true;
      GET_BF_PVAL_LOGISTIC = false;
   }

   if(GET_GENE_BIC_PVAL_LOGISTIC)
   {
      GET_GENE_BIC_PVAL_LINEAR = true;
      GET_GENE_BIC_PVAL_LOGISTIC = false;
   }

   if(GET_VEGAS_PVAL_LOGISTIC)
   {
      GET_VEGAS_PVAL_LINEAR = true;
      GET_VEGAS_PVAL_LOGISTIC = false;
   }

   if(GET_GATES_LOGISTIC)
   {
      GET_GATES_LINEAR = true;
      GET_GATES_LOGISTIC = false;
   }

   //-----------------//

   if(GET_MINSNP_LOGISTIC)
   {
      GET_MINSNP_LINEAR = true;
      GET_MINSNP_LOGISTIC = false;
   }

   if(GET_BF_LOGISTIC)
   {
      GET_BF_LINEAR = true;
      GET_BF_LOGISTIC = false;
   }

   if(GET_GENE_BIC_LOGISTIC)
   {
      GET_GENE_BIC_LINEAR = true;
      GET_GENE_BIC_LOGISTIC = false;
   }

   if(GET_VEGAS_LOGISTIC)
   {
      GET_VEGAS_LINEAR = true;
      GET_VEGAS_LOGISTIC = false;
   }
}
//--V.1.2 FIX META 
//
void turn_off_bimbam_gwis()
{
  printf("\nFor mode=Summary : Cannot do GWiS or Bimbam as sample size is not specified\n\n");
  GET_GENE_BIC_LINEAR = false;
  GET_GENE_BIC_PVAL_LINEAR = false;
  GET_BF_LINEAR = false;
  GET_BF_PVAL_LINEAR = false;
  GET_GENE_BIC_LOGISTIC = false;
  GET_GENE_BIC_PVAL_LOGISTIC = false;
  GET_BF_LOGISTIC = false;
  GET_BF_PVAL_LOGISTIC = false;
}

void set_precedence(PAR * par)
{
   if(SUMMARY)
      set_linear();

   //If and PVAL option is specified, the corresponding option is automatically set to true.
   if(GET_MINSNP_PVAL_LINEAR)
      GET_MINSNP_LINEAR = true;

   if(GET_MINSNP_PVAL_LOGISTIC)
      GET_MINSNP_LOGISTIC = true;

   if(GET_BF_PVAL_LINEAR)
      GET_BF_LINEAR = true;

   if(GET_BF_PVAL_LOGISTIC)
      GET_BF_LOGISTIC = true;

   if(GET_VEGAS_PVAL_LINEAR)
      GET_VEGAS_LINEAR = true;

   if(GET_VEGAS_PVAL_LOGISTIC)
      GET_VEGAS_LOGISTIC = true;

   if(GET_GENE_BIC_PVAL_LINEAR)
      GET_GENE_BIC_LINEAR = true;

   if(GET_GENE_BIC_PVAL_LOGISTIC)
      GET_GENE_BIC_LOGISTIC = true;

   if(SUMMARY)
   {
      //can do gwis, vegas, gates, bf, minsnp, minsnp-p for linear.
      //can do vegas, gates, minsnp, minsnp-p for logistic.
    
      bool f = GET_GATES_LINEAR || GET_GATES_LOGISTIC || GET_GENE_BIC_LINEAR || GET_GENE_BIC_PVAL_LINEAR || GET_VEGAS_LINEAR || GET_VEGAS_PVAL_LINEAR;
      f = f || GET_MINSNP_LINEAR || GET_MINSNP_PVAL_LINEAR || GET_MINSNP_P_PVAL_LINEAR;
      f = f || GET_MINSNP_LOGISTIC || GET_MINSNP_PVAL_LOGISTIC || GET_MINSNP_P_PVAL_LOGISTIC;
      f = f || GET_VEGAS_LOGISTIC || GET_VEGAS_PVAL_LOGISTIC;
      f = f || GET_BF_LINEAR || GET_BF_PVAL_LINEAR;

      bool p = GET_GENE_BIC_PVAL_LINEAR || GET_VEGAS_PVAL_LINEAR || GET_MINSNP_PVAL_LINEAR || GET_MINSNP_P_PVAL_LINEAR || GET_VEGAS_PVAL_LOGISTIC;
           p = p || GET_MINSNP_PVAL_LOGISTIC || GET_MINSNP_P_PVAL_LOGISTIC || GET_BF_PVAL_LINEAR; //V.1.2 BF FIX

      NEED_SNP_LINEAR_SUMMARY = false;
      NEED_SNP_LINEAR_SUMMARY = GET_GATES_LINEAR || GET_GENE_BIC_LINEAR || GET_GENE_BIC_PVAL_LINEAR || GET_VEGAS_LINEAR || GET_VEGAS_PVAL_LINEAR;
      NEED_SNP_LINEAR_SUMMARY |= GET_MINSNP_LINEAR || GET_MINSNP_PVAL_LINEAR || GET_MINSNP_P_PVAL_LINEAR || GET_BF_LINEAR || GET_BF_PVAL_LINEAR; 

      NEED_SNP_LOGISTIC_SUMMARY = false;
      NEED_SNP_LOGISTIC_SUMMARY = GET_GATES_LOGISTIC || GET_VEGAS_LOGISTIC || GET_VEGAS_PVAL_LOGISTIC;
      NEED_SNP_LOGISTIC_SUMMARY |= GET_MINSNP_LOGISTIC || GET_MINSNP_PVAL_LOGISTIC || GET_MINSNP_P_PVAL_LOGISTIC;

      if(f==false)
      {
         printf("-no appropriate methods specified in summary mode, quitting.....\n");
         printf("-must specify one/more of gwis/bimbam/vegas/gates/minsnp/minsnp-gene methods\n");
         exit(1);
      }

      work_on_genes = true;

      if(SKIPPERM)
      {
         printf("-No permutations to be performed\n");
         GET_GENE_BIC_PVAL_LINEAR = false;
         GET_VEGAS_PVAL_LINEAR = GET_VEGAS_PVAL_LOGISTIC = false;
         GET_MINSNP_PVAL_LINEAR = GET_MINSNP_PVAL_LOGISTIC = false;
         GET_MINSNP_P_PVAL_LINEAR = GET_MINSNP_P_PVAL_LOGISTIC = false;
         GET_BF_PVAL_LINEAR = false;
      }

      //switch all logistic off.
      //V.1.2 FIX META 
      //if(GET_BF_LOGISTIC || GET_BF_PVAL_LOGISTIC) printf("\n-WARNING : Bayes Factor logistic regression cannot run in summary mode, ignored\n\n");
      //if(GET_GENE_BIC_LOGISTIC || GET_GENE_BIC_PVAL_LOGISTIC) printf("\n-WARNING : GWiS logistic regression cannot run in summary mode, ignored\n\n");
      
      GET_SINGLE_SNP_LOGISTIC = false; 
      GET_BF_LOGISTIC = false;
      GET_BF_PVAL_LOGISTIC = false;
      GET_GENE_BIC_LOGISTIC = false;
      GET_GENE_BIC_PVAL_LOGISTIC = false;
 
      //only this linear is off.
      GET_SINGLE_SNP_LINEAR = false; 

      if(!p)
         SKIPPERM = true;

      return;
   }

   //for conflicting options specified, set a precedence of the options.
   if(!GET_MINSNP_PVAL_LINEAR && !GET_MINSNP_PVAL_LOGISTIC && !GET_MINSNP_P_PVAL_LINEAR && !GET_MINSNP_P_PVAL_LOGISTIC && !GET_GENE_BIC_PVAL_LINEAR 
      && !GET_GENE_BIC_PVAL_LOGISTIC && !GET_BF_PVAL_LINEAR && !GET_BF_PVAL_LOGISTIC && !GET_VEGAS_PVAL_LINEAR && !GET_VEGAS_PVAL_LOGISTIC)
   {
      //no permutation options are specified.
      SKIPPERM = true;
   }

   //SKIPPERM = true will always override and other permutation options.
   //So, if SKIPPERM is true, no permutations will be performed at all.
   if(SKIPPERM)
   {
      //block all permutations.
      //report parametric pvalues for min snp.
      GET_MINSNP_PVAL_LINEAR = GET_MINSNP_PVAL_LOGISTIC = false;
      GET_MINSNP_P_PVAL_LINEAR = GET_MINSNP_P_PVAL_LOGISTIC = false; //block min snp p.
      GET_GENE_BIC_PVAL_LINEAR = GET_GENE_BIC_PVAL_LOGISTIC = false;
      GET_BF_PVAL_LINEAR = GET_BF_PVAL_LOGISTIC = false;
      GET_VEGAS_PVAL_LINEAR = GET_VEGAS_PVAL_LOGISTIC = false;
   }

   NEED_SNP_LINEAR = false; //need to do single snp linear regression ?
   if(GET_SINGLE_SNP_LINEAR || GET_BF_LINEAR || GET_VEGAS_LINEAR || GET_GENE_BIC_LINEAR || GET_MINSNP_LINEAR || GET_MINSNP_P_PVAL_LINEAR || GET_GATES_LINEAR)
      NEED_SNP_LINEAR = true;

   NEED_SNP_LOGISTIC = false; //need to do single snp logistic regression ?
   if(GET_SINGLE_SNP_LOGISTIC || GET_BF_LOGISTIC || GET_VEGAS_LOGISTIC || GET_GENE_BIC_LOGISTIC || GET_MINSNP_LOGISTIC || GET_MINSNP_P_PVAL_LOGISTIC || GET_GATES_LOGISTIC)
      NEED_SNP_LOGISTIC = true;
 
   //need to work on genes ? 
   work_on_genes = false; 
   work_on_genes = GET_BF_LINEAR || GET_VEGAS_LINEAR || GET_GENE_BIC_LINEAR || GET_MINSNP_LINEAR || GET_MINSNP_P_PVAL_LINEAR || GET_GATES_LINEAR;
   work_on_genes |= GET_BF_LOGISTIC || GET_VEGAS_LOGISTIC || GET_GENE_BIC_LOGISTIC || GET_MINSNP_LOGISTIC || GET_MINSNP_P_PVAL_LOGISTIC || GET_GATES_LOGISTIC;
   work_on_genes |= (par->single_snp_linear_gene || par->single_snp_logistic_gene);

   if(!NEED_SNP_LINEAR && !NEED_SNP_LOGISTIC)
   {
      //no options are set, default is single snp linear regression.
      GET_SINGLE_SNP_LINEAR = true;
      NEED_SNP_LINEAR = true;
   }

   if(NEED_SNP_LINEAR)
      printf("-will perform linear regression based analysis\n");

   if(NEED_SNP_LOGISTIC)
      printf("-will perform logistic regression based analysis\n");

   if(work_on_genes)
      printf("-will perform gene based association analysis\n");
   else
      printf("-skipping gene based association analysis, will perform single snp association analyses\n");
   if(SKIPPERM)
      printf("-skipping permutations of any method specified\n");
}

void print_what_should_run()
{
   bool f = false;
   printf("--------These should be running ----------\n");
   if(NEED_SNP_LINEAR)
   {
      if(GET_SINGLE_SNP_LINEAR)       {printf("      ->  Single snp linear regression\n");f=true;}

      if(GET_MINSNP_PVAL_LINEAR)      {printf("      ->  Min Snp linear regression with permutations\n");f=true;}
      else if(GET_MINSNP_LINEAR)      {printf("      ->  Min Snp linear regression\n");f=true;}

      if(GET_MINSNP_P_PVAL_LINEAR)    {printf("      ->  Min Snp Gene linear regression with permutations\n");f=true;}

      if(GET_GENE_BIC_PVAL_LINEAR)    {printf("      ->  GWiS linear regression with permutations\n");f=true;}
      else if(GET_GENE_BIC_LINEAR)    {printf("      ->  GWiS linear regression\n");f=true;}

      if(GET_BF_PVAL_LINEAR)          {printf("      ->  Bayes Factor linear regression with permutations\n");f=true;}
      else if(GET_BF_LINEAR)          {printf("      ->  Bayes Factor linear regression\n");f=true;}

      if(GET_VEGAS_PVAL_LINEAR)       {printf("      ->  Vegas linear regression with permutations\n");f=true;}
      else if(GET_VEGAS_LINEAR)       {printf("      ->  Vegas linear regression\n");f=true;}

      if(GET_GATES_LINEAR)            {printf("      ->  Gates linear regression\n");f=true;}
   }
   if(NEED_SNP_LOGISTIC)
   { 
      if(GET_SINGLE_SNP_LOGISTIC)     {printf("      ->  Single snp logistic regression\n");f=true;}

      if(GET_MINSNP_PVAL_LOGISTIC)    {printf("      ->  Min Snp logistic regression with permutations\n");f=true;}
      else if(GET_MINSNP_LOGISTIC)    {printf("      ->  Min Snp logistic regression\n");f=true;}

      if(GET_MINSNP_P_PVAL_LOGISTIC)  {printf("      ->  Min Snp Gene logistic regression with permutations\n");f=true;}

      if(GET_GENE_BIC_PVAL_LOGISTIC)  {printf("      ->  GWiS logistic regression with permutations\n");f=true;}
      else if(GET_GENE_BIC_LOGISTIC)  {printf("      ->  GWiS logistic regression\n");f=true;}

      if(GET_BF_PVAL_LOGISTIC)        {printf("      ->  Bayes Factor logistic regression with permutations\n");f=true;}
      else if(GET_BF_LOGISTIC)        {printf("      ->  Bayes Factor logistic regression\n");f=true;}

      if(GET_VEGAS_PVAL_LOGISTIC)     {printf("      ->  Vegas logistic regression with permutations\n");f=true;}
      else if(GET_VEGAS_LOGISTIC)     {printf("      ->  Vegas logistic regression\n");f=true;}

      if(GET_GATES_LOGISTIC)          {printf("      ->  Gates logistic regression\n");f=true;}
   }
   //V.1.2 FIX META
   if(SUMMARY)
   {
      if(GET_BF_PVAL_LINEAR)          {printf("      ->  Bayes Factor and permutations with summary\n");f=true;}
      else if(GET_BF_LINEAR)          {printf("      ->  Bayes Factor with summary\n");f=true;}

      if(GET_GENE_BIC_PVAL_LINEAR)    {printf("      ->  GWiS and permutations with summary\n");f=true;}
      else if(GET_GENE_BIC_LINEAR)    {printf("      ->  GWiS with summary\n");f=true;}

      if(GET_VEGAS_PVAL_LINEAR)       {printf("      ->  Vegas and permutations with summary\n");f=true;}
      else if(GET_VEGAS_LINEAR)       {printf("      ->  Vegas with summary\n");f=true;}

      if(GET_GATES_LINEAR || GET_GATES_LOGISTIC)        {printf("      ->  Gates with summary\n");f=true;}
      if(GET_MINSNP_PVAL_LINEAR)      {printf("      ->  Min Snp and permutations with summary\n");f=true;}

      else if(GET_MINSNP_LINEAR)      {printf("      ->  Min Snp with summary\n");f=true;}
      if(GET_MINSNP_P_PVAL_LINEAR)    {printf("      ->  Min Snp Gene and permutations with summary\n");f=true;}

      if(GET_VEGAS_PVAL_LOGISTIC)     {printf("      ->  Vegas and permutations with summary\n");f=true;}
      else if(GET_VEGAS_LOGISTIC)     {printf("      ->  Vegas with summary\n");f=true;}

      if(GET_MINSNP_PVAL_LOGISTIC)    {printf("      ->  Min Snp and permutations with summary\n");f=true;}
      else if(GET_MINSNP_LOGISTIC)    {printf("      ->  Min Snp with summary\n");f=true;}

      if(GET_MINSNP_P_PVAL_LOGISTIC)  {printf("      ->  Min Snp Gene and permutations with summary\n");f=true;}
   }
   if(!SUMMARY)
   { 
      if(!f)
      {
         printf("-no appropriate methods to run in genotype mode, quitting.....\n");
         exit(1);
      }
   }
   else
   {
      if(!f)
      {
         printf("-no appropriate methods to run in summary mode, quitting.....\n");
         exit(1);
      }
   }
   printf("------------------------------------------\n");
}

void print_all_values(PAR par) 
{
   printf("NO_SIGN_FLIP = %d\n",NO_SIGN_FLIP);
   printf("BOUND_BETA = %d\n",BOUND_BETA);  
   printf("maf_cutoff = %lg\n",maf_cutoff);
   printf("n_hat_cutoff = %d\n",n_hat_cutoff);
   printf("r2_cutoff = %lg\n",r2_cutoff);
   printf("SIGMA_A = %lg\n",SIGMA_A);
   printf("SKIPPERM = %d\n",SKIPPERM);
   printf("IMPUTE2_input = %d\n",IMPUTE2_input);
   printf("COMPUTE_LD = %d\n",COMPUTE_LD);
   printf("ESTIMATE_PHENO_VAR = %d\n",ESTIMATE_PHENO_VAR);
   printf("ESTIMATE_N = %d\n",ESTIMATE_N);
   printf("USE_WEIGHTED_LD = %d\n",USE_WEIGHTED_LD);
   printf("MAX_MISSINGNESS = %lg\n",MAX_MISSINGNESS);
   printf("MISSING_VAL = %lg\n",MISSING_VAL);
   printf("N_PERM = %d\n",N_PERM);
   printf("FLANK = %d\n",FLANK);
   printf("SUMMARY = %d\n",SUMMARY);
   printf("n_threads = %d\n",n_threads);

   printf("chr = %d\n",par.chr);
   printf("pheno_var = %lg\n",par.pheno_var);
   printf("nsample = %d\n",par.n_sample);
   printf("maf-cutoff = %lg\n",par.maf_cutoff);
   printf("num-covariates = %d\n",par.ncov);
   printf("random_seed = %d\n",par.random_seed);
}

//main function
int main(int nARG, char *ARGV[])
{
  PAR par;

  //++urgent
  bool ldfile_decomp = false;
  bool hapfile_decomp = false;
  bool metafile_decomp = false;
  //--urgent

  OUTFILE outfile;
  printf("-FAST : tool for gene based Fast Association Tests, version 1.8.mc\n");
  printf("-Pritam Chanda, Hailiang Huang, Dan Arking and Joel Bader\n");
  printf("-Available from https://bitbucket.org/pchanda/fast/downloads\n");
  printf("-Parsing input parameters :\n");
  init_par(&par,&outfile);
  parseArgs(nARG, ARGV, &par);
  checkArgs(&par);
  printf("-Num Threads = %d\n",n_threads);
  int ncpu = numCPU_linux();
  printf("-Num of available cores = %d\n",ncpu);
  printf("-MAF cutoff = %lg\n",maf_cutoff);
  if(ncpu < n_threads)
  {
     printf("\n-Warning : requested no. of threads %d > no. of available cores %d\n\n",n_threads,ncpu);//V.1.7.mc
  }

  N_PERM=par.max_perm;
  N_PERM_MIN=par.n_perm_min;
  N_CUTOFF = 0.5*N_PERM_MIN;

  printf("min-perm = %d cutoff = %d\n",N_PERM_MIN,N_CUTOFF);

  if(N_PERM<N_PERM_MIN)
  {
     printf("-Minimum of %d permutations will be performed\n",N_PERM_MIN);
     N_PERM = N_PERM_MIN;
  }
  FLANK=par.flank;
  SKIPPERM=par.skip_perm;
  //exit(0);

  set_precedence(&par);
  getEnv();
  checkArgs_1(&par); //V.1.3.mc
 
  if(SUMMARY) //V.1.4.mc
  {
     NO_SIGN_FLIP = true;
     BOUND_BETA = true;
  }
  else
  {
     NO_SIGN_FLIP = false;
     BOUND_BETA = false;
  }

  if(VERBOSE)
     print_all_values(par);
 
  init_mutex();
  time_t start_time;
  time(&start_time);

  if(par.gene_set==NULL)
  {
     if(work_on_genes)
     {
        printf("-No gene set provided for the methods requested to be run\n");
        printf(" quitting....\n");
        exit(1);
     }
     work_on_genes = false;
  }

  printf("-to read %d covariates, flank = %d\n",par.ncov,FLANK);

  // input files
  char indiv_id_file[MAX_FILENAME_LEN] = "";
  char tfam_file[MAX_FILENAME_LEN] = "";
  char refseq_file[MAX_FILENAME_LEN]="";
  char mlinfo_file[MAX_FILENAME_LEN]="";
  char tped_file[MAX_FILENAME_LEN]="";
  char snp_info_file[MAX_FILENAME_LEN]="";

  //V.1.5.mc, support for impute2 input files.
  char impute2_geno_file[MAX_FILENAME_LEN]="";
  char impute2_info_file[MAX_FILENAME_LEN]="";

  char allele_info_file[MAX_FILENAME_LEN]=""; //for SUMMARY data to provide allele information

  // output files
  // all snps
  char allSNP_file_linear[MAX_FILENAME_LEN]="";   //single snp linear parametric pvalue
  char allSNP_file_logistic[MAX_FILENAME_LEN]=""; //single snp logistic parametric pvalue 

  //min snp
  char snp_pval_file_linear[MAX_FILENAME_LEN]=""; //minsnp linear permutation pvalue         
  char snp_pval_file_logistic[MAX_FILENAME_LEN]=""; //minsnp linear permutation pvalue         

  //min snp p
  char snp_bonf_pval_file_linear[MAX_FILENAME_LEN]=""; //minsnp p linear permutation pvalue
  char snp_bonf_pval_file_logistic[MAX_FILENAME_LEN]=""; //minsnp p logistic permutation pvalue

  //bf and optional permutation pvalue 
  char bf_pval_file_linear[MAX_FILENAME_LEN]="";
  char bf_pval_file_logistic[MAX_FILENAME_LEN]="";

  //vegas score and optional permutation pvalue
  char vegas_pval_file_linear[MAX_FILENAME_LEN]="";
  char vegas_pval_file_logistic[MAX_FILENAME_LEN]="";

  //gates score and parametric pvalue
  char gates_pval_file_linear[MAX_FILENAME_LEN]="";
  char gates_pval_file_logistic[MAX_FILENAME_LEN]="";

  char gene_SNP_file[MAX_FILENAME_LEN]="";

  //BIC linear
  //char BIC_linear_result_file[MAX_FILENAME_LEN]="";
  char BIC_linear_perm_result_file[MAX_FILENAME_LEN]="";

  //BIC logistic
  //char BIC_logistic_result_file[MAX_FILENAME_LEN]="";
  char BIC_logistic_perm_result_file[MAX_FILENAME_LEN]="";
  
  //char LD_file[MAX_FILENAME_LEN]="";
  char EXCLUDE_file[MAX_FILENAME_LEN]="";
  char META_file[MAX_FILENAME_LEN]=""; //meta-analysis input file.

  char log_file[MAX_FILENAME_LEN]=""; //V.1.5.mc

  //check if the output files can be created.
  if(!file_can_be_created(par.output))
  {
     printf("-cannot create output file in %s\n",par.output);
     exit(1);
  }

  //input files
  if(work_on_genes)
     strcpy(refseq_file, par.gene_set);

  if(!SUMMARY){
    strcpy(tfam_file, par.trait_file);
    sprintf(indiv_id_file, "%s", par.indivfile);
    //V.1.5.mc
    if(IMPUTE2_input==true)
    {
      sprintf(impute2_geno_file, "%s", par.impute2_geno_file);
      sprintf(impute2_info_file, "%s", par.impute2_info_file);
    }
    else
    {
      sprintf(tped_file, "%s", par.tped_file);
      sprintf(snp_info_file, "%s", par.snpinfo_file);
      sprintf(mlinfo_file, "%s", par.mlinfo_file);
    }
  }else{
    sprintf(EXCLUDE_file,  "%s", par.multipos_file);
    sprintf(META_file,"%s",par.summary_file);
    //sprintf(LD_file,  "%s.ld", par.ldfile);
    sprintf(allele_info_file, "%s", par.allelefile);
  }

  //output files
  //++V.1.2 FIX META 
  if(!SUMMARY) 
  {
     sprintf(allSNP_file_linear,"%s.chr%d.%s", par.output,par.chr,"allSNP.Linear.txt"); //Pritam
     sprintf(allSNP_file_logistic,"%s.chr%d.%s", par.output,par.chr, "allSNP.Logistic.txt"); //Pritam

     sprintf(snp_pval_file_linear,"%s.chr%d.%s", par.output,par.chr, "minSNP.Linear.txt");
     sprintf(snp_pval_file_logistic,"%s.chr%d.%s", par.output,par.chr, "minSNP.Logistic.txt");

     sprintf(snp_bonf_pval_file_linear,"%s.chr%d.%s", par.output,par.chr, "minSNP_Gene.Linear.txt");
     sprintf(snp_bonf_pval_file_logistic,"%s.chr%d.%s", par.output,par.chr, "minSNP_Gene.Logistic.txt");

     sprintf(bf_pval_file_linear,"%s.chr%d.%s", par.output,par.chr, "BF.Linear.txt");
     sprintf(bf_pval_file_logistic,"%s.chr%d.%s", par.output,par.chr, "BF.Logistic.txt");

     sprintf(vegas_pval_file_linear,"%s.chr%d.%s", par.output,par.chr,  "Vegas.Linear.txt");
     sprintf(vegas_pval_file_logistic,"%s.chr%d.%s", par.output,par.chr,  "Vegas.Logistic.txt");

     sprintf(gates_pval_file_linear,"%s.chr%d.%s", par.output,par.chr,  "Gates.Linear.txt");
     sprintf(gates_pval_file_logistic,"%s.chr%d.%s", par.output,par.chr,  "Gates.Logistic.txt");

     sprintf(gene_SNP_file, "%s.chr%d.%s",par.output, par.chr,"geneSNP.txt");
     sprintf(log_file, "%s.chr%d.%s",par.output, par.chr,"log.txt"); //V.1.5.mc

     sprintf(BIC_linear_perm_result_file, "%s.chr%d.%s",par.output,par.chr,   "GWiS.Linear.txt");
     sprintf(BIC_logistic_perm_result_file, "%s.chr%d.%s",par.output,par.chr, "GWiS.Logistic.txt");
  }
  else
  {
     sprintf(allSNP_file_linear,"%s.chr%d.%s", par.output,par.chr, "allSNP.Summary.txt"); //Pritam
     sprintf(allSNP_file_logistic,"%s.chr%d.%s", par.output,par.chr, "allSNP.Summary.txt"); //Pritam

     sprintf(snp_pval_file_linear,"%s.chr%d.%s", par.output,par.chr, "minSNP.Summary.txt");
     sprintf(snp_pval_file_logistic,"%s.chr%d.%s", par.output,par.chr, "minSNP.Summary.txt");

     sprintf(snp_bonf_pval_file_linear,"%s.chr%d.%s", par.output,par.chr, "minSNP_Gene.Summary.txt");
     sprintf(snp_bonf_pval_file_logistic,"%s.chr%d.%s", par.output,par.chr, "minSNP_Gene.Summary.txt");

     sprintf(bf_pval_file_linear,"%s.chr%d.%s", par.output,par.chr, "BF.Summary.txt");
     sprintf(bf_pval_file_logistic,"%s.chr%d.%s", par.output,par.chr, "BF.Summary.txt");

     sprintf(vegas_pval_file_linear,"%s.chr%d.%s", par.output,par.chr,  "Vegas.Summary.txt");
     sprintf(vegas_pval_file_logistic,"%s.chr%d.%s", par.output,par.chr,  "Vegas.Summary.txt");

     sprintf(gates_pval_file_linear,"%s.chr%d.%s", par.output,par.chr,  "Gates.Summary.txt");
     sprintf(gates_pval_file_logistic,"%s.chr%d.%s", par.output,par.chr,  "Gates.Summary.txt");

     sprintf(gene_SNP_file, "%s.chr%d.%s",par.output,par.chr, "geneSNP.txt");
     sprintf(log_file, "%s.chr%d.%s",par.output, par.chr,"log.txt"); //V.1.5.mc

     sprintf(BIC_linear_perm_result_file, "%s.chr%d.%s",par.output,par.chr,   "GWiS.Summary.txt");
     sprintf(BIC_logistic_perm_result_file, "%s.chr%d.%s",par.output,par.chr, "GWiS.Summary.txt");
  }
  //--V.1.2 FIX META 
  
  //read genes from CCDS.txt or similar file
  GENE* gene = NULL;
  int nGene = 0;
  if(work_on_genes)
  {
     if(!file_exists(refseq_file))
     {
       if(refseq_file!=NULL)
          printf("-cannot find gene-set file [%s] , please provide one with --gene-set option\nquitting....\n",refseq_file);
       else
          printf("-cannot find gene-set file , please provide one with --gene-set option\nquitting....\n");
       exit(1);
     }
     nGene = readGene(refseq_file, &gene, par.chr);
     printf("-%d total genes loaded\n", nGene);
     if(nGene == 0){
       printf("-Time: %s", getTime());
       return EXIT_SUCCESS;
     }
  }

  P2 = NULL;
  P3 = NULL;
  P4 = NULL;
  P5 = NULL;
  P6 = NULL;

  //open input files

  //V.1.5.mc
  FILE *fp_impute2_geno =NULL;
  FILE *fp_impute2_info = NULL;

  FILE *fp_tped =NULL;
  FILE *fp_snp_info = NULL;
  FILE *fp_mlinfo =NULL;

  FILE *fp_pos=NULL; //V.1.4.mc
  FILE *fp_hap = NULL; //V.1.4.mc

  FILE *fp_allele_info =NULL; //for SUMMARY

  FILE *fp_ld=NULL;
  FILE *fp_exclude=NULL;
  FILE *fp_frq=NULL;
  int ncols = 0; //V.1.4.mc

  if(!SUMMARY)
  {
    //V.1.5.mc
    if(IMPUTE2_input==false)
    {
       //file with the genotype
       if(file_exists(tped_file))
       {
          if(!checkgz(tped_file))
             fp_tped = fopen(tped_file, "r");
          else
             fp_tped = gzopen(tped_file);
       }
       else
       {
          if(tped_file!=NULL)
             printf("Cannot find tped file %s, quitting\n",tped_file);
          else
             printf("Cannot find tped file, quitting\n");
          exit(1);
       }

       char sline[MAX_LINE_WIDTH];
       strcpy(sline, "");
       //file with info about SNPs
       if(file_exists(snp_info_file))
       {
         if(!checkgz(snp_info_file))
            fp_snp_info = fopen(snp_info_file, "r");
         else
         {
            fp_snp_info = gzopen(snp_info_file);
         }
       }
       else
       {
          if(snp_info_file!=NULL)
            printf("Cannot find snp info file %s, quitting\n",snp_info_file);
          else
            printf("Cannot find snp info file, quitting\n");
          exit(1);
       }
       if(fp_snp_info != NULL)
       {
         //printf("fp_snp_info = %p\n",fp_snp_info);
         fgets(sline, 1000, fp_snp_info);
         //printf("sline=[%s]\n",sline);
         if(sline[0]!='#') {printf("-Header line missing in snp_info file, please add one starting with '#'\n");exit(1);}
       }
       //file with more info about SNPs
       if(file_exists(mlinfo_file))
       {
          if(!checkgz(mlinfo_file))
             fp_mlinfo= fopen(mlinfo_file, "r");
          else
             fp_mlinfo= gzopen(mlinfo_file);
       }
       else
       {
         if(mlinfo_file!=NULL)
            printf("Cannot find mlinfo file %s, quitting\n",mlinfo_file);
         else
            printf("Cannot find mlinfo file, quitting\n");
         exit(1);
       }
       //skip the header line
       strcpy(sline, "");
       if(fp_mlinfo != NULL)
         fgets(sline, MAX_LINE_WIDTH, fp_mlinfo);
       if(sline[0]!='#') {printf("-Header line missing in mlinfo file, please add one starting with '#'\n");exit(1);}
    }
    else
    {
       //impute2 geno file 
       if(file_exists(impute2_geno_file))
       {
          if(!checkgz(impute2_geno_file))
             fp_impute2_geno = fopen(impute2_geno_file, "r");
          else
             fp_impute2_geno = gzopen(impute2_geno_file);
       }
       else
       {
          if(impute2_geno_file!=NULL)
             printf("Cannot find impute2 geno file %s, quitting\n",impute2_geno_file);
          else
             printf("Cannot find impute2 geno file, quitting\n");
          exit(1);
       }
       char sline[MAX_LINE_WIDTH];
       strcpy(sline, "");
       //file with info about SNPs
       if(file_exists(impute2_info_file))
       {
          if(!checkgz(impute2_info_file))
             fp_impute2_info = fopen(impute2_info_file, "r");
          else
             fp_impute2_info = gzopen(impute2_info_file);
       }
       else
       {
          if(impute2_info_file!=NULL)
             printf("Cannot find impute2 info file %s, quitting\n",impute2_info_file);
          else
            printf("Cannot find impute2 info file, quitting\n");
          exit(1);
       }
       if(fp_impute2_info != NULL)
       {
         fgets(sline, MAX_LINE_WIDTH, fp_impute2_info);//strip header
       }
    }
  }
  else
  {
    char sline_ld[MAX_LINE_WIDTH]="";

    if(file_exists(par.ldfile))
    {
       //++urgent
       if(checkgz(par.ldfile))
       {
          char * newf = decompress(par.output,par.chr,par.ldfile,"ld");
          strcpy(par.ldfile,newf);
          ldfile_decomp = true;
       }
       fp_ld=fopen(par.ldfile,"r");
       //--urgent

       fgets(sline_ld, MAX_LINE_WIDTH, fp_ld); //get rid of header.
       if(sline_ld[0]!='#')
       {
          printf("-Warning : Header line missing in LD file, please add one starting with '#'\n");
          fclose(fp_ld);
          if(!checkgz(par.ldfile))
             fp_ld=fopen(par.ldfile,"r");
          else
             fp_ld=gzopen(par.ldfile);
       }
    }
    else if(!COMPUTE_LD)
    {
       if(par.ldfile!=NULL)
          printf("\n **** Warning : Cannot find LD file %s, assuming no LD between snps *** \n\n",par.ldfile); //NO LD BUG
       else
          printf("\n **** Warning : Cannot find LD file, assuming no LD between snps *** \n\n"); //NO LD BUG
       fp_ld = NULL;
    }

    if(file_exists(EXCLUDE_file))
    {
       if(!checkgz(EXCLUDE_file))
          fp_exclude=fopen(EXCLUDE_file,"r");
       else
          fp_exclude=gzopen(EXCLUDE_file);
    }
    else
    {
       if(EXCLUDE_file!=NULL)
          printf("-Cannot find multipos file %s, assuming no SNPs have ambiguous positions\n",EXCLUDE_file);
       else
          printf("-Cannot find multipos file, assuming no SNPs have ambiguous positions\n");
       fp_exclude = NULL;
    }

    if(COMPUTE_LD) //+V.1.4.mc
    {
      printf("-Will compute LD on the fly using haplotypes\n");
      if(file_exists(par.posfile))
      {
       bool isgz = false; //BIG BUG
       if(!checkgz(par.posfile))
          fp_pos=fopen(par.posfile,"r");
       else
       {
          isgz = true;
          fp_pos=gzopen(par.posfile);
       }
       fgets(sline_ld, MAX_LINE_WIDTH, fp_pos); //get rid of header.
       if(sline_ld[0]!='#')
       {
         printf("-Warning : Header line missing in haplotype position file, please add one starting with '#'\n");
         if(!isgz)
           fclose(fp_pos);
         else
           pclose(fp_pos);
         if(!checkgz(par.posfile))
            fp_pos=fopen(par.posfile,"r");
         else
            fp_pos=gzopen(par.posfile);
       }
      }
      else
      {
         if(par.posfile!=NULL)
            printf("COMPUTE_LD is true : Cannot find haplotype index position file %s, quitting\n",par.posfile);
         else
            printf("COMPUTE_LD is true : Cannot find haplotype index position file, quitting\n");
         exit(1);
      }

      if(file_exists(par.hapfile))
      {
         //++urgent
         if(checkgz(par.hapfile))
         {
            char* newf = decompress(par.output,par.chr,par.hapfile,"hap");
            strcpy(par.hapfile,newf);
            hapfile_decomp = true;
         }
         fp_hap=fopen(par.hapfile,"r");
         //--urgent

         fpos_t prevPos;
         fgetpos(fp_hap, &prevPos);
         fgets(sline_ld, MAX_LINE_WIDTH, fp_hap);
         ncols = count_columns(sline_ld);
         fsetpos(fp_hap, &prevPos);
         fclose(fp_hap);

         if(!checkgz(par.hapfile))
            fp_hap=fopen(par.hapfile,"r");
         else
            fp_hap=gzopen(par.hapfile);
      } 
      else
      {
         if(par.hapfile!=NULL)
           printf("COMPUTE_LD is true : Cannot find SNP haplotype file %s, quitting\n",par.hapfile);
         else
           printf("COMPUTE_LD is true : Cannot find SNP haplotype file, quitting\n");
         exit(1);
      }

      if(USE_WEIGHTED_LD)
      { 
        if(file_exists(par.hap_wt_file))
        {
          get_haplotype_weights(par.hap_wt_file, ncols);
        }
        else
        {
           if(par.hap_wt_file!=NULL)
              printf("USE_WEIGHTED_LD is true : Cannot find haplotype weights file %s, quitting\n",par.hap_wt_file);
           else
              printf("USE_WEIGHTED_LD is true : Cannot find haplotype weights file, quitting\n");
          exit(1);
        }
      } 
    }//-V.1.4.mc

    if(file_exists(META_file))
    {
       //++ urgent
       if(checkgz(META_file))
       {
          char* newf = decompress(par.output,par.chr,META_file,"hap");
          strcpy(META_file,newf);
          metafile_decomp = true;
       }
       fp_frq=fopen(META_file,"r");
       //-- urgent

       char sline[MAX_LINE_WIDTH];
       fgets(sline, MAX_LINE_WIDTH, fp_frq);
       if(sline[0]!='#') {printf("-Header line missing in summary file, please add one starting with '#'\n");exit(1);}

       //find out if the summary file is in simple format : chr snp bp pval.
       int n_tok = count_tokens(sline);
       if(n_tok==4) { SIMPLE_SUMMARY = true; printf("Detected data format: Simple summary\n");}
       else if(n_tok!=10) { printf("Wrong file format for summary file, quitting\n");exit(1);}
    }
    else
    {
       if(META_file!=NULL)
          printf("Cannot find snp summary file %s, quitting\n",META_file);
       else
          printf("Cannot find snp summary file, quitting\n");
       exit(1);
    }

    if(!COMPUTE_LD) //V.1.4.mc
    {
      if(file_exists(allele_info_file))
      {
       char sline_info[MAX_LINE_WIDTH];
       if(!checkgz(allele_info_file))
          fp_allele_info= fopen(allele_info_file, "r");
       else
          fp_allele_info= gzopen(allele_info_file);
       fgets(sline_info, MAX_LINE_WIDTH, fp_allele_info);
       if(sline_info[0]!='#') {printf("-Header line missing in allele info file, please add one starting with '#'\n");exit(1);}
      }
      else
      {
         if(fp_ld!=NULL)
         {
           if(allele_info_file!=NULL)
              printf("Cannot find allele info file %s when ld-file is specified, quitting\n",allele_info_file);
           else
              printf("Cannot find allele info file when ld-file is specified, quitting\n");
           exit(1);
         }
         fp_allele_info = NULL;
      }
    }
  }

  PHENOTYPE* phenotype;
  //get the phenotype
  
  if(!SUMMARY){
   
    if(!file_exists(indiv_id_file))
    {
       if(indiv_id_file!=NULL)
          printf("-Cannot find individual id file [%s] , please provide one with --indiv-file option\nquitting....\n",indiv_id_file);
       else
          printf("-Cannot find individual id file , please provide one with --indiv-file option\nquitting....\n");
       exit(1);
    }
    if(!file_exists(tfam_file))
    {
       if(tfam_file!=NULL)
          printf("-Cannot find phenotype file [%s] , please provide one with --trait-file option\nquitting....\n",tfam_file);
       else
          printf("-Cannot find phenotype file, please provide one with --trait-file option\nquitting....\n");
       exit(1);
    }

    phenotype = getPhenotype(tfam_file, indiv_id_file, par.ncov);
  
    if(par.scalepheno)
       scalePheno(phenotype); //Pritam added, both pheno_array_org and pheno_array_reg are scaled.
    else if (par.quantilepheno)
       standardizePheno(phenotype);
    regCov(phenotype); //Pritam, the residuals from pheno_array_org are stored in pheno_array_reg.

  }else{
    phenotype = (PHENOTYPE*) malloc(sizeof(PHENOTYPE));
    phenotype->pheno_array_org = NULL;
    phenotype->pheno_array_reg = NULL;
    phenotype->pheno_array_log = NULL;

    //phenotype->mean= par.pheno_mean; //V.1.4.mc
  
    if(ESTIMATE_PHENO_VAR || ESTIMATE_N)
    {
       double est_var = GSL_NEGINF;
       int est_N = -1;
       if(SIMPLE_SUMMARY==false)
       {
          estimate_phenotype_variance_and_N(META_file,&est_var,&est_N);
       }
       else
       {
          //Either pheno_var or sample size is unestimated.
          if(ESTIMATE_N)
          {
            bool f = GET_BF_LINEAR || GET_GENE_BIC_LINEAR;
            if(f) turn_off_bimbam_gwis();
          }
          if(ESTIMATE_PHENO_VAR)
             est_var = 1.0; //assume unit variance, should not affect any method.
       }

       if(ESTIMATE_PHENO_VAR)
          phenotype->tss_per_n = est_var;
       else
          phenotype->tss_per_n = par.pheno_var;

       if(ESTIMATE_N)
          phenotype->N_sample = est_N;
       else
          phenotype->N_sample= par.n_sample;
    }
    else
    {
       phenotype->tss_per_n = par.pheno_var;
       phenotype->N_sample= par.n_sample;
    }
   
    if(!SUMMARY || VERBOSE)
    {
       printf("-phenotype TSS per indiv: %f\n", phenotype->tss_per_n);
       printf("-phenotype nsample: %d\n", phenotype->N_sample);
    }

    phenotype->N_na = 0;
    phenotype->N_indiv = 0;
  }

  print_what_should_run();

  //open output files
  outfile.fp_allSNP_linear =  NULL;   //single snp parametric pvalue
  outfile.fp_allSNP_logistic =  NULL; //single snp parametric pvalue
  outfile.fp_snp_pval_linear = NULL;  //minsnp permutation pvalue
  outfile.fp_snp_pval_logistic = NULL; //minsnp paermutation pvalue
  outfile.fp_snp_perm_pval_linear = NULL; //min snp p permutation pvalue
  outfile.fp_snp_perm_pval_logistic = NULL; //min snp p permutation pvalue

  fp_log = fopen(log_file,"w");//V.1.5.mc 
 
  if(work_on_genes)
     outfile.fp_gene_snp = fopen(gene_SNP_file, "w");

  if(NEED_SNP_LINEAR || NEED_SNP_LINEAR_SUMMARY)
      outfile.fp_allSNP_linear = fopen(allSNP_file_linear, "w");

  if(NEED_SNP_LOGISTIC || NEED_SNP_LOGISTIC_SUMMARY)
      outfile.fp_allSNP_logistic = fopen(allSNP_file_logistic, "w");
  
  if(GET_MINSNP_LINEAR || GET_MINSNP_PVAL_LINEAR)
      outfile.fp_snp_pval_linear = fopen(snp_pval_file_linear, "w");

  if(GET_MINSNP_LOGISTIC || GET_MINSNP_PVAL_LOGISTIC)
      outfile.fp_snp_pval_logistic = fopen(snp_pval_file_logistic, "w");
   
  if(GET_MINSNP_P_PVAL_LINEAR)
      outfile.fp_snp_perm_pval_linear = fopen(snp_bonf_pval_file_linear, "w");

  if(GET_MINSNP_P_PVAL_LOGISTIC)
      outfile.fp_snp_perm_pval_logistic = fopen(snp_bonf_pval_file_logistic, "w");

  //outfile.fp_bic_logistic_result = NULL;
  outfile.fp_bic_logistic_perm_result = NULL;

  if(GET_GENE_BIC_PVAL_LOGISTIC || GET_GENE_BIC_LOGISTIC)
  {
    //outfile.fp_bic_logistic_result = fopen(BIC_logistic_result_file, "w");
    //if(GET_GENE_BIC_PVAL_LOGISTIC)
    outfile.fp_bic_logistic_perm_result = fopen(BIC_logistic_perm_result_file, "w");
  }
  
  //outfile.fp_bic_linear_result = NULL;
  outfile.fp_bic_linear_perm_result=NULL;

  if(GET_GENE_BIC_PVAL_LINEAR || GET_GENE_BIC_LINEAR)
  {
    //outfile.fp_bic_linear_result = fopen(BIC_linear_result_file, "w");
    //if(GET_GENE_BIC_PVAL_LINEAR)
    outfile.fp_bic_linear_perm_result = fopen(BIC_linear_perm_result_file, "w");
  }

  outfile.fp_bf_pval_linear = NULL;
  outfile.fp_bf_pval_logistic = NULL;

  if(GET_BF_PVAL_LINEAR || GET_BF_LINEAR)
    outfile.fp_bf_pval_linear =  fopen(bf_pval_file_linear, "w");

  if(GET_BF_PVAL_LOGISTIC || GET_BF_LOGISTIC)
    outfile.fp_bf_pval_logistic =  fopen(bf_pval_file_logistic, "w");

  outfile.fp_vegas_pval_linear = NULL;
  outfile.fp_vegas_pval_logistic = NULL;

  if(GET_VEGAS_PVAL_LINEAR || GET_VEGAS_LINEAR)
    outfile.fp_vegas_pval_linear =  fopen(vegas_pval_file_linear, "w");
  
  if(GET_VEGAS_PVAL_LOGISTIC || GET_VEGAS_LOGISTIC)
    outfile.fp_vegas_pval_logistic =  fopen(vegas_pval_file_logistic, "w");

  outfile.fp_gates_pval_linear = NULL;
  outfile.fp_gates_pval_logistic = NULL;

  if(GET_GATES_LINEAR || GET_GATES_LINEAR)
    outfile.fp_gates_pval_linear =  fopen(gates_pval_file_linear, "w");

  if(GET_GATES_LOGISTIC || GET_GATES_LOGISTIC)
    outfile.fp_gates_pval_logistic =  fopen(gates_pval_file_logistic, "w");

  int i;
  //state of the random number generator
  gsl_rng *r, *saved_r_state;

  //can be init to any seed. 
  saved_r_state=initRand(1);

  r =initRand(par.random_seed);

  //circular queue
  C_QUEUE * snp_queue =  cq_init(MAX_SNP_PER_GENE, sizeof(SNP));

  if(CQ_DEBUG)
    printf("Circular Q initialized, size = %lu\n", snp_queue->nsize);

  SNP* curSNP= NULL;

  int nSNP = 0;
  SNP_CNT snp_cnt;
  //For SUMMARY data. FIX MONOMORPHIC SNPS V.1.2
  snp_cnt.nSNPAssigned=0;
  snp_cnt.nSNPRedundent=0;
  snp_cnt.nSNPMulti=0;
  snp_cnt.nSNPnoLD=0;
  snp_cnt.nSNPambig=0;
  //snp_cnt.nSNPsmallSample=0; not used, V.1.5.mc
  snp_cnt.nSNPnoA1=0;
  snp_cnt.nSNPA1MissMatch=0;
  snp_cnt.nSignFlipped=0;
  snp_cnt.nSignKept=0;
  snp_cnt.nInvSE=0; //V.1.2 VEGAS FIX, invalid standard error.

  //For Genotype data. FIX MONOMORPHIC SNPS V.1.2
  snp_cnt.nSNP_missingness=0;
  snp_cnt.nSNP_esamplesize=0;
  snp_cnt.nSNP_quality=0;
  snp_cnt.nSNP_mono=0;

  snp_cnt.nSNP_small_maf=0;


  int curSNP_id = 0;
  int nReadyGene=0;
  int nGeneFinished = 0;
  int nGeneLeft = nGene;
  int nGeneNoSNP=0;
  double c0, c1, rss;
  c0 = c1 = rss = 0.0;
  double se = 0.0;
  GENE** readyGene = NULL;
  GENE* startGene = NULL;
  double * LD_dat = NULL;
  double * Cov_dat = NULL;
  double * pvalCorr_dat = NULL;

  gsl_rng_memcpy(saved_r_state, r);

  if(work_on_genes)
  {
     if(readyGene==NULL) readyGene = (GENE**) malloc(nGene*sizeof(GENE*));
     startGene = gene->next;
     if(LD_dat==NULL) LD_dat = malloc(MAX_SNP_PER_GENE*MAX_SNP_PER_GENE*sizeof(double));
     if(Cov_dat==NULL) Cov_dat = malloc(MAX_SNP_PER_GENE*MAX_SNP_PER_GENE*sizeof(double));
     if(GET_GATES_LINEAR || GET_GATES_LOGISTIC)
     {
        if(pvalCorr_dat==NULL) pvalCorr_dat = malloc(MAX_SNP_PER_GENE*MAX_SNP_PER_GENE*sizeof(double));
     }
  }

  int max_snp_per_gene=MAX_SNP_PER_GENE;
  gsl_matrix_view  geneLD;
  gsl_matrix_view  geneCov;
  gsl_matrix_view genePvalCorr;

  if(VERBOSE)
    printf("start to print file headers\n");

    //print output file headers
    if(NEED_SNP_LINEAR)
    {
      //V.1.6.mc
       //fprintf(outfile.fp_allSNP_linear, "SNP.id\tChr\tPos\tA1\tA2\tBeta\tSe\tF_stat\tLog10BF\tMaf\tQual\tESampleSize\tNGenes\tFmiss\tPval\n");
       fprintf(outfile.fp_allSNP_linear, "SNP.id\tChr\tPos\tNonCoded.Allele\tCoded.Allele\tBeta\tSe\tChi2\tLog10BF\tCoded.Af\tQual\tESampleSize\tNGenes\tFmiss\tPval\n");
    }

    if(NEED_SNP_LINEAR_SUMMARY)
    {
       //V.1.2 FIX META
       fprintf(outfile.fp_allSNP_linear, "SNP.id\tChr\tPos\tNonCoded.Allele\tCoded.Allele\tBeta\tSe\tChi2\tLog10BF\tCoded.Af\tQual\tESampleSize\tNGenes\tNmiss\tPval\n");
    }

    if(NEED_SNP_LOGISTIC)
    {
       //V.1.6.mc 
       //fprintf(outfile.fp_allSNP_logistic, "SNP.id\tChr\tPos\tA1\tA2\tBeta\tSe\tWald\tLog10BF\tMaf\tQual\tESampleSize\tNGenes\tFmiss\tPval\n");
       fprintf(outfile.fp_allSNP_logistic, "SNP.id\tChr\tPos\tNonCoded.Allele\tCoded.Allele\tBeta\tSe\tChi2\tLog10BF\tCoded.Af\tQual\tESampleSize\tNGenes\tFmiss\tPval\n");
    }

    if(NEED_SNP_LOGISTIC_SUMMARY)
    {
       //V.1.2 FIX META
       fprintf(outfile.fp_allSNP_logistic, "SNP.id\tChr\tPos\tNonCoded.Allele\tCoded.Allele\tBeta\tSe\tChi2\tLog10BF\tCoded.Af\tQual\tESampleSize\tNGenes\tNmiss\tPval\n");
    }

    if(!SKIPPERM)
    {
        if(GET_MINSNP_PVAL_LINEAR || GET_MINSNP_LINEAR)
           fprintf(outfile.fp_snp_pval_linear, "Chr\tGene.id\tName\tStart\tEnd\tLength\tSNPs\tTests\tSNP.id\tSNP.pos\tSNP.maf\tSNP.qual\tSNP.chi2\tN.tot\tN.better\tPval\tIsBest\n");

        if(GET_MINSNP_PVAL_LOGISTIC || GET_MINSNP_LOGISTIC)
           fprintf(outfile.fp_snp_pval_logistic, "Chr\tGene.id\tName\tStart\tEnd\tLength\tSNPs\tTests\tSNP.id\tSNP.pos\tSNP.maf\tSNP.qual\tSNP.chi2\tN.tot\tN.better\tPval\tIsBest\n");

        if(GET_MINSNP_P_PVAL_LINEAR)
           fprintf(outfile.fp_snp_perm_pval_linear, "Chr\tGene.id\tName\tStart\tEnd\tLength\tSNPs\tTests\tSNP.id\tSNP.pos\tSNP.maf\tSNP.qual\tSNP.chi2\tN.tot\tN.better\tPval\tIsBest\n");

        if(GET_MINSNP_P_PVAL_LOGISTIC)
           fprintf(outfile.fp_snp_perm_pval_logistic, "Chr\tGene.id\tName\tStart\tEnd\tLength\tSNPs\tTests\tSNP.id\tSNP.pos\tSNP.maf\tSNP.qual\tSNP.chi2\tN.tot\tN.better\tPval\tIsBest\n");
    }
    else //skip permutations, report parametric pvalues.
    {
        if(GET_MINSNP_LINEAR)
        {
           //fprintf(outfile.fp_snp_pval_linear, "Chr\tGeneID\tName\tStart\tEnd\tLength\tSNPs\tTests\tSNP.name\tSNP.pos\tSNP.MAF\tSNP.qual\tchi2\tpval\tisBest\n");
           fprintf(outfile.fp_snp_pval_linear, "Chr\tGene.id\tName\tStart\tEnd\tLength\tSNPs\tTests\tSNP.id\tSNP.pos\tSNP.maf\tSNP.qual\tSNP.chi2\tN.tot\tN.better\tPval\tIsBest\n");
        }

        if(GET_MINSNP_LOGISTIC)
        {
           //fprintf(outfile.fp_snp_pval_logistic, "Chr\tGeneID\tName\tStart\tEnd\tLength\tSNPs\tTests\tSNP.name\tSNP.pos\tSNP.MAF\tSNP.qual\tchi2\tpval\tisBest\n");
           fprintf(outfile.fp_snp_pval_logistic, "Chr\tGene.id\tName\tStart\tEnd\tLength\tSNPs\tTests\tSNP.id\tSNP.pos\tSNP.maf\tSNP.qual\tSNP.chi2\tN.tot\tN.better\tPval\tIsBest\n");
        }
    }

    if(GET_BF_PVAL_LINEAR || GET_BF_LINEAR)
      fprintf(outfile.fp_bf_pval_linear, "Chr\tGene.id\tName\tStart\tEnd\tLength\tSNPs\tTests\tlog10(BF_sum)\tN.tot\tN.better\tPval\n");//V.1.2 sum FIX

    if(GET_BF_PVAL_LOGISTIC || GET_BF_LOGISTIC)
      fprintf(outfile.fp_bf_pval_logistic, "Chr\tGene.id\tName\tStart\tEnd\tLength\tSNPs\tTests\tlog10(BF_sum)\tN.tot\tN.better\tPval\n");//V.1.2 sum FIX

    if(GET_VEGAS_PVAL_LINEAR || GET_VEGAS_LINEAR)
      fprintf(outfile.fp_vegas_pval_linear, "Chr\tGene.id\tName\tStart\tEnd\tLength\tSNPs\tTests\tVegas\tN.tot\tN.better\tPval\n");//V.1.2 sum FIX

    if(GET_VEGAS_PVAL_LOGISTIC || GET_VEGAS_LOGISTIC)
      fprintf(outfile.fp_vegas_pval_logistic, "Chr\tGene.id\tName\tStart\tEnd\tLength\tSNPs\tTests\tVegas\tN.tot\tN.better\tPval\n");//V.1.2 sum FIX
 
    if(GET_GENE_BIC_LINEAR || GET_GENE_BIC_PVAL_LINEAR)
       fprintf(outfile.fp_bic_linear_perm_result, "Chr\tGene.id\tName\tStart\tEnd\tLength\tSNPs\tTests\tSNP.id\tSNP.pos\tSNP.maf\tSNP.qual\tK\tSSM\tBIC\tF_stat\tR2\tN.stop\tN.better\tPval\n");

    if(GET_GENE_BIC_LOGISTIC || GET_GENE_BIC_PVAL_LOGISTIC)
       fprintf(outfile.fp_bic_logistic_perm_result, "Chr\tGene.id\tName\tStart\tEnd\tLength\tSNPs\tTests\tSNP.id\tSNP.pos\tSNP.maf\tSNP.qual\tK\tBIC\tChi2\tN.stop\tN.better\tPval\n");

    if(GET_GATES_LINEAR)
       fprintf(outfile.fp_gates_pval_linear, "Chr\tGene.id\tName\tStart\tEnd\tLength\tSNPs\tTests\tGates\tPval\n");

    if(GET_GATES_LOGISTIC)
       fprintf(outfile.fp_gates_pval_logistic, "Chr\tGene.id\tName\tStart\tEnd\tLength\tSNPs\tTests\tGates\tPval\n");

    if(work_on_genes)
       fprintf(outfile.fp_gene_snp, "SNP.id\tSNP.chr\tSNP.bp\tGene.id\tGene.name\tGene.start\tGene.end\tSNP.maf\tSNP.qual\tSNP.ESampleSize\n");

    if(VERBOSE){
      printf("done printing file headers\n");
  }

  if(SHUFFLEBUF){

    printf("-Start to prepare phenotype shuffle buffer at %s", getTime());

    pheno_buff[NSHUFFLEBUF-1] = gsl_vector_alloc(phenotype->N_sample);
    gsl_vector_memcpy(pheno_buff[NSHUFFLEBUF-1], phenotype->pheno_array_org);
	
    for(i=0; i<NSHUFFLEBUF-1; i++){
      shuffle(pheno_buff[NSHUFFLEBUF-1], phenotype->N_sample, r,-1);
      pheno_buff[i] = gsl_vector_alloc(phenotype->N_sample);
      gsl_vector_memcpy(pheno_buff[i], pheno_buff[NSHUFFLEBUF-1]);
    }
    shuffle(pheno_buff[NSHUFFLEBUF-1], phenotype->N_sample, r,-1);

    printf("-End preparing phenotype shuffle buffer at %s", getTime());
  }

  printf("-Start to read SNPs\n");

  struct hashtable* snp_pos_table = NULL; //V.1.4.mc
  snp_pos_table = create_hashtable(16, hash_key, keys_equal_fn);

  //read the SNP positions from the freq file.
  //all header lines must have been stripped off already.
  if(SUMMARY){
    if(SIMPLE_SUMMARY==false)
       Load_POS_TABLE(fp_frq,fp_exclude, fp_allele_info, fp_pos, snp_pos_table); //V.1.4.mc
    else
       Load_POS_TABLE_simple(fp_frq,fp_exclude, fp_pos, fp_hap, snp_pos_table, ncols); //V.1.7
  }

  LOGISTIC_SCRATCH LG; //for logistic regression with single snp.
  BG_SCRATCH BG; //Bayes factor logistic regression.
  init_one_time(&LG,&BG);

  if(NEED_SNP_LOGISTIC)
  {
     init_scratch(&LG, phenotype);
     if(GET_BF_LOGISTIC)
       init_scratch_bf(&BG, phenotype);

     alloc_P();
  }
  
  bool do_snp_linear = false;
  bool do_snp_logistic = false;

  int last_bp = -1;
  //static clock_t time_sum = 0;

  //Start loop per snp
  do{
      if(!SUMMARY)
      {
         if(IMPUTE2_input)
            curSNP = readSNP_impute2(snp_queue,fp_impute2_geno, fp_impute2_info, phenotype, &snp_cnt, par.chr);//V.1.5.mc
         else 
            curSNP = readSNP(snp_queue,fp_tped, fp_snp_info, fp_mlinfo,  phenotype, &snp_cnt);//FIX MONOMORPHIC SNPS V.1.2
      }
      else
      {
         if(SIMPLE_SUMMARY==false)
            curSNP = readSNP_SUMMARY(snp_queue, snp_pos_table , fp_frq, fp_ld, fp_allele_info, &snp_cnt);
         else
            curSNP = readSNP_SUMMARY_simple(snp_queue, snp_pos_table , fp_frq, &snp_cnt,phenotype->N_sample,phenotype->tss_per_n);
      }

      if(work_on_genes)
      {
         //save the last unfinished gene 
         nReadyGene = assignSNP2Gene(&startGene, curSNP,curSNP_id, readyGene, outfile.fp_gene_snp, &snp_cnt);//FIX MONOMORPHIC SNPS V.1.2 
      }

      //do linear regression on this snp based on whether it is mapped to a gene ? 
      if(curSNP != NULL)
      {
         //printf("Doing snp = %s bp=%d nGene=%d flag=%d\n",curSNP->name,curSNP->bp,curSNP->nGene,par.single_snp_linear_gene);
         do_snp_linear   = (!par.single_snp_linear_gene)   || (par.single_snp_linear_gene && curSNP->nGene > 0);
         do_snp_logistic = (!par.single_snp_logistic_gene) || (par.single_snp_logistic_gene && curSNP->nGene > 0);
      }

      if(curSNP != NULL)
      {
         //++FIX CHECK BP ORDER
         if(curSNP->bp < last_bp)
         {
            printf("-error : the snps in all files must be ordered by increasing base-pair positions\n");
            printf("-SNP %s at bp = %d is not in order, previous SNP bp = %d\n",curSNP->name,curSNP->bp,last_bp);
            exit(1);
         }
         last_bp = curSNP->bp;
         //--FIX CHECK BP ORDER

         if(!SUMMARY && NEED_SNP_LINEAR && do_snp_linear)
         {
	    curSNP->geno_tss = tss(curSNP->geno,  phenotype->N_sample,true);
         }
         else if(SUMMARY)
         {
	    //	curSNP->geno_tss = 2*curSNP->MAF *(1-curSNP->MAF) *curSNP->eSampleSize;
	    curSNP->geno_tss = 2*curSNP->MAF *(1-curSNP->MAF) * phenotype->N_sample; //2p(1-p)n
            //printf("Here : %s %lg %lg\n",curSNP->name,curSNP->MAF,curSNP->geno_tss); //REMOVE
         }

	 if(!SUMMARY)
         {
              c0 = c1 = rss = 0.0;
              se = 0.0;
              if(NEED_SNP_LINEAR && do_snp_linear)
              {
	         curSNP->f_stat = runTest(curSNP->geno, phenotype->pheno_array_reg->data, phenotype->N_sample, phenotype->tss_per_n, &c0, &c1, &rss, &se,curSNP->missingness,phenotype->n_covariates);
	         curSNP->sum_pheno_geno = covariance(curSNP->geno, phenotype->pheno_array_reg->data, phenotype->N_sample,true,false) * phenotype->N_sample;
   
                 //printf("%s %g %g \n",curSNP->name,curSNP->f_stat,c1);
                 if(GET_BF_LINEAR)
	         {   
                    curSNP->BF_linear = getlog10BF(phenotype->tss_per_n, curSNP->geno_tss/phenotype->N_sample, phenotype->N_sample, curSNP->sum_pheno_geno/phenotype->N_sample);//FIX HESSIAN LATEST V.1.2
                    //printf("%s , BF lin = %g\n",curSNP->name,curSNP->BF_linear); 
                    //printf("GENOTYPE : %s => %g\n",curSNP->name,curSNP->BF_linear);
                 }

                 
                 //V.1.6.mc
                 if(par.ncov==0) 
                 { 
                    curSNP->pval_linear = gsl_cdf_chisq_Q(curSNP->f_stat,1);
                 }
                 else //V.1.6.mc
                 {
                    double tval = fabs(c1/se);
                    double chisqr = tval*tval;
                    double chisqr_pval = gsl_cdf_chisq_Q(chisqr,1);
                    curSNP->pval_linear = chisqr_pval;
                    //double df = phenotype->N_sample - curSNP->missingness*phenotype->N_sample - par.ncov - 2;
                    //curSNP->pval_linear = 2*gsl_cdf_tdist_Q(tval,df);
                    //printf("Pritam : %s t=%lg     df=%lg t-pval=%lg \n",curSNP->name,tval,df,curSNP->pval_linear);
                    //printf("Pritam : %s chi2=%lg         chi2-pval=%lg \n",curSNP->name,chisqr,chisqr_pval);
                 }

                 curSNP->c1 = c1; //to be used for starting gradient descent in logistic regression.
                 curSNP->c1_se = se;
              }
              //single SNP logistic
              if(NEED_SNP_LOGISTIC && do_snp_logistic)
              {
                 LG.phenotype = phenotype;
                 runLogisticSNP(curSNP, &LG,false,NULL,true); //TBD : this is also needed for all other logistic regression based operations
       
                 if(GET_BF_LOGISTIC)
                 {
                    BG.phenotype = phenotype;
                    //clock_t s1 = clock();
                    if(USE_BG_FR)
                       curSNP->BF_logistic = runBFLogisticSNP_fr(curSNP, &BG, false);
                    else
                       curSNP->BF_logistic = runBFLogisticSNP_newton(curSNP, &BG, false);
                    //printf("%s , BF log = %g\n",curSNP->name,curSNP->BF_logistic); 
                    //time_sum += (clock()-s1);
                 }
                 if(LOGISTIC_DEBUG_LVL==1)
                    printf("%s se=%.7f beta=%.7f wald=%.7f loglik=%.7f bf=%.7f\n",curSNP->name,curSNP->se_logistic,curSNP->beta_logistic,curSNP->wald,curSNP->loglik_logistic,curSNP->BF_logistic);
              }
              //--Pritam
	 }
         else //SUMMARY
         {
           if(SIMPLE_SUMMARY==false)
           {
              double pheno_tss_per_n = phenotype->tss_per_n; //V.1.4.mc
	      double z = curSNP->beta/curSNP->se;
              //both f_stat and wald are same for meta data.
	      curSNP->f_stat = z*z;
              curSNP->wald = z*z;
	      curSNP->sum_pheno_geno = z*sqrt(curSNP->geno_tss/phenotype->N_sample*pheno_tss_per_n)*phenotype->N_sample/sqrt(z*z+phenotype->N_sample-2);
	      curSNP->BF_linear = getlog10BF(pheno_tss_per_n, curSNP->geno_tss/phenotype->N_sample, phenotype->N_sample, curSNP->sum_pheno_geno/phenotype->N_sample);//FIX HESSIAN LATEST V.1.2
             //printf("SUMMARY : %s => %g\n",curSNP->name,curSNP->BF_linear);
              curSNP->pval_linear = curSNP->metaP;//Vegas and SUMMARY Linear
              curSNP->pval_logistic = curSNP->metaP;//Vegas and SUMMARY Logistic
           }
           else //get all from the pvalue
           {
              double chisqr = gsl_cdf_chisq_Qinv (curSNP->metaP,1);
              double z = sqrt(chisqr);
              curSNP->f_stat = chisqr;
              curSNP->wald = chisqr;
              if(VERBOSE)
                 printf("Computed %s pval = %lg, chisqr = %lg phen_var = %lg N_sample = %d\n",curSNP->name,curSNP->metaP,chisqr,phenotype->tss_per_n,phenotype->N_sample);
              if(phenotype->tss_per_n > 0 && phenotype->N_sample > 0) //if these are provided by the user.
              {
                 double pheno_tss_per_n = phenotype->tss_per_n;
                 //printf("1) z=%lg geno_tss = %lg N=%d pheno_tss = %lg\n",z,curSNP->geno_tss,phenotype->N_sample,pheno_tss_per_n);
	         curSNP->sum_pheno_geno = z*sqrt(curSNP->geno_tss/phenotype->N_sample*pheno_tss_per_n)*phenotype->N_sample/sqrt(z*z+phenotype->N_sample-2);
	         curSNP->BF_linear = getlog10BF(pheno_tss_per_n, curSNP->geno_tss/phenotype->N_sample, phenotype->N_sample, curSNP->sum_pheno_geno/phenotype->N_sample);
                 //printf("2) %lg %lg\n",curSNP->sum_pheno_geno,curSNP->BF_linear);
              }
              else
              {
                 curSNP->sum_pheno_geno = GSL_NEGINF;
                 curSNP->BF_linear = GSL_NEGINF;
              }
              curSNP->pval_linear = curSNP->metaP;//Vegas and SUMMARY Linear
              curSNP->pval_logistic = curSNP->metaP;//Vegas and SUMMARY Logistic
           }
	 }
      } //end if(curSNP != NULL)

      if(curSNP != NULL)
      {
          if(curSNP->nGene>0 && curSNP->missingness > 0)
          {
            if(!MISSING_DATA)
               printf("-at least one SNP in a gene has missing data\n");
            MISSING_DATA = true;
          }

          if(curSNP->nGene>0)
          {
	     snp_cnt.nSNPAssigned++;
          }
     
	  if(!SUMMARY)
          {
              if(NEED_SNP_LOGISTIC && do_snp_logistic)
              { 
                  if(GET_BF_LOGISTIC)
	              fprintf(outfile.fp_allSNP_logistic, "%s\t%d\t%d\t%c\t%c\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%d\t%lg\t%lg\n", curSNP->name,curSNP->chr,curSNP->bp,
                                                                               curSNP->A1,
                                                                               curSNP->A2, 
                                                                               curSNP->beta_logistic,
                                                                               curSNP->se_logistic,
                                                                               curSNP->wald,
                                                                               curSNP->BF_logistic,
                                                                               curSNP->AF1, 
                                                                               curSNP->R2, 
                                                                               curSNP->eSampleSize, 
                                                                               curSNP->nGene,
                                                                               curSNP->missingness,
                                                                               curSNP->pval_logistic);
                  else
	              fprintf(outfile.fp_allSNP_logistic, "%s\t%d\t%d\t%c\t%c\t%lg\t%lg\t%lg\t-\t%lg\t%lg\t%lg\t%d\t%lg\t%lg\n", curSNP->name,curSNP->chr,curSNP->bp,
                                                                               curSNP->A1,
                                                                               curSNP->A2, 
                                                                               curSNP->beta_logistic,
                                                                               curSNP->se_logistic,
                                                                               curSNP->wald,
                                                                               curSNP->AF1, 
                                                                               curSNP->R2, 
                                                                               curSNP->eSampleSize, 
                                                                               curSNP->nGene,
                                                                               curSNP->missingness,
                                                                               curSNP->pval_logistic);
              }
              if(NEED_SNP_LINEAR && do_snp_linear)
              {
                  if(GET_BF_LINEAR)
	             fprintf(outfile.fp_allSNP_linear, "%s\t%d\t%d\t%c\t%c\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\t%d\t%lg\t%lg\n", curSNP->name,curSNP->chr,curSNP->bp,curSNP->A1,curSNP->A2, 
                                                                                     c1,
                                                                                     se, 
                                                                                     curSNP->f_stat,
                                                                                     curSNP->BF_linear,  
                                                                                     curSNP->AF1, 
                                                                                     curSNP->R2, 
                                                                                     curSNP->eSampleSize, 
                                                                                     curSNP->nGene,
                                                                                     curSNP->missingness,
                                                                                     curSNP->pval_linear);
                  else
	             fprintf(outfile.fp_allSNP_linear, "%s\t%d\t%d\t%c\t%c\t%lg\t%lg\t%lg\t-\t%lg\t%lg\t%lg\t%d\t%lg\t%lg\n", curSNP->name,curSNP->chr,curSNP->bp,curSNP->A1,curSNP->A2, 
                                                                                     c1,
                                                                                     se, 
                                                                                     curSNP->f_stat,
                                                                                     curSNP->AF1, 
                                                                                     curSNP->R2, 
                                                                                     curSNP->eSampleSize, 
                                                                                     curSNP->nGene,
                                                                                     curSNP->missingness,
                                                                                     curSNP->pval_linear);
              }
         }
	 else
         {
            char Beta[100] = "-";
            char Se[100] = "-";
            if(gsl_finite(curSNP->beta)==1)
            {
               sprintf(Beta,"%lg",curSNP->beta);
               sprintf(Se,"%lg",curSNP->se);
            }

            if(NEED_SNP_LINEAR_SUMMARY && do_snp_linear)
            {
                if(GET_BF_LINEAR)
	           fprintf(outfile.fp_allSNP_linear, "%s\t%d\t%d\t%c\t%c\t%s\t%s\t%lg\t%lg\t%lg\t%lg\t%lg\t%d\t%d\t%lg\n", curSNP->name,curSNP->chr,curSNP->bp,curSNP->A1,curSNP->A2, 
                                                                               Beta,
                                                                               Se, 
                                                                               curSNP->f_stat, 
                                                                               curSNP->BF_linear, 
                                                                               curSNP->MAF, 
                                                                               curSNP->R2,
                                                                               curSNP->eSampleSize,
                                                                               curSNP->nGene,
                                                                               curSNP->nmiss,
                                                                               curSNP->pval_linear);
                else
	           fprintf(outfile.fp_allSNP_linear, "%s\t%d\t%d\t%c\t%c\t%s\t%s\t%lg\t-\t%lg\t%lg\t%lg\t%d\t%d\t%lg\n", curSNP->name,curSNP->chr,curSNP->bp,curSNP->A1,curSNP->A2, 
                                                                               Beta,
                                                                               Se, 
                                                                               curSNP->f_stat, 
                                                                               curSNP->MAF, 
                                                                               curSNP->R2,
                                                                               curSNP->eSampleSize,
                                                                               curSNP->nGene,
                                                                               curSNP->nmiss,
                                                                               curSNP->pval_linear);
            }
            if(NEED_SNP_LOGISTIC_SUMMARY && do_snp_logistic)
            {
	        fprintf(outfile.fp_allSNP_logistic, "%s\t%d\t%d\t%c\t%c\t%s\t%s\t%lg\t-\t%lg\t%lg\t%lg\t%d\t%d\t%lg\n", curSNP->name,curSNP->chr,curSNP->bp,curSNP->A1,curSNP->A2, 
                                                                               Beta,
                                                                               Se, 
                                                                               curSNP->wald, 
                                                                               curSNP->MAF, 
                                                                               curSNP->R2,
                                                                               curSNP->eSampleSize,
                                                                               curSNP->nGene,
                                                                               curSNP->nmiss,
                                                                               curSNP->pval_logistic);
            }
         }
         nSNP++;
      }//end if(curSNP != NULL)

    //work on genes that have all snps loaded
    if(nReadyGene > 0 && work_on_genes) 
    { 
        for (i =0; i< nReadyGene; i++)
        {
	    if(readyGene[i]->nSNP == 0){
	       nGeneNoSNP++;
	       continue;
	    }
	
	    if(VERBOSE)
	       printf("working on %s, %d total snps\n", readyGene[i]->ccds, readyGene[i]->nSNP);
	
            nGeneFinished++;
	
	    if(VERBOSE)
	       printf("calculating LD\n");

	    if(max_snp_per_gene<readyGene[i]->nSNP){
	             free(LD_dat);LD_dat=NULL;
	             free(Cov_dat);Cov_dat=NULL;
                     if(pvalCorr_dat!=NULL) 
                     {
                        free(pvalCorr_dat);
                        pvalCorr_dat = NULL;
                     }
	             max_snp_per_gene=readyGene[i]->nSNP;
	             LD_dat = malloc(max_snp_per_gene*max_snp_per_gene*sizeof(double));
	             Cov_dat = malloc(max_snp_per_gene*max_snp_per_gene*sizeof(double));

                     if(GET_GATES_LINEAR || GET_GATES_LOGISTIC)
                     {
                        //printf("GATES: allocated pvalCorr_dat\n");
	                pvalCorr_dat = malloc(max_snp_per_gene*max_snp_per_gene*sizeof(double));
	                if( pvalCorr_dat==NULL)
                        {
	                   printf("Failed to rezied the  pvalue Correlation matrix.  Possibly out of memory. Number of SNPs: %d\n", max_snp_per_gene);
	                   abort();
	                }else
	                   printf("-Successfully resized the pvalue Correlation matrix to %d x %d\n", max_snp_per_gene, max_snp_per_gene);
                     }
	             if( LD_dat == NULL|| Cov_dat== NULL){
	                printf("Failed to rezied the LD or Cov or pvalue Correlation matrix.  Possibly out of memory. Number of SNPs: %d\n", max_snp_per_gene);
	                abort();
	             }else
	                printf("-Successfully resized the LD matrix to %d x %d\n", max_snp_per_gene, max_snp_per_gene);
	    }

	    geneLD=gsl_matrix_view_array (LD_dat, readyGene[i]->nSNP, readyGene[i]->nSNP);
	    geneCov=gsl_matrix_view_array (Cov_dat, readyGene[i]->nSNP, readyGene[i]->nSNP);

            if(GET_GATES_LINEAR || GET_GATES_LOGISTIC)
	       genePvalCorr=gsl_matrix_view_array (pvalCorr_dat, readyGene[i]->nSNP, readyGene[i]->nSNP);

	    if(!SUMMARY)
            {
               if(GET_GATES_LINEAR || GET_GATES_LOGISTIC)//FIX GATES V.1.2
               {
                     //printf("GATES: computing pvalCorr_dat\n");
                     getGeneLD(readyGene[i], &geneLD.matrix, &geneCov.matrix, &genePvalCorr.matrix, snp_queue, phenotype->N_sample);//FIX GATES V.1.2
               }
               else
                     getGeneLD(readyGene[i], &geneLD.matrix, &geneCov.matrix, NULL, snp_queue, phenotype->N_sample);//FIX GATES V.1.2
	    }
            else
            {
               if(GET_GATES_LINEAR || GET_GATES_LOGISTIC)//FIX GATES V.1.2
                     getGeneLD_SUMMARY(readyGene[i], &geneLD.matrix, &geneCov.matrix, &genePvalCorr.matrix, snp_queue, ncols, fp_hap);//FIX GATES V.1.2. V.1.4.mc
               else
                     getGeneLD_SUMMARY(readyGene[i], &geneLD.matrix, &geneCov.matrix, NULL, snp_queue, ncols, fp_hap);//FIX GATES V.1.2. V.1.4.mc
	    }
	    if(VERBOSE)
	           printf("-calculating eSNP\n");
	    readyGene[i]->eSNP = getGeneESNP_dan(readyGene[i]->LD);
	           printf("-calculating eSNP for %s = %lg\n",readyGene[i]->name,readyGene[i]->eSNP);
	          //	getGeneESNP_matrix(readyGene[i]->LD, readyGene[i]->nSNP);
	          //reduced to a function to print the eigen values of 
	          //printCovEigen(readyGene[i]->Cov, readyGene[i]->nSNP, readyGene[i]->chr, readyGene[i]->name,fp_cor_eigen_vector );

	          //start to do permutations
	    printf("-Start to work on %s (n=%d) at %s",readyGene[i]->name , readyGene[i]->nSNP, getTime());
            fflush(stdout);
          
            //printf("-Gene %s took %g seconds to run\n",readyGene[i]->name,((double)(time_sum))/CLOCKS_PER_SEC);
            //time_sum = 0; 
 
	    gsl_rng_memcpy(r, saved_r_state);

            if(!SUMMARY)
            {
              if(GET_GATES_LINEAR || GET_GATES_LOGISTIC)
                 get_GATES_statistic(readyGene[i],snp_queue);
            }
            else
            {
              if(GET_GATES_LINEAR || GET_GATES_LOGISTIC)
                 get_GATES_statistic_SUMMARY(readyGene[i],snp_queue);
            }

            if(GET_GATES_LINEAR)
            {
               double pval = readyGene[i]->gates_linear; //because cdf(x) = x-a/(b-a), here a=0, b=1.
	       fprintf(outfile.fp_gates_pval_linear, "%d\t%s\t%s\t%d\t%d\t%d\t%d\t%g\t%g\t%g\n", readyGene[i]->chr, readyGene[i]->ccds, readyGene[i]->name, readyGene[i]->bp_start, readyGene[i]->bp_end, readyGene[i]->bp_end - readyGene[i]->bp_start+1, readyGene[i]->nSNP, readyGene[i]->eSNP, readyGene[i]->gates_linear,pval);
            }

            if(GET_GATES_LOGISTIC)
            {
               double pval = readyGene[i]->gates_logistic; //because cdf(x) = x-a/(b-a), here a=0, b=1.
	       fprintf(outfile.fp_gates_pval_logistic, "%d\t%s\t%s\t%d\t%d\t%d\t%d\t%g\t%g\t%g\n", readyGene[i]->chr, readyGene[i]->ccds, readyGene[i]->name, readyGene[i]->bp_start, readyGene[i]->bp_end, readyGene[i]->bp_end - readyGene[i]->bp_start+1, readyGene[i]->nSNP, readyGene[i]->eSNP, readyGene[i]->gates_logistic,pval);
            }

	    getPerm(snp_queue, 
                          readyGene[i], 
                          phenotype, 
                          r,
                          outfile,
                          &LG
                          );
	    printf("-End working on %s (n=%d) at %s", readyGene[i]->name, readyGene[i]->nSNP ,getTime());fflush(stdout);
	    cleanSNPinGene(readyGene[i], snp_queue);
	    if(VERBOSE)
	       printf("gene finished\n");
            
	    readyGene[i]->LD=NULL;
            readyGene[i]->pvalCorr = NULL;//Pritam
      } //for (i =0; i< nReadyGene; i++)
	  
      if(VERBOSE)
	printf("clean up\n");
      
      cleanSNPQ(snp_queue);
      
      printf("-%d snps scanned, %d genes finished, %d genes have no SNP,  %d gene remaining\n\n", nSNP, nGeneFinished,nGeneNoSNP,  nGeneLeft -= nReadyGene);
    } //if(nReadyGene > 0)

    if(!work_on_genes)
       nGeneLeft -= nReadyGene;
    
    if(curSNP!= NULL)
    {
       if(curSNP->nGene>0 && work_on_genes)
       { 
 	  curSNP_id++;
       }
       else
       {
	  if(curSNP->geno != NULL){
	    free(curSNP->geno);
	    curSNP->geno=NULL;
   	  }

          if(curSNP->ref_geno!=NULL){ //+V.1.4.mc
            free(curSNP->ref_geno);
            printf("        FREEING ref geno of %s\n",curSNP->name);
            curSNP->ref_geno = NULL;
          }

          if(curSNP->correlated_snps!=NULL)
          {
            hashtable_destroy(curSNP->correlated_snps, 1);
            curSNP->correlated_snps = NULL;
          }//-V.1.4.mc
     
	  if(curSNP->r != NULL){
	    free(curSNP->r);
	    curSNP->r=NULL;
	  }
	  cq_shift(snp_queue);
	  if(CQ_DEBUG)
	    printf("shifted, (start, end) = (%lu, %lu)\n", snp_queue->start, snp_queue->end);
       }
    }//if(curSNP!= NULL)

    if(curSNP==NULL)
    {
         printf("-no more snps left, done....\n");
         break;
    }
     
    if(nGeneLeft<=0) //V.1.6.mc THOROUGH_TEST
    {
       if(par.single_snp_linear_gene_parsed==true && GET_SINGLE_SNP_LOGISTIC==false) {
          printf("-No more genes left, not continuing with single snp analysis\n");
          break;
       }
       if(par.single_snp_logistic_gene_parsed==true && GET_SINGLE_SNP_LINEAR==false) {
          printf("-No more genes left, not continuing with single snp analysis\n");
          break;
       }
       if(par.single_snp_linear_gene_parsed==true && par.single_snp_logistic_gene_parsed==true) {
          printf("-No more genes left, not continuing with single snp analysis\n");
          break;
       }
    }

  //}while(nGeneLeft > 0 || GET_SINGLE_SNP_LINEAR || GET_SINGLE_SNP_LOGISTIC); //V.1.6.mc
  }while(nGeneLeft > 0 || NEED_SNP_LINEAR || NEED_SNP_LOGISTIC); 

  if(NEED_SNP_LOGISTIC)
  {
     //gsl_multimin_fdfminimizer_free(LG.s);
     //LG.s = NULL;
     //printf("%LG.h = %x\n",LG.h);
     free_scratch(&LG);
    
     free_P();
  } 
  if(GET_BF_LOGISTIC)
  {
     free_scratch_bf(&BG);
  }

  //++FIX MONOMORPHIC SNPS V.1.2
  if(SUMMARY)
  {
     printf("-%d out of %d genes tested\n", nGeneFinished, nGene);
     if(!COMPUTE_LD)//V.1.5.mc
        printf("-Out of %d SNPs read:\n", nSNP + snp_cnt.nSNPRedundent+snp_cnt.nSNPMulti+ snp_cnt.nSNPnoA1+ snp_cnt.nSNPA1MissMatch + snp_cnt.nInvSE);
     else
        printf("-Out of %d SNPs read:\n", nSNP + snp_cnt.nSNPRedundent+snp_cnt.nSNPnoLD + snp_cnt.nSNPambig + snp_cnt.nSNPMulti + snp_cnt.nSNPA1MissMatch + snp_cnt.nInvSE);
     //printf("---%d are processed \n", (nSNP - snp_cnt.nSNP_esamplesize - snp_cnt.nSNP_quality - snp_cnt.nSNP_small_maf) );
     printf("---%d dropped because of mapped to multiple positions\n", snp_cnt.nSNPMulti);
     if(!COMPUTE_LD) //V.1.5.mc
     {
         //if a snp is not in ref panel, we DO NOT throw it away when COMPUTE_LD = true.
         printf("---%d dropped because of no allele coding \n", snp_cnt.nSNPnoA1); //LD BUG

         printf("---%d dropped because of no LD profile \n", snp_cnt.nSNPnoLD); //LD BUG
     }
     printf("---%d dropped because of the allele mismatch\n", snp_cnt.nSNPA1MissMatch);
     printf("---%d dropped because of the strand ambiguity\n", snp_cnt.nSNPambig);
     printf("---%d dropped because of in high LD with another SNP\n", snp_cnt.nSNPRedundent);
     printf("---%d dropped because of invalid standard error\n", snp_cnt.nInvSE); //V.1.2 VEGAS FIX, invalid standard error.
     printf("---%d dropped because of effective Sample Size < %d\n",snp_cnt.nSNP_esamplesize,n_hat_cutoff); //counted in nSNP
     printf("---%d dropped because of low imputation quality < %g\n",snp_cnt.nSNP_quality,r2_cutoff); //counted in nSNP
     printf("---%d dropped because of low MAF < %f\n",snp_cnt.nSNP_small_maf,maf_cutoff); //counted in nSNP
     printf("---%d mapped to at least one gene\n", snp_cnt.nSNPAssigned);
     printf("-beta signs\n");
     printf("---%d SNPs flippped\n---%d SNPs stay same\n", snp_cnt.nSignFlipped, snp_cnt.nSignKept);

     if(!COMPUTE_LD)//V.1.5.mc
     {
       if(snp_cnt.nSNPAssigned < 0.5*(nSNP + snp_cnt.nSNPRedundent+snp_cnt.nSNPnoLD+snp_cnt.nSNPMulti+ snp_cnt.nSNPnoA1 + snp_cnt.nSNPA1MissMatch + snp_cnt.nInvSE))
          printf("\n-WARNING: LESS THAN 50%% OF SNPs used in FAST for gene based analysis\n\n");
     }
     else
     {
       if(snp_cnt.nSNPAssigned < 0.5*(nSNP + snp_cnt.nSNPRedundent+snp_cnt.nSNPMulti+ snp_cnt.nSNPnoA1 + snp_cnt.nSNPA1MissMatch + snp_cnt.nInvSE))
          printf("\n-WARNING: LESS THAN 50%% OF SNPs used in FAST\n\n");
     }
  }
  else
  {
     printf("-%d out of %d genes tested\n", nGeneFinished, nGene);
     printf("-Out of %d SNPs read:\n", nSNP+snp_cnt.nSNP_mono);
     //printf("---%d are processed \n", nSNP);
     printf("---%d dropped because of missingness > %f\n",snp_cnt.nSNP_missingness,MAX_MISSINGNESS);
     printf("---%d dropped because of effective Sample Size < %d\n",snp_cnt.nSNP_esamplesize,n_hat_cutoff);
     printf("---%d dropped because of snp imputation quality < %f\n",snp_cnt.nSNP_quality,r2_cutoff);
     printf("---%d dropped because of low MAF < %f\n",snp_cnt.nSNP_small_maf,maf_cutoff);
     printf("---%d dropped because monomorphic\n",snp_cnt.nSNP_mono);
     printf("---%d mapped to at least one gene\n",snp_cnt.nSNPAssigned);
     if(snp_cnt.nSNPAssigned < 0.5*(nSNP+snp_cnt.nSNP_mono))
     {
        printf("\n-WARNING: LESS THAN 50%% OF SNPs used in FAST\n\n");
     }
  }
  //--FIX MONOMORPHIC SNPS V.1.2

  if(nGeneFinished < 0.8*nGene){
    printf("\n-WARNING: LESS THAN 80%% OF GENES TESTED\n\n");
  }

  //close input files;
  if(!SUMMARY){
    //V.1.5.mc
    if(fp_impute2_geno!=NULL)
       fclose(fp_impute2_geno);
    if(fp_impute2_info!=NULL)
       fclose(fp_impute2_info);

    if(fp_tped!=NULL)
       fclose(fp_tped);
    if(fp_snp_info!=NULL)
       fclose(fp_snp_info);
    if(fp_mlinfo!=NULL)
       fclose(fp_mlinfo);
  }else{

    if(fp_ld!=NULL) //FIX V.1.3.mc 
       fclose(fp_ld);
    fclose(fp_frq);
    if(fp_allele_info!=NULL)
      fclose(fp_allele_info);
    if(fp_exclude!=NULL)
      fclose(fp_exclude);
    if(fp_pos!=NULL)
      fclose(fp_pos);
    if(fp_hap!=NULL)
      fclose(fp_hap);

    //++urgent
    if(ldfile_decomp) remove(par.ldfile);
    if(hapfile_decomp) remove(par.hapfile);
    if(metafile_decomp) remove(META_file);
    //--urgent
  }

  //close output files
  if(outfile.fp_allSNP_linear!=NULL)
     fclose(outfile.fp_allSNP_linear);

  if(outfile.fp_allSNP_logistic!=NULL)
     fclose(outfile.fp_allSNP_logistic);

  if(outfile.fp_snp_pval_linear != NULL)
    fclose(outfile.fp_snp_pval_linear);

  if(outfile.fp_snp_pval_logistic != NULL)
    fclose(outfile.fp_snp_pval_logistic);

  if(outfile.fp_snp_perm_pval_linear != NULL)
    fclose(outfile.fp_snp_perm_pval_linear);

  if(outfile.fp_snp_perm_pval_logistic != NULL)
    fclose(outfile.fp_snp_perm_pval_logistic);

  if(outfile.fp_bf_pval_linear!=NULL)
    fclose(outfile.fp_bf_pval_linear);

  if(outfile.fp_bf_pval_logistic!=NULL)
    fclose(outfile.fp_bf_pval_logistic);

  if(outfile.fp_vegas_pval_linear !=NULL)
    fclose(outfile.fp_vegas_pval_linear);

  if(outfile.fp_vegas_pval_logistic !=NULL)
    fclose(outfile.fp_vegas_pval_logistic);
 
  if(outfile.fp_gene_snp!=NULL)
     fclose(outfile.fp_gene_snp);

  if(fp_log!=NULL) //V.1.5.mc
     fclose(fp_log);

  //if(outfile.fp_bic_linear_result!=NULL)
  //  fclose(outfile.fp_bic_linear_result);

  //if(outfile.fp_bic_logistic_result!=NULL)
  //  fclose(outfile.fp_bic_logistic_result);

  if(outfile.fp_bic_linear_perm_result != NULL)
    fclose(outfile.fp_bic_linear_perm_result);

  if(outfile.fp_bic_logistic_perm_result != NULL)
    fclose(outfile.fp_bic_logistic_perm_result);

  if(outfile.fp_gates_pval_linear!=NULL)
    fclose(outfile.fp_gates_pval_linear);

  if(outfile.fp_gates_pval_logistic!=NULL)
    fclose(outfile.fp_gates_pval_logistic);

  free(readyGene);
  gsl_rng_free (r);
  gsl_rng_free (saved_r_state);

  if(phenotype->pheno_array_org != NULL)
  { 
    gsl_vector_free(phenotype->pheno_array_org);
    phenotype->pheno_array_org=NULL;
  }

  if(phenotype->pheno_array_reg != NULL)
  { 
    gsl_vector_free(phenotype->pheno_array_reg);
    phenotype->pheno_array_reg=NULL;
  }

  if(phenotype->pheno_array_log != NULL)
  { 
    gsl_vector_free(phenotype->pheno_array_log);
    phenotype->pheno_array_log = NULL;
  }

  for(i=0;i<phenotype->n_covariates;i++)
  {
     gsl_vector_free(phenotype->covariates[i]);
     phenotype->covariates[i]=NULL; 
  }

  free(phenotype);

  destroy_mutex();

  if(snp_pos_table!=NULL) //V.1.4.mc
    hashtable_destroy(snp_pos_table, 1);
  
  if(LD_dat!=NULL)
     free(LD_dat);

  if(Cov_dat!=NULL)
     free(Cov_dat);

  if(pvalCorr_dat!=NULL)
     free(pvalCorr_dat);
 
  if(haplotype_weights!=NULL)
    free(haplotype_weights);

  printf("-Finished at %s", getTime());
  time_t end_time;
  time(&end_time);
  double diff = difftime(end_time,start_time);
  if(diff>86400)
    printf("-FAST took %g days\n",diff/86400);
  else if(diff>3600)
    printf("-FAST took %g hours\n",diff/3600);
  else if(diff>60)
    printf("-FAST took %g minutes\n",diff/60);
  else 
    printf("-FAST took %g seconds\n",diff);
  
  return 0;
}
