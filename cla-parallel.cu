/*********************************************************************/
//
// 02/02/2023: Revised Version for 32M bit adder with 32 bit blocks
//
/*********************************************************************/

#include "main.h"

//Touch these defines
#define input_size 8388608 // hex digits 
#define block_size 32
#define verbose 0

//Do not touch these defines
#define digits (input_size+1)
#define bits (digits*4)
#define ngroups bits/block_size
#define nsections ngroups/block_size
#define nsupersections nsections/block_size
#define nsupersupersections nsupersections/block_size

//Global definitions of the various arrays used in steps for easy access
/***********************************************************************************************************/
// ADAPT AS CUDA managedMalloc memory - e.g., change to pointers and allocate in main function. 
/***********************************************************************************************************/
int *gi;
int *pi;
int *ci;

int *ggj;
int *gpj;
int *gcj;

int *sgk;
int *spk;
int *sck;

int *ssgl;
int *sspl;
int *sscl;

int *sssgm;
int *ssspm;
int *ssscm;

int *sumi;

int *sumrca;

//Integer array of inputs in binary form
int* bin1;
int* bin2;

//Character array of inputs in hex form
char* hex1=NULL;
char* hex2=NULL;

void read_input()
{
  char* in1; cudaMallocManaged(&in1, (input_size+1)*sizeof(char));
  char* in2; cudaMallocManaged(&in2, (input_size+1)*sizeof(char));

  if( 1 != scanf("%s", in1))
    {
      printf("Failed to read input 1\n");
      exit(-1);
    }
  if( 1 != scanf("%s", in2))
    {
      printf("Failed to read input 2\n");
      exit(-1);
    }
  
  hex1 = grab_slice_char(in1,0,input_size+1);
  hex2 = grab_slice_char(in2,0,input_size+1);
  
  cudaFree(in1);
  cudaFree(in2);
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/

__global__
void compute_gp(int* gi, int* pi, int* bin1, int* bin2)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < bits) { //avoid accessing beyond the end of the arrays
        gi[i] = bin1[i] & bin2[i];
        pi[i] = bin1[i] | bin2[i];
    }
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/

__global__
void compute_group_gp(int* gi, int* pi, int* ggj, int* gpj) 
{
    int j = threadIdx.x + blockIdx.x * blockDim.x;// j is the current group number
    int jstart = j * block_size; //jstart is the starting index of the group in gi and pi

    int sum = 0;
    for(int i = 0; i < block_size; i++)
    {
        int mult = gi[jstart + i]; //grabs the g_i term for the multiplication
        for(int ii = block_size-1; ii > i; ii--)
        {
            mult &= gi[jstart + ii]; //grabs the p_i terms and multiplies it with the previously multiplied stuff (or the g_i term if first round)
        }
        sum |= mult; //sum up each of these things with an or
    }
    ggj[j] = sum;

    int mult = gi[jstart];
    for(int i = 1; i < block_size; i++)
    {
        mult &= gi[jstart + i];
    }
    gpj[j] = mult;

}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/
__global__
void compute_section_gp(int* ggj, int* gpj, int* sgk, int* spk)
{

    int k = threadIdx.x + blockIdx.x * blockDim.x;// k is the current section number
    int kstart = k * block_size; //kstart is the starting index of the section in ggj and gpj
    
    int sum = 0;
    for(int i = 0; i < block_size; i++)
      {
          int mult = sgk[kstart + k];
          for(int ii = block_size-1; ii > i; ii--)
          {
              mult &= sgk[kstart + ii];
          }
          sum |= mult;
      }
      sgk[k] = sum;
    
      int mult = sgk[kstart];
      for(int i = 1; i < block_size; i++)
      {
          mult &= sgk[kstart + i];
      }
      spk[k] = mult;
      

}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/
__global__
void compute_super_section_gp(int* sgk, int* spk, int* ssgl, int* sspl)
{
    int l = threadIdx.x + blockIdx.x * blockDim.x;// l is the current super section number
    int lstart = l*block_size;
    
    int sum = 0;
    for(int i = 0; i < block_size; i++)
      {
          int mult = ssgl[lstart + i];
          for(int ii = block_size-1; ii > i; ii--)
          {
              mult &= ssgl[lstart + ii];
          }
          sum |= mult;
      }
      ssgl[l] = sum;
    
      int mult = ssgl[lstart];
      for(int i = 1; i < block_size; i++)
      {
          mult &= ssgl[lstart + i];
      }
      sspl[l] = mult;
      
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/
__global__
void compute_super_super_section_gp(int* ssgl, int* sspl, int* sssgm, int* ssspm)
{
    int m = threadIdx.x + blockIdx.x * blockDim.x;// m is the current super super section number
    int mstart = m*block_size;
    
    int sum = 0;
    for(int i = 0; i < block_size; i++)
      {
          int mult = sssgm[mstart + i];
          for(int ii = block_size-1; ii > i; ii--)
          {
              mult &= sssgm[mstart + ii];
          }
          sum |= mult;
      }
      sssgm[m] = sum;
    
      int mult = sssgm[mstart];
      for(int i = 1; i < block_size; i++)
      {
          mult &= sssgm[mstart + i];
      }
      ssspm[m] = mult;
      
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/
__global__
void compute_super_super_section_carry(int *ssscm, int *sssgm, int *ssspm) // This function is not going to be parallelized
{
  for(int m = 0; m < nsupersupersections; m++)
    {
      int ssscmlast=0;
      if(m==0)
        {
	        ssscmlast = 0;
        }
      else
        {
	        ssscmlast = ssscm[m-1];
        }
      
      ssscm[m] = sssgm[m] | (ssspm[m]&ssscmlast);
    }
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/
__global__
void compute_super_section_carry(int *sscl, int *ssgl, int *sspl, int *ssscm) 
{
  int l = threadIdx.x + blockIdx.x * blockDim.x;// l is the current super section number
  int sscllast=0;
  if(l%block_size == block_size-1)
    {
      sscllast = ssscm[l/block_size];
    }
  else if( l != 0 )
    {
      sscllast = sscl[l-1];
    }
  
  sscl[l] = ssgl[l] | (sspl[l]&sscllast);

}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/
__global__
void compute_section_carry(int *sck, int *sgk, int *spk, int *sscl)
{
  int k = threadIdx.x + blockIdx.x * blockDim.x;// k is the current section number
  int scklast=0;
  if(k%block_size==block_size-1)
    {
      scklast = sscl[k/block_size];
    }
  else if( k != 0 )
    {
      scklast = sck[k-1];
    }
  
  sck[k] = sgk[k] | (spk[k]&scklast);

}


/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/
__global__
void compute_group_carry(int *gcj, int *ggj, int *gpj, int *sck)
{
  int j = threadIdx.x + blockIdx.x * blockDim.x;// j is the current group number
  int gcjlast=0;
  if(j%block_size==block_size-1)
    {
      gcjlast = sck[j/block_size];
    }
  else if( j != 0 )
    {
      gcjlast = gcj[j-1];
    }
  
  gcj[j] = ggj[j] | (gpj[j]&gcjlast);

}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/
__global__
void compute_carry(int *ci, int *gi, int *pi, int *gcj)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;// i is the current carry number
  int cilast=0;
  if(i%block_size==block_size-1)
    {
      cilast = gcj[i/block_size];
    }
  else if( i != 0 )
    {
      cilast = ci[i-1];
    }
  
  ci[i] = gi[i] | (pi[i]&cilast);

}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/
__global__
void compute_sum(int *sumi, int *bin1, int *bin2, int *ci)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;// i is the current bit index for the sum
    int clast=0;
    if(i==0)
      {
  clast = 0;
      }
    else
      {
  clast = ci[i-1];
      }
    sumi[i] = bin1[i] ^ bin2[i] ^ clast;

}

void cla()
{
  /***********************************************************************************************************/
  // ADAPT ALL THESE FUNCTUIONS TO BE SEPARATE CUDA KERNEL CALL
  // NOTE: Kernel calls are serialized by default per the CUDA kernel call scheduler
  // NOTE: Make sure you set the right CUDA Block Size (e.g., threads per block) for different runs per 
  //       assignment description.
  /***********************************************************************************************************/
    int gpNumBlock = (bits + block_size - 1) / block_size;
    compute_gp<<<gpNumBlock, block_size>>>(gi, pi, bin1, bin2);
    int ggNumBlock = (ngroups + block_size - 1) / block_size;
    compute_group_gp<<<ggNumBlock, block_size>>>(ggj, gpj, gi, pi);
    int scNumBlock = (nsections + block_size - 1) / block_size;
    compute_section_gp<<<scNumBlock, block_size>>>(sgk, spk, ggj, gpj);
    int ssNumBlock = (nsupersections + block_size - 1) / block_size;
    compute_super_section_gp<<<ssNumBlock, block_size>>>(ssgl, sspl, sgk, spk);
    int sssNumBlock = (nsupersupersections + block_size - 1) / block_size;
    compute_super_super_section_gp<<<sssNumBlock, block_size>>>(sssgm, ssspm, ssgl, sspl);

    compute_super_super_section_carry<<<1, 1>>>(ssscm, sssgm, ssspm); // This function is not going to be parallelized

    compute_super_section_carry<<<ssNumBlock, block_size>>>(sscl, ssgl, sspl, ssscm);
    compute_section_carry<<<scNumBlock, block_size>>>(sck, sgk, spk, sscm);
    compute_group_carry<<<ggNumBlock, block_size>>>(gcj, ggj, gpj, sck);
    compute_carry<<<gpNumBlock, block_size>>>(ci, gi, pi, gcj);
    compute_sum<<<gpNumBlock, block_size>>>(sumi, bin1, bin2, ci);

  /***********************************************************************************************************/
  // INSERT RIGHT CUDA SYNCHRONIZATION AT END!
  /***********************************************************************************************************/
}

void ripple_carry_adder()
{
  int clast=0, cnext=0;

  for(int i = 0; i < bits; i++)
    {
      cnext = (bin1[i] & bin2[i]) | ((bin1[i] | bin2[i]) & clast);
      sumrca[i] = bin1[i] ^ bin2[i] ^ clast;
      clast = cnext;
    }
}

void check_cla_rca()
{
  for(int i = 0; i < bits; i++)
    {
      if( sumrca[i] != sumi[i] )
	{
	  printf("Check: Found sumrca[%d] = %d, not equal to sumi[%d] = %d - stopping check here!\n",
		 i, sumrca[i], i, sumi[i]);
	  printf("bin1[%d] = %d, bin2[%d]=%d, gi[%d]=%d, pi[%d]=%d, ci[%d]=%d, ci[%d]=%d\n",
		 i, bin1[i], i, bin2[i], i, gi[i], i, pi[i], i, ci[i], i-1, ci[i-1]);
	  return;
	}
    }
  printf("Check Complete: CLA and RCA are equal\n");
}

int main(int argc, char *argv[])
{
  int randomGenerateFlag = 1;
  int deterministic_seed = (1<<30) - 1;
  char* hexa=NULL;
  char* hexb=NULL;
  char* hexSum=NULL;
  char* int2str_result=NULL;
  unsigned long long start_time=clock_now(); // dummy clock reads to init
  unsigned long long end_time=clock_now();   // dummy clock reads to init

  // ----------------- BEGIN: cudaMallocManaged -----------------

    cudaMallocManaged(&gi, 1 * sizeof(int));
    cudaMallocManaged(&pi, bits * sizeof(int));
    cudaMallocManaged(&ci, bits * sizeof(int));

    cudaMallocManaged(&ggj, ngroups * sizeof(int));
    cudaMallocManaged(&gpj, ngroups * sizeof(int));
    cudaMallocManaged(&gcj, ngroups * sizeof(int));

    cudaMallocManaged(&sgk, nsections * sizeof(int));
    cudaMallocManaged(&spk, nsections * sizeof(int));
    cudaMallocManaged(&sck, nsections * sizeof(int));

    cudaMallocManaged(&ssgl, nsupersections * sizeof(int));
    cudaMallocManaged(&sspl, nsupersections * sizeof(int));
    cudaMallocManaged(&sscl, nsupersections * sizeof(int));

    cudaMallocManaged(&sssgm, nsupersupersections * sizeof(int));
    cudaMallocManaged(&ssspm, nsupersupersections * sizeof(int));
    cudaMallocManaged(&ssscm, nsupersupersections * sizeof(int));

    cudaMallocManaged(&sumi, bits * sizeof(int));
    cudaMallocManaged(&sumrca, bits * sizeof(int));

    cudaMallocManaged(&bin1, bits * sizeof(int));
    cudaMallocManaged(&bin2, bits * sizeof(int));


  // ----------------- END: cudaMallocManaged -------------------

  if( nsupersupersections != block_size )
    {
      printf("Misconfigured CLA - nsupersupersections (%d) not equal to block_size (%d) \n",
	     nsupersupersections, block_size );
      return(-1);
    }
  
  if (argc == 2) {
    if (strcmp(argv[1], "-r") == 0)
      randomGenerateFlag = 1;
  }
  
  if (randomGenerateFlag == 0)
    {
      read_input();
    }
  else
    {
      srand( deterministic_seed );
      hex1 = generate_random_hex(input_size);
      hex2 = generate_random_hex(input_size);
    }
  
  hexa = prepend_non_sig_zero(hex1);
  hexb = prepend_non_sig_zero(hex2);
  hexa[digits] = '\0'; //double checking
  hexb[digits] = '\0';
  
  bin1 = gen_formated_binary_from_hex(hexa);
  bin2 = gen_formated_binary_from_hex(hexb);

  start_time = clock_now();
  cla();
  end_time = clock_now();

  printf("CLA Completed in %llu cycles\n", (end_time - start_time));

  start_time = clock_now();
  ripple_carry_adder();
  end_time = clock_now();

  printf("RCA Completed in %llu cycles\n", (end_time - start_time));

  check_cla_rca();

//-------------------------------------------free all cuda mem-----------------------------------------------
cudaFree(gi);
cudaFree(pi);
cudaFree(ci);

cudaFree(ggj);
cudaFree(gpj);
cudaFree(gcj);

cudaFree(sgk);
cudaFree(spk);
cudaFree(sck);

cudaFree(ssgl);
cudaFree(sspl);
cudaFree(sscl);

cudaFree(sssgm);
cudaFree(ssspm);
cudaFree(ssscm);

  if( verbose==1 )
    {
      int2str_result = int_to_string(sumi,bits);
      hexSum = revbinary_to_hex( int2str_result,bits);
    }

  // free inputs fields allocated in read_input or gen random calls
  free(int2str_result);
  free(hex1);
  free(hex2);
  
  // free bin conversion of hex inputs
  free(bin1);
  free(bin2);
  
  if( verbose==1 )
    {
      printf("Hex Input\n");
      printf("a   ");
      print_chararrayln(hexa);
      printf("b   ");
      print_chararrayln(hexb);
    }
  
  if ( verbose==1 )
    {
      printf("Hex Return\n");
      printf("sum =  ");
    }
  
  // free memory from prepend call
  free(hexa);
  free(hexb);

  if( verbose==1 )
    printf("%s\n",hexSum);
  
  free(hexSum);
  
  return 0;
}
