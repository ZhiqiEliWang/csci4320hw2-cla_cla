/*********************************************************************/
//
// 02/02/2023: Revised Version for 32M bit adder with 32 bit blocks
//
/*********************************************************************/

#include "main.h"
#include <assert.h>

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
  if (j < ngroups) { //avoid accessing beyond the end of the arrays
    int jstart = j * block_size; //jstart is the starting index of the group in gi and pi

    int sum = 0;
    for(int i = 0; i < block_size; i++)
    {
        int mult = gi[jstart + i]; //grabs the g_i term for the multiplication
        for(int ii = block_size-1; ii > i; ii--)
        {
            mult &= pi[jstart + ii]; //grabs the p_i terms and multiplies it with the previously multiplied stuff (or the g_i term if first round)
            assert(jstart + ii < bits);
        }
        sum |= mult; //sum up each of these things with an or
    }
    ggj[j] = sum;

    int mult = pi[jstart];
    for(int i = 1; i < block_size; i++)
    {
        mult &= pi[jstart + i];
    }
    gpj[j] = mult;
  }
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/
__global__
void compute_section_gp(int* ggj, int* gpj, int* sgk, int* spk)
{
  
  int k = threadIdx.x + blockIdx.x * blockDim.x;// k is the current section number
  if (k < nsections) { //avoid accessing beyond the end of the arrays
    int kstart = k * block_size; //kstart is the starting index of the section in ggj and gpj
    
    int sum = 0;
    for(int i = 0; i < block_size; i++)
      {
          int mult = ggj[kstart + i];
          for(int ii = block_size-1; ii > i; ii--)
          {
              mult &= gpj[kstart + ii];
          }
          sum |= mult;
      }
      sgk[k] = sum;
    
      int mult = gpj[kstart];
      for(int i = 1; i < block_size; i++)
      {
          mult &= gpj[kstart + i];
      }
      spk[k] = mult;
  }
      

}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/
__global__
void compute_super_section_gp(int* sgk, int* spk, int* ssgl, int* sspl)
{
    int l = threadIdx.x + blockIdx.x * blockDim.x;// l is the current super section number
    if (l < nsupersections) { //avoid accessing beyond the end of the arrays
    int lstart = l*block_size;
    
    int sum = 0;
    for(int i = 0; i < block_size; i++)
      {
          int mult = sgk[lstart + i];
          for(int ii = block_size-1; ii > i; ii--)
          {
              mult &= spk[lstart + ii];
          }
          sum |= mult;
      }
      ssgl[l] = sum;
    
      int mult = spk[lstart];
      for(int i = 1; i < block_size; i++)
      {
          mult &= spk[lstart + i];
      }
      sspl[l] = mult;
    }
      
}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/
__global__
void compute_super_super_section_gp(int* ssgl, int* sspl, int* sssgm, int* ssspm)
{
    int m = threadIdx.x + blockIdx.x * blockDim.x;// m is the current super super section number
    if (m < nsupersupersections) { //avoid accessing beyond the end of the arrays
    int mstart = m*block_size;
    
    int sum = 0;
    for(int i = 0; i < block_size; i++)
      {
          int mult = ssgl[mstart + i];
          for(int ii = block_size-1; ii > i; ii--)
          {
              mult &= sspl[mstart + ii];
          }
          sum |= mult;
      }
      sssgm[m] = sum;
    
      int mult = sspl[mstart];
      for(int i = 1; i < block_size; i++)
      {
          mult &= sspl[mstart + i];
      }
      ssspm[m] = mult;
    }
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
  if (l < nsupersections) { //avoid accessing beyond the end of the arrays
    for (int offset=0; offset<block_size; offset++){
      if (l == 0 && offset == 0)
        {
          sscllast = 0;
        }
      else if(offset == 0)
        {
          sscllast = ssscm[l];
        }
      else if( offset != 0 )
        {
          // l*block_size + offset - 1 is the index of the previous section in the super section
          sscllast = sscl[l*block_size + offset - 1];
        }
      
        sscl[l*block_size + offset] = ssgl[l*block_size + offset] | (sspl[l*block_size + offset]&sscllast);
    }
  }

}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/
__global__
void compute_section_carry(int *sck, int *sgk, int *spk, int *sscl)
{
  int k = threadIdx.x + blockIdx.x * blockDim.x;// k is the current section number
  int scklast=0;
  if (k < nsections) { //avoid accessing beyond the end of the arrays
    for (int offset=0; offset<block_size; offset++){
      if (k == 0 && offset == 0)
        {
          scklast = 0;
        }
      else if(offset == 0)
        {
          scklast = sscl[k];
        }
      else if( offset != 0 )
        {
          scklast = sck[k*block_size + offset - 1];
        }
      
        sck[k*block_size + offset] = sgk[k*block_size + offset] | (spk[k*block_size + offset]&scklast);
    }
  }
}


/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/
__global__
void compute_group_carry(int *gcj, int *ggj, int *gpj, int *sck)
{
  int j = threadIdx.x + blockIdx.x * blockDim.x;// j is the current group number
  int gcjlast=0;
  if (j < ngroups) { //avoid accessing beyond the end of the arrays
    for (int offset=0; offset<block_size; offset++){
      if (j == 0 && offset == 0)
        {
          gcjlast = 0;
        }
      else if(offset == 0)
        {
          gcjlast = sck[j];
        }
      else if( offset != 0 )
        {
          gcjlast = gcj[j*block_size + offset - 1];
        }
      
        gcj[j*block_size + offset] = ggj[j*block_size + offset] | (gpj[j*block_size + offset]&gcjlast);
    }
  }

}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/
__global__
void compute_carry(int *ci, int *gi, int *pi, int *gcj)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;// i is the current bit index for the carry
  int cilast=0;
  if (i < bits) { //avoid accessing beyond the end of the arrays
    for (int offset=0; offset<block_size; offset++){
      if (i == 0 && offset == 0)
        {
          cilast = 0;
        }
      else if(offset == 0)
        {
          cilast = gcj[i];
        }
      else if( offset != 0 )
        {
          cilast = ci[i*block_size + offset - 1];
        }
      
        ci[i*block_size + offset] = gi[i*block_size + offset] | (pi[i*block_size + offset]&cilast);
    }
  }

}

/***********************************************************************************************************/
// ADAPT AS CUDA KERNEL 
/***********************************************************************************************************/
__global__
void compute_sum(int *sumi, int *bin1, int *bin2, int *ci)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;// i is the current bit index for the sum
    int clast=0;
    if (i < bits) { //avoid accessing beyond the end of the arrays
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

}

void cla()
{
  int threadPerBlock = 1024;
  /***********************************************************************************************************/
  // ADAPT ALL THESE FUNCTUIONS TO BE SEPARATE CUDA KERNEL CALL
  // NOTE: Kernel calls are serialized by default per the CUDA kernel call scheduler
  // NOTE: Make sure you set the right CUDA Block Size (e.g., threads per block) for different runs per 
  //       assignment description.
  /***********************************************************************************************************/
    int gpNumBlock = (bits + threadPerBlock - 1) / threadPerBlock;
    compute_gp<<<gpNumBlock, threadPerBlock>>>(gi, pi, bin1, bin2);
    printf("compute_gp done\n");
    int ggNumBlock = (ngroups + threadPerBlock - 1) / threadPerBlock;
    compute_group_gp<<<ggNumBlock, threadPerBlock>>>(gi, pi, ggj, gpj);
    printf("compute_group_gp done\n");
    int scNumBlock = (nsections + threadPerBlock - 1) / threadPerBlock;
    compute_section_gp<<<scNumBlock, threadPerBlock>>>(ggj, gpj, sgk, spk);
    printf("compute_section_gp done\n");
    int ssNumBlock = (nsupersections + threadPerBlock - 1) / threadPerBlock;
    compute_super_section_gp<<<ssNumBlock, threadPerBlock>>>(sgk, spk, ssgl, sspl);
    printf("compute_super_section_gp done\n");
    int sssNumBlock = (nsupersupersections + threadPerBlock - 1) / threadPerBlock;
    compute_super_super_section_gp<<<sssNumBlock, threadPerBlock>>>(ssgl, sspl, sssgm, ssspm);
    printf("compute_super_super_section_gp done\n");

    compute_super_super_section_carry<<<1, 1>>>(ssscm, sssgm, ssspm); // This function is not going to be parallelized
    printf("compute_super_super_section_carry done\n");
    compute_super_section_carry<<<ssNumBlock, threadPerBlock>>>(sscl, ssgl, sspl, ssscm);
    printf("compute_super_section_carry done\n");
    compute_section_carry<<<scNumBlock, threadPerBlock>>>(sck, sgk, spk, sscl);
    printf("compute_section_carry done\n");
    compute_group_carry<<<ggNumBlock, threadPerBlock>>>(gcj, ggj, gpj, sck);
    printf("compute_group_carry done\n");
    compute_carry<<<gpNumBlock, threadPerBlock>>>(ci, gi, pi, gcj);
    printf("compute_carry done\n");
    compute_sum<<<gpNumBlock, threadPerBlock>>>(sumi, bin1, bin2, ci);
    printf("compute_sum done\n");

    cudaDeviceSynchronize(); // This is the right place to insert the CUDA synchronization



    printf("\n\nsscl[0] = %d\nbin1[0] = %d\nbin2[0] = %d\n" ,sscl[0], bin1[0], bin2[0]);
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

    cudaMallocManaged(&gi, bits * sizeof(int));
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
