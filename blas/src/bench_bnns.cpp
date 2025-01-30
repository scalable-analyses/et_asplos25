#include <cstdlib>
#include <chrono>
#include <random>
#include <bnns.h>
#include <iostream>

void gemm( int64_t i_m,
           int64_t i_n,
           int64_t i_k,
           int64_t i_num_repetitions_initial = 10,
           double  i_target_time = 10.0 ) {
  std::chrono::steady_clock::time_point l_tp0, l_tp1;
  std::chrono::duration< double > l_dur;
  int64_t l_num_flops = 0;
  int64_t l_num_repetitions = 0;
  double l_time = 0;
  double l_gflops = 0;

  // allocate memory
  float * l_a = nullptr;
  float * l_b = nullptr;
  float * l_c = nullptr;

  posix_memalign( (void**) &l_a, 128, i_m * i_k * sizeof(float) );
  posix_memalign( (void**) &l_b, 128, i_k * i_n * sizeof(float) );
  posix_memalign( (void**) &l_c, 128, i_m * i_n * sizeof(float) );

  // init the matrices using a normal distribution with mean 0 and variance 1
  std::random_device l_rd;
  std::mt19937 l_gen( l_rd() );
  std::normal_distribution<float> l_dist( 0.0, 1.0 );

  for( int64_t l_en = 0; l_en < i_m * i_k; l_en++ ) {
    l_a[l_en] = l_dist( l_gen );
  }

  for( int64_t l_en = 0; l_en < i_k * i_n; l_en++ ) {
    l_b[l_en] = l_dist( l_gen );
  }

  for( int64_t l_en = 0; l_en < i_m * i_n; l_en++ ) {
    l_c[l_en] = l_dist( l_gen );
  }

  // descriptor for input A
  BNNSNDArrayDescriptor l_desc_a;
  l_desc_a.flags = BNNSNDArrayFlags( 0 );
  l_desc_a.layout = BNNSDataLayoutColumnMajorMatrix;
  l_desc_a.size[0] = i_m;
  l_desc_a.size[1] = i_k;
  for( int64_t l_di = 2; l_di < BNNS_MAX_TENSOR_DIMENSION; l_di++ ) {
    l_desc_a.size[l_di] = 0;
  }
  for( int64_t l_di = 0; l_di < BNNS_MAX_TENSOR_DIMENSION; l_di++ ) {
    l_desc_a.stride[l_di] = 0;
  }
  l_desc_a.data_type = BNNSDataTypeFloat32;
  l_desc_a.table_data_type = BNNSDataTypeFloat32;
  l_desc_a.data_scale = 1;
  l_desc_a.data_bias = 0;
  l_desc_a.data = nullptr;
  l_desc_a.table_data = nullptr;

  // descriptor for input B
  BNNSNDArrayDescriptor l_desc_b;
  l_desc_b.flags = BNNSNDArrayFlags( 0 );
  l_desc_b.layout = BNNSDataLayoutColumnMajorMatrix;
  l_desc_b.size[0] = i_k;
  l_desc_b.size[1] = i_n;
  for( int64_t l_di = 2; l_di < BNNS_MAX_TENSOR_DIMENSION; l_di++ ) {
    l_desc_b.size[l_di] = 0;
  }
  for( int64_t l_di = 0; l_di < BNNS_MAX_TENSOR_DIMENSION; l_di++ ) {
    l_desc_b.stride[l_di] = 0;
  }
  l_desc_b.data_type = BNNSDataTypeFloat32;
  l_desc_b.table_data_type = BNNSDataTypeFloat32;
  l_desc_b.data_scale = 1;
  l_desc_b.data_bias = 0;
  l_desc_b.data = nullptr;
  l_desc_b.table_data = nullptr;


  // output descriptor
  BNNSNDArrayDescriptor l_desc_c;
  l_desc_c.flags = BNNSNDArrayFlags( 0 );
  l_desc_c.layout = BNNSDataLayoutColumnMajorMatrix;
  l_desc_c.size[0] = i_m;
  l_desc_c.size[1] = i_n;
  for( int64_t l_di = 2; l_di < BNNS_MAX_TENSOR_DIMENSION; l_di++ ) {
    l_desc_c.size[l_di] = 0;
  }
  for( int64_t l_di = 0; l_di < BNNS_MAX_TENSOR_DIMENSION; l_di++ ) {
    l_desc_c.stride[l_di] = 0;
  }
  l_desc_c.data_type = BNNSDataTypeFloat32;
  l_desc_c.table_data_type = BNNSDataTypeFloat32;
  l_desc_c.data_scale = 1;
  l_desc_c.data_bias = 0;
  l_desc_c.data = nullptr;
  l_desc_c.table_data = nullptr;

  // einsum string
  std::string l_einsum_string = "a_ip, b_qi -> o_pq";

  BNNSLayerParametersTensorContraction l_layer_params;
  l_layer_params.operation = l_einsum_string.c_str();
  l_layer_params.alpha = 1.0;
  l_layer_params.beta = 0.0;
  l_layer_params.iA_desc = l_desc_a;
  l_layer_params.iB_desc = l_desc_b;
  l_layer_params.o_desc  = l_desc_c;

  BNNSFilter l_layer = BNNSFilterCreateLayerTensorContraction( &l_layer_params,
                                                               NULL );

  // warmup
  BNNSFilterApplyTwoInput( l_layer, l_a, l_b, l_c );

  // determine number of repetitions
  l_tp0 = std::chrono::steady_clock::now();
  for( int64_t l_re = 0; l_re < i_num_repetitions_initial; l_re++) {
    BNNSFilterApplyTwoInput( l_layer, l_a, l_b, l_c );
  }
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_num_repetitions = std::max( 1.0, (i_target_time * i_num_repetitions_initial) / l_dur.count() );
  l_num_flops = 2 * i_m * i_n * i_k * l_num_repetitions;

  // benchmark
  l_tp0 = std::chrono::steady_clock::now();

  for( int64_t l_re = 0; l_re < l_num_repetitions; l_re++ ) {
    BNNSFilterApplyTwoInput( l_layer, l_a, l_b, l_c );
  }

  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time = l_dur.count();
  l_gflops = 1.0E-9 * l_num_flops / l_time;

  std::cout << "  #repetitions: " << l_num_repetitions << std::endl;
  std::cout << "  time: " << l_time << std::endl;
  std::cout << "  gflops: " << l_gflops << std::endl;

  // free memory
  free( l_a );
  free( l_b );
  free( l_c );

}

int main( int     i_argc,
          char  * i_argv[] ) {
  std::cout << "running bench_bnns" << std::endl;

  gemm( 128, 128, 128 );

  // float l_a[ 4 ] = { 1, 2, 3, 4 };
  // float l_b[ 4 ] = { 3, 4, 3, 4 };


  // // input descriptor
  // BNNSNDArrayDescriptor l_input_desc;
  // l_input_desc.flags = BNNSNDArrayFlags( 0 );
  // l_input_desc.layout = BNNSDataLayoutRowMajorMatrix;
  // l_input_desc.size[0] = 2;
  // l_input_desc.size[1] = 2;
  // l_input_desc.data_type = BNNSDataTypeFloat32;
  // l_input_desc.table_data_type = BNNSDataTypeFloat32;
  // l_input_desc.data_scale = 1;
  // l_input_desc.data_bias = 0;

  // // output descriptor
  // BNNSNDArrayDescriptor l_output_desc;
  // l_output_desc.flags = BNNSNDArrayFlags( 0 );
  // l_output_desc.layout = BNNSDataLayoutRowMajorMatrix;
  // l_output_desc.size[0] = 2;
  // l_output_desc.size[1] = 2;
  // l_output_desc.data_type = BNNSDataTypeFloat32;
  // l_output_desc.table_data_type = BNNSDataTypeFloat32;
  // l_output_desc.data_scale = 1;
  // l_output_desc.data_bias = 0;

  // // einsum string
  // std::string l_einsum_string = "a_pi, b_iq -> o_pq";

  // BNNSLayerParametersTensorContraction l_layer_params;
  // l_layer_params.operation = l_einsum_string.c_str();
  // l_layer_params.alpha = 10.0;
  // l_layer_params.beta = 0.0;
  // l_layer_params.iA_desc = l_input_desc;
  // l_layer_params.iB_desc = l_input_desc;
  // l_layer_params.o_desc  = l_output_desc;

  // BNNSFilter l_layer = BNNSFilterCreateLayerTensorContraction( &l_layer_params,
  //                                                             NULL );

  // float l_c[4] = {0};

  // BNNSFilterApplyTwoInput( l_layer, l_a, l_b, l_c );

  // std::cout << l_c[0] << " " << l_c[1] << " " << l_c[2] << " " << l_c[3] << std::endl;


  std::cout << "finished running bench_bnns!" << std::endl;
  return EXIT_SUCCESS;
}