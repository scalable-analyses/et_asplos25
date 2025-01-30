#include <cstdlib>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <set>
#include <map>
#include <tblis/tblis.h>
#include <ATen/ATen.h>
#include <vector>

/*
 * Functions are c&p from EinsumExpressionAscii.cpp
 */

void split_string( std::string                const & i_input,
                   std::string                const & i_separation,
                   std::vector< std::string >       & o_output ) {
  std::string l_string = i_input;
  int64_t l_off = 0;
  int64_t l_size_string = l_string.size();
  while( l_off < l_size_string ) {
    l_off = l_string.find( i_separation );
    if( l_off < 0 ) break;
    o_output.push_back( l_string.substr( 0, l_off ) );
    l_string.erase( 0, l_off + i_separation.size() );
  }
  if( l_string.size() > 0 ) {
    o_output.push_back( l_string );
  }
}

void parse_tensors( std::string                const & i_expr_string,
                    std::vector< std::string >       & o_tensors ) {
  o_tensors.clear();

  std::string l_expr = i_expr_string;

  l_expr.erase( std::remove( l_expr.begin(),
                             l_expr.end(),
                             ' '),
                l_expr.end());
  std::vector< std::string > l_tensors_tmp;

  split_string( l_expr,
                std::string("->"),
                l_tensors_tmp );

  split_string( l_tensors_tmp[0],
                std::string(","),
                o_tensors );
  o_tensors.push_back( l_tensors_tmp[1] );
}

void parse_dim_sizes( std::string            const & i_dim_sizes_string,
                      std::vector< int64_t >       & o_dim_sizes ) {
  o_dim_sizes.clear();

  std::string l_sizes = i_dim_sizes_string;

  l_sizes.erase( std::remove( l_sizes.begin(),
                              l_sizes.end(),
                              ' '),
                 l_sizes.end());

  std::vector< std::string > l_dim_sizes_tmp;
  split_string( l_sizes,
                std::string(","),
                l_dim_sizes_tmp );
  
  o_dim_sizes.resize( l_dim_sizes_tmp.size() );
  for( std::size_t l_di = 0; l_di < l_dim_sizes_tmp.size(); l_di++ ) {
    o_dim_sizes[l_di] = std::stoi( l_dim_sizes_tmp[l_di] );
  }
}

void parse_dim_ids( std::string               const & i_expr_string,
                    std::map< char, int64_t >       & o_map_dim_name_to_id ) {
  o_map_dim_name_to_id.clear();

  std::vector< std::string > l_tensors;
  parse_tensors( i_expr_string, l_tensors );
  int64_t l_num_tensors = l_tensors.size();

  std::set< char > l_dim_names_set;
  for( int64_t l_te = 0; l_te < l_num_tensors; l_te++ ) {
    std::string l_tensor = l_tensors[l_te];

    for( std::size_t l_ch = 0; l_ch < l_tensor.size(); l_ch++ ) {
      l_dim_names_set.insert( l_tensor[l_ch] );
    }
  }
  std::vector< char > l_dim_names( l_dim_names_set.begin(),
                                   l_dim_names_set.end() );

  for( std::size_t l_di = 0; l_di < l_dim_names.size(); l_di++ ) {
    o_map_dim_name_to_id.insert( { l_dim_names[l_di], l_di } );
  }
}

int main( int     i_argc,
          char  * i_argv[] ) {
  std::cout << "running bench_tblis" << std::endl;

  std::string l_expression_string;
  std::string l_dim_sizes_string;
  std::string l_precision = "FP32";
  if( i_argc > 2 ) {
    l_expression_string = i_argv[1];
    l_dim_sizes_string = i_argv[2];
  }
  else if( i_argc > 3 ) {
    l_precision = i_argv[3];
  }
  else {
    std::cerr << "Usage: bench_tblis <einsum_str> <dim_sizes> [precision]" << std::endl;
    return EXIT_FAILURE;
  }

  at::ScalarType l_dtype_at;
  if( l_precision == "FP32" ) {
    l_dtype_at = at::ScalarType::Float;
  }
  else if( l_precision == "FP64" ) {
    l_dtype_at = at::ScalarType::Double;
  }
  else {
    std::cerr << "Invalid precision: " << l_precision << std::endl;
    return EXIT_FAILURE;
  }

  /**
   * parse input tensors and output tensors
   **/
  std::vector< std::string > l_tensors;
  parse_tensors( l_expression_string,
                 l_tensors );
  int64_t l_num_tensors = l_tensors.size();

  std::cout << "parsed tensors:" << std::endl;
  for( int64_t l_te = 0; l_te < l_num_tensors; l_te++ ) {
    std::cout << "  " << l_tensors[l_te] << std::endl;
  }

  /*
   * parse dimension sizes
   */
  std::vector< int64_t > l_dim_sizes;
  parse_dim_sizes( l_dim_sizes_string,
                   l_dim_sizes );

  /*
   * create mapping from dimension name to id
   */
  std::map< char, int64_t > m_map_dim_name_to_id;
  parse_dim_ids( l_expression_string,
                 m_map_dim_name_to_id );

  std::cout << "parsed dimension sizes:" << std::endl;
  // iterate over keys of map dim name to id
  for( std::map< char, int64_t >::iterator l_di = m_map_dim_name_to_id.begin(); l_di != m_map_dim_name_to_id.end(); l_di++ ) {
    char l_dim_name = l_di->first;
    int64_t l_dim_id = l_di->second;
    int64_t l_dim_size = l_dim_sizes[ l_dim_id ];

    std::cout << "  " << l_dim_name << ": " <<  l_dim_size << std::endl;
  }

  /*
   * create the tensors' data
   */
  std::vector< int64_t > l_string_num_dims( l_num_tensors );
  for( int64_t l_te = 0; l_te < l_num_tensors; l_te++ ) {
    l_string_num_dims[l_te] = l_tensors[l_te].size();
  }

  std::vector< int64_t > l_string_dim_ids;
  for( int64_t l_te = 0; l_te < l_num_tensors; l_te++ ) {
    std::string l_tensor = l_tensors[l_te];

    for( std::size_t l_ch = 0; l_ch < l_tensor.size(); l_ch++ ) {
      int64_t l_dim_id = m_map_dim_name_to_id[ l_tensor[l_ch] ];
      l_string_dim_ids.push_back( l_dim_id );
    }
  }

  std::vector< at::Tensor > l_data;
  int64_t l_off = 0;
  for( int64_t l_te = 0; l_te < l_num_tensors; l_te++ ) {
    // assemble size of the tensor
    std::vector< int64_t > l_sizes;
    int64_t l_num_dims = l_string_num_dims[l_te];
    for( int64_t l_di = 0; l_di < l_num_dims; l_di++ ) {
      int64_t l_dim_id = l_string_dim_ids[l_off + l_di];
      int64_t l_size = l_dim_sizes[ l_dim_id ];
      l_sizes.push_back( l_size );
    }
    l_off += l_string_num_dims[l_te];

    l_data.push_back( at::randn( l_sizes, l_dtype_at ) );
  }

  std::vector< void * > l_data_ptrs;
  for( std::size_t l_te = 0; l_te < l_data.size(); l_te++ ) {
    l_data_ptrs.push_back( l_data[l_te].data_ptr() );
  }

  /*
   * performance data structures
   */
  std::chrono::steady_clock::time_point l_tp0, l_tp1;
  std::chrono::duration< double > l_dur;
  int64_t l_num_flops = 0;
  double l_time = 0;
  double l_gflops = 0;

  /*
   * create tblis tensors
   */
  std::vector< tblis::len_type >    l_sizes_left;
  std::vector< tblis::stride_type > l_strides_left;

  std::vector< tblis::len_type >    l_sizes_right;
  std::vector< tblis::stride_type > l_strides_right;

  std::vector< tblis::len_type >    l_sizes_out;
  std::vector< tblis::stride_type > l_strides_out;

  l_off = 0;
  l_sizes_left.resize( l_string_num_dims[0] );
  l_strides_left.resize( l_string_num_dims[0] );

  for( int64_t l_di = l_string_num_dims[0]-1; l_di >= 0; l_di-- ) {
    int64_t l_dim_id = l_string_dim_ids[l_off + l_di];
    l_sizes_left[l_di] = l_dim_sizes[ l_dim_id ];
    l_strides_left[l_di] = (l_di == l_string_num_dims[0]-1) ? 1 : l_strides_left[l_di+1]*l_sizes_left[l_di+1];
  }

  l_off += l_string_num_dims[0];
  l_sizes_right.resize( l_string_num_dims[1] );
  l_strides_right.resize( l_string_num_dims[1] );

  for( int64_t l_di = l_string_num_dims[1]-1; l_di >= 0; l_di-- ) {
    int64_t l_dim_id = l_string_dim_ids[l_off + l_di];
    l_sizes_right[l_di] = l_dim_sizes[ l_dim_id ];
    l_strides_right[l_di] = (l_di == l_string_num_dims[1]-1) ? 1 : l_strides_right[l_di+1]*l_sizes_right[l_di+1];
  }

  l_off += l_string_num_dims[1];
  l_sizes_out.resize( l_string_num_dims[2] );
  l_strides_out.resize( l_string_num_dims[2] );

  for( int64_t l_di = l_string_num_dims[2]-1; l_di >= 0; l_di-- ) {
    int64_t l_dim_id = l_string_dim_ids[l_off + l_di];
    l_sizes_out[l_di] = l_dim_sizes[ l_dim_id ];
    l_strides_out[l_di] = (l_di == l_string_num_dims[2]-1) ? 1 : l_strides_out[l_di+1]*l_sizes_out[l_di+1];
  }


  tblis::tblis_tensor l_tensor_left;
  tblis::tblis_init_tensor_s( &l_tensor_left,
                              l_string_num_dims[0],
                              l_sizes_left.data(),
                              (float *) l_data_ptrs[0],
                              l_strides_left.data() );

  tblis::tblis_tensor l_tensor_right;
  tblis::tblis_init_tensor_s(&l_tensor_right,
                              l_string_num_dims[1],
                              l_sizes_right.data(),
                              (float *) l_data_ptrs[1],
                              l_strides_right.data() );

  tblis::tblis_tensor l_tensor_out;
  tblis::tblis_init_tensor_scaled_s( &l_tensor_out,
                                     0.0,
                                     l_string_num_dims[2],
                                     l_sizes_out.data(),
                                     (float *) l_data_ptrs[2],
                                     l_strides_out.data() );

  // perform warm-up contraction
  tblis::tblis_tensor_mult( NULL,
                            NULL,
                            &l_tensor_left,
                            l_tensors[0].c_str(),
                            &l_tensor_right,
                            l_tensors[1].c_str(),
                            &l_tensor_out,
                            l_tensors[2].c_str() );

  at::Tensor l_out_tblis = l_data.back().clone();

  // benchmark performance
  std::cout <<  "\n*** benchmarking tblis ***" << std::endl;

  l_tp0 = std::chrono::steady_clock::now();
  for( int l_run = 0; l_run < 10; l_run++ ) {
    tblis::tblis_tensor_mult( NULL,
                              NULL,
                              &l_tensor_left,
                              l_tensors[0].c_str(),
                              &l_tensor_right,
                              l_tensors[1].c_str(),
                              &l_tensor_out,
                              l_tensors[2].c_str() );
  }
  l_tp1 = std::chrono::steady_clock::now();

  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time= l_dur.count();

  l_num_flops = 10 * 2;
  for( std::size_t l_di = 0; l_di < l_dim_sizes.size(); l_di++ ) {
    l_num_flops *= l_dim_sizes[l_di];
  }
  
  l_gflops = 1.0E-9 * l_num_flops / l_time;

  std::cout << "  #flops: " << l_num_flops << std::endl;\
  std::cout << "  time :  " << l_time << std::endl;
  std::cout << "  gflops: " << l_gflops << std::endl;
  std::cout << "CSV_DATA: "
            << "einsum_ir,"
            << "\"" << l_expression_string << "\","
            << "\"" << l_dim_sizes_string << "\","
            << l_num_flops << ","
            << l_time << ","
            << l_gflops
            << std::endl;

  /*
   * run at::einsum
   */
  std::vector< at::Tensor > l_data_in( l_num_tensors-1 );
  for( int64_t l_te = 0; l_te < l_num_tensors - 1; l_te++ ) {
    l_data_in[l_te] = l_data[l_te];
  }

  at::Tensor l_out_aten = at::einsum( l_expression_string, l_data_in );

  /*
   * compare solution
   */
  std::cout << std::endl;
  std::cout << "*** comparing solution ***:" << std::endl;
  std::cout << "  maximum absolute entry in ATen solution:      " << at::max( at::abs( l_out_aten  ) ).item() << std::endl;
  std::cout << "  maximum absolute entry in einsum_ir solution: " << at::max( at::abs( l_out_tblis ) ).item() << std::endl;
  std::cout << "  maximum element-wise difference:              " << at::max( at::abs( l_out_aten - l_out_tblis ) ).item() << std::endl;
  if( !at::allclose( l_out_aten, l_out_tblis ) ) {
    std::cerr << "warning: einsum_ir solution is not close to at:einsum!" << std::endl;
    return EXIT_FAILURE;
  }


  return EXIT_SUCCESS;
}