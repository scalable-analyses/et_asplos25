#include <cstdlib>
#include <chrono>
#include <random>
#if defined(BLAS_OPENBLAS)
#include <cblas.h>
#elif defined(BLAS_NVPL)
#include <nvpl_blas_cblas.h>
#endif
#include <libxsmm.h>
#include <ATen/ATen.h>
#include <iostream>

void sgemm_bench( int64_t i_m,
                  int64_t i_n,
                  int64_t i_k,
                  int64_t i_lda,
                  int64_t i_ldb,
                  int64_t i_ldc,
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

  posix_memalign( (void**) &l_a, 128, i_lda * i_k * sizeof(float) );
  posix_memalign( (void**) &l_b, 128, i_ldb * i_n * sizeof(float) );
  posix_memalign( (void**) &l_c, 128, i_ldc * i_n * sizeof(float) );

  // init the matrices using a normal distribution with mean 0 and variance 1
  std::random_device l_rd;
  std::mt19937 l_gen( l_rd() );
  std::normal_distribution<float> l_dist( 0.0, 1.0 );

  for( int64_t l_en = 0; l_en < i_lda * i_k; l_en++ ) {
    l_a[l_en] = l_dist( l_gen );
  }

  for( int64_t l_en = 0; l_en < i_ldb * i_n; l_en++ ) {
    l_b[l_en] = l_dist( l_gen );
  }

  for( int64_t l_en = 0; l_en < i_ldc * i_n; l_en++ ) {
    l_c[l_en] = l_dist( l_gen );
  }

  // ATen data structures
  at::Tensor l_a_aten = at::from_blob( l_a,
                                       {i_k, i_m},
                                       at::ScalarType::Float );

  at::Tensor l_b_aten = at::from_blob( l_b,
                                       {i_n, i_k},
                                       at::ScalarType::Float );

  at::Tensor l_c_aten = at::from_blob( l_c,
                                       {i_n, i_m},
                                       at::ScalarType::Float );

  // warmup and verification
  at::Tensor l_c_aten_ref = l_c_aten + at::matmul( l_b_aten,
                                                   l_a_aten );

  cblas_sgemm( CblasColMajor,
               CblasNoTrans,
               CblasNoTrans,
               i_m,
               i_n,
               i_k,
               1,
               l_a,
               i_lda,
               l_b,
               i_ldb,
               1,
               l_c,
               i_ldc );

  std::cout << "  max error: " << at::max( at::abs( l_c_aten_ref - l_c_aten ) ).item() << std::endl;

  // determine number of repetitions
  l_tp0 = std::chrono::steady_clock::now();
  for( int64_t l_re = 0; l_re < i_num_repetitions_initial; l_re++) {
    cblas_sgemm( CblasColMajor,
                 CblasNoTrans,
                 CblasNoTrans,
                 i_m,
                 i_n,
                 i_k,
                 1,
                 l_a,
                 i_lda,
                 l_b,
                 i_ldb,
                 1,
                 l_c,
                 i_ldc );
  }
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_num_repetitions = std::max( 1.0, (i_target_time * i_num_repetitions_initial) / l_dur.count() );
  l_num_flops = 2 * i_m * i_n * i_k * l_num_repetitions;

  l_tp0 = std::chrono::steady_clock::now();

  for( int64_t l_re = 0; l_re < l_num_repetitions; l_re++ ) {
    cblas_sgemm( CblasColMajor,
                 CblasNoTrans,
                 CblasNoTrans,
                 i_m,
                 i_n,
                 i_k,
                 1,
                 l_a,
                 i_lda,
                 l_b,
                 i_ldb,
                 1,
                 l_c,
                 i_ldc );
  }

  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time = l_dur.count();
  l_gflops = 1.0E-9 * l_num_flops / l_time;

  std::cout << "  #repetitions: " << l_num_repetitions << std::endl;
  std::cout << "  time: " << l_time << std::endl;
  std::cout << "  gflops: " << l_gflops << std::endl;
  std::cout << "CSV_DATA: "
            << "sgemm,"
            << i_m << ","
            << i_n << ","
            << i_k << ","
            << i_lda << ","
            << i_ldb << ","
            << i_ldc << ","
            << l_num_flops << ","
            << l_time << ","
            << l_gflops
            << std::endl;

  // free memory
  free( l_a );
  free( l_b );
  free( l_c );
}

void sgemm_xsmm_bench( int64_t i_m,
                       int64_t i_n,
                       int64_t i_k,
                       int64_t i_lda,
                       int64_t i_ldb,
                       int64_t i_ldc,
                       int64_t i_num_repetitions_initial = 10,
                       double  i_target_time = 10.0 ) {
  std::chrono::steady_clock::time_point l_tp0, l_tp1;
  std::chrono::duration< double > l_dur;
  int64_t l_num_flops = 0;
  int64_t l_num_repetitions = 0;
  double l_time = 0;
  double l_gflops = 0;

  // generate GEMM kernel
  libxsmm_gemm_shape l_shape_brgemm;
  libxsmm_bitfield l_flags_brgemm = LIBXSMM_GEMM_FLAGS('N', 'N');
  libxsmm_bitfield l_prefetch_flags_brgemm = 0;

  l_shape_brgemm = libxsmm_create_gemm_shape( i_m,
                                              i_n,
                                              i_k,
                                              i_lda,
                                              i_ldb,
                                              i_ldc,
                                              libxsmm_datatype::LIBXSMM_DATATYPE_F32,
                                              libxsmm_datatype::LIBXSMM_DATATYPE_F32,
                                              libxsmm_datatype::LIBXSMM_DATATYPE_F32,
                                              libxsmm_datatype::LIBXSMM_DATATYPE_F32 );

  libxsmm_gemm_batch_reduce_config l_brconfig;
  l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_NONE;
  l_brconfig.br_stride_a_hint = 0;
  l_brconfig.br_stride_b_hint = 0;
  l_brconfig.br_unroll_hint = 0;

  libxsmm_xmmfunction l_xmm_gemm_beta_1;
  l_xmm_gemm_beta_1.gemm = libxsmm_dispatch_brgemm( l_shape_brgemm,
                                                    l_flags_brgemm,
                                                    l_prefetch_flags_brgemm,
                                                    l_brconfig );

  // allocate memory
  float * l_a = nullptr;
  float * l_b = nullptr;
  float * l_c = nullptr;

  posix_memalign( (void**) &l_a, 128, i_lda * i_k * sizeof(float) );
  posix_memalign( (void**) &l_b, 128, i_ldb * i_n * sizeof(float) );
  posix_memalign( (void**) &l_c, 128, i_ldc * i_n * sizeof(float) );

  // init the matrices using a normal distribution with mean 0 and variance 1
  std::random_device l_rd;
  std::mt19937 l_gen( l_rd() );
  std::normal_distribution<float> l_dist( 0.0, 1.0 );

  for( int64_t l_en = 0; l_en < i_lda * i_k; l_en++ ) {
    l_a[l_en] = l_dist( l_gen );
  }

  for( int64_t l_en = 0; l_en < i_ldb * i_n; l_en++ ) {
    l_b[l_en] = l_dist( l_gen );
  }

  for( int64_t l_en = 0; l_en < i_ldc * i_n; l_en++ ) {
    l_c[l_en] = l_dist( l_gen );
  }

  // ATen data structures
  at::Tensor l_a_aten = at::from_blob( l_a,
                                       {i_k, i_m},
                                       at::ScalarType::Float );

  at::Tensor l_b_aten = at::from_blob( l_b,
                                       {i_n, i_k},
                                       at::ScalarType::Float );

  at::Tensor l_c_aten = at::from_blob( l_c,
                                       {i_n, i_m},
                                       at::ScalarType::Float );

  // warmup and verification
  at::Tensor l_c_aten_ref = l_c_aten + at::matmul( l_b_aten,
                                                   l_a_aten );

  libxsmm_gemm_param l_gemm_param;
  l_gemm_param.a.primary = l_a;
  l_gemm_param.b.primary = l_b;
  l_gemm_param.c.primary = l_c;
  l_xmm_gemm_beta_1.gemm( &l_gemm_param );

  std::cout << "  max error: " << at::max( at::abs( l_c_aten_ref - l_c_aten ) ).item() << std::endl;

  // determine number of repetitions
  l_tp0 = std::chrono::steady_clock::now();
  for( int64_t l_re = 0; l_re < i_num_repetitions_initial; l_re++) {
    l_gemm_param.a.primary = l_a;
    l_gemm_param.b.primary = l_b;
    l_gemm_param.c.primary = l_c;
    l_xmm_gemm_beta_1.gemm( &l_gemm_param );
  }
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_num_repetitions = std::max( 1.0, (i_target_time * i_num_repetitions_initial) / l_dur.count() );
  l_num_flops = 2 * i_m * i_n * i_k * l_num_repetitions;

  l_tp0 = std::chrono::steady_clock::now();

  for( int64_t l_re = 0; l_re < l_num_repetitions; l_re++ ) {
    l_gemm_param.a.primary = l_a;
    l_gemm_param.b.primary = l_b;
    l_gemm_param.c.primary = l_c;
    l_xmm_gemm_beta_1.gemm( &l_gemm_param );
  }

  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time = l_dur.count();
  l_gflops = 1.0E-9 * l_num_flops / l_time;

  std::cout << "  #repetitions: " << l_num_repetitions << std::endl;
  std::cout << "  time: " << l_time << std::endl;
  std::cout << "  gflops: " << l_gflops << std::endl;
  std::cout << "CSV_DATA: "
            << "sgemm_xsmm,"
            << i_m << ","
            << i_n << ","
            << i_k << ","
            << i_lda << ","
            << i_ldb << ","
            << i_ldc << ","
            << l_num_flops << ","
            << l_time << ","
            << l_gflops
            << std::endl;

  // free memory
  free( l_a );
  free( l_b );
  free( l_c );
}

void cgemm_bench( int64_t i_m,
                  int64_t i_n,
                  int64_t i_k,
                  int64_t i_lda,
                  int64_t i_ldb,
                  int64_t i_ldc,
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

  posix_memalign( (void**) &l_a, 128, i_lda * i_k * 2 * sizeof(float) );
  posix_memalign( (void**) &l_b, 128, i_ldb * i_n * 2 * sizeof(float) );
  posix_memalign( (void**) &l_c, 128, i_ldc * i_n * 2 * sizeof(float) );

  // init the matrices using a normal distribution with mean 0 and variance 1
  std::random_device l_rd;
  std::mt19937 l_gen( l_rd() );
  std::normal_distribution<float> l_dist( 0.0, 1.0 );

  for( int64_t l_en = 0; l_en < 2 * i_lda * i_k; l_en++ ) {
    l_a[l_en] = l_dist( l_gen );
  }

  for( int64_t l_en = 0; l_en < 2 * i_ldb * i_n; l_en++ ) {
    l_b[l_en] = l_dist( l_gen );
  }

  for( int64_t l_en = 0; l_en < 2 * i_ldc * i_n; l_en++ ) {
    l_c[l_en] = l_dist( l_gen );
  }

  // ATen data structures
  at::Tensor l_a_aten = at::from_blob( l_a,
                                       {i_k, i_m},
                                       at::ScalarType::ComplexFloat );

  at::Tensor l_b_aten = at::from_blob( l_b,
                                       {i_n, i_k},
                                       at::ScalarType::ComplexFloat );

  at::Tensor l_c_aten = at::from_blob( l_c,
                                       {i_n, i_m},
                                       at::ScalarType::ComplexFloat );

  // warmup and verification
  at::Tensor l_c_aten_ref = l_c_aten + at::matmul( l_b_aten,
                                                   l_a_aten );

  float l_alpha[2] = {1, 0};
  float l_beta[2] = {1, 0};

  cblas_cgemm( CblasColMajor,
               CblasNoTrans,
               CblasNoTrans,
               i_m,
               i_n,
               i_k,
               l_alpha,
               l_a,
               i_lda,
               l_b,
               i_ldb,
               l_beta,
               l_c,
               i_ldc );

  std::cout << "  max error: " << at::max( at::abs( l_c_aten_ref - l_c_aten ) ).item() << std::endl;

  // determine number of repetitions
  l_tp0 = std::chrono::steady_clock::now();
  for( int64_t l_re = 0; l_re < i_num_repetitions_initial; l_re++) {
    cblas_cgemm( CblasColMajor,
                 CblasNoTrans,
                 CblasNoTrans,
                 i_m,
                 i_n,
                 i_k,
                 l_alpha,
                 l_a,
                 i_lda,
                 l_b,
                 i_ldb,
                 l_beta,
                 l_c,
                 i_ldc );
  }
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_num_repetitions = std::max( 1.0, (i_target_time * i_num_repetitions_initial) / l_dur.count() );
  l_num_flops = 2 * i_m * i_n * i_k * l_num_repetitions;

  l_tp0 = std::chrono::steady_clock::now();

  for( int64_t l_re = 0; l_re < l_num_repetitions; l_re++ ) {
    cblas_cgemm( CblasColMajor,
                 CblasNoTrans,
                 CblasNoTrans,
                 i_m,
                 i_n,
                 i_k,
                 l_alpha,
                 l_a,
                 i_lda,
                 l_b,
                 i_ldb,
                 l_beta,
                 l_c,
                 i_ldc );
  }

  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time = l_dur.count();
  l_gflops = 1.0E-9 * l_num_flops / l_time;

  std::cout << "  #repetitions: " << l_num_repetitions << std::endl;
  std::cout << "  time: " << l_time << std::endl;
  std::cout << "  gflops (3m): " << 3*l_gflops << std::endl;
  std::cout << "  gflops (4m): " << 4*l_gflops << std::endl;
  std::cout << "CSV_DATA: "
            << "cgemm,"
            << i_m << ","
            << i_n << ","
            << i_k << ","
            << i_lda << ","
            << i_ldb << ","
            << i_ldc << ","
            << 4*l_num_flops << ","
            << l_time << ","
            << 4*l_gflops
            << std::endl;

  // free memory
  free( l_a );
  free( l_b );
  free( l_c );
}

void cgemm_soa_3m( int64_t         i_m,
                   int64_t         i_n,
                   int64_t         i_k,
                   int64_t         i_lda,
                   int64_t         i_ldb,
                   int64_t         i_ldc,
                   float   const * i_a_real,
                   float   const * i_a_imag,
                   float   const * i_b_real,
                   float   const * i_b_imag,
                   float         * o_p1,
                   float         * o_p2,
                   float         * o_p3,
                   float         * o_q1,
                   float         * o_q2,
                   float         * o_q3,
                   float         * io_c_real,
                   float         * io_c_imag ) {
  for( int64_t l_k = 0; l_k < i_k; l_k++ ) {
#pragma omp simd
    for( int64_t l_m = 0; l_m < i_m; l_m++ ) {
      o_p1[l_k * i_m + l_m] = i_a_real[l_k * i_lda + l_m] + i_a_imag[l_k * i_lda + l_m];
    }
  }

  for( int64_t l_n = 0; l_n < i_n; l_n++ ) {
#pragma omp simd
    for( int64_t l_k = 0; l_k < i_k; l_k++ ) {
      o_p2[l_n * i_k + l_k] = i_b_imag[l_n * i_ldb + l_k] - i_b_real[l_n * i_ldb + l_k];
      o_p3[l_n * i_k + l_k] = i_b_real[l_n * i_ldb + l_k] + i_b_imag[l_n * i_ldb + l_k];
    }
  }

  cblas_sgemm( CblasColMajor,
               CblasNoTrans,
               CblasNoTrans,
               i_m,
               i_n,
               i_k,
               1,
               o_p1,
               i_m,
               i_b_real,
               i_ldb,
               0,
               o_q1,
               i_m );

  cblas_sgemm( CblasColMajor,
               CblasNoTrans,
               CblasNoTrans,
               i_m,
               i_n,
               i_k,
               1,
               i_a_real,
               i_lda,
               o_p2,
               i_k,
               0,
               o_q2,
               i_m );

  cblas_sgemm( CblasColMajor,
               CblasNoTrans,
               CblasNoTrans,
               i_m,
               i_n,
               i_k,
               1,
               i_a_imag,
               i_lda,
               o_p3,
               i_k,
               0,
               o_q3,
               i_m );

  for( int64_t l_n = 0; l_n < i_n; l_n++ ) {
#pragma omp simd
    for( int64_t l_m = 0; l_m < i_m; l_m++ ) {
      io_c_real[l_n * i_ldc + l_m] += o_q1[l_n * i_m + l_m] - o_q3[l_n * i_m + l_m];
      io_c_imag[l_n * i_ldc + l_m] += o_q1[l_n * i_m + l_m] + o_q2[l_n * i_m + l_m];
    }
  }
}
                

void cgemm_soa_3m_bench( int64_t i_m,
                         int64_t i_n,
                         int64_t i_k,
                         int64_t i_lda,
                         int64_t i_ldb,
                         int64_t i_ldc,
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

  float * l_p1 = nullptr;
  float * l_p2 = nullptr;
  float * l_p3 = nullptr;

  float * l_q1 = nullptr;
  float * l_q2 = nullptr;
  float * l_q3 = nullptr;

  posix_memalign( (void**) &l_a, 128, 2 * i_lda * i_k * sizeof(float) );
  posix_memalign( (void**) &l_b, 128, 2 * i_ldb * i_n * sizeof(float) );
  posix_memalign( (void**) &l_c, 128, 2 * i_ldc * i_n * sizeof(float) );

  posix_memalign( (void**) &l_p1, 128, i_k * i_m * sizeof(float) );
  posix_memalign( (void**) &l_p2, 128, i_n * i_k * sizeof(float) );
  posix_memalign( (void**) &l_p3, 128, i_n * i_k * sizeof(float) );

  posix_memalign( (void**) &l_q1, 128, i_n * i_m * sizeof(float) );
  posix_memalign( (void**) &l_q2, 128, i_n * i_m * sizeof(float) );
  posix_memalign( (void**) &l_q3, 128, i_n * i_m * sizeof(float) );

  // init the matrices using a normal distribution with mean 0 and variance 1
  std::random_device l_rd;
  std::mt19937 l_gen( l_rd() );
  std::normal_distribution<float> l_dist( 0.0, 1.0 );

  for( int64_t l_en = 0; l_en < 2 * i_lda * i_k; l_en++ ) {
    l_a[l_en] = l_dist( l_gen );
  }

  for( int64_t l_en = 0; l_en < 2 * i_ldb * i_n; l_en++ ) {
    l_b[l_en] = l_dist( l_gen );
  }

  for( int64_t l_en = 0; l_en < 2 * i_ldc * i_n; l_en++ ) {
    l_c[l_en] = l_dist( l_gen );
  }

  float * l_a_real = l_a;
  float * l_a_imag = l_a + i_lda * i_k;
  float * l_b_real = l_b;
  float * l_b_imag = l_b + i_ldb * i_n;
  float * l_c_real = l_c;
  float * l_c_imag = l_c + i_ldc * i_n;

  // ATen data structures
  at::Tensor l_a_aten = at::from_blob( l_a_real,
                                       {2, i_k, i_m},
                                       at::ScalarType::Float );
  l_a_aten = l_a_aten.permute( {1, 2, 0} ).contiguous();
  l_a_aten = at::view_as_complex( l_a_aten );

  at::Tensor l_b_aten = at::from_blob( l_b_real,
                                       {2, i_n, i_k},
                                       at::ScalarType::Float );
  l_b_aten = l_b_aten.permute( {1, 2, 0} ).contiguous();
  l_b_aten = at::view_as_complex( l_b_aten );

  at::Tensor l_c_aten = at::from_blob( l_c_real,
                                       {2, i_n, i_m},
                                       at::ScalarType::Float );
  l_c_aten = l_c_aten.permute( {1, 2, 0} ).contiguous();
  l_c_aten = at::view_as_complex( l_c_aten );

  // warmup and verification
  at::Tensor l_c_aten_ref = l_c_aten + at::matmul( l_b_aten,
                                                   l_a_aten );

  cgemm_soa_3m( i_m,
                i_n,
                i_k,
                i_lda,
                i_ldb,
                i_ldc,
                l_a_real,
                l_a_imag,
                l_b_real,
                l_b_imag,
                l_p1,
                l_p2,
                l_p3,
                l_q1,
                l_q2,
                l_q3,
                l_c_real,
                l_c_imag );

  l_c_aten = at::from_blob( l_c_real,
                            {2, i_n, i_m},
                            at::ScalarType::Float );
  l_c_aten = l_c_aten.permute( {1, 2, 0} ).contiguous();
  l_c_aten = at::view_as_complex( l_c_aten );

  std::cout << "  max error: " << at::max( at::abs( l_c_aten_ref - l_c_aten ) ).item() << std::endl;

  // determine number of repetitions
  l_tp0 = std::chrono::steady_clock::now();
  for( int64_t l_re = 0; l_re < i_num_repetitions_initial; l_re++) {
    cgemm_soa_3m( i_m,
                  i_n,
                  i_k,
                  i_lda,
                  i_ldb,
                  i_ldc,
                  l_a_real,
                  l_a_imag,
                  l_b_real,
                  l_b_imag,
                  l_p1,
                  l_p2,
                  l_p3,
                  l_q1,
                  l_q2,
                  l_q3,
                  l_c_real,
                  l_c_imag );
  }
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_num_repetitions = std::max( 1.0, (i_target_time * i_num_repetitions_initial) / l_dur.count() );
  l_num_flops = 2 * i_m * i_n * i_k * l_num_repetitions;

  l_tp0 = std::chrono::steady_clock::now();

  for( int64_t l_re = 0; l_re < l_num_repetitions; l_re++ ) {
    cgemm_soa_3m( i_m,
                  i_n,
                  i_k,
                  i_lda,
                  i_ldb,
                  i_ldc,
                  l_a_real,
                  l_a_imag,
                  l_b_real,
                  l_b_imag,
                  l_p1,
                  l_p2,
                  l_p3,
                  l_q1,
                  l_q2,
                  l_q3,
                  l_c_real,
                  l_c_imag );
  }

  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time = l_dur.count();
  l_gflops = 1.0E-9 * l_num_flops / l_time;

  std::cout << "  #repetitions: " << l_num_repetitions << std::endl;
  std::cout << "  time: " << l_time << std::endl;
  std::cout << "  gflops (3m): " << 3 * l_gflops << std::endl;
  std::cout << "  gflops (4m): " << 4 * l_gflops << std::endl;
  std::cout << "CSV_DATA: "
            << "cgemm_soa_3m,"
            << i_m << ","
            << i_n << ","
            << i_k << ","
            << i_lda << ","
            << i_ldb << ","
            << i_ldc << ","
            << 4*l_num_flops << ","
            << l_time << ","
            << 4*l_gflops
            << std::endl;

  // free memory
  free( l_a );
  free( l_b );
  free( l_c );

  free( l_p1 );
  free( l_p2 );
  free( l_p3 );
  free( l_q1 );
  free( l_q2 );
  free( l_q3 );
}

void cgemm_soa_4m( int64_t         i_m,
                   int64_t         i_n,
                   int64_t         i_k,
                   int64_t         i_lda,
                   int64_t         i_ldb,
                   int64_t         i_ldc,
                   float   const * i_a_real,
                   float   const * i_a_imag,
                   float   const * i_b_real,
                   float   const * i_b_imag,
                   float         * io_c_real,
                   float         * io_c_imag ) {
  cblas_sgemm( CblasColMajor,
               CblasNoTrans,
               CblasNoTrans,
               i_m,
               i_n,
               i_k,
               1.0,
               i_a_real,
               i_lda,
               i_b_real,
               i_ldb,
               1.0,
               io_c_real,
               i_ldc );

  cblas_sgemm( CblasColMajor,
               CblasNoTrans,
               CblasNoTrans,
               i_m,
               i_n,
               i_k,
               -1.0,
               i_a_imag,
               i_lda,
               i_b_imag,
               i_ldb,
               1.0,
               io_c_real,
               i_ldc );

  cblas_sgemm( CblasColMajor,
               CblasNoTrans,
               CblasNoTrans,
               i_m,
               i_n,
               i_k,
               1.0,
               i_a_real,
               i_lda,
               i_b_imag,
               i_ldb,
               1.0,
               io_c_imag,
               i_ldc );

  cblas_sgemm( CblasColMajor,
               CblasNoTrans,
               CblasNoTrans,
               i_m,
               i_n,
               i_k,
               1.0,
               i_a_imag,
               i_lda,
               i_b_real,
               i_ldb,
               1.0,
               io_c_imag,
               i_ldc );
}

void cgemm_soa_4m_bench( int64_t i_m,
                         int64_t i_n,
                         int64_t i_k,
                         int64_t i_lda,
                         int64_t i_ldb,
                         int64_t i_ldc,
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

  posix_memalign( (void**) &l_a, 128, 2 * i_lda * i_k * sizeof(float) );
  posix_memalign( (void**) &l_b, 128, 2 * i_ldb * i_n * sizeof(float) );
  posix_memalign( (void**) &l_c, 128, 2 * i_ldc * i_n * sizeof(float) );

  // init the matrices using a normal distribution with mean 0 and variance 1
  std::random_device l_rd;
  std::mt19937 l_gen( l_rd() );
  std::normal_distribution<float> l_dist( 0.0, 1.0 );

  for( int64_t l_en = 0; l_en < 2 * i_lda * i_k; l_en++ ) {
    l_a[l_en] = l_dist( l_gen );
  }

  for( int64_t l_en = 0; l_en < 2 * i_ldb * i_n; l_en++ ) {
    l_b[l_en] = l_dist( l_gen );
  }

  for( int64_t l_en = 0; l_en < 2 * i_ldc * i_n; l_en++ ) {
    l_c[l_en] = l_dist( l_gen );
  }

  float * l_a_real = l_a;
  float * l_a_imag = l_a + i_lda * i_k;
  float * l_b_real = l_b;
  float * l_b_imag = l_b + i_ldb * i_n;
  float * l_c_real = l_c;
  float * l_c_imag = l_c + i_ldc * i_n;

  // ATen data structures
  at::Tensor l_a_aten = at::from_blob( l_a_real,
                                       {2, i_k, i_m},
                                       at::ScalarType::Float );
  l_a_aten = l_a_aten.permute( {1, 2, 0} ).contiguous();
  l_a_aten = at::view_as_complex( l_a_aten );

  at::Tensor l_b_aten = at::from_blob( l_b_real,
                                       {2, i_n, i_k},
                                       at::ScalarType::Float );
  l_b_aten = l_b_aten.permute( {1, 2, 0} ).contiguous();
  l_b_aten = at::view_as_complex( l_b_aten );

  at::Tensor l_c_aten = at::from_blob( l_c_real,
                                       {2, i_n, i_m},
                                       at::ScalarType::Float );
  l_c_aten = l_c_aten.permute( {1, 2, 0} ).contiguous();
  l_c_aten = at::view_as_complex( l_c_aten );

  // warmup and verification
  at::Tensor l_c_aten_ref = l_c_aten + at::matmul( l_b_aten,
                                                   l_a_aten );

  cgemm_soa_4m( i_m,
                i_n,
                i_k,
                i_lda,
                i_ldb,
                i_ldc,
                l_a_real,
                l_a_imag,
                l_b_real,
                l_b_imag,
                l_c_real,
                l_c_imag );

  l_c_aten = at::from_blob( l_c_real,
                            {2, i_n, i_m},
                            at::ScalarType::Float );
  l_c_aten = l_c_aten.permute( {1, 2, 0} ).contiguous();
  l_c_aten = at::view_as_complex( l_c_aten );

  std::cout << "  max error: " << at::max( at::abs( l_c_aten_ref - l_c_aten ) ).item() << std::endl;


  // determine number of repetitions
  l_tp0 = std::chrono::steady_clock::now();
  for( int64_t l_re = 0; l_re < i_num_repetitions_initial; l_re++) {
    cgemm_soa_4m( i_m,
                  i_n,
                  i_k,
                  i_lda,
                  i_ldb,
                  i_ldc,
                  l_a_real,
                  l_a_imag,
                  l_b_real,
                  l_b_imag,
                  l_c_real,
                  l_c_imag );
  }
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_num_repetitions = std::max( 1.0, (i_target_time * i_num_repetitions_initial) / l_dur.count() );
  l_num_flops = 2 * i_m * i_n * i_k * l_num_repetitions;

  l_tp0 = std::chrono::steady_clock::now();

  for( int64_t l_re = 0; l_re < l_num_repetitions; l_re++ ) {
    cgemm_soa_4m( i_m,
                  i_n,
                  i_k,
                  i_lda,
                  i_ldb,
                  i_ldc,
                  l_a_real,
                  l_a_imag,
                  l_b_real,
                  l_b_imag,
                  l_c_real,
                  l_c_imag );
  }

  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time = l_dur.count();
  l_gflops = 1.0E-9 * l_num_flops / l_time;

  std::cout << "  #repetitions: " << l_num_repetitions << std::endl;
  std::cout << "  time: " << l_time << std::endl;
  std::cout << "  gflops (3m): " << 3 * l_gflops << std::endl;
  std::cout << "  gflops (4m): " << 4 * l_gflops << std::endl;
  std::cout << "CSV_DATA: "
            << "cgemm_soa_4m,"
            << i_m << ","
            << i_n << ","
            << i_k << ","
            << i_lda << ","
            << i_ldb << ","
            << i_ldc << ","
            << 4*l_num_flops << ","
            << l_time << ","
            << 4*l_gflops
            << std::endl;

  // free memory
  free( l_a );
  free( l_b );
  free( l_c );
}

void cgemm_soa_4m_xsmm_bench( int64_t i_m,
                              int64_t i_n,
                              int64_t i_k,
                              int64_t i_lda,
                              int64_t i_ldb,
                              int64_t i_ldc,
                              int64_t i_num_repetitions_initial = 10,
                              double  i_target_time = 10.0 ) {
  std::chrono::steady_clock::time_point l_tp0, l_tp1;
  std::chrono::duration< double > l_dur;
  int64_t l_num_flops = 0;
  int64_t l_num_repetitions = 0;
  double l_time = 0;
  double l_gflops = 0;

  // generate binary kernel
  libxsmm_meltw_binary_shape l_shape_binary = libxsmm_create_meltw_binary_shape( i_m,
                                                                                 i_n,
                                                                                 i_ldc,
                                                                                 i_m,
                                                                                 i_ldc,
                                                                                 libxsmm_datatype::LIBXSMM_DATATYPE_F32,
                                                                                 libxsmm_datatype::LIBXSMM_DATATYPE_F32,
                                                                                 libxsmm_datatype::LIBXSMM_DATATYPE_F32,
                                                                                 libxsmm_datatype::LIBXSMM_DATATYPE_F32 );

  libxsmm_meltwfunction_binary m_xmm_binary_sub = libxsmm_dispatch_meltw_binary( libxsmm_meltw_binary_type::LIBXSMM_MELTW_TYPE_BINARY_SUB,
                                                                                 l_shape_binary,
                                                                                 LIBXSMM_MELTW_FLAG_BINARY_NONE );

  // generate GEMM kernels
  libxsmm_gemm_shape l_shape_brgemm;
  libxsmm_bitfield l_flags_brgemm = LIBXSMM_GEMM_FLAGS('N', 'N');
  libxsmm_bitfield l_prefetch_flags_brgemm = 0;

  l_shape_brgemm = libxsmm_create_gemm_shape( i_m,
                                              i_n,
                                              i_k,
                                              i_lda,
                                              i_ldb,
                                              i_ldc,
                                              libxsmm_datatype::LIBXSMM_DATATYPE_F32,
                                              libxsmm_datatype::LIBXSMM_DATATYPE_F32,
                                              libxsmm_datatype::LIBXSMM_DATATYPE_F32,
                                              libxsmm_datatype::LIBXSMM_DATATYPE_F32 );

  libxsmm_gemm_batch_reduce_config l_brconfig;
  l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_NONE;
  l_brconfig.br_stride_a_hint = 0;
  l_brconfig.br_stride_b_hint = 0;
  l_brconfig.br_unroll_hint = 0;

  libxsmm_xmmfunction l_xmm_gemm_beta_1;
  l_xmm_gemm_beta_1.gemm = libxsmm_dispatch_brgemm( l_shape_brgemm,
                                                    l_flags_brgemm,
                                                    l_prefetch_flags_brgemm,
                                                    l_brconfig );

  libxsmm_xmmfunction l_xmm_gemm_beta_0;
  l_flags_brgemm |= LIBXSMM_BASIC_GEMM_FLAG_BETA_0;
  l_xmm_gemm_beta_0.gemm = libxsmm_dispatch_brgemm( l_shape_brgemm,
                                                    l_flags_brgemm,
                                                    l_prefetch_flags_brgemm,
                                                    l_brconfig );

  // allocate memory
  float * l_a = nullptr;
  float * l_b = nullptr;
  float * l_c = nullptr;
  float * l_tmp = nullptr;

  posix_memalign( (void**) &l_a, 128, 2 * i_lda * i_k * sizeof(float) );
  posix_memalign( (void**) &l_b, 128, 2 * i_ldb * i_n * sizeof(float) );
  posix_memalign( (void**) &l_c, 128, 2 * i_ldc * i_n * sizeof(float) );
  posix_memalign( (void**) &l_tmp, 128, i_m * i_n * sizeof(float) );

  // init the matrices using a normal distribution with mean 0 and variance 1
  std::random_device l_rd;
  std::mt19937 l_gen( l_rd() );
  std::normal_distribution<float> l_dist( 0.0, 1.0 );

  for( int64_t l_en = 0; l_en < 2 * i_lda * i_k; l_en++ ) {
    l_a[l_en] = l_dist( l_gen );
  }

  for( int64_t l_en = 0; l_en < 2 * i_ldb * i_n; l_en++ ) {
    l_b[l_en] = l_dist( l_gen );
  }

  for( int64_t l_en = 0; l_en < 2 * i_ldc * i_n; l_en++ ) {
    l_c[l_en] = l_dist( l_gen );
  }

  float * l_a_real = l_a;
  float * l_a_imag = l_a + i_lda * i_k;
  float * l_b_real = l_b;
  float * l_b_imag = l_b + i_ldb * i_n;
  float * l_c_real = l_c;
  float * l_c_imag = l_c + i_ldc * i_n;

  // ATen data structures
  at::Tensor l_a_aten = at::from_blob( l_a_real,
                                       {2, i_k, i_m},
                                       at::ScalarType::Float );
  l_a_aten = l_a_aten.permute( {1, 2, 0} ).contiguous();
  l_a_aten = at::view_as_complex( l_a_aten );

  at::Tensor l_b_aten = at::from_blob( l_b_real,
                                       {2, i_n, i_k},
                                       at::ScalarType::Float );
  l_b_aten = l_b_aten.permute( {1, 2, 0} ).contiguous();
  l_b_aten = at::view_as_complex( l_b_aten );

  at::Tensor l_c_aten = at::from_blob( l_c_real,
                                       {2, i_n, i_m},
                                       at::ScalarType::Float );
  l_c_aten = l_c_aten.permute( {1, 2, 0} ).contiguous();
  l_c_aten = at::view_as_complex( l_c_aten );

  // warmup and verification
  at::Tensor l_c_aten_ref = l_c_aten + at::matmul( l_b_aten,
                                                   l_a_aten );

  libxsmm_gemm_param l_gemm_param;
  libxsmm_meltw_binary_param l_binary_param;
  
  // real += real * real
  l_gemm_param.a.primary = l_a_real;
  l_gemm_param.b.primary = l_b_real;
  l_gemm_param.c.primary = l_c_real;
  l_xmm_gemm_beta_1.gemm( &l_gemm_param );

  // imag += real * imag
  l_gemm_param.a.primary = l_a_real;
  l_gemm_param.b.primary = l_b_imag;
  l_gemm_param.c.primary = l_c_imag;
  l_xmm_gemm_beta_1.gemm( &l_gemm_param );

  // imag += imag * real
  l_gemm_param.a.primary = l_a_imag;
  l_gemm_param.b.primary = l_b_real;
  l_gemm_param.c.primary = l_c_imag;
  l_xmm_gemm_beta_1.gemm( &l_gemm_param );

  // real -= imag * imag
  l_gemm_param.a.primary = l_a_imag;
  l_gemm_param.b.primary = l_b_imag;
  l_gemm_param.c.primary = l_tmp;
  l_xmm_gemm_beta_0.gemm( &l_gemm_param );

  l_binary_param.in0.primary = l_c;
  l_binary_param.in1.primary = l_tmp;
  l_binary_param.out.primary = l_c;
  m_xmm_binary_sub( &l_binary_param );

  l_c_aten = at::from_blob( l_c_real,
                            {2, i_n, i_m},
                            at::ScalarType::Float );
  l_c_aten = l_c_aten.permute( {1, 2, 0} ).contiguous();
  l_c_aten = at::view_as_complex( l_c_aten );

  std::cout << "  max error: " << at::max( at::abs( l_c_aten_ref - l_c_aten ) ).item() << std::endl;


  // determine number of repetitions
  l_tp0 = std::chrono::steady_clock::now();
  for( int64_t l_re = 0; l_re < i_num_repetitions_initial; l_re++) {
    // real += real * real
    l_gemm_param.a.primary = l_a_real;
    l_gemm_param.b.primary = l_b_real;
    l_gemm_param.c.primary = l_c_real;
    l_xmm_gemm_beta_1.gemm( &l_gemm_param );

    // imag += real * imag
    l_gemm_param.a.primary = l_a_real;
    l_gemm_param.b.primary = l_b_imag;
    l_gemm_param.c.primary = l_c_imag;
    l_xmm_gemm_beta_1.gemm( &l_gemm_param );

    // imag += imag * real
    l_gemm_param.a.primary = l_a_imag;
    l_gemm_param.b.primary = l_b_real;
    l_gemm_param.c.primary = l_c_imag;
    l_xmm_gemm_beta_1.gemm( &l_gemm_param );

    // real -= imag * imag
    l_gemm_param.a.primary = l_a_imag;
    l_gemm_param.b.primary = l_b_imag;
    l_gemm_param.c.primary = l_tmp;
    l_xmm_gemm_beta_0.gemm( &l_gemm_param );

    l_binary_param.in0.primary = l_c;
    l_binary_param.in1.primary = l_tmp;
    l_binary_param.out.primary = l_c;
    m_xmm_binary_sub( &l_binary_param );
  }
  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_num_repetitions = std::max( 1.0, (i_target_time * i_num_repetitions_initial) / l_dur.count() );
  l_num_flops = 2 * i_m * i_n * i_k * l_num_repetitions;

  l_tp0 = std::chrono::steady_clock::now();

  for( int64_t l_re = 0; l_re < l_num_repetitions; l_re++ ) {
    // real += real * real
    l_gemm_param.a.primary = l_a_real;
    l_gemm_param.b.primary = l_b_real;
    l_gemm_param.c.primary = l_c_real;
    l_xmm_gemm_beta_1.gemm( &l_gemm_param );

    // imag += real * imag
    l_gemm_param.a.primary = l_a_real;
    l_gemm_param.b.primary = l_b_imag;
    l_gemm_param.c.primary = l_c_imag;
    l_xmm_gemm_beta_1.gemm( &l_gemm_param );

    // imag += imag * real
    l_gemm_param.a.primary = l_a_imag;
    l_gemm_param.b.primary = l_b_real;
    l_gemm_param.c.primary = l_c_imag;
    l_xmm_gemm_beta_1.gemm( &l_gemm_param );

    // real -= imag * imag
    l_gemm_param.a.primary = l_a_imag;
    l_gemm_param.b.primary = l_b_imag;
    l_gemm_param.c.primary = l_tmp;
    l_xmm_gemm_beta_0.gemm( &l_gemm_param );

    l_binary_param.in0.primary = l_c;
    l_binary_param.in1.primary = l_tmp;
    l_binary_param.out.primary = l_c;
    m_xmm_binary_sub( &l_binary_param );
  }

  l_tp1 = std::chrono::steady_clock::now();
  l_dur = std::chrono::duration_cast< std::chrono::duration< double> >( l_tp1 - l_tp0 );
  l_time = l_dur.count();
  l_gflops = 1.0E-9 * l_num_flops / l_time;

  std::cout << "  #repetitions: " << l_num_repetitions << std::endl;
  std::cout << "  time: " << l_time << std::endl;
  std::cout << "  gflops (3m): " << 3 * l_gflops << std::endl;
  std::cout << "  gflops (4m): " << 4 * l_gflops << std::endl;
  std::cout << "CSV_DATA: "
            << "cgemm_soa_4m_xsmm,"
            << i_m << ","
            << i_n << ","
            << i_k << ","
            << i_lda << ","
            << i_ldb << ","
            << i_ldc << ","
            << 4*l_num_flops << ","
            << l_time << ","
            << 4*l_gflops
            << std::endl;

  // free memory
  free( l_a );
  free( l_b );
  free( l_c );
}

int main( int     i_argc,
          char  * i_argv[] ) {
  std::cout << "running bench_gemm!" << std::endl;

  bool l_bias = true;
  bool l_relu = true;

  int64_t l_m = 64;
  int64_t l_n = 64;
  int64_t l_k = 64;
  int64_t l_lda = 64;
  int64_t l_ldb = 64;
  int64_t l_ldc = 64;
  std::string l_bench = "all";
  int64_t l_num_repetitions_initial = 10;
  double  l_target_time = 10.0;

  if( i_argc == 1 ) {}
  else if( i_argc >= 7 ) {
    l_m   = std::stoi( i_argv[1] );
    l_n   = std::stoi( i_argv[2] );
    l_k   = std::stoi( i_argv[3] );
    l_lda = std::stoi( i_argv[4] );
    l_ldb = std::stoi( i_argv[5] );
    l_ldc = std::stoi( i_argv[6] );
  }
  else {
    std::cerr << "usage: ./bench_gemm M N K LDA LDB LDC BENCH TIME" << std::endl;
    std::cerr << "example ./bench_gemm 64 64 64 64 64 64 all 10.0" << std::endl;
    return EXIT_FAILURE;
  }

  if( i_argc >= 8 ) {
    l_bench = i_argv[7];
    l_target_time = std::stod( i_argv[8] );
  }

  std::cout << "  M:     " << l_m << std::endl;
  std::cout << "  N:     " << l_n << std::endl;
  std::cout << "  K:     " << l_k << std::endl;
  std::cout << "  LDA:   " << l_lda << std::endl;
  std::cout << "  LDB:   " << l_ldb << std::endl;
  std::cout << "  LDC:   " << l_ldc << std::endl;
  std::cout << "  BENCH: " << l_bench << std::endl;
  std::cout << "  TIME:  " << l_target_time << std::endl;
  std::cout << std::endl;

  if( l_bench == "all" || l_bench == "sgemm" ) {
    std::cout << "*** running SGEMM ***" << std::endl;
    sgemm_bench( l_m,
                l_n,
                l_k,
                l_lda,
                l_ldb,
                l_ldc,
                l_num_repetitions_initial,
                l_target_time );
    std::cout << std::endl;
  }

  if( l_bench == "all" || l_bench == "sgemm_xsmm" ) {
    std::cout << "*** running SGEMM_XSMM ***" << std::endl;
    sgemm_xsmm_bench( l_m,
                      l_n,
                      l_k,
                      l_lda,
                      l_ldb,
                      l_ldc,
                      l_num_repetitions_initial,
                      l_target_time );
    std::cout << std::endl;
  }

  if( l_bench == "all" || l_bench == "cgemm" ) {
    std::cout << "*** running CGEMM ***" << std::endl;
    cgemm_bench( l_m,
                 l_n,
                 l_k,
                 l_lda,
                 l_ldb,
                 l_ldc,
                 l_num_repetitions_initial,
                 l_target_time );
    std::cout << std::endl;
  }

  if( l_bench == "all" || l_bench == "cgemm_soa_3m" ) {
    std::cout << "*** running CGEMM_SOA_3M ***" << std::endl;
    cgemm_soa_3m_bench( l_m,
                        l_n,
                        l_k,
                        l_lda,
                        l_ldb,
                        l_ldc,
                        l_num_repetitions_initial,
                        l_target_time );
    std::cout << std::endl;
  }

  if( l_bench == "all" || l_bench == "cgemm_soa_4m" ) {
    std::cout << "*** running CGEMM_SOA_4M ***" << std::endl;
    cgemm_soa_4m_bench( l_m,
                        l_n,
                        l_k,
                        l_lda,
                        l_ldb,
                        l_ldc,
                        l_num_repetitions_initial,
                        l_target_time );
    std::cout << std::endl;
  }

  if( l_bench == "all" || l_bench == "cgemm_soa_4m_xsmm" ) {
    std::cout << "*** running CGEMM_SOA_4M_XSMM ***" << std::endl;
    cgemm_soa_4m_xsmm_bench( l_m,
                            l_n,
                            l_k,
                            l_lda,
                            l_ldb,
                            l_ldc,
                            l_num_repetitions_initial,
                            l_target_time );
    std::cout << std::endl;
  }

  std::cout << "finished running bench_gemm!" << std::endl;
  return EXIT_SUCCESS;
}