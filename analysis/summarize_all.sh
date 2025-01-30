machines="grace spr ryzen_8700g"

for machine in $machines
do
  mkdir -p data/${machine}/fc
  log_dir_tpp=logs
  # append _ht if SPR
  if [ $machine == "spr" ]; then
    log_dir_tpp=logs_ht
  fi

  python summarize.py --input ../${machine}/${log_dir_tpp}/fc/einsum_ir_backend_tpp_reorder_dims_0.log --code einsum_ir --backend tpp --output data/${machine}/fc/einsum_ir_backend_tpp.csv
done

for machine in $machines
do
  mkdir -p data/${machine}/tccg_blocked_reordered

  log_dir_tpp=logs
  # append _ht if SPR
  if [ $machine == "spr" ]; then
    log_dir_tpp=logs_ht
  fi

  python summarize.py --input ../${machine}/logs/tccg_blocked_reordered/opt_einsum_backend_torch.log --code opt_einsum --backend torch --output data/${machine}/tccg_blocked_reordered/opt_einsum_backend_torch.csv

  python summarize.py --input ../${machine}/${log_dir_tpp}/tccg_blocked_reordered/einsum_ir_backend_tpp_reorder_dims_0.log --code einsum_ir --backend tpp --output data/${machine}/tccg_blocked_reordered/einsum_ir_backend_tpp.csv

  python summarize.py --input ../${machine}/logs/tccg_blocked_reordered/einsum_ir_backend_tpp_reorder_dims_0.log --code at::einsum --backend eigen --output data/${machine}/tccg_blocked_reordered/aten_backend_eigen.csv

  python summarize.py --input ../${machine}/logs/tccg_blocked_reordered/einsum_ir_backend_blas_reorder_dims_0.log --code einsum_ir --backend blas --output data/${machine}/tccg_blocked_reordered/einsum_ir_backend_blas.csv

  python summarize.py --input ../${machine}/logs/tccg_blocked_reordered/einsum_ir_backend_tblis_reorder_dims_0.log --code einsum_ir --backend tblis --output data/${machine}/tccg_blocked_reordered/einsum_ir_backend_tblis.csv

  python summarize.py --input ../${machine}/logs/tccg_blocked_reordered/tvm.log --code tvm --backend ansor --output data/${machine}/tccg_blocked_reordered/tvm_backend_ansor.csv

  trees="syn tt fctn tw getd trn mera tnlm"

  for tree in ${trees}
  do
    mkdir -p data/${machine}/${tree}

    log_dir_tpp=logs
    # append _ht if SPR
    if [ $machine == "spr" ]; then
      log_dir_tpp=logs_ht
    fi

    python summarize.py --input ../${machine}/logs/${tree}/opt_einsum_backend_torch.log --code opt_einsum --backend torch --output data/${machine}/${tree}/opt_einsum_backend_torch.csv

    python summarize.py --input ../${machine}/${log_dir_tpp}/${tree}/einsum_ir_backend_tpp_reorder_dims_1.log --code einsum_ir --backend tpp --output data/${machine}/${tree}/einsum_ir_backend_tpp.csv

    python summarize.py --input ../${machine}/logs/${tree}/einsum_ir_backend_tpp_reorder_dims_1.log --code at::einsum --backend eigen --output data/${machine}/${tree}/aten_backend_eigen.csv

    python summarize.py --input ../${machine}/logs/${tree}/einsum_ir_backend_blas_reorder_dims_1.log --code einsum_ir --backend blas --output data/${machine}/${tree}/einsum_ir_backend_blas.csv

    python summarize.py --input ../${machine}/logs/${tree}/einsum_ir_backend_tblis_reorder_dims_1.log --code einsum_ir --backend tblis --output data/${machine}/${tree}/einsum_ir_backend_tblis.csv

    python summarize.py --input ../${machine}/logs/${tree}/tvm.log --code tvm --backend ansor --output data/${machine}/${tree}/tvm_backend_ansor.csv
  done
done

python summarize.py --input ../grace/logs/fc/nvpl.log --code sgemm --backend nvpl --output data/grace/fc/sgemm_backend_nvpl.csv

python summarize.py --input ../ryzen_8700g/logs/fc/openblas.log --code sgemm --backend openblas --output data/ryzen_8700g/fc/sgemm_backend_openblas.csv