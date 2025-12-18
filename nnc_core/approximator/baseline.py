import copy
import numpy as np
import deepCABAC # O módulo C++
import nnc_core
from nnc_core.nnr_model import NNRModelAccess
from nnc_core.coder import hls, baseline
from .. import common
import pandas as pd
import os # Para criar a pasta de log


def approx(approx_info, model_info, approx_data_in):
    
    # Cria a cópia do dicionário de saída
    approx_data_out = {k: copy.copy(v) for k, v in approx_data_in.items()}
    
    model_access = NNRModelAccess(model_info)

    # --- FASE 1: Coletar informações para todos os blocos ---
    block_info_list_for_cpp = []
    output_qindex_arrays = {} # Dicionário para guardar referências aos arrays de saída

    print("Coletando informações dos blocos para C++...")
    
    for block_or_param in model_access.blocks_and_params():
        for par_type, param, _ in block_or_param.param_generator(approx_data_in["compressed_parameter_types"]):
            if (par_type in approx_info["to_approximate"]) and (param not in approx_data_in["approx_method"]):
                
                original_weights = approx_data_in["parameters"][param]
                
                # --- IMPORTANTE: Pré-alocar o array de saída NumPy ---
                # O C++ escreverá diretamente aqui. DEVE ser C-contíguo.
                quantizedValues = np.zeros_like(original_weights, dtype=np.int32, order='C')
                output_qindex_arrays[param] = quantizedValues # Guarda a referência
                
                # Calcular qStepSize (lógica do bindings.cpp)
                qp = approx_info['qp'][param]
                qpDensity = approx_data_in['qp_density']
                k = 1 << qpDensity
                mul = k + (qp & (k-1))
                shift = qp >> qpDensity
                qStepSize = mul * pow(2.0, shift - qpDensity)

                # Monta o dicionário para este bloco
                block_info = {
                    'param_name': param,
                    'weights': original_weights, # Array NumPy original
                    'qindex': quantizedValues,   # Array NumPy de saída (pré-alocado)
                    'qStepSize': qStepSize,
                    'lambdaScale': approx_info["lambda_scale"],
                    'dq_flag': approx_info['dq_flag'][param],
                    'maxNumNoRem': approx_info["cabac_unary_length_minus1"],
                    'scan_order': approx_data_in["scan_order"].get(param, 0),
                    'qp': qp, # QP original
                    'qpDensity': approx_data_in['qp_density']
                }
                block_info_list_for_cpp.append(block_info)
    
    print(f"Total de {len(block_info_list_for_cpp)} blocos preparados. Chamando C++ Pthreads...")
    
    # --- FASE 2: Chamada ÚNICA para a função C++ paralela ---
    # Certifique-se que a pasta de log existe
    os.makedirs("C:\\Henrique", exist_ok=True) 

    # Chama a nova função C++ que faz o trabalho pesado em paralelo
    cpp_results = deepCABAC.quantize_all_blocks_parallel(block_info_list_for_cpp)

    print("C++ Pthreads concluído. Processando resultados...")

    # --- FASE 3: Atualizar o dicionário de saída ---
    for result_dict in cpp_results:
        param = result_dict['param_name']
        final_qp = result_dict['final_qp']

        final_dq_flag = result_dict['dq_flag'] # Pega o dq_flag retornado pelo C++
        
        original_qp = approx_info['qp'][param]
        if final_qp != original_qp:
             print("INFO: QP for {} has been clipped from {} to {} to avoid int32_t overflow!".format(param, original_qp, final_qp))
        
        # Atualiza o dicionário de saída
        approx_data_out['qp'][param] = final_qp
        approx_data_out['parameters'][param] = output_qindex_arrays[param] # Usa o array que foi modificado
        approx_data_out['approx_method'][param] = 'uniform'
        
        approx_data_out['dq_flag'][param] = final_dq_flag # Atualiza o dicionário dq_flag

    print("Todos os resultados foram coletados.")
    return approx_data_out

def rec(param, approx_data):
    assert approx_data['parameters'][param].dtype == np.int32

    decoder = deepCABAC.Decoder()
    values = approx_data['parameters'][param]

    approx_data["parameters"][param] = np.zeros(values.shape, dtype=np.float32)
    decoder.dequantLayer(approx_data["parameters"][param], values, approx_data["qp_density"], approx_data["qp"][param], approx_data['scan_order'].get(param, 0))


    del approx_data["approx_method"][param]
