#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <locale>
#include <iomanip>
#include <algorithm>
#include <cmath>

#define HAVE_STRUCT_TIMESPEC 1
#include <thread>
#include <pthread.h> // Pthreads
#include <atomic> 

#include <Lib/CommonLib/TypeDef.h>
#include <Lib/CommonLib/Quant.h>
#include <Lib/CommonLib/Scan.h> // Se for usar logging CSV

namespace py = pybind11;

// Estrutura BlockQuantInfo (sem mudanças)
struct BlockQuantInfo {
    std::string param_name;
    py::array_t<float32_t, py::array::c_style | py::array::forcecast> weights_array;
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> qindex_array;
    uint32_t numWeights;
    uint32_t layerWidth;
    float32_t qStepSize;
    float32_t lambdaScale;
    uint8_t dq_flag;
    uint32_t maxNumNoRem;
    int32_t scan_order;
    int32_t original_qp;
    int32_t qpDensity;
};

// --- ESTRUTURA DE DADOS PARA THREADS (MODIFICADA PARA ATOMIC) ---
struct ThreadWorkerDataAtomic {
    int thread_id;
    const std::vector<BlockQuantInfo>* all_block_infos; // Ponteiro para TODOS os blocos
    std::vector<int32_t>* all_final_qps;            // Ponteiro para TODOS os resultados
    std::atomic<int>* next_block_idx_ptr;           // Ponteiro para o contador ATÔMICO compartilhado
    int total_num_blocks;                           // Número total de blocos (para condição de parada)
};

// Função Worker (sem mudanças significativas na lógica principal)
void* quantize_blocks_pthread_worker_atomic(void* arg) {
    ThreadWorkerDataAtomic* data = static_cast<ThreadWorkerDataAtomic*>(arg);

    // Loop principal da thread: pega e processa blocos até acabar
    while (true) {
        // Pega o próximo índice de forma atômica e incrementa o contador
        int block_idx = (*(data->next_block_idx_ptr))++;

        // Verifica se o índice pego é válido
        if (block_idx >= data->total_num_blocks) {
            break; // Não há mais blocos para esta thread, sai do loop
        }

        // Processa o bloco com o índice obtido
        const BlockQuantInfo& info = (*(data->all_block_infos))[block_idx];

        // --- Obtém ponteiros  ---
        float32_t* pWeights = nullptr;
        int32_t* pQIndex = nullptr;
        try {
            py::buffer_info bi_weights = info.weights_array.request(true);
            py::buffer_info bi_qindex = info.qindex_array.request(true);
            pWeights = static_cast<float32_t*>(bi_weights.ptr);
            pQIndex = static_cast<int32_t*>(bi_qindex.ptr);
        } catch (const std::exception& e) {
             std::cerr << "[Thread " << data->thread_id << "] ERRO buffers NumPy para " << info.param_name << " (idx=" << block_idx << "): " << e.what() << std::endl;
             continue; // Pula este bloco, tenta pegar o próximo
        }
        // --- Fim da obtenção de ponteiros ---

        int32_t current_qp = info.original_qp;
        float32_t current_qStepSize = info.qStepSize;

        // Chamada quantize
        int32_t success = quantize(
            pWeights,               // Ponteiro para os pesos originais
            pQIndex,                // Ponteiro para o array onde os níveis serão escritos
            current_qStepSize,      // O qStep calculado (pode ter sido ajustado)
            info.layerWidth,        // O stride
            info.numWeights,        // O número total de pesos
            DIST_MSE,               // O tipo de distorção (assumindo MSE como antes)
            info.lambdaScale,       // O fator lambda
            info.dq_flag,           // O flag TCQ/URQ
            info.maxNumNoRem,       // Parâmetro do CABAC
            info.scan_order         // A ordem de varredura
        );

        // Lógica de ajuste de QP
        if (!success) {
             if (!success) {
                  // Protege escrita concorrente no cerr
                 #pragma omp critical // Usa pragma omp critical mesmo sem omp parallel for
                 {
                     std::cerr << "[Thread " << data->thread_id << "] ERRO FATAL: Overflow para " << info.param_name << " mesmo após ajuste!" << std::endl;
                 }
             }
        }

        // Escreve o QP final
        (*(data->all_final_qps))[block_idx] = current_qp;

        // --- Logging CSV (Opcional) ---
    } // Fim do loop sobre block_indices_to_process

    pthread_exit(nullptr);
    return nullptr;
}


py::list quantize_all_blocks_parallel_pthreads(py::list py_block_info_list) {

    // 1. Extrair informações do Python
    std::vector<BlockQuantInfo> block_infos;
    std::vector<int32_t> final_qps;
    try {
        block_infos.reserve(py_block_info_list.size());
        for (const auto& item : py_block_info_list) {
            py::dict block_dict = item.cast<py::dict>();
            BlockQuantInfo info; // <-- Declarada dentro do loop

            info.param_name = block_dict["param_name"].cast<std::string>();
            info.weights_array = block_dict["weights"].cast<py::array_t<float32_t, py::array::c_style | py::array::forcecast>>();
            info.qindex_array = block_dict["qindex"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
            py::buffer_info bi_weights = info.weights_array.request();
            info.numWeights = 1; info.layerWidth = 1;
            for (py::ssize_t i = 0; i < bi_weights.ndim; ++i) { info.numWeights *= bi_weights.shape[i]; if (i > 0) info.layerWidth *= bi_weights.shape[i];}
            if (bi_weights.ndim <= 1) info.layerWidth = 1;
            info.qStepSize = block_dict["qStepSize"].cast<float32_t>();
            info.lambdaScale = block_dict["lambdaScale"].cast<float32_t>();
            info.dq_flag = block_dict["dq_flag"].cast<uint8_t>();
            info.maxNumNoRem = block_dict["maxNumNoRem"].cast<uint32_t>();
            info.scan_order = block_dict["scan_order"].cast<int32_t>();
            info.original_qp = block_dict["qp"].cast<int32_t>();
            info.qpDensity = block_dict["qpDensity"].cast<int32_t>();
            if (info.layerWidth == 1 || info.numWeights == info.layerWidth) info.scan_order = 0;

            block_infos.push_back(std::move(info)); // push_back DENTRO do loop
        }
        final_qps.resize(block_infos.size());
    } catch (const std::exception& e) {
        py::gil_scoped_acquire acquire_gil;
        throw std::runtime_error(std::string("Erro ao extrair dados do Python: ") + e.what());
    }

    // --- Gerenciamento de Threads Pthreads com Contador Atômico ---
    int num_blocks = static_cast<int>(block_infos.size());
    if (num_blocks == 0) return py::list();

    int num_threads = std::thread::hardware_concurrency(); 
    if (num_threads == 0) { // Fallback se a detecção falhar
        num_threads = 12; // Ou um valor padrão razoável como 8
        std::cout << "[Pthreads] Aviso: Não foi possível detectar o número de núcleos, usando " << num_threads << " threads." << std::endl;
    } else {
         std::cout << "[Pthreads] Detectado " << num_threads << " threads de hardware." << std::endl;
         // Você pode optar por usar todos ou limitar (ex: num_threads = std::max(1, num_threads - 1); // deixa um núcleo livre)
    }
    std::cout << "[Pthreads Atomic] Usando " << num_threads << " threads para " << num_blocks << " blocos." << std::endl;
    std::vector<pthread_t> threads(num_threads);
    std::vector<ThreadWorkerDataAtomic> thread_worker_data(num_threads);
    std::atomic<int> next_block_idx(0);
    std::vector<std::vector<int>> block_indices_per_thread(num_threads);
    std::vector<bool> thread_launched_successfully(num_threads, false); 
    py::gil_scoped_release release_gil;

    // Lança as threads
    for (int i = 0; i < num_threads; ++i) {
        // Preenche a estrutura de dados para esta thread
        thread_worker_data[i].thread_id = i;
        thread_worker_data[i].all_block_infos = &block_infos;
        thread_worker_data[i].all_final_qps = &final_qps;
        thread_worker_data[i].next_block_idx_ptr = &next_block_idx; // Passa ponteiro p/ contador
        thread_worker_data[i].total_num_blocks = num_blocks;       // Passa o total de blocos

        // Cria a thread, passando a nova função worker
        int rc = pthread_create(&threads[i], nullptr, quantize_blocks_pthread_worker_atomic, &thread_worker_data[i]);
        if (rc == 0) { // Sucesso na criação
             thread_launched_successfully[i] = true; // Marca como sucesso
        } else {
             #pragma omp critical // Protege cerr
             {
                std::cerr << "ERRO: pthread_create falhou para thread " << i << " com código " << rc << std::endl;
             }
        }
    }

    // Espera (Join) as threads terminarem
    std::cout << "[Pthreads Atomic] Esperando threads terminarem..." << std::endl;
    for (int i = 0; i < num_threads; ++i) {
         // Usa o flag booleano para decidir se faz join
         if (thread_launched_successfully[i]) {
            pthread_join(threads[i], nullptr);
         }
    }
    std::cout << "[Pthreads Atomic] Todas as threads terminaram." << std::endl;
    py::gil_scoped_acquire acquire_gil;

    // Monta a lista de resultados
    py::list py_results;
    for (int i = 0; i < num_blocks; ++i) {
        py::dict result_dict;
        result_dict["param_name"] = block_infos[i].param_name;
        result_dict["final_qp"] = final_qps[i];
        result_dict["dq_flag"] = block_infos[i].dq_flag;
        py_results.append(result_dict);
    }
    return py_results;

}
