# Importar a biblioteca nnc, que agora está instalada no seu ambiente
import nnc

# ---- DEFINIÇÃO DOS CAMINHOS DO EXEMPLO ----
# Caminho para o modelo original de exemplo
input_model = './example/squeezenet1_1_pytorch_zoo.pt'

# Caminho onde o ficheiro comprimido será guardado
bitstream_output = './example/bitstream_squeezenet1_1.nnc'

# Caminho onde o modelo reconstruído será guardado
reconstructed_output = './example/reconstructed_squeezenet1_1.pt'


# ---- EXECUÇÃO DO PROCESSO ----

print(f"Passo 1: A comprimir o modelo '{input_model}'...")

# Chama a função de compressão
nnc.compress_model(input_model, bitstream_path=bitstream_output)

print(f"Compressão concluída! Bitstream guardado em '{bitstream_output}'")
print("-" * 30)


print(f"Passo 2: A descomprimir o bitstream '{bitstream_output}'...")

# Chama a função de descompressão
nnc.decompress_model(bitstream_output, model_path=reconstructed_output)

print(f"Descompressão concluída! Modelo reconstruído guardado em '{reconstructed_output}'")