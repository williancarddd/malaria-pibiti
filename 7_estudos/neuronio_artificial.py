def neuronio_artificial(s_entrada: [int], p_sinaptico: [int], f_ativacao, limiar ):
  n = len(s_entrada)
  combinacao_linear = 0
  for i in range(0, n):
    combinacao_linear += s_entrada[i]*p_sinaptico[i]
  
  potencial_ativacao = combinacao_linear  - limiar

  y = f_ativacao(potencial_ativacao)

  return y

def funcao_degrau(potencial_ativacao):
    if potencial_ativacao >= 0:
        return 1
    else:
        return 0
    
def funcao_degrau_bipolar(potencial_ativacao):
   if potencial_ativacao > 0:
      return 1
   if potencial_ativacao == 0:
      return 0
   else:
      return -1

def funcao_rampa_simetrica(potencial_ativacao, a):
   if potencial_ativacao > a:
      return a
   if potencial_ativacao >= -a and potencial_ativacao <= a:
      return 0
   if potencial_ativacao == a:
      return -a

# Entradas e pesos sinápticos para a função OR
entradas = [1, 0, 1, 1]
pesos = [0.25, 0.25,  0.25, 0.25]  
limiar = 0  


resultado = neuronio_artificial(entradas, pesos, funcao_degrau_bipolar, limiar)


print("Resultado da função OR para entradas", entradas, ":", resultado)


"""
1) Neurônio artificial
Um neurônio artificial  ele funciona com valores de entradas denominado
sinapses e com valores que ponderarão essas entradas , chamado de pesos
sinaptícos. Esse valores são multiplicados e somadados, chamado de 
combinação linear, e , logo em seguida, e aplicado o limiar de ativação,
, que vai dizer se compensa ou não, chamar a função de ativação para calcular
a saida.


2) Função de ativação
Tem o papel de limitar a saída do neurônio dentro de sua imagem.


4) Limiar de ativação
Ele tem uma sua importância em limitar o resultado conforme a função de ativação



"""