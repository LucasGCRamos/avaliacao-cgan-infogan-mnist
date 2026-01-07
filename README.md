## 🧠 Avaliação de cGANs e InfoGANs para Data Augmentation no MNIST

Este repositório contém o desenvolvimento e a avaliação de **modelos generativos condicionais (cGANs e InfoGANs)** aplicados ao conjunto de dados **MNIST**, com foco em **data augmentation generativo** e seu impacto no desempenho de uma **CNN classificadora**.

📌 **Esta atividade refere-se à disciplina de Aprendizagem de Máquina**, integrante do **curso de Capacitação Técnica e Empreendedora em Inteligência Artificial**, ofertado pela **FDTE (Fundação para o Desenvolvimento Tecnológico da Engenharia) da USP**.

---

## 🎯 Objetivo
Investigar como diferentes arquiteturas de GANs condicionais afetam:
- A qualidade das imagens sintéticas geradas
- A diversidade das amostras
- O desempenho de um classificador supervisionado treinado com dados reais + sintéticos

Foram comparados:
- cGAN (MLP e Convolucional)
- InfoGAN (MLP e Convolucional)

---

## 📊 Conjunto de Dados
- **Dataset:** MNIST
- **Imagens:** Dígitos manuscritos (0–9)
- **Resolução:** 28 × 28 pixels (tons de cinza)
- **Divisão:**  
  - 60.000 imagens de treino  
  - 10.000 imagens de teste  

O MNIST foi escolhido por permitir comparações claras entre arquiteturas MLP e convolucionais.

---

## 🧩 Arquiteturas Implementadas

### 1️⃣ cGAN MLP
- Gerador e discriminador baseados em camadas totalmente conectadas  
- Entrada: ruído + rótulo (one-hot)  
- Saída: imagem flatten (784 dimensões)

### 2️⃣ cGAN Convolucional
- Gerador com convoluções transpostas  
- Discriminador totalmente convolucional  
- Geração direta de imagens 28×28

### 3️⃣ InfoGAN MLP
- Arquitetura MLP  
- Uso de código latente categórico e contínuo  
- Treinamento com perda de informação mútua (InfoLoss)

### 4️⃣ InfoGAN Convolucional
- Arquitetura convolucional  
- Maior capacidade de modelar estrutura espacial  
- Código latente interpretável

---

## 🔬 Metodologia

### 🧪 CNN Classificadora Base
Uma CNN convolucional foi treinada inicialmente apenas com dados reais do MNIST, servindo como **baseline**.

### 🔁 Data Augmentation Generativo
Para cada modelo generativo:
- Foram geradas **2.000 imagens sintéticas por classe**
- Os dados sintéticos foram combinados com o conjunto de treino real
- Uma nova CNN foi treinada em cada cenário
- A avaliação foi feita exclusivamente no conjunto de teste real

---

## 📈 Métricas de Avaliação

### 🎨 Qualidade das Imagens Geradas
- **FID (Fréchet Inception Distance)**
- **KID (Kernel Inception Distance)**
- **Precision / Recall para modelos generativos**

### 🤖 Impacto no Classificador
- Acurácia
- Precisão e Revocação
- Matriz de confusão

---

## 📊 Resultados

### 🔹 Qualidade dos Geradores

| Modelo        | FID ↓ | KID ↓ | Precision | Recall |
|--------------|------:|------:|----------:|-------:|
| cGAN MLP     | 0.387 | 0.0035 | 1.000 | 0.000 |
| cGAN Conv    | 0.013 | 0.00007 | 0.400 | 0.443 |
| InfoGAN MLP  | 0.368 | 0.0032 | 0.716 | 0.000 |
| InfoGAN Conv | 0.033 | 0.00012 | 0.255 | 0.357 |

📌 Modelos convolucionais superaram amplamente os MLPs, enquanto os MLPs sofreram colapso de modo.

---

### 🔹 Impacto no Classificador (Acurácia)

| Conjunto de Treinamento | Acurácia |
|------------------------|----------|
| MNIST (baseline)       | 0.9854 |
| MNIST + cGAN MLP       | 0.9829 |
| MNIST + cGAN Conv      | 0.9860 |
| MNIST + InfoGAN MLP    | 0.9873 |
| MNIST + InfoGAN Conv   | 0.9566 |

---

## 🧠 Discussão
- Arquiteturas convolucionais produzem imagens mais realistas e úteis
- A qualidade do gerador é crucial para o sucesso do data augmentation
- InfoGAN aumenta diversidade, mas nem sempre melhora o classificador
- Dados sintéticos de baixa qualidade podem atuar como regularização

---

## ✅ Conclusão
- cGAN convolucional foi o modelo mais eficaz para data augmentation
- InfoGAN não garante melhoria automática no desempenho
- GANs são ferramentas poderosas, desde que estáveis e bem treinadas

---

## 🚀 Trabalhos Futuros
- Avaliação em bases mais complexas (Fashion-MNIST, CIFAR-10)
- Ajuste da proporção entre dados reais e sintéticos
- Análise detalhada de matrizes de confusão
- Estudo aprofundado dos códigos latentes contínuos do InfoGAN
