# Manual de Uso: Demo Plugável

O TwinRank AI possui uma página de **Live Demo** integrada ao portal Streamlit. Nela, donos de e-commerce ou interessados podem testar o poder de um sistema de recomendação **Two-Tower com Deep Learning**, treinando o modelo na hora, sem a necessidade de infraestrutura pesada.

---

## 1. Estrutura dos Arquivos CSV

Para utilizar o modo plugável, você precisará de duas planilhas em formato `.csv`. Caso não possua, o sistema carregará automaticamente planilhas de exemplo (dummy data).

### 1.1 `products.csv` (Cadastro de Produtos)
Esta planilha deve conter os metadados dos itens do seu e-commerce.

**Colunas obrigatórias:**
- `item_id`: Identificador único do produto (texto ou número).
- `name`: Nome do produto (ex: "Tênis Esportivo").
- `category`: Categoria ou subcategoria do produto.
- `price`: Preço do produto.

### 1.2 `orders.csv` (Histórico de Interações/Compras)
Esta planilha deve conter os logs de interações dos seus clientes.

**Colunas obrigatórias:**
- `user_id`: Identificador único do usuário/cliente.
- `item_id`: Identificador único do produto interagido (deve bater com o `products.csv`).
- `event_type`: Tipo de interação (ex: "view", "addtocart", "transaction").
- `timestamp`: Momento em que a interação ocorreu (formato ISO 8601 ou epoch).

---

## 2. Passo a Passo no Streamlit

1. No menu lateral do portal TwinRank AI, acesse a página **🚀 Recomendações**.
2. No menu da esquerda, procure a seção **1. Upload de Dados**.
3. Arraste e solte o seu arquivo `products.csv` no primeiro campo.
4. Arraste e solte o seu arquivo `orders.csv` no segundo campo.

> 💡 **Treinamento Ultra-rápido:** Assim que os dois arquivos forem carregados, o sistema iniciará o treinamento on-the-fly. O TwinRank AI instanciará uma rede neural Two-Tower, aprenderá os embeddings exclusivos para a sua loja com base nas interações e construirá um índice de busca vetorial FAISS. Aguarde alguns segundos.

## 3. Gerando Recomendações

1. Assim que a barra de progresso terminar, a seção **2. Recomendações** aparecerá no menu lateral.
2. Selecione o **ID do Usuário** no qual você quer basear as sugestões.
3. Escolha a quantidade de itens a recomendar na barra **Top K** (ex: 5, 10, 20).
4. Clique no botão azul **"Gerar Recomendações"**.

O sistema executará a inferência ANN (Approximate Nearest Neighbors) através do FAISS e exibirá um grid visual com os melhores produtos para aquele cliente específico!
