# 🌍 Sistema Completo de Previsão de Terremotos \- Japão

Este projeto apresenta um sistema abrangente de Machine Learning para a previsão e análise sísmica, com foco específico na região do Japão. Utilizando dados em tempo real da API do USGS, ele oferece um pipeline completo desde a coleta de dados até a geração de relatórios interativos e monitoramento em tempo real.

## **🎯 Visão Geral do Projeto**

A previsão de terremotos é um dos maiores desafios científicos. Este sistema visa contribuir para essa área complexa através de uma abordagem baseada em dados e aprendizado de máquina. Ele não pretende ser um sistema de alerta definitivo, mas uma ferramenta poderosa para análise, pesquisa e compreensão dos padrões sísmicos.

## **✨ Principais Características**

* **Coleta de Dados Automatizada:**  
  * Integração com a **API gratuita do USGS** (USGS Public APIs) para dados sísmicos globais.  
  * Filtragem geográfica específica para o Japão.  
  * Coleta de **dados históricos de 5 anos** por padrão e **monitoramento em tempo real**.  
* **Engenharia de Features Avançada:**  
  * Criação de features temporais (ano, mês, dia, hora), geográficas (distância do centro sísmico), estatísticas móveis (médias, desvios), features de clustering espacial e de sequência temporal para enriquecer o conjunto de dados.  
* **Modelos de Machine Learning Robustos:**  
  * Implementação de **Random Forest** para capturar relações não-lineares, **Gradient Boosting** para otimização sequencial e **Linear Regression** como baseline.  
  * **Otimização automática de hiperparâmetros** via GridSearch e **validação cruzada** para garantir a robustez dos modelos.  
* **Sistema de Avaliação Completo:**  
  * Métricas de desempenho como **MSE, MAE e R²**.  
  * **Análise de resíduos** e comparação detalhada entre os modelos treinados.  
* **Monitoramento em Tempo Real:**  
  * Análise contínua da atividade sísmica regional.  
  * Sistema de **alertas automáticos** configuráveis.  
  * **Mapa de risco dinâmico** para as principais cidades japonesas.  
  * Capacidade de gerar **predições personalizadas** para coordenadas específicas.  
* **Exportação de Relatórios HTML:**  
  * Geração automática de **relatórios HTML interativos** contendo:  
    * **Métricas detalhadas do modelo escolhido**.  
    * **Análise sísmica regional** (terremotos recentes, magnitudes, localizações).  
    * **Alertas e recomendações** baseadas na atividade sísmica atual.  
    * **Estatísticas gerais** sobre os dados sísmicos processados.  
* **Visualizações Interativas:**  
  * Gráficos e mapas para visualização da distribuição geográfica, análise temporal e comparação de modelos.  
* **Persistência:**  
  * Funcionalidades para salvar e carregar modelos treinados, permitindo reutilização e implementação sem a necessidade de retreinar.  
* **Escalabilidade:**  
  * Arquitetura modular projetada para fácil expansão para outras regiões geográficas.  
* **Cidades Monitoradas:**  
  * Análise específica para as principais cidades japonesas: **Tóquio, Osaka, Kyoto, Yokohama, Kobe, Sendai, Hiroshima e Fukuoka**.

## **🚀 Como Usar**

Para inicializar o sistema e realizar uma análise completa, basta executar:

\# Inicializar e executar análise completa  
system, monitor, results \= main()

Para fazer uma predição para um local específico (ex: Tóquio) e uma profundidade:

\# Fazer predição para um local específico (latitude, longitude, profundidade)  
prediction \= system.predict\_earthquake(35.6762, 139.6503, 10.0)  \# Exemplo: Tóquio  
print(f"Magnitude predita: {prediction\['predicted\_magnitude'\]}")

Para analisar a atividade sísmica recente em uma região (ex: Tóquio):

\# Analisar atividade sísmica regional  
analysis \= monitor.analyze\_seismic\_activity("Tóquio", 35.6762, 139.6503)  
print(f"Terremotos recentes: {analysis\['recent\_earthquakes'\]}")

## **📈 Vantagens Chave**

* **Dados Reais e Oficiais:** Utiliza dados sísmicos autênticos do USGS.  
* **Custo Zero:** Sem custos de API ou licenciamento.  
* **Sistema End-to-End:** Aborda todas as etapas, desde a coleta de dados até a geração de relatórios.  
* **Modular e Extensível:** Facilmente adaptável para novas regiões ou modelos.  
* **Base Científica:** Construído sobre boas práticas de Machine Learning e princípios sismológicos.

## **⚠️ Importante**

A previsão de terremotos é um desafio científico extremamente complexo e em constante evolução. Este sistema é uma ferramenta avançada de **análise e pesquisa**, e **não deve ser utilizado como um sistema de alerta ou previsão definitiva** para tomar decisões críticas de segurança. Ele serve para explorar padrões, gerar insights e auxiliar na compreensão de fenômenos sísmicos.

## **🤝 Contribuições**

Contribuições são bem-vindas\! Sinta-se à vontade para abrir issues ou pull requests para melhorias, novas funcionalidades ou correção de bugs.

