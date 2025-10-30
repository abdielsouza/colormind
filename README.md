# 🎨 Colormind - Gerador de Paletas de Cores com IA

Um aplicativo interativo feito em **Python + Streamlit** que utiliza **K-Means** (do scikit-learn) para extrair as **principais cores** de uma imagem e gerar uma **paleta automática**.  
O projeto é leve e responsivo.

---

## ✨ Recursos

- 📤 Upload de imagens (`.png`, `.jpg`, `.jpeg`)
- 🧠 Extração de cores principais via **K-Means**
- 🎚️ Ajuste de **quantidade de cores** e **nível de precisão**
- 🌗 Alternância entre **modo claro e escuro**
- 🖼️ Visual moderno e responsivo
- 🚀 Deploy direto no **Streamlit Cloud**

---

## 🧩 Tecnologias usadas

| Categoria | Tecnologias |
|------------|--------------|
| Linguagem | Python 3.12+ |
| Framework web | Streamlit |
| Machine Learning | scikit-learn |
| Processamento de imagem | Pillow (PIL), NumPy |

---

## 🧠 Como funciona

1. O usuário envia uma imagem.
2. O app converte a imagem para um array NumPy.
3. O algoritmo **K-Means** identifica os *clusters* de cores mais frequentes.
4. O sistema converte cada cluster em um código **HEX** e mostra visualmente.
5. O usuário pode ajustar o número de cores e a precisão do modelo.

---

## Acesso
Navegue até [o website do Colormind](https://colormind-hbei.onrender.com/) para utilizar as utilidades.