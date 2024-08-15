
---

<div align="center">    
 
#  Analiza algorytmów przetwarzania obrazu wideo z nagraniami twarzy w celu poprawy jakości wykrywania emocji
<!--
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  

ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)

![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)
-->

<!--  
Conference   
-->   
</div>
 
## Opis   
Celem pracy było zbadanie wpływu różnych metod wstępnego przetwarzania obrazu na skuteczność rozpoznawania emocji przez sieci neuronowe na podstawie zdjęć i materiałów wideo. Zbadane metody to normalizacja przestrzenna łączona z takimi metodami jak normalizacja intensywności metodą CLAHE, wykrywanie krawędzi filtrem Sobela i metodą Canny, oraz dodawanie i odejmowanie klatek. W pracy wykorzystano klasyfikatory AlexNet oraz dedykowaną sieć CNN.

## Wyniki

\begin{table}[htbp]
    \centering
    \caption{Najlepsze uzyskane pomiary dla zbiorów CK+, FER2013 i RAF-DB z różnymi technikami przetwarzania obrazów (klasyfikator CNN2D)}
    \begin{tabular}{|m{1.4cm}|m{1.8cm}|m{1.8cm}|m{1.8cm}|m{1.8cm}|m{1.8cm}|m{1.8cm}|}
        \hline
        \textbf{baza \qquad danych} & \textbf{dokładność bazowa} & \textbf{normalizacja bez maski} & \textbf{normalizacja z maską} & \textbf{normalizacja + SOBEL} & \textbf{normalizacja + CANNY} & \textbf{normalizacja + CLAHE} \\
        \hline
        \textbf{CK+} & 0,8283 & 0,8788 & 0,8485 & \textbf{0,9040} & 0,8030 & 0,8788 \\
        \hline
        \textbf{FER2013} & 0,6967 & \textbf{0,6987} & 0,6928 & 0,6562 & 0,6240 & 0,6920 \\
        \hline
        \textbf{RAFDB} & 0,8338 & 0,8342 & \textbf{0,8344} & 0,7120 & 0,6932 & 0,8322 \\
        \hline
    \end{tabular}
    \label{tab:resultsphotos}
\end{table}

\begin{table}[htbp]
    \centering
    \caption{Najlepsze uzyskane pomiary dla zbioru RAVDESS z różnymi technikami przetwarzania obrazów}
    \begin{tabular}{|m{1.8cm}|m{1.8cm}|m{1.8cm}|m{1.8cm}|m{1.8cm}|m{1.8cm}|m{1.8cm}|}
        \hline
        \textbf{klasyfikator} & \textbf{dokładność bazowa} & \textbf{normalizacja bez maski} & \textbf{normalizacja z maską} & \textbf{normalizacja + SOBEL} & \textbf{normalizacja + CANNY} & \textbf{normalizacja + CLAHE} \\
        \hline
        \textbf{CNN2D} & 0,6006 & 0,6147 & 0,6086 & 0,6256 & 0,6343 & \textbf{0,6352} \\
        \hline
        \textbf{AlexNet} & 0,6845 & 0,6842 & 0,6537 & 0,5138 & 0,4502 & \textbf{0,6928} \\
        \hline
    \end{tabular}
    \label{tab:resultsvideos1}
\end{table}

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/YourGithubName/deep-learning-project-template

# install project   
cd deep-learning-project-template 
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd project

# run module (example: mnist as your main contribution)   
python lit_classifier_main.py    
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   

 
