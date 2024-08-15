
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
Celem pracy było zbadanie wpływu różnych metod wstępnego przetwarzania obrazu na skuteczność rozpoznawania emocji przez sieci neuronowe na podstawie zdjęć i materiałów wideo. Zbadane metody to normalizacja przestrzenna łączona z takimi metodami jak normalizacja intensywności metodą CLAHE, wykrywanie krawędzi filtrem Sobela i metodą Canny, oraz dodawanie i odejmowanie klatek. W pracy wykorzystano klasyfikatory AlexNet oraz dedykowaną sieć CNN. Spośród wszystkich przetestowanych metod najskuteczniejsza okazała się ta, oparta o dodawanie klatek, polegająca na przypidsywaniu 3 kolejnych klatek w małych odstępach czasu do kanałów R, G i B. Metoda ta, do działania wykorzystuje także normalizację przestrzenną, pozycjonującą i skalującą twarz w oparciu o pozycję punktów charakterysytcznych twarzy. Ogólny model rozwiązania prezentuje się następująco: 
#

![ModelRoziwazania](https://github.com/user-attachments/assets/05c95074-30d5-4596-9cce-79f8b4bb4965)


## Wyniki
W przypadku wykorzystania metody opartej o dodawnie klatek i klasyfikatora AlexNet osiągnięta dokładność dla bazy RAVDESS wyniosła 70,09%. Pozostałe wyniki przedstawiono w poniższych tabelach.

Wyniki dla baz danych ze zdjęciami:

| **Baza danych** | **Dokładność bazowa** | **Normalizacja bez maski** | **Normalizacja z maską** | **Normalizacja + SOBEL** | **Normalizacja + CANNY** | **Normalizacja + CLAHE** |
|-----------------|-----------------------|----------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| **CK+**         | 0,8283                | 0,8788                     | 0,8485                   | **0,9040**                | 0,8030                   | 0,8788                   |
| **FER2013**     | 0,6967                | **0,6987**                 | 0,6928                   | 0,6562                   | 0,6240                   | 0,6920                   |
| **RAFDB**       | 0,8338                | 0,8342                     | **0,8344**               | 0,7120                   | 0,6932                   | 0,8322                   |

Wyniki dla materiałów wideo (baza RAVDESS - 7 emocji: spokój, radość, smutek, złość, strach, zniesmaczenie, zaskoczenie):

| **Klasyfikator** | **Dokładność bazowa** | **Normalizacja bez maski** | **Normalizacja z maską** | **Normalizacja + SOBEL** | **Normalizacja + CANNY** | **Normalizacja + CLAHE** |
|------------------|-----------------------|----------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| **CNN2D**        | 0,6006                | 0,6147                     | 0,6086                   | 0,6256                   | 0,6343                   | **0,6352**               |
| **AlexNet**      | 0,6845                | 0,6842                     | 0,6537                   | 0,5138                   | 0,4502                   | **0,6928**               |

| **Klasyfikator** | **Dokładność bazowa** | **Odejmowanie klatek z maską** | **Odejmowanie klatek (brak maski)** | **Dodawanie klatek z maską** | **Dodawanie klatek (brak maski)** |
|------------------|-----------------------|--------------------------------|------------------------------------|------------------------------|----------------------------------|
| **CNN2D**        | 0,6006                | 0,4567                         | 0,4922                             | **0,6492**                    | 0,6274                           |
| **AlexNet**      | 0,6552                | 0,5931                         | 0,6200                             | **0,7009**                    | 0,6804                           |



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

 
