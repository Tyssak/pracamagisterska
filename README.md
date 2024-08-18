
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
Celem pracy było zbadanie wpływu różnych metod wstępnego przetwarzania obrazu na skuteczność rozpoznawania emocji przez sieci neuronowe na podstawie zdjęć i materiałów wideo. Zbadane metody to normalizacja przestrzenna łączona z takimi metodami jak normalizacja intensywności metodą CLAHE, wykrywanie krawędzi filtrem Sobela i metodą Canny, oraz dodawanie i odejmowanie klatek. W pracy wykorzystano klasyfikatory AlexNet oraz dedykowaną sieć CNN2D. Spośród wszystkich przetestowanych metod najskuteczniejsza okazała się ta, oparta o dodawanie klatek, polegająca na przypidsywaniu 3 kolejnych klatek w małych odstępach czasu do kanałów R, G i B. Metoda ta, do działania wykorzystuje także normalizację przestrzenną, pozycjonującą i skalującą twarz w oparciu o pozycję punktów charakterysytcznych twarzy. Ogólny model rozwiązania prezentuje się następująco: 
#

![ModelRoziwazania](https://github.com/user-attachments/assets/05c95074-30d5-4596-9cce-79f8b4bb4965)


## Wyniki
W przypadku wykorzystania metody opartej o dodawnie klatek i klasyfikatora AlexNet osiągnięta dokładność dla bazy RAVDESS wyniosła 70,09%. Pozostałe wyniki przedstawiono w poniższych tabelach.

Wyniki dla baz danych ze zdjęciami (klasyfikator CNN2D):

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



## Jak uruchomić
Najpierw zainstaluj wymagane zależności: 
```bash
# Sklonuj repozytorium  
git clone https://github.com/Tyssak/pracamagisterska.git

# Zainstaluj projekt   
cd pracamagisterska
pip install -e .   
pip install -r requirements.txt

# Pobierz plik model_eq.h5 i umieść go w katalogu projektu lub samemu wytrenuj model (parz rozdział trening)
# Uruchom moduł (domyślnie używa pliku model_eq.h5 jako modelu)
python main.py

 ```   
# Lista wszystkich skryptów:
- clasyficate.py: Zawiera klasę Clasyficator, która klasyfikuje emocje na podstawie dostarczonej klatki. Wywoływana przez klasę PreProcessor ze skryptu pre_processor_classify.
- filters.py: Zawiera klasę FiltersOption implementującą różne filtry zastosowane w algorytmie.
- main.py: Skrypt odpowiedzialny za GUI oraz wywołujący klasę PreProcessor ze skryptu pre_processor_classify.
- model_training.ipynb: Skrypt do trenowania modelu. Domyślnie wykorzystuje dataset znajdujący się w folderze /kaggle/input/laczone/wszystkie, który został uprzednio podzielony na dwa podfoldery: jeden w folderze train oraz drugi w folderze test. Domyślna liczba klas w datasecie wynosi 7, a rozmiar obrazów to (227, 227) z trzema kanałami. Domyślnym klasyfikatorem jest AlexNet. Po odkomentowaniu oznaczonej części kodu, można użyć klasyfikatora CNN2D.
- pre_processor_classify.py: Zawiera klasę PreProcessor, implementującą główną logikę przetwarzania wstępnego oraz kilka innych przydatnych funkcji podczas przetwarzania wstępnego.
- pre_processor_prepare_dataset.py: Nieznacznie zmodyfikowana wersja klasy PreProcessor, przystosowana do przygotowania datasetu do treningu.
- prepare_dataset_from_photos.py: Skrypt pobierający zdjęcia ze ścieżki input_directory, a następnie wykonujący na nich preprocessing za pomocą klasy PreProcessor ze skryptu pre_processor_prepare_dataset. Przetworzone zdjęcia są zapisywane do ścieżki save_directory.
- prepare_dataset_from_video.py: Skrypt działający analogicznie jak prepare_dataset_from_photos.py, ale dla folderu z materiałami wideo.
- folder other_useful_scripts: Inne skrypty niewymagane do działania programu, ale przydatne podczas przygotowywania datasetów (np. wyrównanie wielkości klas, parsery, wykreślanie wykresów, czy archiwalne wersje algorytmu wykonującego preprocessing).
# 


# W celu wytrenowania własnego modelu

## Przygotowanie datasetu

Dla datasetu ze zdjęciami:
W skrypcie prepare_dataset_from_photos.py.
wybrać:
input_directory - folder z datasetem treningowym bądź testowym uprzednio podzielonym na klasy - każda z klas w osobnym podfolerze.
filter_option - wybrany filer z listy Enum klasy FilterOption
save_directory - folder do którego zostaną zapisane zdjęcia po wykonaniu preprocessingu

Dla datasetu z materiałami wideo  (obsługiwane formaty: mp4 i mkv)
W skrypcie prepare_dataset_from_photos.py.
wybrać:
input_directory - folder z datasetem treningowym bądź testowym uprzednio podzielonym na klasy - każda z klas w osobnym podfolerze (chyba, że wartość w dataset = 0 lub dataset = 1)
filter_option - wybrany filer z listy Enum klasy FilterOption
save_directory - folder do którego zostaną zapisane zdjęcia po wykonaniu preprocessingu
dataset - domyślna wartość 2. Dla wartości 0 i 1 wykonywany jest podział na klasy na podstawie nazwy pliku (tylko dla datasetu damevo: datset = 0 i RAVDESS: dataset = 1).

## Trening własnego modelu
W modelu 



 
