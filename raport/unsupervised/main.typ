#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Dekodowanie Wyobrażeń Ruchowych z sygnałów EEG o niskiej rozdzielczości przestrzennej: Analiza wykrywania anomali przy użyciu metod nienadzorowanych uczenia maszynowego],
  abstract: [],
  authors: (
    (
      name: "Ignacy Berent",
      email: "275255@student.pwr.edu.pl",
    ),
    (
      name: "Robert Rubaszek",
      email: "",
    ),
  ),
  index-terms: (
    " Unsupervised Machine Learning",
    "Electroence-phalography",
    "Anomaly Detection",
    "Brain-Computer Interface",
    "Motor Imagery",
  ),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

= Wstęp

= Materiały & Metody
== Rejestracja danych
Dane wykorzystane w eksperymencie zostały zebrane za pomocą 14-kanałowego bezprzewodowego zestawu EEG _Emotiv Epoc X_ @EmotivEPOCX. Badanie przeprowadzono z udziałem jednego ochotnika, a cała procedura pomiarowa charakteryzowała się ścisłą sekwencyjnością czasową, w której poszczególne etapy następowały bezpośrednio po sobie:

- *Stan spoczynkowy (baseline):* 2,5 sekundy rejestracji sygnału w stanie relaksu (wytłumienie intencji kognitywnych),
- *Zadanie ruchowe (Motor Imagery):* 2,5 sekundy wyobrażenia ruchu (odpowiednio lewą lub prawą ręką),
- *Stan spoczynkowy (post-baseline):* 2,5 sekundy ponownej rejestracji sygnału w stanie spoczynku.

Łącznie zrealizowano cztery sesje pomiarowe (dwie dla lewej oraz dwie dla prawej ręki) o czasie trwania około 7,5 sekundy każda. Dodatkowo zarejestrowano 10-sekundowy sygnał szumu aparaturowego urządzenia. Surowe zapisy w formacie _.edf_ zostały wyeksportowane do plików tekstowych _.csv_ przy użyciu dedykowanego oprogramowania firmy Emotiv.

== Metody analizy danych
Przetwarzanie sygnału EEG zrealizowano w nienadzorowanej architekturze potokowej typu End-to-End (E2E), integrującej cyfrową filtrację, ekstrakcję cech widmowych, redukcję wymiarowości oraz probabilistyczne modelowanie stanów mózgowych. Ciągły zapis EEG z 13 sprawnych kanałów fizycznych poddano segmentacji za pomocą techniki okna przesuwnego. Długość pojedynczego okna (epoki) ustalono na 0.5 sekundy, z krokiem przesunięcia wynoszącym 0.1 sekundy, co zapewniło 80-procentowe nakładanie się sąsiadujących segmentów sygnału.

W celu ekstrakcji cech zaimplementowano dwa niezależne podejścia, szeroko opisywane w literaturze dotyczącej interfejsów mózg-komputer @Lotte2018:
+ *Bezpośrednia analiza widmowa (Direct PSD)* -- gęstość widmowa mocy estymowana klasyczną metodą Welcha @Welch1967 bezpośrednio z fizycznych kanałów EEG.
+ *Dekompozycja na źródła niezależne (ICA)* -- transformacja sygnału za pomocą algorytmu Infomax do przestrzeni wirtualnych komponentów niezależnych @Makeig1996 przed analizą widmową.

Z wyznaczonych widm częstotliwościowych wyodrębniono średnią moc w pasmach sensorimotorycznych kluczowych dla paradygmatu wyobrażeń ruchowych (_Motor Imagery_): Mu (8-12 Hz) oraz Beta (13-30 Hz) @Pfurtscheller2001. Uzyskane wektory cech poddano standaryzacji statystycznej, a następnie redukcji wymiarowości za pomocą Analizy Składowych Głównych (PCA) w celu uniknięcia zjawiska klęski wymiarowości.

=== Model Mieszanin Gaussowskich (GMM)
Do globalnej, statycznej identyfikacji stanów wewnętrznych bez uwzględnienia zależności czasowej wykorzystano Model Mieszanin Gaussowskich (_Gaussian Mixture Model_) @Reynolds2009. GMM jest nienadzorowanym algorytmem klastrowania probabilistycznego, traktującym gęstość rozkładu cech jako sumę ważoną skończonej liczby składowych gaussowskich.

W tym podejściu wyekstrahowane cechy ze wszystkich zarejestrowanych sesji pomiarowych (zarówno dla wyobrażenia ruchu lewej, jak i prawej ręki) zostały połączone w jedną globalną macierz. Przestrzeń cech zredukowano przy użyciu PCA do 3 składowych głównych. Model skonfigurowano dla dwóch komponentów ($n_"clusters" = 2$) odpowiadających stanom "rest" oraz "task", aplikując pełną macierz kowariancji (`covariance_type="full"`). Pozwoliło to na swobodne modelowanie geometrii elipsoid skupień w przestrzeni skompresowanej.

=== Ukryty Model Markowa (HMM)
Z uwagi na fakt, że procedura pomiarowa narzucała ścisłą sekwencyjność zdarzeń w czasie (odpoczynek -> zadanie -> odpoczynek), do jawnego modelowania dynamiki przejść czasowych zaimplementowano Ukryty Model Markowa (_Hidden Markov Model_) z emisjami gaussowskimi @Rabiner1989.

W przeciwieństwie do potoku GMM, model HMM trenowany był indywidualnie na pojedynczych plikach pomiarowych (z racji konieczności zachowania ciągłości chronologicznej próbek). Z powodu krytycznie małej liczby obserwacji w pojedynczym pliku (~70 okien czasowych), przestrzeń wejściową zredukowano za pomocą PCA do restrykcyjnego poziomu 1 lub 2 składowych głównych. Zapobiegło to numerycznej niestabilności oraz błędom osobliwości podczas estymacji macierzy kowariancji stanów ukrytych, reprezentujących dynamikę przejścia ochotnika między procesami kognitywnymi.

= Wyniki
