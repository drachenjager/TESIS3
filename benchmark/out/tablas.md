## Ranking global (54 experimentos)
| Modelo           |   Victorias (6 modelos) |   Victorias (con Naive) |   R² global |   MAE global |   MAE mediana |   Mejor que Naive (DM p<0.05) |   Peor que Naive (DM p<0.05) |   MAE < Naive (conteo) |
|:-----------------|------------------------:|------------------------:|------------:|-------------:|--------------:|------------------------------:|-----------------------------:|-----------------------:|
| Naive            |                   0.000 |                  35.000 |       0.815 |        0.064 |         0.052 |                         0.000 |                        0.000 |                  0.000 |
| SARIMA           |                  36.000 |                  25.000 |       0.757 |        0.071 |         0.057 |                         0.000 |                        6.000 |                  3.000 |
| Holt-Winters     |                  18.000 |                  18.000 |       0.792 |        0.067 |         0.063 |                         1.000 |                       14.000 |                 14.000 |
| Regresión Lineal |                   7.000 |                   4.000 |       0.711 |        0.077 |         0.065 |                         0.000 |                       17.000 |                  4.000 |
| Random Forest    |                   1.000 |                   1.000 |      -1.196 |        0.176 |         0.161 |                         0.000 |                       52.000 |                  1.000 |
| RNN              |                   0.000 |                   0.000 |       0.198 |        0.099 |         0.088 |                         0.000 |                       45.000 |                  0.000 |
| LSTM             |                   0.000 |                   0.000 |      -1.694 |        0.141 |         0.127 |                         0.000 |                       51.000 |                  0.000 |

## Conteo de primeros lugares por periodo (6 modelos de la tesis)
| P       |   SARIMA |   Holt-Winters |   Regresión Lineal |   Random Forest |   RNN |   LSTM |
|:--------|---------:|---------------:|-------------------:|----------------:|------:|-------:|
| 3 meses |        0 |              7 |                  2 |               1 |     0 |      0 |
| 6 meses |        6 |              4 |                  0 |               0 |     0 |      0 |
| 1 año   |        9 |              0 |                  3 |               0 |     0 |      0 |
| 2 años  |        9 |              1 |                  0 |               0 |     0 |      0 |
| 5 años  |        5 |              3 |                  2 |               0 |     0 |      0 |
| 10 años |        7 |              3 |                  0 |               0 |     0 |      0 |

## Conteo de primeros lugares por periodo (incluyendo Naive)
| P       |   Naive |   SARIMA |   Holt-Winters |   Regresión Lineal |   Random Forest |   RNN |   LSTM |
|:--------|--------:|---------:|---------------:|-------------------:|----------------:|------:|-------:|
| 3 meses |       0 |        0 |              7 |                  2 |               1 |     0 |      0 |
| 6 meses |       6 |        6 |              4 |                  0 |               0 |     0 |      0 |
| 1 año   |       8 |        1 |              0 |                  0 |               0 |     0 |      0 |
| 2 años  |       9 |        7 |              1 |                  0 |               0 |     0 |      0 |
| 5 años  |       5 |        4 |              3 |                  2 |               0 |     0 |      0 |
| 10 años |       7 |        7 |              3 |                  0 |               0 |     0 |      0 |

## R² promedio por periodo
| Modelo           |   3 meses |   6 meses |   1 año |   2 años |   5 años |   10 años |
|:-----------------|----------:|----------:|--------:|---------:|---------:|----------:|
| Naive            |     0.382 |     0.709 |   0.902 |    0.941 |    0.973 |     0.986 |
| SARIMA           |     0.038 |     0.709 |   0.896 |    0.941 |    0.973 |     0.986 |
| Holt-Winters     |     0.282 |     0.723 |   0.852 |    0.938 |    0.973 |     0.986 |
| Regresión Lineal |    -0.123 |     0.598 |   0.894 |    0.938 |    0.973 |     0.985 |
| Random Forest    |    -4.021 |    -4.095 |  -1.833 |    0.872 |    0.945 |     0.954 |
| RNN              |    -2.658 |     0.191 |   0.826 |    0.886 |    0.960 |     0.982 |
| LSTM             |   -10.206 |    -3.061 |   0.303 |    0.854 |    0.964 |     0.984 |

## MAE promedio por periodo
| Modelo           |   3 meses |   6 meses |   1 año |   2 años |   5 años |   10 años |
|:-----------------|----------:|----------:|--------:|---------:|---------:|----------:|
| Naive            |    0.0434 |    0.0483 |  0.0506 |   0.0614 |   0.0878 |    0.0950 |
| SARIMA           |    0.0814 |    0.0483 |  0.0511 |   0.0616 |   0.0878 |    0.0950 |
| Holt-Winters     |    0.0374 |    0.0535 |  0.0631 |   0.0654 |   0.0883 |    0.0956 |
| Regresión Lineal |    0.0999 |    0.0602 |  0.0533 |   0.0642 |   0.0881 |    0.0963 |
| Random Forest    |    0.2023 |    0.2081 |  0.2675 |   0.0934 |   0.1186 |    0.1681 |
| RNN              |    0.1569 |    0.0730 |  0.0660 |   0.0882 |   0.0975 |    0.1097 |
| LSTM             |    0.2336 |    0.1603 |  0.1363 |   0.1039 |   0.1096 |    0.1051 |

## Antes vs. después (promedio por periodo)
|                             |   MAE Apéndice C |   MAE reproducido multipaso |   MAE corregido un paso |   MAE Naive |   R² Apéndice C |   R² corregido |
|:----------------------------|-----------------:|----------------------------:|------------------------:|------------:|----------------:|---------------:|
| ('SARIMA', '3 meses')       |           0.2484 |                      0.2227 |                  0.0814 |      0.0434 |         -3.9204 |         0.0377 |
| ('SARIMA', '6 meses')       |           0.2289 |                      0.2326 |                  0.0483 |      0.0483 |         -5.1900 |         0.7091 |
| ('SARIMA', '1 año')         |           0.3884 |                      0.3759 |                  0.0511 |      0.0506 |         -4.0116 |         0.8961 |
| ('SARIMA', '2 años')        |           0.6314 |                      0.7430 |                  0.0616 |      0.0614 |         -1.9796 |         0.9412 |
| ('SARIMA', '5 años')        |           1.3655 |                      1.3150 |                  0.0878 |      0.0878 |         -2.2924 |         0.9734 |
| ('SARIMA', '10 años')       |           1.3512 |                      1.4052 |                  0.0950 |      0.0950 |         -1.1670 |         0.9855 |
| ('Holt-Winters', '3 meses') |           0.2086 |                      0.1806 |                  0.0374 |      0.0434 |         -2.5979 |         0.2820 |
| ('Holt-Winters', '6 meses') |           0.2168 |                      0.2196 |                  0.0535 |      0.0483 |         -3.1441 |         0.7231 |
| ('Holt-Winters', '1 año')   |           0.4037 |                      0.3917 |                  0.0631 |      0.0506 |         -4.3614 |         0.8523 |
| ('Holt-Winters', '2 años')  |           1.4552 |                      1.4196 |                  0.0654 |      0.0614 |        -10.2282 |         0.9375 |
| ('Holt-Winters', '5 años')  |           1.6960 |                      1.7866 |                  0.0883 |      0.0878 |         -3.6156 |         0.9728 |
| ('Holt-Winters', '10 años') |           1.7626 |                      1.8724 |                  0.0956 |      0.0950 |         -2.3758 |         0.9855 |

## Fidelidad de la reproducción (protocolo original, modelos deterministas)
| Modelo       |   count |   mean |    max |
|:-------------|--------:|-------:|-------:|
| Holt-Winters | 40.0000 | 0.0022 | 0.0413 |
| SARIMA       | 39.0000 | 0.0018 | 0.0319 |

## MAE por elemento (7 modelos)
|                       |   Naive |   SARIMA |   Holt-Winters |   Regresión Lineal |   Random Forest |    RNN |   LSTM | Mejor            |
|:----------------------|--------:|---------:|---------------:|-------------------:|----------------:|-------:|-------:|:-----------------|
| ('P1,T1', 58, 6)      |  0.0428 |   0.0428 |         0.0447 |             0.0385 |          0.0569 | 0.0517 | 0.0953 | Regresión Lineal |
| ('P1,T2', 55, 9)      |  0.0392 |   0.0392 |         0.0343 |             0.0355 |          0.0336 | 0.0666 | 0.1610 | Random Forest    |
| ('P1,T3', 52, 12)     |  0.0357 |   0.0357 |         0.0283 |             0.0348 |          0.1215 | 0.1117 | 0.2509 | Holt-Winters     |
| ('P1,T4', 48, 16)     |  0.0467 |   0.0816 |         0.0353 |             0.0848 |          0.1994 | 0.1595 | 0.2838 | Holt-Winters     |
| ('P1,T5', 45, 19)     |  0.0448 |   0.1088 |         0.0386 |             0.1351 |          0.3259 | 0.1790 | 0.2857 | Holt-Winters     |
| ('P1,T6', 42, 22)     |  0.0431 |   0.1179 |         0.0407 |             0.1418 |          0.2981 | 0.1892 | 0.2828 | Holt-Winters     |
| ('P1,T7', 39, 25)     |  0.0445 |   0.1137 |         0.0394 |             0.1499 |          0.2665 | 0.2514 | 0.2685 | Holt-Winters     |
| ('P1,T8', 36, 28)     |  0.0460 |   0.0990 |         0.0389 |             0.1164 |          0.2418 | 0.1888 | 0.2395 | Holt-Winters     |
| ('P1,T9', 32, 32)     |  0.0481 |   0.0942 |         0.0360 |             0.1619 |          0.2773 | 0.2141 | 0.2351 | Holt-Winters     |
| ('P2,T1', 117, 13)    |  0.0427 |   0.0427 |         0.0349 |             0.0489 |          0.1987 | 0.0752 | 0.2049 | Holt-Winters     |
| ('P2,T2', 111, 19)    |  0.0448 |   0.0448 |         0.0413 |             0.0605 |          0.3133 | 0.0777 | 0.2061 | Holt-Winters     |
| ('P2,T3', 104, 26)    |  0.0435 |   0.0435 |         0.0413 |             0.0629 |          0.2585 | 0.0744 | 0.1735 | Holt-Winters     |
| ('P2,T4', 98, 32)     |  0.0481 |   0.0481 |         0.0461 |             0.0624 |          0.2213 | 0.0722 | 0.1613 | Holt-Winters     |
| ('P2,T5', 91, 39)     |  0.0501 |   0.0501 |         0.0563 |             0.0658 |          0.2019 | 0.0720 | 0.1587 | Naive            |
| ('P2,T6', 85, 45)     |  0.0526 |   0.0526 |         0.0654 |             0.0606 |          0.1930 | 0.0752 | 0.1439 | Naive            |
| ('P2,T7', 78, 52)     |  0.0501 |   0.0501 |         0.0617 |             0.0593 |          0.1734 | 0.0709 | 0.1328 | Naive            |
| ('P2,T8', 72, 58)     |  0.0513 |   0.0513 |         0.0655 |             0.0577 |          0.1653 | 0.0652 | 0.1176 | Naive            |
| ('P2,T9', 65, 65)     |  0.0514 |   0.0514 |         0.0691 |             0.0634 |          0.1471 | 0.0741 | 0.1441 | Naive            |
| ('P3,T1', 232, 25)    |  0.0445 |   0.0437 |         0.0511 |             0.0488 |          0.2663 | 0.0751 | 0.1606 | SARIMA           |
| ('P3,T2', 219, 38)    |  0.0492 |   0.0503 |         0.0589 |             0.0552 |          0.2054 | 0.0680 | 0.1473 | Naive            |
| ('P3,T3', 206, 51)    |  0.0498 |   0.0503 |         0.0623 |             0.0523 |          0.1744 | 0.0588 | 0.1253 | Naive            |
| ('P3,T4', 193, 64)    |  0.0515 |   0.0509 |         0.0620 |             0.0525 |          0.1505 | 0.0608 | 0.1025 | SARIMA           |
| ('P3,T5', 180, 77)    |  0.0517 |   0.0518 |         0.0634 |             0.0531 |          0.4012 | 0.0656 | 0.1379 | Naive            |
| ('P3,T6', 168, 89)    |  0.0495 |   0.0500 |         0.0620 |             0.0517 |          0.3550 | 0.0630 | 0.1395 | Naive            |
| ('P3,T7', 155, 102)   |  0.0523 |   0.0512 |         0.0640 |             0.0529 |          0.2851 | 0.0599 | 0.1297 | SARIMA           |
| ('P3,T8', 142, 115)   |  0.0533 |   0.0559 |         0.0703 |             0.0569 |          0.2529 | 0.0750 | 0.1179 | Naive            |
| ('P3,T9', 129, 128)   |  0.0532 |   0.0555 |         0.0739 |             0.0564 |          0.3163 | 0.0677 | 0.1662 | Naive            |
| ('P4,T1', 468, 51)    |  0.0498 |   0.0498 |         0.0525 |             0.0529 |          0.0662 | 0.0746 | 0.0758 | Naive            |
| ('P4,T2', 442, 77)    |  0.0517 |   0.0517 |         0.0548 |             0.0544 |          0.0773 | 0.0688 | 0.0749 | Naive            |
| ('P4,T3', 416, 103)   |  0.0518 |   0.0518 |         0.0549 |             0.0542 |          0.0698 | 0.0734 | 0.0772 | Naive            |
| ('P4,T4', 390, 129)   |  0.0529 |   0.0529 |         0.0577 |             0.0558 |          0.0894 | 0.0773 | 0.1001 | Naive            |
| ('P4,T5', 364, 155)   |  0.0561 |   0.0561 |         0.0604 |             0.0590 |          0.0902 | 0.0952 | 0.0824 | Naive            |
| ('P4,T6', 338, 181)   |  0.0581 |   0.0581 |         0.0615 |             0.0607 |          0.1001 | 0.0745 | 0.1179 | Naive            |
| ('P4,T7', 312, 207)   |  0.0717 |   0.0722 |         0.0762 |             0.0748 |          0.1148 | 0.1002 | 0.1469 | Naive            |
| ('P4,T8', 286, 233)   |  0.0744 |   0.0757 |         0.0791 |             0.0781 |          0.1159 | 0.1194 | 0.1281 | Naive            |
| ('P4,T9', 260, 259)   |  0.0863 |   0.0863 |         0.0913 |             0.0880 |          0.1170 | 0.1105 | 0.1318 | Naive            |
| ('P5,T1', 1170, 130)  |  0.0525 |   0.0527 |         0.0554 |             0.0535 |          0.0915 | 0.0857 | 0.0636 | Naive            |
| ('P5,T2', 1105, 195)  |  0.0702 |   0.0702 |         0.0704 |             0.0707 |          0.1032 | 0.0726 | 0.0751 | Naive            |
| ('P5,T3', 1040, 260)  |  0.0861 |   0.0861 |         0.0860 |             0.0864 |          0.1111 | 0.0946 | 0.1214 | Holt-Winters     |
| ('P5,T4', 975, 325)   |  0.0883 |   0.0883 |         0.0883 |             0.0886 |          0.1122 | 0.0904 | 0.1044 | Naive            |
| ('P5,T5', 910, 390)   |  0.0996 |   0.0996 |         0.1000 |             0.0997 |          0.1169 | 0.1024 | 0.1136 | Naive            |
| ('P5,T6', 845, 455)   |  0.1024 |   0.1024 |         0.1024 |             0.1023 |          0.1211 | 0.1080 | 0.1240 | Regresión Lineal |
| ('P5,T7', 780, 520)   |  0.0965 |   0.0965 |         0.0967 |             0.0965 |          0.1262 | 0.1020 | 0.1185 | Naive            |
| ('P5,T8', 715, 585)   |  0.0976 |   0.0976 |         0.0980 |             0.0980 |          0.1278 | 0.1159 | 0.1342 | Naive            |
| ('P5,T9', 650, 650)   |  0.0966 |   0.0966 |         0.0972 |             0.0969 |          0.1573 | 0.1057 | 0.1316 | Naive            |
| ('P6,T1', 2344, 260)  |  0.0861 |   0.0861 |         0.0864 |             0.0867 |          0.1012 | 0.1079 | 0.0931 | Naive            |
| ('P6,T2', 2214, 390)  |  0.0996 |   0.0996 |         0.1001 |             0.0999 |          0.1126 | 0.1121 | 0.1057 | Naive            |
| ('P6,T3', 2084, 520)  |  0.0965 |   0.0965 |         0.0966 |             0.0972 |          0.1229 | 0.1065 | 0.1001 | Naive            |
| ('P6,T4', 1953, 651)  |  0.0965 |   0.0965 |         0.0964 |             0.0980 |          0.1497 | 0.1042 | 0.1002 | Holt-Winters     |
| ('P6,T5', 1823, 781)  |  0.0947 |   0.0947 |         0.0951 |             0.0975 |          0.2353 | 0.1217 | 0.1133 | Naive            |
| ('P6,T6', 1693, 911)  |  0.0951 |   0.0951 |         0.0960 |             0.0973 |          0.2160 | 0.1104 | 0.1006 | Naive            |
| ('P6,T7', 1563, 1041) |  0.0950 |   0.0950 |         0.0960 |             0.0967 |          0.2005 | 0.1126 | 0.1070 | Naive            |
| ('P6,T8', 1433, 1171) |  0.0945 |   0.0945 |         0.0954 |             0.0958 |          0.1895 | 0.1033 | 0.1120 | Naive            |
| ('P6,T9', 1302, 1302) |  0.0969 |   0.0969 |         0.0981 |             0.0980 |          0.1853 | 0.1086 | 0.1140 | Naive            |

## R^2 por elemento (7 modelos)
|                       |   Naive |   SARIMA |   Holt-Winters |   Regresión Lineal |   Random Forest |    RNN |    LSTM | Mejor            |
|:----------------------|--------:|---------:|---------------:|-------------------:|----------------:|-------:|--------:|:-----------------|
| ('P1,T1', 58, 6)      |  -1.214 |   -1.214 |         -2.216 |             -0.625 |          -3.089 | -3.003 |  -9.566 | Regresión Lineal |
| ('P1,T2', 55, 9)      |  -0.179 |   -0.179 |         -0.376 |             -0.007 |          -0.009 | -2.643 | -15.269 | Regresión Lineal |
| ('P1,T3', 52, 12)     |  -0.170 |   -0.170 |         -0.150 |             -0.080 |         -10.333 | -9.705 | -44.517 | Regresión Lineal |
| ('P1,T4', 48, 16)     |   0.671 |    0.140 |          0.781 |              0.120 |          -3.873 | -2.119 |  -8.020 | Holt-Winters     |
| ('P1,T5', 45, 19)     |   0.791 |    0.020 |          0.834 |             -0.470 |          -7.614 | -1.640 |  -5.435 | Holt-Winters     |
| ('P1,T6', 42, 22)     |   0.858 |    0.116 |          0.878 |             -0.271 |          -4.721 | -1.294 |  -3.865 | Holt-Winters     |
| ('P1,T7', 39, 25)     |   0.881 |    0.350 |          0.914 |             -0.145 |          -2.770 | -2.162 |  -2.593 | Holt-Winters     |
| ('P1,T8', 36, 28)     |   0.904 |    0.617 |          0.934 |              0.458 |          -1.518 | -0.436 |  -1.297 | Holt-Winters     |
| ('P1,T9', 32, 32)     |   0.894 |    0.659 |          0.940 |             -0.085 |          -2.264 | -0.923 |  -1.295 | Holt-Winters     |
| ('P2,T1', 117, 13)    |  -0.718 |   -0.718 |         -0.314 |             -1.253 |         -24.620 | -4.302 | -26.171 | Holt-Winters     |
| ('P2,T2', 111, 19)    |   0.791 |    0.791 |          0.802 |              0.658 |          -7.015 |  0.413 |  -2.394 | Holt-Winters     |
| ('P2,T3', 104, 26)    |   0.902 |    0.902 |          0.900 |              0.809 |          -2.118 |  0.733 |  -0.294 | Naive            |
| ('P2,T4', 98, 32)     |   0.894 |    0.894 |          0.900 |              0.840 |          -1.218 |  0.782 |  -0.058 | Holt-Winters     |
| ('P2,T5', 91, 39)     |   0.903 |    0.903 |          0.872 |              0.849 |          -0.731 |  0.811 |   0.122 | Naive            |
| ('P2,T6', 85, 45)     |   0.906 |    0.906 |          0.857 |              0.884 |          -0.391 |  0.826 |   0.349 | Naive            |
| ('P2,T7', 78, 52)     |   0.907 |    0.907 |          0.858 |              0.877 |          -0.335 |  0.828 |   0.348 | Naive            |
| ('P2,T8', 72, 58)     |   0.902 |    0.902 |          0.835 |              0.880 |          -0.257 |  0.839 |   0.456 | Naive            |
| ('P2,T9', 65, 65)     |   0.894 |    0.894 |          0.797 |              0.840 |          -0.172 |  0.784 |   0.097 | Naive            |
| ('P3,T1', 232, 25)    |   0.881 |    0.887 |          0.857 |              0.870 |          -2.776 |  0.680 |  -0.241 | SARIMA           |
| ('P3,T2', 219, 38)    |   0.898 |    0.894 |          0.864 |              0.881 |          -0.912 |  0.811 |   0.215 | Naive            |
| ('P3,T3', 206, 51)    |   0.909 |    0.904 |          0.864 |              0.904 |          -0.302 |  0.866 |   0.470 | Naive            |
| ('P3,T4', 193, 64)    |   0.895 |    0.888 |          0.844 |              0.891 |          -0.198 |  0.843 |   0.559 | Naive            |
| ('P3,T5', 180, 77)    |   0.878 |    0.869 |          0.817 |              0.873 |          -5.054 |  0.796 |   0.121 | Naive            |
| ('P3,T6', 168, 89)    |   0.910 |    0.904 |          0.861 |              0.904 |          -3.085 |  0.851 |   0.295 | Naive            |
| ('P3,T7', 155, 102)   |   0.915 |    0.910 |          0.869 |              0.912 |          -1.529 |  0.883 |   0.468 | Naive            |
| ('P3,T8', 142, 115)   |   0.913 |    0.903 |          0.856 |              0.903 |          -0.886 |  0.838 |   0.605 | Naive            |
| ('P3,T9', 129, 128)   |   0.916 |    0.905 |          0.840 |              0.904 |          -1.752 |  0.864 |   0.239 | Naive            |
| ('P4,T1', 468, 51)    |   0.909 |    0.909 |          0.911 |              0.903 |           0.838 |  0.800 |   0.786 | Holt-Winters     |
| ('P4,T2', 442, 77)    |   0.878 |    0.878 |          0.872 |              0.872 |           0.717 |  0.783 |   0.729 | Naive            |
| ('P4,T3', 416, 103)   |   0.916 |    0.916 |          0.910 |              0.912 |           0.841 |  0.834 |   0.810 | Naive            |
| ('P4,T4', 390, 129)   |   0.917 |    0.917 |          0.905 |              0.912 |           0.775 |  0.834 |   0.728 | Naive            |
| ('P4,T5', 364, 155)   |   0.947 |    0.947 |          0.942 |              0.944 |           0.875 |  0.870 |   0.896 | Naive            |
| ('P4,T6', 338, 181)   |   0.971 |    0.971 |          0.969 |              0.970 |           0.916 |  0.955 |   0.890 | Naive            |
| ('P4,T7', 312, 207)   |   0.974 |    0.974 |          0.972 |              0.973 |           0.950 |  0.960 |   0.925 | Naive            |
| ('P4,T8', 286, 233)   |   0.982 |    0.981 |          0.980 |              0.981 |           0.965 |  0.965 |   0.959 | Naive            |
| ('P4,T9', 260, 259)   |   0.978 |    0.978 |          0.976 |              0.978 |           0.968 |  0.972 |   0.963 | Naive            |
| ('P5,T1', 1170, 130)  |   0.917 |    0.917 |          0.912 |              0.916 |           0.745 |  0.810 |   0.884 | Naive            |
| ('P5,T2', 1105, 195)  |   0.970 |    0.970 |          0.970 |              0.970 |           0.948 |  0.969 |   0.968 | Holt-Winters     |
| ('P5,T3', 1040, 260)  |   0.978 |    0.978 |          0.979 |              0.978 |           0.970 |  0.977 |   0.969 | Holt-Winters     |
| ('P5,T4', 975, 325)   |   0.978 |    0.978 |          0.978 |              0.978 |           0.969 |  0.977 |   0.974 | Naive            |
| ('P5,T5', 910, 390)   |   0.973 |    0.973 |          0.972 |              0.973 |           0.965 |  0.972 |   0.966 | Naive            |
| ('P5,T6', 845, 455)   |   0.983 |    0.983 |          0.983 |              0.983 |           0.978 |  0.981 |   0.975 | Regresión Lineal |
| ('P5,T7', 780, 520)   |   0.987 |    0.987 |          0.987 |              0.987 |           0.980 |  0.986 |   0.981 | Naive            |
| ('P5,T8', 715, 585)   |   0.987 |    0.987 |          0.987 |              0.987 |           0.980 |  0.984 |   0.977 | Naive            |
| ('P5,T9', 650, 650)   |   0.988 |    0.988 |          0.988 |              0.988 |           0.971 |  0.986 |   0.979 | Naive            |
| ('P6,T1', 2344, 260)  |   0.978 |    0.978 |          0.979 |              0.978 |           0.972 |  0.971 |   0.976 | Holt-Winters     |
| ('P6,T2', 2214, 390)  |   0.973 |    0.973 |          0.973 |              0.972 |           0.966 |  0.968 |   0.971 | Holt-Winters     |
| ('P6,T3', 2084, 520)  |   0.987 |    0.987 |          0.987 |              0.987 |           0.981 |  0.986 |   0.986 | Holt-Winters     |
| ('P6,T4', 1953, 651)  |   0.988 |    0.988 |          0.988 |              0.988 |           0.972 |  0.987 |   0.988 | Holt-Winters     |
| ('P6,T5', 1823, 781)  |   0.987 |    0.987 |          0.987 |              0.987 |           0.916 |  0.982 |   0.984 | Naive            |
| ('P6,T6', 1693, 911)  |   0.988 |    0.988 |          0.988 |              0.988 |           0.932 |  0.985 |   0.987 | Naive            |
| ('P6,T7', 1563, 1041) |   0.989 |    0.989 |          0.989 |              0.989 |           0.944 |  0.987 |   0.988 | Naive            |
| ('P6,T8', 1433, 1171) |   0.990 |    0.990 |          0.990 |              0.990 |           0.951 |  0.989 |   0.987 | Naive            |
| ('P6,T9', 1302, 1302) |   0.989 |    0.989 |          0.989 |              0.989 |           0.953 |  0.987 |   0.986 | Naive            |

## Rank por elemento (6 modelos, criterio de la tesis)
| Elemento   |   SARIMA |   Holt-Winters |   Regresión Lineal |   Random Forest |   RNN |   LSTM |
|:-----------|---------:|---------------:|-------------------:|----------------:|------:|-------:|
| P1,T1      |        2 |              3 |                  1 |               5 |     4 |      6 |
| P1,T2      |        4 |              3 |                  2 |               1 |     5 |      6 |
| P1,T3      |        3 |              1 |                  1 |               5 |     4 |      6 |
| P1,T4      |        2 |              1 |                  3 |               5 |     4 |      6 |
| P1,T5      |        2 |              1 |                  3 |               6 |     4 |      5 |
| P1,T6      |        2 |              1 |                  3 |               6 |     4 |      5 |
| P1,T7      |        2 |              1 |                  3 |               5 |     4 |      5 |
| P1,T8      |        2 |              1 |                  3 |               6 |     4 |      5 |
| P1,T9      |        2 |              1 |                  3 |               6 |     4 |      5 |
| P2,T1      |        2 |              1 |                  3 |               5 |     4 |      6 |
| P2,T2      |        2 |              1 |                  3 |               6 |     4 |      5 |
| P2,T3      |        1 |              1 |                  3 |               6 |     4 |      5 |
| P2,T4      |        2 |              1 |                  3 |               6 |     4 |      5 |
| P2,T5      |        1 |              2 |                  3 |               6 |     4 |      5 |
| P2,T6      |        1 |              3 |                  2 |               6 |     4 |      5 |
| P2,T7      |        1 |              3 |                  2 |               6 |     4 |      5 |
| P2,T8      |        1 |              4 |                  2 |               6 |     3 |      5 |
| P2,T9      |        1 |              3 |                  2 |               6 |     4 |      5 |
| P3,T1      |        1 |              3 |                  2 |               6 |     4 |      5 |
| P3,T2      |        1 |              3 |                  2 |               6 |     4 |      5 |
| P3,T3      |        1 |              4 |                  2 |               6 |     3 |      5 |
| P3,T4      |        1 |              3 |                  1 |               6 |     3 |      5 |
| P3,T5      |        1 |              3 |                  1 |               6 |     4 |      5 |
| P3,T6      |        1 |              3 |                  2 |               6 |     4 |      5 |
| P3,T7      |        1 |              4 |                  1 |               6 |     3 |      5 |
| P3,T8      |        1 |              3 |                  2 |               6 |     4 |      5 |
| P3,T9      |        1 |              4 |                  2 |               6 |     3 |      5 |
| P4,T1      |        1 |              1 |                  3 |               4 |     5 |      6 |
| P4,T2      |        1 |              2 |                  2 |               6 |     4 |      5 |
| P4,T3      |        1 |              3 |                  2 |               4 |     5 |      6 |
| P4,T4      |        1 |              3 |                  2 |               5 |     4 |      6 |
| P4,T5      |        1 |              3 |                  2 |               5 |     6 |      4 |
| P4,T6      |        1 |              3 |                  2 |               5 |     4 |      6 |
| P4,T7      |        1 |              3 |                  2 |               5 |     4 |      6 |
| P4,T8      |        1 |              3 |                  2 |               4 |     5 |      6 |
| P4,T9      |        1 |              3 |                  2 |               5 |     4 |      6 |
| P5,T1      |        1 |              3 |                  2 |               6 |     5 |      4 |
| P5,T2      |        2 |              1 |                  3 |               6 |     4 |      5 |
| P5,T3      |        2 |              1 |                  3 |               5 |     4 |      6 |
| P5,T4      |        1 |              1 |                  3 |               6 |     4 |      5 |
| P5,T5      |        1 |              3 |                  2 |               6 |     4 |      5 |
| P5,T6      |        2 |              3 |                  1 |               5 |     4 |      6 |
| P5,T7      |        2 |              3 |                  1 |               6 |     4 |      5 |
| P5,T8      |        1 |              3 |                  2 |               5 |     4 |      6 |
| P5,T9      |        1 |              3 |                  2 |               6 |     4 |      5 |
| P6,T1      |        1 |              1 |                  3 |               5 |     6 |      4 |
| P6,T2      |        1 |              2 |                  3 |               6 |     5 |      4 |
| P6,T3      |        2 |              1 |                  3 |               6 |     5 |      4 |
| P6,T4      |        2 |              1 |                  3 |               6 |     5 |      4 |
| P6,T5      |        1 |              2 |                  3 |               6 |     5 |      4 |
| P6,T6      |        1 |              2 |                  3 |               6 |     5 |      4 |
| P6,T7      |        1 |              2 |                  3 |               6 |     5 |      4 |
| P6,T8      |        1 |              2 |                  3 |               6 |     4 |      5 |
| P6,T9      |        1 |              3 |                  2 |               6 |     4 |      5 |

## Diebold-Mariano vs. Naive por elemento
| Elemento   | SARIMA         | Holt-Winters    | Regresión Lineal   | Random Forest   | RNN            | LSTM           |
|:-----------|:---------------|:----------------|:-------------------|:----------------|:---------------|:---------------|
| P1,T1      | n/a            | sin diferencia  | sin diferencia     | sin diferencia  | sin diferencia | sin diferencia |
| P1,T2      | n/a            | sin diferencia  | sin diferencia     | sin diferencia  | sin diferencia | peor que Naive |
| P1,T3      | n/a            | sin diferencia  | sin diferencia     | peor que Naive  | peor que Naive | peor que Naive |
| P1,T4      | peor que Naive | sin diferencia  | peor que Naive     | peor que Naive  | peor que Naive | peor que Naive |
| P1,T5      | peor que Naive | sin diferencia  | peor que Naive     | peor que Naive  | peor que Naive | peor que Naive |
| P1,T6      | peor que Naive | sin diferencia  | peor que Naive     | peor que Naive  | peor que Naive | peor que Naive |
| P1,T7      | peor que Naive | sin diferencia  | peor que Naive     | peor que Naive  | peor que Naive | peor que Naive |
| P1,T8      | peor que Naive | sin diferencia  | peor que Naive     | peor que Naive  | peor que Naive | peor que Naive |
| P1,T9      | peor que Naive | mejor que Naive | peor que Naive     | peor que Naive  | peor que Naive | peor que Naive |
| P2,T1      | n/a            | sin diferencia  | sin diferencia     | peor que Naive  | peor que Naive | peor que Naive |
| P2,T2      | n/a            | sin diferencia  | peor que Naive     | peor que Naive  | peor que Naive | peor que Naive |
| P2,T3      | n/a            | sin diferencia  | peor que Naive     | peor que Naive  | peor que Naive | peor que Naive |
| P2,T4      | n/a            | sin diferencia  | peor que Naive     | peor que Naive  | peor que Naive | peor que Naive |
| P2,T5      | n/a            | sin diferencia  | peor que Naive     | peor que Naive  | peor que Naive | peor que Naive |
| P2,T6      | n/a            | peor que Naive  | peor que Naive     | peor que Naive  | peor que Naive | peor que Naive |
| P2,T7      | n/a            | peor que Naive  | peor que Naive     | peor que Naive  | peor que Naive | peor que Naive |
| P2,T8      | n/a            | peor que Naive  | peor que Naive     | peor que Naive  | peor que Naive | peor que Naive |
| P2,T9      | n/a            | peor que Naive  | peor que Naive     | peor que Naive  | peor que Naive | peor que Naive |
| P3,T1      | sin diferencia | sin diferencia  | sin diferencia     | peor que Naive  | peor que Naive | peor que Naive |
| P3,T2      | sin diferencia | sin diferencia  | sin diferencia     | peor que Naive  | peor que Naive | peor que Naive |
| P3,T3      | sin diferencia | peor que Naive  | sin diferencia     | peor que Naive  | peor que Naive | peor que Naive |
| P3,T4      | sin diferencia | peor que Naive  | sin diferencia     | peor que Naive  | peor que Naive | peor que Naive |
| P3,T5      | sin diferencia | peor que Naive  | sin diferencia     | peor que Naive  | peor que Naive | peor que Naive |
| P3,T6      | sin diferencia | peor que Naive  | sin diferencia     | peor que Naive  | peor que Naive | peor que Naive |
| P3,T7      | sin diferencia | peor que Naive  | sin diferencia     | peor que Naive  | peor que Naive | peor que Naive |
| P3,T8      | sin diferencia | peor que Naive  | sin diferencia     | peor que Naive  | peor que Naive | peor que Naive |
| P3,T9      | sin diferencia | peor que Naive  | sin diferencia     | peor que Naive  | peor que Naive | peor que Naive |
| P4,T1      | n/a            | sin diferencia  | sin diferencia     | peor que Naive  | peor que Naive | peor que Naive |
| P4,T2      | n/a            | sin diferencia  | sin diferencia     | peor que Naive  | peor que Naive | peor que Naive |
| P4,T3      | n/a            | sin diferencia  | sin diferencia     | peor que Naive  | peor que Naive | peor que Naive |
| P4,T4      | n/a            | sin diferencia  | sin diferencia     | peor que Naive  | peor que Naive | peor que Naive |
| P4,T5      | n/a            | sin diferencia  | peor que Naive     | peor que Naive  | peor que Naive | peor que Naive |
| P4,T6      | n/a            | sin diferencia  | sin diferencia     | peor que Naive  | peor que Naive | peor que Naive |
| P4,T7      | sin diferencia | peor que Naive  | sin diferencia     | peor que Naive  | peor que Naive | peor que Naive |
| P4,T8      | sin diferencia | peor que Naive  | sin diferencia     | peor que Naive  | peor que Naive | peor que Naive |
| P4,T9      | sin diferencia | peor que Naive  | sin diferencia     | peor que Naive  | sin diferencia | peor que Naive |
| P5,T1      | sin diferencia | sin diferencia  | sin diferencia     | peor que Naive  | peor que Naive | peor que Naive |
| P5,T2      | n/a            | sin diferencia  | sin diferencia     | peor que Naive  | sin diferencia | sin diferencia |
| P5,T3      | n/a            | sin diferencia  | sin diferencia     | peor que Naive  | sin diferencia | peor que Naive |
| P5,T4      | sin diferencia | sin diferencia  | sin diferencia     | peor que Naive  | sin diferencia | peor que Naive |
| P5,T5      | sin diferencia | sin diferencia  | sin diferencia     | peor que Naive  | sin diferencia | peor que Naive |
| P5,T6      | n/a            | sin diferencia  | sin diferencia     | peor que Naive  | sin diferencia | peor que Naive |
| P5,T7      | n/a            | sin diferencia  | sin diferencia     | peor que Naive  | sin diferencia | peor que Naive |
| P5,T8      | n/a            | sin diferencia  | sin diferencia     | peor que Naive  | peor que Naive | peor que Naive |
| P5,T9      | n/a            | sin diferencia  | sin diferencia     | peor que Naive  | peor que Naive | peor que Naive |
| P6,T1      | n/a            | sin diferencia  | sin diferencia     | peor que Naive  | peor que Naive | peor que Naive |
| P6,T2      | n/a            | sin diferencia  | sin diferencia     | peor que Naive  | peor que Naive | peor que Naive |
| P6,T3      | sin diferencia | sin diferencia  | sin diferencia     | peor que Naive  | peor que Naive | sin diferencia |
| P6,T4      | n/a            | sin diferencia  | sin diferencia     | peor que Naive  | peor que Naive | peor que Naive |
| P6,T5      | n/a            | sin diferencia  | peor que Naive     | peor que Naive  | peor que Naive | peor que Naive |
| P6,T6      | sin diferencia | sin diferencia  | peor que Naive     | peor que Naive  | peor que Naive | peor que Naive |
| P6,T7      | sin diferencia | sin diferencia  | sin diferencia     | peor que Naive  | peor que Naive | peor que Naive |
| P6,T8      | n/a            | sin diferencia  | sin diferencia     | peor que Naive  | peor que Naive | peor que Naive |
| P6,T9      | sin diferencia | sin diferencia  | sin diferencia     | peor que Naive  | peor que Naive | peor que Naive |

## Órdenes elegidos por auto_arima (54 experimentos)
|                        |   Orden auto_arima |
|:-----------------------|-------------------:|
| (0, 1, 0)x(0, 0, 0, 5) |                 36 |
| (0, 1, 1)x(0, 0, 1, 5) |                  7 |
| (1, 0, 0)x(0, 0, 0, 5) |                  6 |
| (0, 1, 1)x(0, 0, 0, 5) |                  2 |
| (2, 1, 2)x(0, 0, 0, 5) |                  2 |
| (0, 1, 0)x(0, 0, 1, 5) |                  1 |

## Variabilidad entre semillas (RNN/LSTM): desviación estándar del MAE
| Modelo   |   ('MAE desv.', 'mean') |   ('MAE desv.', 'max') |   ('R² desv.', 'mean') |   ('R² desv.', 'max') |
|:---------|------------------------:|-----------------------:|-----------------------:|----------------------:|
| LSTM     |                  0.0107 |                 0.0417 |                 0.3855 |                6.0674 |
| RNN      |                  0.0088 |                 0.0238 |                 0.1657 |                1.8430 |