# UAV Acoustic Model Roadmap

## Конечная цель

Создать воспроизводимую и экспериментально проверяемую систему акустического
оценивания направления на БПЛА: от геометрии и синтеза многоканального сигнала
до калиброванных bearing-измерений, последовательной фильтрации и полевой
валидации. Все этапы должны сохранять соглашение `tau_ij = T_i - T_j`, единицы
SI, явные допущения и автоматическую численную приёмку.

Одна микрофонная станция измеряет только направление (bearing). Её temporal
bearing tracking может сглаживать последовательность направлений, но не даёт
наблюдаемую 3D-дальность. 3D localization требует как минимум нескольких
пространственно разнесённых и синхронизированных станций либо независимого
источника информации о дальности; это отдельная многопозиционная задача.

## Этапы

| Этап | Статус | Цель | Основные deliverables | Критерий приёмки | Зависит от |
|---|---|---|---|---|---|
| S0 | Done | Зафиксировать геометрию и знаки TDOA | `model/geometry.py`, `model/tdoa.py`, базовые тесты | единичный `u`, антисимметрия и циклы TDOA | — |
| S1 | Done | Локальная чувствительность и условная CRLB | Jacobian, Fisher, rank/condition, CRLB | analytic/numeric Jacobian и корректное вырождение | S0 |
| S2 | Done | Проверить WLS и ковариационные модели | TOA/TDOA covariance, Monte Carlo notebook | invariance и малошумовое соответствие CRLB | S1 |
| S3 | Done | Валидировать детерминированное распространение | spherical/plane/second-order, fractional delay | far-field boundary, два согласованных delay backend | S0 |
| S4 | Done | Оценивать TDOA из сигналов | GCC-PHAT, cycle projection, AWGN studies | знак, sub-sample точность, tails/coverage | S3 |
| S5 | Done | Реализовать прямую пространственную оценку | equal-weight far-field SRP-PHAT | reference/vector agreement и paired comparison | S4 |
| S6 | Done | Учесть кинематику источника | trajectories, retarded-time propagation, moving study | analytic/numeric emission time и Doppler tests | S3–S5 |
| S7 | Done | Получить причинную последовательность кадров | continuous stream, chunking, sequential DOA | единый signal/noise stream и causal frame estimates | S6 |
| S7A | Done | Калибровать неопределённость bearing-измерений | spherical residual, calibration/evaluation study, covariance/quality CSV | split isolation, PSD `R`, calibration-bias-centered evaluation NIS, notebook | S7 |
| S7B | Next | Последовательно фильтровать bearing одной станции | отдельно валидируемый bearing tracker и consistency study | причинность, calibrated `R`, matched no-filter baseline | S7A |
| S8 | Planned | Добавить измеренную спектральную модель БПЛА | dataset interface, signal model, held-out validation | train/evaluation separation и воспроизводимость | S4, S7A |
| S9 | Planned | Проверить сложный акустический фон | цветной/коррелированный noise и interferers | контролируемые сценарии и failure reporting | S8 |
| S10 | Planned | Добавить физику среды | температура, ветер и пространственно меняющийся `c` | независимые limiting-case tests | S3, S8 |
| S11 | Planned | Добавить отражения и многолучёвость | room/ground reflection scenarios | direct-path baseline и bias/tail analysis | S9–S10 |
| S12 | Planned | Выполнить 3D localization нескольких станций | synchronization, bearing fusion, station geometry | observability, covariance propagation, 3D ground truth | S7A, S10 |
| S13 | Planned | Лабораторная и полевая приёмка | versioned datasets, calibration protocol, final report | независимый ground truth и полностью воспроизводимый benchmark | S8–S12 |

## Текущий переход

S7A завершён как калиброванный benchmark неопределённости отдельных bearing-
измерений. Он сохраняет `R` и `mu_cal` только из calibration split; evaluation
NIS центрируется исключительно по `mu_cal`, а raw normalized squared error
хранится отдельно. S7A не реализует tracking и не является signal-level CRLB.
Следующий этап — S7B; его bearing tracker должен использовать принятые
split-calibrated `R,mu_cal` и сравниваться с matched последовательностью
несглаженных измерений.
