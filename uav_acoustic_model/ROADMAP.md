# UAV Acoustic Model Roadmap

## Конечная цель

Создать воспроизводимую и экспериментально проверяемую систему из трёх
пространственно разнесённых микрофонных станций, которая по калиброванным
bearing-измерениям определяет и затем причинно отслеживает 3D-координаты
движущегося БПЛА. Все этапы должны сохранять соглашение
`tau_ij = T_i - T_j`, единицы SI, явные допущения и автоматическую численную
приёмку.

Одно мгновенное измерение одной микрофонной станции определяет только bearing,
но не дальность. В instantaneous temporal модели сохраняется масштабная
неоднозначность. Идеальная exact retarded-time модель с известным конечным `c`,
строго постоянной и нерадиальной скоростью может формально дать слабую
информацию о масштабе, однако это не является практически устойчивой заменой
нескольким пространственно разнесённым синхронизированным станциям или другому
независимому источнику дальности.

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
| S7B | Done | Задать ENU, station poses, общий measurement contract и статическую 3D bearing-триангуляцию | `StationPose`, `BearingMeasurement`, constrained spherical WLS, observability/Monte Carlo/visualization | deterministic invariance/Jacobian gates, exact nullspace constraints, dimensionless projected-KKT optimality, cross-platform CI, явное вырождение, full static study | S7A |
| S7C | In progress | Реализовать центральный причинный dynamic 3D tracker по проверенным подэтапам | S7C-A…S7C-D | отдельная приёмка measurement model, event stream, filter и robustness benchmark | S7B |
| S7C-A | Done | Проверить retarded-time bearing measurement model для 6D constant-velocity state | dynamic state, retarded prediction/residual/Jacobian, observability notebook | analytic/numeric emission time и Jacobian, radial/nonradial/instantaneous rank diagnostics, invariance | S7B |
| S7C-B | Next | Задать причинный поток асинхронных событий и offline batch reference | event contract, ordering/dropout rules, batch baseline | available-time causality и отсутствие future access | S7C-A |
| S7C-C | Planned | Реализовать первый центральный EKF baseline | EKF и matched no-filter/static baselines | consistency и causal Monte Carlo без скрытого truth | S7C-B |
| S7C-D | Planned | Проверить dropout/outlier/out-of-sequence robustness | controlled benchmark и failure reporting | reproducible stress gates и явные ограничения | S7C-C |
| S8 | Planned | Добавить измеренные сигналы БПЛА и held-out datasets | dataset interface, signal model, held-out validation | train/evaluation separation и воспроизводимость | S4, S7A |
| S9 | Planned | Проверить сложный акустический фон | цветной/коррелированный noise и interferers | контролируемые сценарии и failure reporting | S8 |
| S10 | Planned | Добавить физику среды | температура, ветер и пространственно меняющийся `c` | независимые limiting-case tests | S3, S8 |
| S11 | Planned | Добавить отражения и многолучёвость | room/ground reflection scenarios | direct-path baseline и bias/tail analysis | S9–S10 |
| S12 | Planned | Реализовать real-time hardware, синхронизацию и transport станций | clock calibration, station I/O, network protocol | воспроизводимая синхронизация и latency/failure audit | S7C, S10–S11 |
| S13 | Planned | Выполнить multi-station field validation | versioned datasets, calibration protocol, final report | независимый 3D ground truth и полностью воспроизводимый benchmark | S8–S12 |

## Уровни зрелости

| Уровень | Определение |
|---|---|
| M0 | Проверенная математика и соглашения |
| M1 | Воспроизводимая синтетическая валидация |
| M2 | Валидация на записанных данных |
| M3 | Real-time отдельная микрофонная станция |
| M4 | Real-time многопозиционная 3D-система |
| M5 | Полевая валидация с независимым ground truth |

## Текущий переход

S7A завершён как калиброванный benchmark неопределённости отдельных bearing-
измерений. S7B завершён после повторного cross-platform corrective gate:
compatibility проверяется по финальному constrained solution, а projected-KKT
остаётся dimensionless Newton-correction metric, инвариантной к rigid
transforms и масштабу сцены. S7C имеет статус `In progress`: S7C-A завершён
после corrective gate radial/nonradial/instantaneous observability. S7C-B —
`Next`; он задаст causal event stream и offline batch reference, а S7C-C
будет отдельным EKF baseline, а S7C-D — отдельным robustness benchmark. В
S7C-A tracking и state update не добавляются.

Одно мгновенное bearing-измерение одной станции не определяет дальность.
Temporal retarded-time модель при строгом constant velocity и известном
конечном `c` может формально получить дополнительную слабую информацию о
масштабе, но это не заменяет практически устойчивую multi-station 3D
localization. Несколько известных поз в общей ENU-системе обеспечивают
геометрическую устойчивость; эти задачи и их критерии не взаимозаменяемы.
