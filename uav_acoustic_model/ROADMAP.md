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
| S7C-B | Done | Задать причинный поток асинхронных событий и offline batch reference | event contract, ordering/dropout rules, constrained retarded-time batch baseline | available-time causality, отсутствие future access, offline/final-prefix agreement и independent-sequence benchmark | S7C-A |
| S7C-C | In progress | Реализовать центральную рекурсивную оценку по проверенным подэтапам | S7C-C1 strict-CV baseline и явно включаемый stochastic-history вариант для манёвров | отдельные causal/Jacobian/history/coverage gates; синтетический benchmark не завершает полевую приёмку | S7C-B |
| S7C-C1 | Done | Реализовать причинный retarded-time EKF baseline при строгой constant velocity | `retarded_ekf.py`, deterministic tests, 96-sequence matched study и notebook | `Q=0`, causal batch initialization, Joseph update, PD covariance contract, sequence-level NIS/NEES/coverage | S7C-B |
| S7C-D | In progress | Проверить dropout/outlier/out-of-sequence robustness | controlled benchmark, opt-in robust variants, confirmation/recovery и failure reporting | reproducible stress gates, causal event accounting, availability/error trade-off и явные ограничения | S7C-C |
| S7C-D1 | Done | Измерить пределы принятого strict-CV C1 при потерях, паузах станций, задержках и выбросах без изменения фильтра | фиксированный протокол, paired 200-block/1800-run study, epoch/sequence/profile CSV и notebook | deterministic/smoke gates, честные denominators/coverage, полный reproducibility audit | S7C-C1 |
| S7C-D2 | Done | Добавить явно включаемые robust initialization и pre-update NIS gate без изменения C1 default | consensus diagnostics, four-way ablation, paired held-out benchmark, CSV и notebook | baseline reproducibility, separate/combined mechanism tests, held-out whole-sequence comparison, full gates | S7C-D1 |
| S8 | In progress | Проверить тракт с записанным приближением source signal | versioned manifest/loader, independent-session paired recorded/broadband pilot, source/session/origin split audit | воспроизводимый provenance, calibration-only uncertainty, session-level denominators и явная граница held-out source benchmark vs field validation | S4, S7A, synthetic S7C pilot |
| S8-GZ | Done (integration pilot) | Записать две заданные траектории одного видимого источника в Gazebo и обработать офлайн тремя станциями | общая ENU-сцена, post-step pose CSV, finite-support адаптер, фиксированный GCC/SRP/tracker, 3D viewer | Gazebo→CSV доказан, прямая/поворот и 50→100 Гц сравнены численно, отказы показаны | S6, synthetic S7C pilot |
| S8-GZ-R | Done (replay contract) | Сделать повторную обработку Gazebo независимой от текущего scene JSON и разделить event IDs опытов | версия `experiment.json`, SHA записи/кода/результатов, переносимый `run_id`, явные init/migration/new, проверяемый viewer | новый каталог проходит запись→init→обработку→viewer без старых результатов; два исходных CSV сохранены, bearing/tracking/RMSE совпали с `03dc7d7` и `b6a20f8` в `1e-12`; Linux/Windows CI | S8-GZ |
| S8-GZ-PX4 | Done (one-flight integration pilot) | Записать один полёт X500 под PX4 SITL и обработать его прежней акустической системой | read-only Gazebo observer, MAVSDK actions/telemetry, MAVLink Offboard setpoints, flight manifest, phases, offline replay и viewer | `pilot_003`: фактические взлёт/прямая/поворот/посадка, 3954 отсчёта без пропусков, 17.02 с аудио, 141/150 валидных публикаций каждого метода, прежние два контроля численно совпали | S8-GZ-R |
| S8-GZ-RANGE | Done (conditional range study) | Завершить PX4/Gazebo integration pilot условным исследованием влияния геометрии, ослабления и манёвра | вторая запись с противоположными поворотами, фиксированный протокол, режим `1/r` с фиксированным фоном, 120 последовательных аудиозапусков, таблицы и notebook | SHA всех входов/результатов, общий source/noise pairing, фактический station SNR, confirmation/availability/P95/failures, пороги 2/5/10 м без экстраполяции | S8-GZ-PX4 |
| S8-GZ-ATTR | Done (known-case diagnostic) | Разделить наблюдаемые механизмы ошибок опубликованного range study без изменения алгоритмов | восемь зафиксированных аудиопотоков, original/ideal-bearing/zero-bias replay, журналы initialization/update/lifecycle, локальный geometry benchmark и notebook | точное воспроизведение 16 original method cases; SHA входов; 8 audio restores; 48 вариантов; причинные выводы с явными границами интерпретации | S8-GZ-RANGE |
| S9 | Planned | Проверить сложный акустический фон | цветной/коррелированный noise и interferers | контролируемые сценарии и failure reporting | S8 |
| S10 | Planned | Добавить физику среды | температура, ветер и пространственно меняющийся `c` | независимые limiting-case tests | S3, S8 |
| S11 | Planned | Добавить отражения и многолучёвость | room/ground reflection scenarios | direct-path baseline и bias/tail analysis | S9–S10 |
| S12 | Planned | Реализовать real-time hardware, синхронизацию и transport станций | clock calibration, station I/O, network protocol | воспроизводимая синхронизация и latency/failure audit | S7C, S10–S11 |
| S13 | Planned | Выполнить multi-station field validation | versioned datasets, calibration protocol, final report | независимый 3D ground truth и полностью воспроизводимый benchmark | S8–S12 |

## Уровни зрелости

S8-GZ подтверждает только подключение уже принятого синтетического
акустического конвейера к кинематическому движению Gazebo. Он не меняет статус
S8 recorded-source validation, S7C-C/D или уровень полевой зрелости. Динамика
полёта, автопилот, ROS 2, PX4, ветер и отражения в этот pilot не входят.
S8-GZ-PX4 отдельно добавляет динамический полёт X500 под PX4 и фактическую
запись Gazebo. Его акустический источник остаётся синтетическим broadband;
один полёт не меняет статус S8 или уровень полевой зрелости.
S8-GZ-RANGE завершает текущую PX4/Gazebo ветвь двумя физическими записями и
условным диапазонным исследованием их переносов. Оно не повышает зрелость до
полевой: source level не калиброван, фон синтетический, а три реализации шума
не оценивают редкие отказы. Контроль с постоянным принятым SNR не отделяет
геометрическую обусловленность от ошибок инициализации и сопровождения,
поэтому его P95 не интерпретируется как чистый геометрический вклад.

Принятая матрица S8-GZ-RANGE содержит 120/120 аудиозапусков. Для основного
критерия confirmation ≤3 s, availability ≥0,90 после первых 3 s и conditional
P95 ≤5 m наибольшая прошедшая точка фиксированного фона равна 100 m при
`SNR_ref=0 dB` и 200 m при `SNR_ref=10/20 dB` для обеих записей и GCC/SRP.
Контроль с принятым SNR +10 dB также проходит до 200 m, показывая ограничение
геометрии на следующей точке. Точные границы для P95 2/5/10 m, явные отказы и
ограничения приведены в `LOCALIZATION_RANGE_REPORT.md`.

S8-GZ-ATTR использует восемь известных случаев опубликованной серии для
диагностики, а не для независимого подтверждения качества. Сравнение с
ideal-bearing и zero-bias показывает взаимодействие акустических ошибок,
радиально слабой геометрии, инициализации и update starvation; разности между
вариантами не являются аддитивным разложением. Рабочие GCC, SRP и tracker не
изменены. Следующее изменение должно быть заморожено на отдельных development
cases и проверено на held-out streams до любого заявления об улучшении.

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
измерений. S7B принят после повторного cross-platform corrective gate и
интеграции formal-model поправок:
compatibility проверяется по финальному constrained solution, а projected-KKT
остаётся dimensionless Newton-correction metric, инвариантной к rigid
transforms и масштабу сцены. S7C имеет статус `In progress`: S7C-A завершён
после corrective gate radial/nonradial/instantaneous observability. S7C-B —
`Done`: causal event stream и offline/causal-prefix batch reference прошли
повторный входной gate на базе `c281523d` с явным `tangent_frame` и устойчивым
retarded-time root. Внутри существующих S2/S4/S6/S7A/S7C-B выполнен
corrective audit без нового номера этапа: глобальный far-field WLS search,
support singular covariance/NIS, pole-safe bearing coordinates, GCC fractional
bound/Nyquist, moving-source TDOA semantics, finite trajectory support,
Doppler band gate и collision-free RNG provenance закреплены regressions и
пересчитанными артефактами. Этот пакет не меняет статусы завершённых этапов и
не является расширением физики. S7C-C имеет статус `In progress`, а
ограниченный S7C-C1 EKF baseline завершил собственную приёмку: 96 независимых
sequences, `96/96` successful initialization/final-valid, полный pytest и
15/15 notebooks PASS. Это не завершает весь S7C-C и не добавляет `Q>0` или
поддержку манёвров. S7C-D1 завершён как отдельный стресс-бенчмарк уже принятого
C1: 200 независимых base blocks, 1800 paired profile-runs и полный
reproducibility/event-partition audit. Опубликованный D2 сохраняется отдельным
вариантом; текущая corrective-реализация внутри S7C-D добавила подтверждение
initial hypothesis независимыми последующими bearings и bounded causal
recovery после наблюдаемой потери согласованности. Held-out benchmark использует
новый seed, но не объявляет поддержку манёвров. S7C-D остаётся `In progress`;
дальнейшие D-проверки манёвренных моделей зависят от соответствующего
расширения S7C-C.

Исправление event contract внутри S7C-D на `d281d730` закрыто: конфликт payload
активного поколения прекращает confirmed publication и запускает свежую
причинную reinitialization, исторический конфликт остаётся audit-событием,
а 18-event bound относится только к построению гипотезы. Новая работа
внутри S7C-C добавляет явно включаемый `Q>0` augmented-history фильтр для
заранее заданных гладких манёвров. C1, опубликованный D2 и
`confirmed_recovery` остаются воспроизводимыми вариантами. Статистический
benchmark на прямых bearing-событиях не является валидацией акустического
frontend, окружающей среды или полевой системы; S7C-C и общий S7C остаются
`In progress`.

Текущее исправление внутри S7C-C ограничивает transient augmented history во
время длинных прогнозов, учитывает peak owned history arrays до marginalization
и разделяет фактическое время принятого update от времени его первого
отображения во внешней публикации. Это корректировка текущей реализации, а не
новый исследовательский подэтап; статусы таблицы не изменяются.

Следующая интеграционная работа внутри того же S7C соединяет проверенные
компоненты в цепочку `continuous three-station audio → GCC/SRP → calibrated
BearingMeasurement → causal retarded-time 3D tracking`. Пилот использует один
общий source stream, непрерывные station channel/noise arrays, отдельные
calibration/evaluation sequences и прежний frozen `Qc`; полевая физика и новый
фильтр не добавляются. Из-за стоимости dense augmented history tracker получает
предобъявленное truth-free frame subset, тогда как bearing-метрики используют
все кадры. Это текущая работа внутри S7C, не новый подэтап.

Корректирующий validation-contract gate этой же интеграционной работы завершил
три обязательных проверки без изменения `Qc`, NIS-порогов или алгоритма:
evaluation-time uncertainty теперь выбирается только по
`(station_id, estimator_variant)` из pooled calibration split; удлинённая
`4.5 s` запись с фиксированным манёвром `[2.5,3.5) s` действительно содержит
confirmed tracking до манёвра и updates во время него для 10/12 потоков, а два
отказа остаются явными; известная emitted band edge `10 kHz` проходит
Doppler/Nyquist guard, тогда как aliasing-case отклоняется. Phase reporting
различает evaluator-only emission-time разметку updates и processing-time
разметку state errors. Это исправление текущей работы, не новый подэтап;
S7C и S7C-C остаются `In progress` до последующих синтетических и полевых gates.

Синтетическая часть S7C может считаться завершённой только когда одновременно:

1. continuous multistation streams, координаты ENU и causal timestamps имеют
   deterministic cross-platform regressions;
2. GCC и SRP на одинаковом аудио формируют truth-free calibrated measurements
   с непересекающимся calibration/evaluation provenance;
3. causal tracker публикует 3D state с явными invalid gaps, failures, resets,
   denominators, runtime и memory diagnostics;
4. временная/межстанционная корреляция и empirical covariance coverage измерены
   и ограничения независимой measurement model явно задокументированы;
5. полный pytest, новый notebook и Windows/Linux CI зелёные.

Эти критерии не заменяют последующие реальные аудиоданные, синхронизацию
оборудования, среду и полевую проверку из S8–S13.

Сквозной синтетический pilot `continuous three-station audio → GCC/SRP →
BearingMeasurement → causal 3D tracking` завершён как отдельный интеграционный
gate: его validation contract и Windows/Linux CI зелёные. Это завершение
синтетической интеграции, но не статистическая квалификация редких хвостов и не
полевая валидация. Текущий S8 добавляет только записанное приближение source
signal и не добавляет новую среду, отражения, ветер или фильтр.

Recorded-source pilot внутри S8 завершён в ограниченном
`single_session_integration_demonstration` scope. S8 остаётся `In progress`:
для held-out статистической приёмки нужны разные исходные записи/сеансы в
calibration и evaluation. Непересекающиеся интервалы одной записи этой
зависимости не устраняют.

Следующий S8 gate добавил четыре provenance-verified исходных session:
2 calibration и 2 evaluation без пересечения `session_id`/`origin_asset_id`.
Manifest audit больше не доверяет ручному boolean-флагу, а выводит
независимость из фактического состава. Ограниченный paired recorded/broadband
benchmark завершён на source-session уровне, но causal tracker не подтвердил
ни один из 16 method-session запусков в коротком `2 s`/9-event протоколе.
Поэтому bearing-level held-out результат воспроизводим, а position/velocity
accuracy и posterior coverage остаются непроверенными. S8 сохраняет статус
`In progress`: это отрицательный integration result, не полевая валидация и не
основание менять `Qc`, NIS gates или алгоритм по evaluation.

Корректирующий gate S8 проверяет прежде всего **достаточность расписания**:
идеальные retarded bearings при исторических 2 с также не подтверждаются,
тогда как фиксированные 4.5 с дают подтверждение и последующие коррекции.
Новая запись и разделение сеансов не меняются; длительность и явный бюджет
batch-оптимизаций зафиксированы до повторного audio evaluation в
`S8_TRACKING_FEASIBILITY_PROTOCOL.md`. Исторические CSV остаются отдельными.
S8 остаётся `In progress` независимо от результата ограниченного пилота:
статистическая и полевая валидация не входят в этот gate.

Повторный ограниченный эксперимент подтвердил 10/16 method-session запусков,
90 последующих corrections приняты, 20 отклонены как
`emission_outside_history`; ещё 6 запусков исчерпали заранее установленный
четырёх-fit бюджет, включая один accurate broadband/-6 dB поток. Это
показывает одновременно, что исправленное расписание позволяет сопровождение,
и что cost cap ухудшает доступность. Для записанного mini-quadcopter сеанса
bearing errors велики, а для другого recorded-сеанса доступное сопровождение
существенно менее точно broadband control. Ни один из этих двух сеансов не
заменяет популяционную статистическую или полевую валидацию; S8 остаётся
`In progress` без нового подэтапа или новой модели среды.

Анализ внутри того же S8 (`analysis/s8-calibration-transfer`) проверил контракт
сферического calibration bias/R с residual трекера, показал непереносимость
pooled recorded-source bias на один held-out сеанс и сохранил truth-free причины
batch-fit отказов до исчерпания бюджета. Диагностический `bias=0` replay на
уже просмотренной evaluation **не выбирает** новый рабочий алгоритм. Первые
сохранённые данные содержали только pooled calibration моменты. По отдельному
разрешению пользователя повторно обработаны исключительно два исходных
calibration-сеанса: сохранены session-grain residuals, выполнены точное
разложение pooled covariance и направленная leave-one-session-out проверка.
Выявлены широкий режим ошибок в одной записи и сильная асимметрия переноса
bias/R между двумя сеансами; paired broadband контроль намного стабильнее.
Это завершённая **диагностика этих двух записей**, не статистическая
квалификация: следующий gate требует новых независимых исходных сеансов до
выбора operational calibration. S8 остаётся `In progress`; физика, фильтр,
`Qc` и пороги не менялись, полевые выводы не заявляются.

Одно мгновенное bearing-измерение одной станции не определяет дальность.
Temporal retarded-time модель при строгом constant velocity и известном
конечном `c` может формально получить дополнительную слабую информацию о
масштабе, но это не заменяет практически устойчивую multi-station 3D
localization. Несколько известных поз в общей ENU-системе обеспечивают
геометрическую устойчивость; эти задачи и их критерии не взаимозаменяемы.
