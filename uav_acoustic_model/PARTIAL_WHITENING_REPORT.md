# S8: нормированное частичное спектральное выравнивание

## Итог

Завершены **16/16 audio streams**, **8 tracker computations** и **0 verified β=1 reuses**: всего восемь заранее заданных парных сравнений. Технических остановок **не было**. Время внешних workers **2197.147 с** (36.62 мин) из 21600 с. Старые исследования/15 из 24/полёты не повторяли.

| Метод | Выбранное β | Harmonic L, baseline→selected | Relative reduction | Перспективен для калибровки? |
|---|---:|---:|---:|---|
| GCC | 0.5 | 51.184% → 15.154% | 70.394% | да |
| SRP | 0.5 | 44.783% → 18.958% | 57.668% | да |

Эффект рассматривается отдельно по методу. Выполнение frontend criterion не является квалификацией uncertainty или доказательством устойчивого real-UAV tracking. Фактическая harmonic доступность/ошибка приведены ниже. Ambiguity gate выключен.

**Ответ на вопрос эксперимента:** направления улучшились у обоих методов,
но устойчивое сопровождение harmonic-источника **не получено**.
GCC harmonic L 51,184%→15,154% (−70,394%), SRP 44,783%→18,958% (−57,668%).
Оба варианта удовлетворяют именно критерию перспективности для дальнейшей
калибровки. Baseline и selected harmonic availability 0%: tracking-preservation
при таком baseline не доказывает успешное сопровождение. Причины отказов
сохранены; 128→106 fits не заменяют подтверждённый track.
Broadband confirmation/availability не изменились, RMSE немного выросла,
P95 немного снизилась. Совместного изменения фильтра/R/gate/budget не было.

## Что изменено и математическая модель

$$C_{ij}(k)=X_i(k)X_j^*(k),\quad a_{ij}(k)=|C_{ij}(k)|/\max_{\ell\in K}|C_{ij}(\ell)|,$$

$$\Psi_{ij}^{(\beta)}(k)=\Psi_{ij}^{base}(k)a_{ij}(k)^{1-\beta},\quad\beta\in\{1,0.75,0.5\}.$$

K=200–10000 Гц. Base floor=1e-8 и прежние silence/energy invalid сохранены. β=1 явно единичные дополнительные веса, включая zero bins; нулевой максимум не создаёт произвольную задержку. На активных bins это `C/|C|^β/M^(1−β)`, в отличие от ненормированного PHAT-β без деления на M. Нормировка удаляет влияние амплитудного масштаба пары, дополнительной нормировки корреляции нет.

Прежние oriented `tau_ij=T_i−T_j`, physical lag bounds, fractional/Nyquist convention, GCC peak refinement, equal-pair WLS, cubic lag-SRP search сохранены. Opt-in находится в `analysis/`; принятые GCC/SRP и исторические SHA не изменены. Truth/class/range/future не входят в frontend.

## Протокол и происхождение

- База `430cad8af3278dfe35759a7d19c0a26048073e29`; implementation/protocol SHA `ccd44d706ed89c8598b976ff16c21b5b554a1a8f`; selection freeze commit `a056e99bcdbdf980b80aabb366776093ae561f2d` до evaluation.
- Development: существующая spatial запись; evaluation: radial. Каждый split: broadband / nonstationary harmonic × 200/700 м×две новые акустические реализации. Это известные движения, а не новые независимые полёты.
- Source/noise seeds и harmonic modulation параметры заранее разные между split и отличаются от прежних. Source banks общие по split/class/replicate для двух distances: зависимые cases, не iid frames.
- Каждый поток синтезирован один раз; все β/GCC/SRP получают те же непрерывные audio/noise samples. fs = 48 kHz, rectangular frame 1024, hop 512, stride 32; прежняя ENU station/microphone geometry.
- Retarded-time physical generator, 1/r,  +10 dB reference на 100 м, непрерывный независимый фон. В manifest отдельно сохранены historical processing snapshot и effective synthesis overrides.
- Прежние bias/R/Qc/history/construction/three-station confirmation и оптимизационные бюджеты. R калибрована для baseline; по evaluation новую R не оценивали.
- 16 audio максимум / 8 tracker максимум, 1200 с/audio, 600 с/replay, 21600 с total, один worker. Parent timeout, lock, attempts before start, SHA/completion полного schedule и сохранённые stdout/stderr.

## Development selection

$$L=(N_{invalid}+N_{valid,error>10^\circ})/N_{planned}.$$

Минимум harmonic L, равный вес stream/station cells, только all frames. Guardrails: broadband L+не более 1 п.п.; conditional P95+не более max(10%,0.1°); harmonic valid loss ≤ 1 п.п. При равенстве L (atol 1e-12) β ближайшее к 1. Baseline допустим. Все три кандидата/feasibility сохранены в `development_tradeoff.csv.gz`; выбор общий для станций/range/source class. Code и SHA всех development artifacts/source banks frozen отдельным manifest/commit. Evaluation grid не расширялся.

## Evaluation: all frames и tracker stride

Полные per-method/class/distance/station/replicate таблицы: `evaluation_metrics.csv.gz`, отдельно all/stride. Сохранены planned/invalid/valid и L; conditional median/P95 (linear quantile); >5/10/30° conditional и all-planned; invalid reasons; durations каждой серии >10° и invalid-or-large. Undefined conditional errors не заменены нулём.

Equal cell macro all-frame сравнение:

| Метод | Class | β | L | Valid fraction | Conditional median / P95, deg |
|---|---|---:|---:|---:|---:|
| GCC | broadband | 1 | 0.110% | 100.000% | 0.279 / 0.605 |
| GCC | broadband | 0.5 | 0.000% | 100.000% | 0.243 / 0.523 |
| GCC | nonstationary_harmonic | 1 | 51.184% | 100.000% | 15.602 / 72.428 |
| GCC | nonstationary_harmonic | 0.5 | 15.154% | 100.000% | 4.867 / 19.435 |
| SRP | broadband | 1 | 0.000% | 100.000% | 0.277 / 0.593 |
| SRP | broadband | 0.5 | 0.000% | 100.000% | 0.243 / 0.520 |
| SRP | nonstationary_harmonic | 1 | 44.783% | 100.000% | 10.289 / 56.437 |
| SRP | nonstationary_harmonic | 0.5 | 18.958% | 100.000% | 5.829 / 15.912 |

## Tracking: evaluation 200 м / replicate 0

Оба класса/GCC/SRP/baseline/selected, только three-station confirmation. Полное causal publication schedule и шесть журналов на computation. β=1 reuse сохраняет SHA baseline completion, bearing input и exact input event IDs; нового replay нет. Fit counts — по всем поколениям, не только final generation.

| Class | Метод | Вариант | β | Confirmation, s | Valid publications | Conditional RMSE / P95, m | Fits | Failure/status |
|---|---|---|---:|---:|---:|---:|---:|---|
| broadband | GCC | baseline | 1 | 1.055 | 94.545% | 2.105 / 3.689 | 2 | confirmed |
| broadband | GCC | selected | 0.5 | 1.055 | 94.545% | 2.139 / 3.622 | 2 | confirmed |
| nonstationary_harmonic | GCC | baseline | 1 | — | 0.000% | — / — | 128 | computational_budget_exceeded |
| nonstationary_harmonic | GCC | selected | 0.5 | — | 0.000% | — / — | 106 | tentative_initialization_unconfirmed |
| broadband | SRP | baseline | 1 | 1.055 | 94.545% | 2.101 / 3.680 | 2 | confirmed |
| broadband | SRP | selected | 0.5 | 1.055 | 94.545% | 2.134 / 3.611 | 2 | confirmed |
| nonstationary_harmonic | SRP | baseline | 1 | — | 0.000% | — / — | 128 | computational_budget_exceeded |
| nonstationary_harmonic | SRP | selected | 0.5 | — | 0.000% | — / — | 106 | tentative_initialization_unconfirmed |

`paired_tracking_review.csv.gz` дополнительно сохраняет first-confirmation error, accepted/rejected updates,  >10/50 м/all epochs и tracking-preservation criterion. `tracking_errors.png` оставляет отсутствующие оценки как NaN-разрывы. Нулевая availability не означает нулевую error.

## Механизм и ограничения

Для заранее заданных frames 0/32/last и S0–S2 каждого потока сохранены actual magnitude/active bins/additional weights/weighted spectra, six lag correlations и selected delay, SRP diagnostic 5° map/refined direction. На рисунках SRP используется отдельный диапазон цвета в исходных score units для каждой карты; корреляционные кривые показаны на общей шкале. Алгоритмической нормировки это не добавляет. Notebook показывает заранее выбранный harmonic evaluation 200 м r0 /frame 0/S0. Development — все 3 β, evaluation — baseline/frozen only.
Частичное выравнивание изменяет относительное влияние spectral bins. Сохранённые кривые/карты позволяют проверить изменение конкурирующих maxima. Без отдельного decomposition опыта нельзя причинно отделить подавление noise-dominated bins от изменений периодических sidelobes. Чистый тон показывает ambiguity; уникальный DOA не обещается.
Известные trajectories и synthetic harmonic transfer не являются независимыми real-UAV sessions. Только две реализации, общие banks/distances и overlapping frames: population claims и iid-frame confidence intervals не заявлены. У baseline R могут быть неверные uncertainty properties на новом frontend; выбранное β требует отдельной calibration прежде qualification. Ветер/отражения/новые gate/filter/adaptive R не добавлены.

## Проверки и просмотр

Математические/контрактные tests: exact β=1 baseline parity; sign/fractional delay; independent spectral sum с Nyquist; cubic SRP versus direct weighted spatial sum с fourth-order bound; gain invariance выше baseline floor; silence/zero spectrum; tonal ambiguity; seed/matrix separation; selection constraints/tie; shared-stream serialization/disjoint variant IDs; init/empty reports; lock/budget stop; запрет evaluation до committed freeze. Targeted  + actual autocrlf checkout: **27 passed**; новый notebook **7/7**
code cells, без ошибок; все четыре figures просмотрены. Полный pytest:
**645 passed** за 210,61 с; `pip check` без конфликтов, `git diff --check` чист.
Фактический Linux/Windows CI сообщается с итоговым SHA. Исследовательская матрица не повторялась.

```bash
cd uav_acoustic_model
python -m analysis.partial_whitening_study verify
python -m analysis.partial_whitening_study report
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/partial_whitening.ipynb
```

Полный frozen порядок: `PARTIAL_WHITENING_RUNBOOK.md`; accepted output не перезаписывать. S8 остаётся In progress, без нового обязательного подэтапа.
