# S8: нормированное частичное спектральное выравнивание

Замороженный протокол до новых аудиопотоков. База
`430cad8af3278dfe35759a7d19c0a26048073e29`, ветка
`feature/partial-whitening-localization`. Исторические данные и код не менять.

## Единственное изменение frontend

Для ориентированной пары `(i,j)`, задержка `T_i-T_j`:

$$C_{ij}(k)=X_i(k)X_j^*(k),\qquad
 a_{ij}(k)=\frac{|C_{ij}(k)|}{\max_{\ell\in\mathcal K}|C_{ij}(\ell)|},$$
$$\Psi_{ij}^{(\beta)}(k)=\Psi_{ij}^{\rm base}(k)a_{ij}(k)^{1-\beta},
 \qquad\beta\in\{1,0.75,0.5\}.$$

Рабочая полоса `K`: 200–10000 Гц, прежний relative spectral floor `1e-8`.
При β=1 дополнительные веса явно равны единице, включая нулевые bins;
нулевые baseline-компоненты остаются нулевыми. Нулевой максимум/тишина
возвращают прежний invalid. Для β<1 веса вне K равны нулю.
На активных bins это `C/|C|^β / M^(1-β)`, где M — максимум пары в K.
В отличие от ненормированного PHAT-β `C/|C|^β`, нормированная версия не
меняет относительные веса пар через их амплитуду; корреляции после IFFT
дополнительно не нормируются. Прежние bounds, знак, Nyquist weight,
параболическое уточнение GCC, cubic SRP, spatial search и equal pair weights.

Новый модуль явно opt-in в `analysis/`; принятые production files не меняются.
Физика, calibration bias/R, Qc/history, construction/confirmation/budgets
не меняются. Ambiguity gate выключен. R калибрована для baseline: это проверка
непосредственного frontend, не квалификация uncertainty нового метода.

## Математическая приёмка до матрицы

Exact β=1 parity, включая нули; positive/negative fractional delay;
независимая прямая спектральная сумма с односторонними DC/Nyquist weights;
SRP cubic versus direct weighted spatial sum с четвертым порядком error bound;
amplitude invariance выше прежнего RMS/spectral floor; silence/zero band invalid.
Чистый тон в изолированной узкой полосе показывает periodic competing peaks;
уникальное направление чистого тона во всех геометриях не требуется.

## Матрица и единственный синтез

Development: spatial_manoeuvre; evaluation: radial_approach_depart.
Каждый split: broadband / nonstationary harmonic × 200/700 м × 2 реализации.
8+8 новых потоков. Известные записи Gazebo, новые акустические реализации:
не новые независимые полёты. Source banks общие по split/class/replicate
для двух дальностей; зависимости объявлены, frames не iid trials.

Root `SeedSequence`: 20261004, experiment namespace 1101. Source координаты
`[root,1101,0,split,class,replicate]`, noise
`[root,1101,1,split,class,replicate,distance_index]`; 64-bit seeds в manifest.
Separation проверяется с известными seeds предшествующих JSON artifacts.
Harmonic split-параметры заранее разные, точные значения в manifest.

Синтез каждого потока ровно один раз. Все β/GCC/SRP читают те же непрерывные
channel/noise samples; overlapping frames — views, не новый synthesis.
fs=48000 Гц, frame=1024, hop=512, tracker stride=32. Прежний rectangular
frame (дополнительного окна frontend не вводится). Общая прежняя ENU geometry.
Retarded time, 1/r attenuation, fixed_reference_snr, +10 dB на 100 м,
reference_source_rms=1; source support с полной prehistory и FIR=129.
Historical processing snapshot + реальные effective synthesis overrides
сохранены отдельно, чтобы nominal metadata не подменяли физическую модель.

## Development selection; никаких evaluation labels

$$L=(N_{\rm invalid}+N_{\rm valid,error>10^\circ})/N_{\rm planned}.$$

Равный вес stream/station cells, отдельно по классу, только all frames для
selection. Conditional P95 — linear percentile внутри каждого cell, затем
equal cell macro; отсутствие valid — undefined, не ноль.
Для каждого метода одно β, common для class/range/station:

1. Минимизировать harmonic macro L.
2. Broadband macro L ≤ baseline +0.01.
3. Broadband macro conditional P95 ≤ baseline +max(0.1×baseline,0.1°).
4. Harmonic macro valid fraction ≥ baseline −0.01.
5. Tie (absolute 1e-12 для вычисленного macro L): β ближайшее к1.

Baseline β=1 всегда допустим; undefined P95 не квалифицирует новый β.
Сохранить все три кандидата. До evaluation: отдельный selection manifest
с SHA code/development/source banks, затем **отдельный Git commit**.
Runner проверяет byte-identical committed manifest до любого eval worker.
Evaluation сохраняет/сравнивает только baseline и выбранный β каждого метода.
Если β=1 выбран, это одна и та же ветвь, не новая реализация.

## Evaluation и tracking

Для каждого method/class/range/station/replicate: planned/invalid/valid,
L, valid fraction, median/P95, >5/10/30° conditional и all-planned fractions,
длительности серий >10° и invalid-or-large, отдельно all frames и stride.
Заранее выбранная диагностика: frame0, frame32, последний; S0–S2,
для всех development β и evaluation baseline/selected. Сохранить реальные
спектральные веса/magnitudes, lag curves, SRP 5° map и refined direction.
Изменение шумовых частот и periodic ambiguity — разные объяснения:
без отдельного доказательства механизма не делать causal claim.

Tracker: только evaluation 200м/replicate0, оба class, GCC/SRP,
baseline/selected, three-station confirmation. До8 replay, те же R/bias/Qc,
полные six journals, causal publication schedule, count fits по всем поколениям.
При selected β=1 проверить identity входа и reuse baseline completion SHA,
не выполнять новый tracker. Missing estimates не соединять линиями.

Перспективность для дальнейшей calibration — **отдельно по методу**:
≥25% relative harmonic L reduction; сохранение всех broadband/valid constraints;
нет потерь confirmation, availability loss ≤0.05, нет роста >50м/all epochs.
Это не автоматически устойчивый harmonic track: report отдельно показывает
сам факт confirmation/availability и failure reasons.

## Жёсткие вычислительные пределы и stop

16 audio, до8 tracker computations, один тяжёлый внешний worker.
1200с/audio, 600с/replay, суммарно21600с worker wall. Parent timeout,
exclusive lock, attempt зарегистрирован до запуска, logs/checkpoints/completion
и полные planned denominators. Ни один technical stop автоматически не
повторять; бюджеты не расширять. Старую 15/24 matrix и полёты не повторять.

Финал: targeted + full pytest, новый saved-artifact notebook, pip check,
diff check, Linux/Windows CI; S8 remains In progress. Отрицательный результат
допустим; grid не расширять по evaluation, новый gate/filter/adaptive R не добавлять.
