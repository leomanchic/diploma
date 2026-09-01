# Состояние проекта UAV Acoustic Model

Последнее обновление: 2026-09-01

## Текущий этап

Текущий этап: **S7C-A — Retarded-time Bearing Measurement Model** в составе
S7C. Цель подэтапа — математически проверить отображение constant-velocity
6D-состояния в асинхронные bearing-измерения трёх станций с физическим
запаздывающим временем и аналитическим Jacobian.

Статус: **Done** после corrective gate наблюдаемости одной станции.
В ветке
`feature/s7c-retarded-bearing-model` реализованы immutable 6D state,
аналитическое/численное emission time, retarded bearing prediction,
spherical tangent residual/Jacobian и локальная диагностика stacked 6D
observability. Локальный полный gate: **286 passed in 43.51s**, `pip check`
PASS, все 13 notebooks/84 code cells заново выполнены в чистых kernels и
проходят `nbformat` audit без error-output, невыполненных cells и пропущенных
cell IDs. Следующий подэтап — **S7C-B**.
EKF/UKF, particle filter, state update и tracking не реализованы.

### Журнал S7C-A

- 2026-09-01 — corrective gate завершён: S7C-A переведён в **Done**, S7C-B —
  **Next**. Полный pytest: `286 passed in 43.51s`; `pip check`: `No broken
  requirements found`; `git diff --check`: PASS. Все 13 notebooks заново
  выполнены отдельными свежими kernels: `array_comparison` 9.867 s,
  `bearing_uncertainty_validation` 55.561 s,
  `far_field_fractional_delay_validation` 85.827 s, `gcc_phat_monte_carlo`
  25.721 s, `gcc_phat_validation` 7.166 s, `gcc_statistical_validation`
  1609.543 s, `monte_carlo_crlb_validation` 102.484 s, `moving_source_3d`
  6.341 s, `moving_source_validation` 6.410 s,
  `multistation_static_validation` 85.160 s,
  `retarded_bearing_model_validation` 9.872 s,
  `sequential_doa_validation` 9.429 s и `srp_phat_validation` 7.355 s.
  Строгий аудит 13 notebooks/84 code cells: invalid nbformat `0`, error-output
  `0`, unexecuted nonempty code cells `0`, missing cell IDs `0`.
- 2026-09-01 — полный свежий notebook run перезаписал измеряемые runtime-поля,
  а не физические эталоны: `fractional_delay_benchmark.csv` изменил только
  четыре timing/speedup columns; `multistation_static_summary.csv` — только
  `mean_runtime_per_estimate_s` (max absolute change `6.132234373126266e-4
  s`); оба sequential CSV — только runtime/availability-derived latency
  columns (max runtime delta `5.1549999807321e-3 s`). В
  `gcc_doa_summary.csv` deterministic Monte Carlo-метрики, seeds и основная
  схема неизменны; только два noiseless diagnostic bias fields изменились на
  не более `5.2154667089186735e-9 deg` и `8.537721090694117e-7 deg` из-за
  floating-point решения. Эти изменения являются результатом обязательного
  свежего исполнения, а не подгонкой эталонов под tests.
- 2026-09-01 — one-station observability теперь разделена математически.
  Радиальный пример имеет rank `4`, `s_min=1.5312424663145318e-18`.
  Нерадиальный finite-`c=343 m/s` пример имеет формальный rank `6`,
  `s_min=1.6270620060085154e-8`, SI-scaled condition
  `2670001.46786563`. Независимый instantaneous Jacobian имеет rank `5` и
  scale-null residual `1.0408340855860843e-16`. При `c=343, 3430, 34300 m/s`
  слабые singular values равны `1.6270620059271063e-8`,
  `1.611393764476606e-9`, `1.6098350716521786e-10`, а spectral distances до
  instantaneous limit — `5.131610546002005e-3`, `5.13609396293407e-4`,
  `5.136613363563169e-5`. Формальный rank 6 не интерпретируется как
  практически устойчивое single-station ranging.
- 2026-09-01 — открыт corrective gate после независимого аудита вывода о
  single-station observability. Требуется разделить мгновенный bearing,
  радиальное движение, формальный full-rank нерaдиальный retarded-time случай
  и практическую устойчивость. До полного pytest и повторного исполнения всех
  13 notebooks S7C-A имеет статус **In review**; S7C-B не начинается.
- 2026-09-01 — реализованы независимый instantaneous Jacobian без передачи
  `np.inf` в production solver и три one-station сценария. Профильный gate:
  `19 passed in 4.20s`. Для нерaдиального движения finite-difference mismatch
  `5.663698676716677e-13`, то есть `3.480935978961724e-5` от smallest
  singular value `1.6270620060085154e-8`; SVD tolerance остаётся стандартным
  `max(shape)*eps*s_max` и не подбирается под ожидаемый rank.
- 2026-09-01 — усиленный one-station temporal gate выявил существующую
  cross-platform нестабильность двух near-zero angular checks на Ubuntu:
  `arccos(dot)` квантовал почти совпадающие directions в
  `1.2074182697257333e-6 deg`. Порог `1e-6 deg` не ослаблен; общая скалярная
  метрика заменена математически эквивалентной устойчивой формулой
  `atan2(||u×v||,u^T v)`, сохраняющей first-order resolution около нуля.
  Retarded-time формулы и сохранённые S7B CSV не менялись.
- 2026-09-01 — commit `6696014d1ee4dba63cb56babbeb97fbfb38499af`
  прошёл обе CI jobs: [Ubuntu Python 3.12](https://github.com/leomanchic/diploma/actions/runs/33476775610/job/99757566879)
  и [Windows Python 3.12](https://github.com/leomanchic/diploma/actions/runs/33476775610/job/99757566671).
  S7C-A переведён в **Done**, S7C-B — в **Next**.
- 2026-09-01 — локальная приёмка завершена. Pinned environment:
  `numpy=2.4.6`, `scipy=1.17.1`; полный pytest: `283 passed in 44.59s`;
  `pip check` и `git diff --check` PASS. Проверены 13 notebooks и 84 code
  cells: `nbformat` valid, error outputs `0`, unexecuted cells `0`.
  `notebooks/retarded_bearing_model_validation.ipynb` программно выполнен.
- 2026-09-01 — randomized FD audit использовал seed `20260901`, 1000 valid
  scenes: 950 с `|v|=0…60 m/s`, 50 stress scenes с `|v|=0.75c…0.9c`, range
  `10…500 m`. Максимумы: retarded-equation residual `1.1102230246251565e-15
  s`, analytic/numeric emission-time difference `4.884981308350689e-15 s`,
  `dt_e/dx` absolute mismatch `6.915884531721872e-12`, local-direction
  Jacobian mismatch `4.486511717693986e-10`, tangent-residual Jacobian
  mismatch `2.725225094202255e-9`, relative mismatch по компонентам с
  `|J_numeric|>1e-7` — `3.861468973416781e-6`. Допуски `2e-8` absolute и
  `2e-5` relative не ослаблялись после запуска.
- 2026-09-01 — observability examples: четыре temporal bearings одной станции
  для радиального constant-velocity движения имеют rank `4` и две численно
  нулевые singular directions (smallest singular value
  `1.5312424663145318e-18`); три станции и три reception epochs имеют rank `6`, condition
  `9.577896723107909`, smallest singular value `0.010160665945304471`.
  Почти коллинеарные станции и окно `0.002 s` сохраняют numerical rank `6`,
  но ухудшают condition до `4316.373666947092`, smallest singular value до
  `8.72927229583684e-6`. Это local parameterization diagnostic заданного
  candidate state, не estimator, covariance или CRLB.
- 2026-09-01 — S7B CSV подтверждены побитово неизменными: geometry SHA-256
  `190F5470FAC0F4C46A478C5EB2EBEB8790C201A256E1CB620CA0D423BB24FF05`,
  static SHA-256
  `7B9C3E7722689488F0F37C5C65DE9EAB760653F00BC1C2ED46596CB76A2013B3`.
- 2026-09-01 — статическая и динамическая модели переведены на одну общую
  производную spherical tangent residual по predicted direction; targeted
  regression S7B остался зелёным: `28 passed`.
- 2026-09-01 — уравнение использует только синхронизированный
  `reception_center_timestamp_s`:
  `t_r=t_e+||q(t_e)-p_k||/c`. `available_timestamp_s` участвует только в
  причинном отборе событий, а `StationPose.clock_*` не применяется повторно.
- 2026-09-01 — добавлены проверки analytic/numeric emission time, статического
  предела, причинности, rigid/time/rebase invariance, pole/antipode и
  finite-difference Jacobian. Targeted результат: `22 passed`.

### Формулы и допущения S7C-A

- `q(t_e)=q0+v(t_e-t0)`,
  `t_r=t_e+||q(t_e)-p_k||/c`, `|v|<c`, `t_e<t_r`, range `>0`.
- Для `Delta=t_e-t0`, `u=(q(t_e)-p_k)/range` и
  `gamma=1+u^T v/c`:
  `dt_e/dx=-[u^T,Delta u^T]/(c gamma)`,
  `dq_e/dx=[I,Delta I]+v(dt_e/dx)`,
  `du_world/dx=(I-uu^T)dq_e/dx/range`,
  `du_local/dx=Q_k^T du_world/dx`.
- Residual:
  `tangent_residual(u_pred_local,u_measured_local)-mu_cal` в радианах дуги.
  Производная spherical residual общая со статическим S7B. Pole и antipode
  отклоняются явно; raw azimuth/elevation subtraction не применяется.
- Timestamps считаются уже синхронизированными. `reception_center_timestamp_s`
  входит в propagation equation; `available_timestamp_s` только ограничивает
  причинную доступность; `StationPose.clock_offset_s/clock_drift_s_per_s` не
  применяются второй раз.
- Среда однородна, неподвижна, `c` постоянно. Не моделируются process noise,
  ускорение внутри constant-velocity state, clock uncertainty, ветер,
  отражения, signal-level fusion и tracking.

### Команды проверки S7C-A

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "from validation.retarded_bearing_validation import run_retarded_bearing_validation; run_retarded_bearing_validation()"
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 notebooks\retarded_bearing_model_validation.ipynb
git diff --check origin/main...HEAD
```

### Журнал S7B

- 2026-08-31 — открыт неблокирующий housekeeping после S7B: GitHub Actions
  переводится на `actions/checkout@v7` и `actions/setup-python@v7`, checkout
  получает `fetch-depth: 0`, а whitespace gate проверяет branch diff от
  merge-base через `git diff --check origin/main...HEAD`. Узкий SciPy SR1
  `RuntimeWarning: overflow encountered in scalar divide` локализован во
  внутреннем reciprocal почти нулевого update denominator и точечно подавлен
  только для модуля `scipy.optimize._hessian_update_strategy`; final
  constraint/KKT/observability/forward-ray gates не меняются. Математика,
  допуски и CSV не изменялись. Локально: полный pytest **259 passed in
  40.25s**, `pip check` PASS, `git diff --check` PASS, `results/` без diff.
  Pinned Linux `numpy=2.4.6`/`scipy=1.17.1` повторил 1000-scene randomized
  gate: **1 passed in 18.69s**, прежний SR1 warning отсутствует. Из-за
  housekeeping-объёма notebooks не перезаписывались и CSV не пересчитывались;
  новый merge-base whitespace gate дополнительно обнаружил и удалил только
  trailing blank EOF lines в двух существующих multistation tests.
  Cross-platform CI matrix остаётся финальным acceptance gate. S7B остаётся
  `Done`, S7C не начат.
- 2026-08-31 — повторный diagnostic commit `6defb6b` прошёл обе CI jobs:
  [Ubuntu Python 3.12](https://github.com/leomanchic/diploma/actions/runs/33384439147/job/99463763226)
  и [Windows Python 3.12](https://github.com/leomanchic/diploma/actions/runs/33384439147/job/99463763057).
  Одиночный Ubuntu failure commit `147a84c` не воспроизвёлся ни в read-only
  Linux Docker full suite, ни в 10 последовательных 1000-scene gates, ни в
  повторном CI. Workflow сохраняет публичную failure-аннотацию для будущих
  отказов. S7B переведён в **Done**, S7C — в **Next**, но S7C не начат.
- 2026-08-31 — финальный status commit `147a84c` временно вернул S7B в
  **In review**: [Windows Python 3.12](https://github.com/leomanchic/diploma/actions/runs/33382997494/job/99459311852)
  прошёл, но [Ubuntu Python 3.12](https://github.com/leomanchic/diploma/actions/runs/33382997494/job/99459312050)
  упал на pytest. Предыдущий кодовый commit `aac36f1` прошёл обе платформы,
  поэтому выполнено повторное pinned Linux воспроизведение в read-only Docker
  mount: полный suite дал **259 passed in 44.25s**. Дополнительно 10 отдельных
  Linux pytest-процессов выполнили один и тот же 1000-scene randomized gate:
  **10/10 PASS**, суммарно 10000 сцен и false-invalid `0`. В workflow добавлена
  публичная failure-аннотация последних 120 строк pytest для точной диагностики
  любого повторного CI failure; S7C не начат.
- 2026-08-31 — cross-platform gate commit `aac36f1` завершён зелёной GitHub
  Actions matrix: [Ubuntu Python 3.12](https://github.com/leomanchic/diploma/actions/runs/33382562395/job/99457989049)
  и [Windows Python 3.12](https://github.com/leomanchic/diploma/actions/runs/33382562395/job/99457989250)
  установили зависимости строго из `requirements.txt` и успешно выполнили
  pytest, `pip check`, `git diff --check`. S7B переведён в **Done**, S7C — в
  **Next**, но S7C не начат.
- 2026-08-31 — реализован cross-platform KKT corrective gate, CI ещё не
  запускался, поэтому S7B остаётся **In review**. Теперь отдельно сохраняются
  `raw_projected_gradient_norm=||Z^T grad J||`, dimensionless
  `scaled_projected_kkt_residual=0.5*sqrt(g_Z^T solve(I_Z,g_Z))`,
  `optimizer_success` и `optimizer_message`; default scaled tolerance — `1e-6`.
  `optimizer_success=False` больше не является самостоятельной причиной
  invalid при конечной позиции, выполненных exact constraints, full local
  observability, forward rays и scaled-KKT PASS. Локальный pinned audit
  (`numpy=2.4.6`, `scipy=1.17.1`, seed `20260831`) на **1000** совместимых
  rank-1 сценах: false-invalid `0`, preliminary exceedances `13`, max final
  constraint `2.458754593756406e-16 rad`, max raw projected gradient
  `7.746729289788835e-9`, max scaled KKT `5.233099289902724e-9`. Профильные
  tests: **20 passed**; полный pytest: **259 passed**. Добавлены регрессии
  trials `771/793`, diagnostic-only optimizer-exit test, truly-suboptimal
  `xtol` rejection и rigid/scale/permutation invariance checks. `pip check`:
  `No broken requirements found`; `git diff --check`: PASS. Все **12/12**
  notebooks выполнены в изолированной копии: `nbformat` valid, error-output
  `0`, unrun nonempty code cells `0`, missing cell IDs `0`. Оба полноранговых
  multistation CSV побитово неизменны относительно `9b2e749`. Добавлена GitHub
  Actions matrix Ubuntu/Windows Python 3.12 со строгой установкой из
  `requirements.txt`; до её зелёного результата S7B остаётся **In review**.
  S7C не начат.
- 2026-08-31 — открыт cross-platform corrective gate к commit `9b2e749`.
  Pinned Linux full pytest дал `1 failed, 254 passed`: совместимые trials
  `771/793` имеют final constraints `4.9117431969933596e-17` и
  `7.253739856973443e-18 rad`, но отвергаются из-за platform-dependent
  optimizer exit/raw-gradient threshold. До dimensionless Newton-correction
  KKT, deterministic regressions, 1000-scene gate и зелёной Ubuntu/Windows
  GitHub Actions matrix S7B имеет статус **In review**; S7C не начинается.
- 2026-08-31 — numerical-robustness приёмка S7B завершена. Полный pytest:
  **255 passed in 44.01s**; профильный gate: **23 passed in 22.15s**;
  `pip check`: `No broken requirements found`; `git diff --check`: PASS.
  `multistation_static_validation.ipynb` программно выполнен без ошибок;
  аудит всех **12/12 notebooks**: `nbformat` valid, error-output `0`, unrun
  nonempty code cells `0`, missing cell IDs `0`. Только этот notebook
  импортирует изменённый triangulator. Полноранговый
  `multistation_static_summary.csv` после regeneration совпал с tip
  `5309cf0` по всем 68 non-runtime полям точно (`max abs/rel=0/0`), после
  чего runtime-only rewrite убран; committed CSV остаётся побитово неизменным.

  Итоговый randomized audit: 1000 совместимых двухстанционных rank-1 сцен,
  seed `20260831`, false-invalid **0**, preliminary residual выше `1e-10`
  встречался 13 раз, max final constraint residual
  `2.458754593756406e-16 rad`, max covariance-scaled projected-KKT
  `7.746729289788835e-9` при acceptance `1e-8`. Targeted regression:
  preliminary `1.5262609922398919e-10 rad`, final `5.040341608778079e-17
  rad`, KKT `4.476730845207426e-14`, valid. Заведомо несовместимый случай:
  preliminary/final residual `0.0361929475478882 rad`,
  `incompatible_exact_constraints`, all-NaN covariance. S7B=`Done`,
  S7C=`Planned (Next)`; tracking не добавлен.
- 2026-08-31 — исправлен preliminary-feasibility control flow и добавлен
  scaled projected-KKT gate `||Z^T grad J||`. На воспроизведённой совместимой
  rank-1 сцене preliminary residual `1.5262609922398919e-10 rad` больше
  tolerance, но constrained SR1 solve даёт final residual `5.04e-17 rad` и
  KKT `4.48e-14`, поэтому результат valid. Искусственный optimizer
  `success=True` с сообщением `xtol`, feasible constraints и большим KKT
  корректно возвращает `projected_kkt_not_satisfied`. Randomized gate на
  **1000** совместимых двухстанционных rank-1 сценах, seed `20260831`:
  false-invalid `0`, preliminary exceedances `13`, max preliminary residual
  `2.202425200253308e-10 rad`, max final residual
  `2.458754593756406e-16 rad`, max scaled projected-KKT
  `7.746729289788835e-9`. Заведомо несовместимый gate и общая приёмка ещё
  выполняются; S7B остаётся **In review**, S7C не начинается. Профильный
  pytest: **23 passed in 22.15s**.
- 2026-08-31 — открыт corrective gate к tip `5309cf0`: preliminary
  `least_squares` ошибочно мог окончательно объявить совместимые exact
  constraints несовместимыми при residual `1.5381543336219373e-10 rad`,
  лишь немного превышающем tolerance `1e-10 rad`, хотя последующий
  constrained solve уменьшает residual до `1.2497494302150491e-17 rad`.
  До исправления control flow, scaled projected-KKT diagnostic, targeted и
  1000-scene randomized gates S7B имеет статус **In review**; S7C не
  начинается.
- 2026-08-31 — корректирующая приёмка S7B завершена. Финальный полный pytest:
  **252 passed in 18.42s**; `pip check`: `No broken requirements found`;
  `git diff --check`: PASS. Все **12/12 notebooks** выполнены через
  `nbconvert` на отдельных executed-копиях без перезаписи пользовательского
  `moving_source_3d.ipynb`; единый аудит: `nbformat` valid, error-output `0`,
  unrun nonempty code cells `0`, missing cell IDs `0`. Финальный
  `multistation_static_validation.ipynb` дополнительно выполнен in-place:
  12 cells/9 code, те же четыре нулевых audit-counts.

  Численная constrained-WLS проверка: три exact nonparallel bearings при
  `R=0` дали position error `8.93e-15 m`, constraint residual `0 rad` и
  нулевую covariance; dimensions `(positive, exact, constraint-rank,
  free, local-rank)=(0,6,3,0,3)`. Rank-1 `R` дал max exact residual
  `4.85e-17 rad` и max `A C A^T=3.40e-21`. Несовместимые exact bearings
  явно вернули invalid с `incompatible_exact_constraints` и residual
  `3.6193e-2 rad`. Расстояние finite-variance решения до constrained при
  `lambda=1e-4,1e-6,1e-8,1e-10,1e-12 rad^2` монотонно уменьшилось:
  `7.5187e-2, 1.0957e-3, 1.1011e-5, 1.1011e-7, 1.1011e-9 m`.
  Station-permutation mismatch: `0 m / 3.47e-18 m²`; global rigid-transform
  mismatch: `7.11e-15 m / 7.63e-17 m²`.

  CSV имеют **126×69** и **9×24**. Полноранговая summary после перехода от
  эквивалентного symmetric whitening к буквальному
  `Lambda+^(-1/2) U+^T r` не побитово совпала с `4ff91c4`, но categorical,
  coverage и failure fields не изменились. Среди scalar metrics max absolute
  difference `2.031e-3` относится к condition `1.670e5` (relative
  `1.216e-8`), max relative difference `3.976e-7` — к covariance-ratio
  diagnostic; position RMSE max absolute/relative differences равны
  `1.016e-6 m / 9.075e-9`. Wall-clock runtime исключён из сравнения. Exact
  constraint acceptance использует явный numerical solver
  tolerance `1e-10 rad`, но covariance eigenvalues не заменяются epsilon и
  nullspace не игнорируется. Запрошенные критерии не ослаблялись. S7B=`Done`,
  S7C=`Planned (Next)`; tracking в этом commit не добавлен.
- 2026-08-31 — полный static study повторён после constrained-WLS fix:
  `multistation_static_summary.csv` имеет **126×69**, geometry CSV —
  **9×24**. Первое сравнение при эквивалентном symmetric whitener было
  побитово идентично commit `4ff91c4`; финальная буквальная spectral-coordinate
  реализация изменила только floating-point младшие разряды с maxima,
  перечисленными в финальном gate выше, без categorical/coverage/failure
  изменений. Исключено `mean_runtime_per_estimate_s`, потому что это новое
  wall-clock измерение.
  Новые geometry-поля: `world_frame_definition=ENU_x_east_y_north_z_up_m` и
  `station_position_reference=microphone_array_centroid`. Этап остаётся
  **In review** до notebook и общих regression gates.
- 2026-08-31 — constrained WLS реализован и прошёл профильный gate:
  **20 passed in 3.45s**. Для каждого `R` отдельно сохраняются `U+`,
  положительные `Lambda+` и `U0`; objective использует whitened residual
  `Lambda+^(-1/2) U+^T r`, а `U0^T r=0` передаётся оптимизатору как equality
  constraint. Несовместимые constraints дают
  `failure_reason=incompatible_exact_constraints` и `NaN` position
  covariance. Diagnostics разделяют finite-variance information,
  constraint Jacobian/rank и reduced information на допустимом manifold.
  Geometry record расширен полями `world_frame_definition` и
  `station_position_reference`, поэтому после регенерации CSV будет иметь
  фактическую, а не заявленную, ширину 24. Этап остаётся **In review** до
  full pytest/notebook/pip/diff gates.
- 2026-08-31 — открыт корректирующий acceptance gate к commit `4ff91c4`.
  Обнаружена математическая ошибка: прежнее применение `R^+` трактовало
  covariance nullspace как нулевой вес, тогда как вырожденная Gaussian-модель
  требует точных ограничений `U0^T r(q)=0`. До constrained WLS, новых
  deterministic/limit tests, пересчёта CSV/notebook и полной повторной
  приёмки S7B имеет статус **In review**, S7C не реализуется. Фактический
  geometry CSV commit `4ff91c4` содержит **9×22**, несмотря на ошибочную
  запись `9×24`; будут добавлены и документированы два поля мировой системы,
  после чего схема действительно станет 24-column.
- 2026-08-31 — предыдущая приёмка, зафиксированная в commit `4ff91c4`
  и теперь superseded корректирующим gate выше. После исправления временного
  контракта тогда полный pytest дал **247 passed in 20.33s**; `pip check`:
  `No broken requirements found`. Все **12/12 committed notebooks** выполнены
  через `nbconvert`; единый аудит дал `nbformat` valid, error-output `0`,
  unrun nonempty code cells `0`, missing cell IDs `0` для каждого. Размеры
  В том отчёте размеры CSV были ошибочно записаны как `126×69` и `9×24`:
  фактическая geometry-таблица commit `4ff91c4` имела **9×22**. Все 13 прежних
  контролируемых CSV также сохранили ожидаемое число строк. `git diff
  --check` проходил. Эта приёмка не учитывала exact nullspace constraints и
  потому не является финальной; dynamic 3D tracking не был реализован.
- 2026-08-31 — исправлена промежуточная ошибка временной семантики нового
  API: первоначально static triangulator по умолчанию требовал одинаковые
  reception timestamps. Это неверно для разнесённых станций, потому что один
  source state/emission может иметь разные propagation delays. Теперь
  обязательная association задаётся общими `sequence_id/frame_index`, разные
  reception timestamps разрешены, а их равенство доступно только как явная
  опциональная проверка. Профильный gate **11 passed in 7.60s**, полный
  **247 passed**. Никакого true emission time в online measurement не
  добавлено.
- 2026-08-31 — количественная invariance/Jacobian проверка после финальной
  коррекции: ideal position max abs error `7.11e-15 m`, station permutation
  position/covariance mismatch `5.33e-15 m / 5.20e-18 m²`, global
  rotation+translation position/covariance mismatch `1.42e-14 m /
  2.60e-18 m²`, analytic/numeric Jacobian max abs mismatch `2.63e-11`,
  coordinate-scale covariance relative Frobenius mismatch `9.78e-17`.
- 2026-08-30 — полный статический study выполнен и затем воспроизводимо
  повторён внутри нового notebook. Использованы 9 физических конфигураций,
  **128 calibration + 256 evaluation** независимых direct-bearing
  realizations на конфигурацию, то есть 3456 уникальных role/config
  realizations; семь matched scenarios дают **24192 spherical WLS estimates**
  на common random numbers. `multistation_static_summary.csv` содержит
  **126 строк × 69 полей** (63/63 calibration/evaluation),
  `multistation_geometry_summary.csv` — **9 × 24**. Девять calibration и
  девять evaluation seeds уникальны по роли, overlap `0`. GCC/SRP, truth и
  true range online не использовались.
- 2026-08-30 — в ideal-known-pose three-station evaluation position RMSE
  находится в `0.2964…38.5498 m`, P95 `0.5040…64.8676 m`, P99
  `0.6390…134.4900 m`; диапазон отражает dimensionless geometry sweep, а не
  одну дальность. Worst-conditioned geometry — `near_collinear_far`:
  median information condition `1692.90`, RMSE `38.55 m`, P95 `64.87 m`.
  Лучший `equilateral_near` даёт condition `5.37`, RMSE `0.296 m`, P95
  `0.504 m`. На данной sampled grid ideal-three RMSE ниже обоих выбранных
  two-station subsets во всех 9 конфигурациях, но это не универсальное
  утверждение: результат зависит от intersection angle и measurement quality.
- 2026-08-30 — local Gaussian covariance benchmark численно проверен, а не
  принят по отсутствию исключений. Для 9 ideal-three evaluation групп
  `trace(C_emp)/trace(mean C_pred)=0.810…1.195`; nominal 95% ellipsoid
  coverage `0.840…0.965`. Ухудшение coverage в elongated/nearly-collinear
  случаях показывает предел локальной Gaussian linearization. Один
  `5°/-2°` erroneous bearing не отбрасывается скрыто: median scenario RMSE
  `19.66 m`, worst RMSE `815.59 m`, maximum failure fraction `0.0547`.
  Position/orientation/covariance mismatch также сохранены отдельными
  сценариями. P99 при 256 evaluation realizations — sampled diagnostic,
  не operational tail guarantee.
- 2026-08-30 — `multistation_static_validation.ipynb` успешно выполнен за
  ~74 s: 12/12 cells имеют ID, `nbformat` valid, error-output `0`, unrun
  nonempty code `0`. Notebook повторяет CSV study, строит RMSE/intersection-
  angle и covariance/coverage diagnostics, а также good/poor ENU scenes с
  validation-only truth marker. Обновлены `README.md`, `AGENTS.md` и
  `ROADMAP.md`; dynamic 3D tracking, retarded-time fusion, synchronization,
  ветер, отражения, SRP-Harmonics и hardware I/O всё ещё не реализованы.
  Полный pytest после документации и аудит остальных committed notebooks ещё
  не выполнены, поэтому S7B пока не объявлен завершённым.
- 2026-08-30 — реализован bearing-level Monte Carlo
  `validation/multistation_static_study.py`. Девять физических конфигураций
  охватывают equilateral/elongated/nearly-collinear station layouts,
  baseline `10/20/40/50 m`, несколько `range/baseline` и
  `altitude/baseline`, aligned/varied local orientations. Семь matched
  сценариев на одних tangent-noise draws сравнивают ideal 3/2 stations,
  отказ одной из трёх, position/orientation calibration mismatch, bearing
  covariance mismatch и один `5°/-2°` erroneous bearing. Шум генерируется
  непосредственно в сферической tangent plane с известными `mu,R`, не через
  GCC/SRP. Calibration/evaluation bearing-noise seeds раздельны; `mu_cal,R`
  fit только по calibration. Число 2D residual components явно не называется
  independent trials. Smoke gate с `12/16` realizations: все coverage `1.0`,
  outlier RMSE больше ideal-three более чем втрое; профильный набор study +
  deterministic tests **15 passed in 3.33s**.
- 2026-08-30 — создан `visualization/multistation_scene.py`: ENU stations и
  их локальные оси, bearing rays без фиктивной station range, closest-point
  residuals, estimated 3D point и local 95% Gaussian covariance ellipsoid.
  True position разрешена только с явным `validation_mode=True`. Создан
  12-cell `multistation_static_validation.ipynb`; `nbformat` valid до
  исполнения. Визуализационный/study gate **4 passed in 3.88s**. Для full
  study зафиксированы **128 calibration + 256 evaluation независимых
  bearing-level realizations на физическую конфигурацию**; P99 по 256
  evaluation realizations будет только sampled diagnostic. Full study,
  notebook execution и общая приёмка ещё не выполнены.
- 2026-08-30 — в исходной реализации commit `4ff91c4` был создан
  `estimators/bearing_triangulation.py`: прозрачный
  closest-rays baseline
  `q=pinv(sum w(I-dd^T)) sum w(I-dd^T)p` и основной spherical weighted
  nonlinear least squares с residual
  `tangent_residual(u_pred_local,u_hat)-mu_cal`. Initial point не использует
  truth; ground/`z>=0` constraint отсутствует; source обязан лежать впереди
  каждого принятого луча. Результат сохраняет residuals, ranges, raw
  Jacobian, position information/eigen/rank/condition, ненаблюдаемые мировые
  направления и local Gaussian covariance benchmark только при полном ранге.
  Историческая реализация обрабатывала singular/zero `R` только спектральной
  pseudoinverse без epsilon и тем самым ошибочно отбрасывала zero-variance
  constraints; это поведение заменено constrained WLS в корректирующем gate
  2026-08-31 выше.
- 2026-08-30 — выведен и реализован аналитический `dr/dq`; его log-map часть
  использует `a=u^T y`, `v=y-au`, `theta=atan2(||v||,a)` и производную
  `(theta/||v||)v`, включая connection terms меняющегося az/el tangent basis.
  Центральные разности подтверждают Якобиан также при переходе azimuth через
  `-pi/pi`. Обнаружена численная потеря точности прежнего малого log-map через
  `arccos(dot)`; она исправлена на устойчивый `atan2(||projection||,dot)`.
  Добавлены deterministic tests для 2/3 станций, разных локальных ориентаций,
  global translation/rotation, station permutation, backward bearing,
  collinear/nearly-parallel геометрии, масштаба covariance и `R,mu_cal`.
  Старый `test_cycle_projection` больше не использует хрупкий фиксированный
  `2e-19`: roundoff-допуск равен machine epsilon, умноженному на норму данных
  и явный малый operation factor. Профильный gate **18 passed in 0.64s**;
  полный регрессионный gate **243 passed in 17.31s**. Monte Carlo/CSV/notebook
  ещё не выполнены, этап остаётся в работе.
- 2026-08-30 — реализованы `model/station.py` и `model/measurements.py`.
  `StationPose` задаёт правую ENU-систему (`x=East,y=North,z=Up`), проверяет
  `Q^TQ=I`, `det(Q)=+1`, хранит решётку относительно centroid и вычисляет
  `r_world=p+Q r_local`; numpy-массивы защищены от изменения. Общий
  `BearingMeasurement` содержит только online-доступные timestamps, локальный
  единичный bearing, tangent `R`/`mu_cal`, estimator/quality/valid metadata и
  намеренно не имеет truth/error/emission-time полей. Singular PSD covariance
  разрешена без epsilon; invalid record хранит all-NaN direction/covariance,
  а не фиктивное измерение. Проверены local/world round-trip, глобальные
  перенос/поворот, permutation микрофонов, неверные rotations, centroid и
  неизменяемость: профильный gate **13 passed in 0.47s**. Триангуляция и
  observability ещё не реализованы, этап остаётся в работе.
- 2026-08-30 — новое ТЗ прочитано полностью вместе с `AGENTS.md`,
  `PROJECT_STATUS.md`, `README.md` и `ROADMAP.md`. Рабочее дерево было чистым
  на точном commit `a48d000e30bc446c1df0995d90f56bde4a9458bc`; поскольку этот
  commit не слит в `main`, создана отдельная ветка
  `feature/multistation-foundation` непосредственно от него. Зафиксирована
  граница этапа: сначала bearing-level статическая триангуляция и
  наблюдаемость, без смешивания с ошибками GCC/SRP и без dynamic tracking.
  Реализация, deterministic/smoke/full gates, CSV и notebook ещё не
  выполнены, поэтому этап не объявляется завершённым.

### Формулы и ограничения S7B

Для station pose `p_k,Q_k` и локального измерения `u_hat_k` мировой луч и
предсказание имеют вид

`d_k=Q_k u_hat_k`,
`u_pred,k(q)=Q_k^T(q-p_k)/||q-p_k||`.

Основной residual и спектральное разложение covariance:

`r_k(q)=tangent_residual(u_pred,k(q),u_hat_k)-mu_cal,k`,
`R_k=U_{+,k} Lambda_{+,k} U_{+,k}^T + U_{0,k} 0 U_{0,k}^T`.

Constrained WLS решает

`min_q sum ||Lambda_{+,k}^(-1/2) U_{+,k}^T r_k(q)||^2`

при точных equality constraints

`U_{0,k}^T r_k(q)=0`.

Finite-variance information равна
`I_+=sum H_k^T U_{+,k} Lambda_{+,k}^-1 U_{+,k}^T H_k`. Пусть `A` —
Якобиан всех точных constraints, а столбцы `Z` образуют `null(A)`. Тогда
combined local observability rank равен
`rank(A)+rank(Z^T I_+ Z)`, а при полном combined rank локальная covariance
на допустимом manifold равна `C_q=Z(Z^T I_+ Z)^-1 Z^T`. Если `Z` пуст,
точные constraints локально фиксируют все три координаты и `C_q=0`. Если
constraints несовместимы либо combined rank меньше трёх, result invalid и
covariance содержит `NaN`; сохраняются constraint/reduced-information ranks,
eigenvalues и ненаблюдаемые мировые направления. Это не точная CRLB
signal-level модели. Closest-rays baseline использует pseudoinverse; обычное
обращение вырожденной матрицы, скрытое `z>=0`, ground constraint и arbitrary
epsilon отсутствуют.

Проверено: ENU/local-world transforms, proper rotations, centroid/permutation,
2/3 stations, forward rays, wrap `-pi/pi`, analytic Jacobian, global
translation/rotation, station permutation, масштабирование, singular/zero
angular covariance, collinear/nearly-parallel geometry, calibration-only
`R,mu_cal`, disjoint seeds и empirical-vs-predicted covariance.

Не реализовано: dynamic/asynchronous 3D tracking, retarded-time fusion,
clock synchronization/transport, signal-level multi-station GCC/SRP fusion,
ветер, температура, отражения, SRP-Harmonics, hardware I/O и field data.
Статический estimator принимает только bearings, заранее ассоциированные с
одним source state (`sequence_id/frame_index`); он не должен применяться к
движущемуся источнику с несогласованными states.

Ослабления и статистические ограничения: быстрый smoke использует `12/16`,
полный study — `128/256` independent calibration/evaluation realizations на
физическую конфигурацию; P99 по 256 samples является только sampled
diagnostic. Семь scenarios используют common random numbers, поэтому 24192
estimator calls не являются 24192 уникальными noise realizations. Допуск
cycle projection масштабируется как `64*eps*||data||`; он заменил хрупкое
фиксированное `2e-19`, а не был подобран под один остаток. Финальных
проваленных критериев нет. Промежуточно исправлены малый log-map
`arccos→atan2` и неверное равенство reception timestamps.

### Журнал S7A

- 2026-08-30 — корректирующий gate bias-centered NIS полностью принят перед
  bearing tracking. CSV и notebook перегенерированы: **216 covariance / 1368
  quality rows**, 36/36 calibration/evaluation физических групп. Во всех
  216 строках сохранён ненулевой `mu_cal` (`calibration_bias_norm_deg`
  `0.01052…0.83711°`), centered/raw P95 различаются, bias-correction flag
  истинный, centering source равен только `calibration`, evaluation mean use
  равен `0`, а `chi_square_comparison_statistic=centered_nis`. После коррекции
  centered NIS P95 / `chi-square(2)` P95 на 108 evaluation группах имеет
  диапазон `0.10165…2.33270`, median `1.15129`; `88/108` групп лежат в
  диагностическом диапазоне `0.5…1.5`. Median raw normalized squared error
  P95 / `chi-square(2)` P95 равна `1.15347`, но raw statistic с chi-square не
  сравнивается. Реальный set-аудит всех 108/108 sequence/source/noise seeds
  дал overlap `0/0/0`; декларативных замен этому контролю нет.
  Финальная проверка после выполнения всех notebook: **226 passed in 17.48s**,
  `pip check` — `No broken requirements found`; все **11/11 notebook** имеют
  `nbformat` valid, error-output `0`, unrun code `0`, missing ID `0`.
  Ослаблений допусков и проваленных критериев нет; прежнее ограничение на
  tail confidence intervals при трёх независимых sequence/group сохраняется.
  `ROADMAP.md`: S7A=`Done`, S7B=`Next`. Tracking/EKF/UKF не реализованы.
  Итог: **calibrated bearing measurement benchmark, not tracking and not a
  signal-level CRLB**.
- 2026-08-30 — начат и пройден математический gate bias-centered NIS перед
  S7B. Матрица `R` и `mu_cal` по-прежнему оцениваются только на calibration
  residuals. Для обоих split теперь
  `NIS_centered=(r-mu_cal)^T R^+ (r-mu_cal)`; evaluation mean нигде не
  участвует в центрировании. Прежняя величина `r^T R^+ r` сохранена отдельно
  как `raw_normalized_squared_error`, а `chi-square(2)` сравнивается только с
  centered NIS. Добавлены calibration bias/centering source/bias-correction
  fields. Sequence/source/noise disjointness теперь вычисляется реальным
  пересечением множеств; любой ненулевой overlap останавливает study, CSV
  хранит четыре overlap counts и audit flag. Профильный результат:
  **17 passed in 5.55s**. Полные CSV/notebook ещё не перегенерированы, поэтому
  корректирующий gate пока не объявлен завершённым; tracking/EKF/UKF не
  реализуются.
- 2026-08-29 — финальная приёмка S7A завершена. После полного повторного
  выполнения notebook весь pytest: **223 passed in 17.25s**; `pip check`:
  `No broken requirements found`. Все **11/11 notebook** выполнены через
  `nbconvert`, проходят `nbformat.validate`, имеют error-output `0`, unrun
  непустых code cells `0` и missing cell ID `0`. Итоговый CSV-аудит:
  covariance/quality `216/1368` строк, обязательных полей не пропущено,
  `36/36` физических calibration/evaluation групп; уникальных sequence,
  source и noise seeds `108/108` на split, межролевой overlap `0/0/0`.
  Нарушений PSD/symmetry `0/0`, evaluation covariance fit `0`, online truth
  use `0`, ложных independent-frame claims `0`, probability claims для
  quality `0`. Финальных проваленных критериев нет. Статистическое ограничение
  явно сохранено: три независимые sequence на группу не дают узких tail
  confidence intervals; CI не вычислялись, поэтому frame bootstrap не
  применялся. `ROADMAP.md`: S7A=`Done`, S7B=`Next`. Итоговая формулировка:
  **calibrated bearing measurement benchmark, not tracking and not a
  signal-level CRLB**.
- 2026-08-29 — полный S7A benchmark выполнен после PASS gates и повторно
  воспроизведён внутри `bearing_uncertainty_validation.ipynb`. Сетка:
  36 физических групп × `(3 calibration + 3 evaluation)` = **216 независимых
  continuous sequences**, по 6000 reception samples (`0.125 s`) и 20 overlap
  frames каждая. Итого по каждой split/group/method 60 статистически зависимых
  residual samples, но ровно 3 независимые sequence units. Сохранены **216
  covariance rows** (108 calibration + 108 evaluation) и **1368 quality rows**.
  Все 108 calibration/evaluation sequence seeds уникальны внутри роли;
  sequence/source/noise overlap между ролями равен `0`. Coverage всех строк
  `1.0`, antipodal count `0`; все `R` symmetric, PSD и rank 2 без epsilon,
  minimum eigenvalue `9.655e-8 rad^2`, maximum condition number `150.135`.
  Notebook: 8 cells, `nbformat` valid, error-output `0`, unrun code `0`,
  missing ID `0`. Полная регрессия и остальные notebook ещё не повторены.
- 2026-08-29 — первоначальная **raw, нецентрированная** диагностика
  `r^T R^+ r` не подтверждала универсальную Gaussian model. Эти числа
  superseded корректирующим gate 2026-08-30 и теперь хранятся только как
  `raw_normalized_squared_error`, без chi-square comparison. Для 108 evaluation
  групп прежнее отношение raw P95 к `chi-square(2)` P95
  имеет диапазон `0.0967…2.7851`, median `1.1535`; только `77/108` групп лежат
  в диагностическом диапазоне `0.5…1.5`, а fraction выше chi-square P95 лежит
  в `0…0.30`. Median P95-ratio по SNR равен `1.029/1.067/1.371` для
  `-6/5/20 dB`. Это benchmark хвостов, **не утверждение Gaussian distribution**.
  Средний evaluation RMSE tetrahedral для ref-3/all-6/SRP составляет на
  `-6 dB` `4.394/1.669/0.741°`, на `5 dB` `0.237/0.177/0.179°`, на `20 dB`
  `0.098/0.071/0.071°`; square соответственно `1.873/1.796/1.303°`,
  `0.410/0.350/0.350°`, `0.309/0.304/0.309°`. Это усреднение по двум сигналам
  и трём траекториям, не критерий превосходства метода.
- 2026-08-29 — offline quality/error analysis показывает контекстную, а не
  вероятностную связь. Средний по группам модуль Spearman наиболее велик у
  SRP score margin (`0.214`), GCC mean/min peak ratio (`0.184/0.181`) и GCC
  curvature (`0.166`); максимум отдельных групп достигает `0.655/0.763`.
  Знак может меняться (например, на square/high-SNR систематические эффекты
  дают положительную связь peak ratio с error), поэтому универсальная online
  probability calibration не заявляется. Все group-level SRP curvature means
  конечны; минимальная mean eigenvalue `0.445`. Профильный deterministic/study/
  SRP/sequential набор: **34 passed in 8.99s**.
- 2026-08-29 — observable quality добавлена в общий frame-wise API без truth:
  GCC сохраняет peak ratios/curvatures/spectral energies используемых пар,
  их агрегаты, boundary count и valid pair count; reference-3 использует
  только три опорные пары, all-6 — шесть. SRP сохраняет peak score,
  coarse-grid score margin и симметричную локальную `-H(score)` с собственными
  значениями в координатах радиан дуги; на elevation-boundary Hessian явно
  недоступен. Ни одна величина не называется вероятностью. Регрессия
  SRP/moving/sequential: **27 passed in 8.36s**.
- 2026-08-29 — реализован независимый sequence-level calibration/evaluation
  pipeline `validation/bearing_uncertainty_study.py`. Основная сетка содержит
  36 групп: tetrahedral/square × SNR `-6/5/20 dB` × stationary/transverse/
  piecewise × random broadband/deterministic multisine, `fs=48 kHz`,
  `L/H=1024/256`. Calibration/evaluation получают разные role-coded sequence,
  source и noise seeds; все методы внутри sequence используют один stream.
  Deterministic multisine сохраняет фиксированный спектр, но получает
  воспроизводимый seed-dependent общий phase offset, поэтому splits имеют
  разные waveform realizations. Smoke gate с `2+2` независимыми sequence
  прошёл: 6 covariance и 38 quality records, все `R` symmetric/PSD, seed
  overlap `0`, evaluation-fit use `0`, online truth use `0`. Профильные тесты:
  **13 passed in 1.85s**. Для полного benchmark после PASS зафиксированы
  **3 calibration + 3 evaluation независимых sequence на группу**, duration
  `0.125 s`; overlap frames используются как зависимые residual samples, не
  как independent trials. Полный study и notebook ещё не выполнены.
- 2026-08-29 — реализован `model/bearing_statistics.py`. Для нормированных
  `u,u_hat` вычисляются `theta=acos(clip(u^T u_hat,-1,1))`, сферический
  `Log_u(u_hat)=theta*(u_hat-(u^T u_hat)u)/sin(theta)` и его координаты в
  ортонормированном базисе `e_az,e_el`; единицы residual — радианы дуги.
  Малые углы обрабатываются через норму касательной проекции, а почти
  антиподальные направления явно дают `AntipodalDirectionError`, поскольку
  log-map там не единственен. Sample covariance строится без произвольного
  epsilon, сохраняет eigen/rank/condition/correlation diagnostics; NIS
  использует `R^+`. Deterministic gate: **8 passed in 0.14s**. В wrap-тесте
  elevation-компонент ограничен величиной второго порядка `delta_phi^2`:
  одинаковый elevation двух точек не означает, что соединяющая их геодезическая
  лежит на параллели. Calibration/evaluation study ещё не выполнен.
- 2026-08-29 — полностью прочитаны `AGENTS.md`, `PROJECT_STATUS.md` и
  `README.md`. Создан `ROADMAP.md` с этапами S0–S13, зависимостями и явным
  различием single-station bearing tracking и multi-station 3D localization.
  S7A отмечен `In progress`, S7B — `Planned (Next)`. Начата реализация
  сферического tangent residual; deterministic/smoke gates, полный study,
  notebook и общая приёмка ещё не выполнены.

### Точная схема последовательности

- Один source array `s[n]` синтезируется на весь source-time support. Из него
  одним retarded-time проходом получаются непрерывные clean channels
  `x_m[n]`; одна noise matrix `w_m[n]` создаётся один раз на всю
  последовательность, после чего `y_m[n]=x_m[n]+w_m[n]`.
- Frame `k` является view `y_m[kH : kH+L]` общего массива, где
  `L=1024`, `H=256`, overlap `L-H=768` (`75%`). При `N=12000` samples
  число frame равно `1+floor((N-L)/H)=43` на последовательность. Общие
  overlap samples побитово одинаковы; frame не ресинтезируются.
- Для start/end reception timestamps `t_s,t_f` центр равен
  `t_c=(t_s+t_f)/2`. Истинный timestamp bearing — centroid emission time
  `t_e`, решающий `t_c=t_e+||q(t_e)-centroid(r)||/c`, а не обычный reception
  time. Physical delay равен `t_c-t_e`; acquisition latency для оценки,
  относимой к центру frame, равна `t_f-t_c`; available timestamp равен
  `t_f+t_algorithm`. Total emission-to-available latency равна сумме этих
  трёх составляющих.
- Frames обрабатываются по возрастанию `k`. Estimator получает только текущий
  frame, координаты и `fs`; truth, future samples и будущие DOA ему не
  передаются. Методы используют один frame hash и один shared six-pair GCC
  frontend, затем отдельные reference-3/all-6/SRP backends.
- Основные параметры: `fs=48000 Hz`, duration `0.25 s`, chunk `4096`,
  Kaiser FIR `129`; maximum interpolation weight matrix содержит
  `4096*129=528384` элементов вместо зависимости от полной длины stream.
- Диагностические последовательности: stationary, constant-velocity
  transverse/receding, circular, piecewise-linear maneuver, azimuth wrap
  `359° -> 0°`, а также low-SNR `-12 dB` с явно маркированным all-channel
  data dropout. Все траектории дозвуковые, не пересекают решётку и остаются
  внутри построенного source-time support.
- 43 overlap frame каждой последовательности статистически зависимы и не
  называются независимыми trials. Frame-level и sequence-level метрики
  хранятся отдельно в CSV.

### Журнал текущего этапа

- 2026-08-29 — финальная приёмка завершена. Полный pytest после выполнения
  notebook: **209 passed in 16.56s**; `pip check`: `No broken requirements
  found`. Все **10/10** notebook программно выполнены и проходят
  `nbformat.validate`; error-output `0`, невыполненных непустых code-ячеек
  `0`, отсутствующих cell ID `0`. CSV counts: GCC `840/574/70`, SRP
  `792/594`, moving `6480`, sequential frame/summary `903/21`. Sequential
  аудит: 301 уникальный статистически зависимый frame, frame-hash violations
  `0`, causality/timestamp violations `0`, truth/future-use flags `0`.
  Stationary coverage `1.0`, truth change `0°`, RMSE
  `0.0514/0.0381/0.0376°`; wrap coverage `1.0`, переход
  `356.81° -> 2.97°`, RMSE `0.0681/0.0577/0.0579°` для
  ref-3/all-6/SRP. В шести штатных sequence coverage `1.0`; low-SNR/dropout
  coverage `0.88372`, пять invalid frame на метод без фиктивного bearing.
  Mean algorithm runtime по штатным sequence находится примерно в диапазонах
  `2.06…2.24 ms` ref-3, `1.48…1.68 ms` all-6 и `3.85…4.42 ms` SRP;
  physical delay `72.874…76.562 ms`, acquisition latency `10.65625 ms`, total
  latency `85.012…91.461 ms`. Новых ослабленных тестовых допусков нет;
  P95/P99 по 43 overlap frames являются sequence diagnostics, не оценками
  хвостов независимой выборки. Console warnings Windows ZMQ/IPython остаются
  нефатальными и не являются notebook error-output. Итог:
  **sequential independent bearings, not tracking**.
- 2026-08-29 — новый `sequential_doa_validation.ipynb` программно выполнен:
  `nbformat` valid, error-output `0`, невыполненных непустых code-ячеек `0`;
  он повторно создал и проверил 903/21 CSV-строку, causality, shared hashes,
  invalid semantics, stationary и wrap. После документации и package exports
  полный pytest: **209 passed in 16.86s**. Это промежуточная приёмка: остальные
  девять notebook ещё должны быть повторно выполнены до завершения этапа.
- 2026-08-29 — реализован полный sequential study и два раздельных CSV.
  Семь потоков по `0.25 s` содержат по 12000 reception samples при `48 kHz`;
  `frame_length=1024`, `hop_length=256`, overlap `768` samples (`75%`) дают
  по **43** перекрывающихся frame и всего **301 статистически зависимый
  frame**, не independent trials. Сохранены **903 frame-level** строки
  (три метода) и **21 sequence-level** агрегат. Для шести штатных
  последовательностей coverage `1.0`; диапазоны conditional RMSE/P95/P99:
  ref-3 `0.0514…0.1060° / 0.0855…0.1625° / 0.0953…0.1990°`, all-6
  `0.0381…0.0880° / 0.0639…0.1474° / 0.0796…0.1661°`, SRP
  `0.0376…0.0886° / 0.0627…0.1485° / 0.0785…0.1680°`. Low-SNR
  (`-12 dB`) stream имеет явный all-channel data-dropout и ровно 5 invalid
  frames на метод, coverage `38/43=0.88372`; invalid bearing/error fields
  остаются пустыми. Stationary truth change строго `0°`; wrap-последовательность
  проходит `356.81° -> 2.97°` с круговым шагом <`1°` и RMSE <`0.14°`.
  Chunk `4096`, FIR `129`, maximum interpolation working set `528384`
  coefficients. Mean physical propagation delay `72.874…76.562 ms`;
  acquisition latency относительно центра frame до последнего принятого
  sample `10.65625 ms` (`frame span=21.3125 ms`, nominal `1024/fs=21.3333 ms`);
  algorithm runtime измеряется отдельно, total emission-to-available latency
  `85.012…91.461 ms`. Causality audit: future sample/DOA use `0`, несовпадений
  frame hash между методами `0`, timestamp violations `0`. Notebook создан и
  проходит `nbformat`; профильные continuous/sequential/moving tests:
  **27 passed in 4.02s**. Полная приёмка всех notebook ещё не выполнена.
- 2026-08-29 — полностью прочитаны `AGENTS.md`, `PROJECT_STATUS.md` и
  `README.md`; начат этап непрерывного потока. В `simulate_moving_source`
  добавлен chunked режим: при блоке `B` и FIR длины `L` интерполяционный
  working set ограничен `B*L`, тогда как emission times, delays и output
  остаются непрерывными по всей последовательности. Реализован
  `simulation/continuous_stream.py`: один source waveform, одна noise matrix
  на весь stream и read-only overlap views исходного `channels`, без
  покадрового ресинтеза. Strict regression для chunked/monolithic:
  channels `atol=3e-12`, emission/delays `atol=2e-15 s`, одинаковый valid
  region. Подтверждены точный overlap, v=0/static agreement и полная seed-
  воспроизводимость. Профильный результат: **13 passed in 1.57s**. Этап не
  завершён до sequential study, полного pytest и notebook-аудита.
- 2026-08-29 — итоговая приёмка reporting-поправок завершена. После полного
  пересчёта `results/moving_source_summary.csv` выполнен весь pytest:
  **198 passed in 14.22s**. Все **9/9** notebook программно перевыполнены и
  прошли `nbformat.validate`; error-output `0`, невыполненных непустых
  code-ячеек `0`. CSV counts: GCC pair/DOA/covariance `840/574/70`, SRP
  DOA/runtime `792/594`, moving source `6480`. `pip check`:
  `No broken requirements found`. Moving-source notebook подтвердил новые
  средние conditional RMSE `3.6054°/2.7701°/0.6064°` для
  reference-3/all-6/SRP при coverage `1.0`; изменение относительно прежних
  чисел ожидаемо из-за общего clean waveform между SNR. Новых ослаблений
  критериев нет; сохранено прежнее явное ограничение: `20 trials/config`
  недостаточно для устойчивого operational P99. Console warnings Windows
  ZMQ/IPython о selector thread, permissions и unencrypted local TCP kernel
  остаются нефатальными и не являются notebook error-output.
- 2026-08-29 — полный moving-source study пересчитан без изменения сетки или
  числа испытаний: **6480 CSV-строк**, **2160 конфигураций**, **43200 paired
  trials**, `20 trials/config`. Численный аудит подтвердил 720 физических
  групп и clean-signal seeds, 2160 noise seeds; в каждой группе один clean
  seed для трёх SNR и три noise seeds. Нарушений
  `reference-3 boundary <= all-6 boundary` нет; all-6 имеет дополнительные
  boundary hits в 21 moving и 17 static конфигурациях, максимальная разница
  fractions `0.15`. Runtime identity `total=shared frontend+backend` выполнена
  с максимальным остатком `8.67e-19 s`; frontend pair count всегда 6, backend
  pair count 3 только для reference-3 и 6 для all-6/SRP. Mean moving frontend
  runtime `0.7950 ms`; mean backend `1.0827/0.6303/3.2462 ms` для
  ref-3/all-6/SRP. Expected static effective-minus-nominal SNR не превышает
  `8.88e-16 dB`; mean realized static отклоняется на `-0.128…+0.156 dB` из-за
  конечной noise realization. Moving effective-minus-nominal имеет диапазон
  `-1.721…+0.975 dB`, поскольку общий noise scale задаётся по matched-static
  clean RMS, а Doppler/time warp меняет moving frame RMS. Все новые numeric
  fields конечны. До полного pytest и выполнения всех notebook статус остаётся
  **в работе**.
- 2026-08-29 — исправлена логика отчётности до полного пересчёта. Для
  `reference_3_gcc_wls` boundary агрегируется только по трём реально
  используемым опорным парам `(0,1)`, `(0,2)`, `(0,3)`; для all-6 — по всем
  шести. Добавлен targeted regression: boundary только на неиспользуемой паре
  даёт `reference-3=False`, `all-6=True`. Runtime разделён на общий
  six-pair GCC frontend, estimator backend и их явно названную сумму; backend
  pair count равен 3/6. Clean-signal seed теперь зависит от всех факторов,
  кроме SNR; noise seed остаётся отдельным. В схему CSV добавлены nominal,
  expected-effective и mean realized effective moving/static SNR, где
  `SNR_eff=20 log10(RMS(clean frame)/RMS(realized noise frame))`. Целевой
  `tests/test_moving_source_study.py`: **7 passed in 4.82s**. Этап пока не
  завершён: полный study и общая приёмка ещё не выполнены.
- 2026-08-29 — итоговая приёмка завершена. Полный pytest после выполнения
  всех notebook: **196 passed in 14.07s**. Все **9/9** notebook прошли
  `nbformat.validate`, имеют error-output `0` и невыполненных непустых
  code-ячеек `0`. Проверены размеры CSV: GCC pair/DOA/covariance
  `840/574/70`, SRP DOA/runtime `792/594`, moving source `6480`.
  Editable reinstall с новым пакетом `visualization` успешен, импорты из
  родительского каталога прошли; `pip check`: `No broken requirements found`.
  Нефатальные Windows warnings
  ZMQ/IPython о selector thread, permissions и локальном unencrypted TCP
  kernel остаются console warnings, но не notebook error-output. Во время
  реализации исправлены: тройное повторение exact count; неверный `floor`
  anchor переменного sinc (static mismatch `4.31e-6` исправлен до
  `1.82e-13`); broadcast shape в тесте static TDOA; `NaN` для неопределённого
  radial lag заменён на `None`; абсолютный tangent threshold заменён
  относительным, устранив round-off lag для approach/recede. Ослабление
  статистики: 20 trials/config дают воспроизводимые RMSE/coverage, но P99
  остаётся sampled diagnostic и не является operational tail estimate.
- 2026-08-29 — полный paired moving-source study выполнен после PASS gates.
  Cartesian product: `2` geometry × `3` motion × `5` speed × `3` distance ×
  `4` frame length × `2` signal × `3` SNR = **2160 конфигураций**;
  `20 trials/config`, **43200 moving/static пар**, **86400 кадров**, **6480
  CSV-строк**. Base seed `20260830`, signal/noise child seeds раздельны,
  noise внутри moving/static пары общий. Все conditional RMSE/P95/P99 и
  diagnostics конечны, coverage всех трёх методов `1.0`, worst boundary-hit
  fraction `0.15` для GCC и `0.05` для SRP; при `v=0` maximum absolute
  moving-static excess строго `0.0°`. Средний RMSE по всему, включая
  `-6 dB`: ref-3 `3.560°`, all-6 `2.728°`, SRP `0.602°`. На `20 dB` средний
  excess трёх методов при `5 м/с` равен примерно `0.016…0.020°`, при
  `30 м/с` — `0.059…0.064°`. Максимальные изменения внутри кадра:
  DOA `7.3204°`, TDOA `74.4492 мкс`; диапазон Doppler factor
  `0.919571…1.095847`. В `-6 dB`, особенно `N=256` deterministic multisine,
  присутствуют sampled P99 выбросы `86.8…145.2°`; `20 trials/config`
  недостаточно для стабильного operational P99 и является явным ослаблением,
  а не выбранным порогом. Исправлен round-off отчёта радиального angular lag:
  для approach/recede он теперь `None`, DOA change точно `0`.
- 2026-08-29 — создан `visualization/moving_scene.py` и два notebook.
  3D-сцена показывает array, истинные trajectory/source/DOA и только bearing
  rays трёх оценок; длина луча использует true range с явной маркировкой
  `visualization only`, оценённая 3D-точка не рисуется. Есть Play/Pause,
  frame slider и два camera rotation sliders. Notebook
  `moving_source_3d.ipynb` синтезирует и локализует 13 независимых кадров;
  `moving_source_validation.ipynb` проверяет CSV и строит RMSE/excess,
  within-frame change и boundary plots. Оба notebook: schema valid,
  error-output `0`, невыполненных code-ячеек `0`. Профильные visualization +
  study tests: **7 passed in 1.98s**.
- 2026-08-29 — реализован каркас парного покадрового исследования
  `validation/moving_source_study.py`. Для каждой moving frame создаётся
  matched stationary frame с тем же исходным сигналом, точно тем же массивом
  AWGN и тем же seed. Истина определяется в centroid emission time выбранного
  центрального reception sample, а не по `q(t_reception)`. Реализованы три
  независимые покадровые оценки: reference-3 GCC+WLS, all-6 equal GCC+WLS и
  equal-weight SRP-PHAT; это не tracking. Signed angular lag определён как
  проекция сферического log-map ошибки на мгновенную касательную DOA и имеет
  значение `None` для чисто радиального движения. Deterministic gate
  tetra/transverse, `v=20 м/с`, `R=50 м`, `N=1024` прошёл: ошибки трёх методов
  `0.06298°`, `0.06575°`, `0.06449°`. Smoke Monte Carlo tetra/transverse,
  `v=20`, `R=25`, `N=512`, random broadband, `10 dB`, `6 trials` прошёл с
  coverage `1.0` для всех методов и RMSE `0.103…0.116°`. Профильные тесты
  study: **5 passed in 1.26s**. Полный Cartesian study ещё не запущен.
- 2026-08-29 — реализован `simulation/moving_source.py`. Точный emission time
  является единственным корнем
  `g(t_e)=t_e+||q(t_e)-r_m||/c-t=0`; производная
  `g'(t_e)=1+v_r/c>0` при `|v|<c`. Общий solver использует векторный Newton
  с bracketed Brent fallback, а для constant velocity независимо реализован
  положительный корень квадратного уравнения для `d=t-t_e`. Естественный
  Doppler равен `dt_e/dt=1/(1+v_r/c)`; отдельный frequency shift не вводится.
  Синтез использует fractional Kaiser-windowed-sinc time warp, zero extension,
  причинность `t_e<t`, общий valid region с FIR guard и опциональное `1/R`.
  Диагностический `frozen_delay` сохраняет постоянные задержки. При `v=0`
  результат совпадает с прежним static Kaiser-FIR в valid region до
  `1.82e-13`; численный/аналитический emission time совпадают до
  `1.11e-16 с`; Doppler tone test: `944.903650` против `944.903581 Гц`,
  ошибка `6.85e-5 Гц`. Проверены монотонность, отсутствие wrap/NaN,
  permutation и translation invariance. Совместная профильная проверка:
  **16 passed in 1.36s**. Изменены `simulation/moving_source.py`,
  `simulation/__init__.py`, `tests/test_moving_source.py`,
  `PROJECT_STATUS.md`.
- 2026-08-29 — реализован кинематический слой `simulation/trajectory.py`:
  `StationaryTrajectory`, `ConstantVelocityTrajectory`, произвольно
  ориентированная `CircularTrajectory` и `PiecewiseLinearTrajectory`.
  Единицы API: `q(t)` — м, `v(t)` — м/с, `a(t)` — м/с², `t` — с;
  скалярное время даёт `(3,)`, массив времени — `(...,3)`. Все конструкторы
  строго проверяют `|v|<c`; piecewise-linear использует постоянную скорость
  внутри сегмента, нулевое обычное ускорение между узлами и не пытается
  представить импульсный скачок скорости в узле конечным `a`. По умолчанию
  выход за knot support запрещён, опциональная экстраполяция продолжает
  крайние сегменты. Профильная проверка: **7 passed in 1.08s**. Изменены
  `simulation/trajectory.py`, `simulation/__init__.py`,
  `tests/test_trajectory.py`, `PROJECT_STATUS.md`.
- 2026-08-29 — полностью прочитаны `AGENTS.md` и `PROJECT_STATUS.md`, этап
  движения начат. Исправлена неоднозначность SRP runtime: поле
  `exact_reference_trial_count` теперь является вкладом в глобальное число
  уникальных exact trials и ненулевое только в строке компонента
  `srp_exact_vectorized_reference`. Отдельное повторяемое поле
  `exact_reference_trials_per_configuration=3` описывает sampling design;
  scope явно равен `first_3_evaluation_trials_per_configuration`, а
  `exact_fast_disagreement_covers_all_evaluation_trials=False`. Существующий
  CSV мигрирован без повторного Monte Carlo: 594 строки, 198 конфигураций,
  `198*3=594` уникальных sampled exact trials; максимум `0.0313649563°`
  относится только к первым 3 из 1000 evaluation trials каждой конфигурации.
  Профильная проверка: **13 passed in 10.41s**. Изменены
  `validation/srp_statistical.py`, `tests/test_srp_phat.py`,
  `results/srp_runtime_summary.csv`, `AGENTS.md`, `PROJECT_STATUS.md`.

- 2026-08-28 — начат двухчастный этап GCC reporting + far-field SRP-PHAT.
  Зафиксированы обязательные P99/P99.9, success counts, явная conditional
  терминология и risk–coverage режимы без отбрасывания, с несколькими
  calibration-percentile thresholds и с soft weighting. Operational SNR
  threshold не будет выбран без явного набора критериев. До PASS части A
  реализация SRP не начинается.
- 2026-08-28 — реализована схема части A. Канонические метрики ошибок успешных
  оценок имеют префикс `conditional_`; добавлены P99/P99.9, success/failure
  counts, coverage и явная conditioning metadata. Risk–coverage содержит
  calibration-covariance без отбрасывания, hard thresholds `P05/P10/P25/P50`
  и soft confidence weighting без удаления пар. Hard rejection теперь
  действительно исключает пары из projection/WLS, а не только проверяет
  связность графа. Промежуточная ошибка идемпотентности из-за `NaN` percentile
  исправлена заменой отсутствующего значения на `None`. Целевые тесты:
  **5 passed in 2.71s**; полный pytest части A: **160 passed in 7.09s**.
  Массовые CSV и численный risk–coverage анализ ещё не пересчитаны, поэтому
  часть A пока не объявлена завершённой.

- 2026-08-28 — принято новое ТЗ полной статистической валидации GCC-PHAT.
  Зафиксировано обязательное разделение calibration/evaluation, минимум
  `1000/2000` реализаций на итоговую конфигурацию, покомпонентные pair-метрики,
  три класса сигналов и термин `Gaussian covariance benchmark` вместо CRLB
  при выбросах/негауссовости. Начата диагностика GCC-пика и прямой эталон.
- 2026-08-28 — диагностика GCC-пика и независимый прямой эталон реализованы.
  Каждый результат содержит `peak_value`, peak-to-second-peak ratio,
  curvature, boundary-hit, invalid/reason, использованную спектральную
  энергию/долю и число bins. Тишина и почти нулевой сигнал возвращают
  `invalid=True`, `delay=NaN`, reason `signal_energy_below_threshold`.
  Неправильная полоса отклоняется как configuration error, а корректная, но
  пустая по FFT-bins полоса возвращает `empty_frequency_band`. Прямое
  вычисление `sum_k Psi[k] exp(j2*pi*f_k*tau)` совпало с FFT по знаку,
  grid-peak и нормированной корреляции. Целевой результат: **30 passed in
  1.54s** для GCC core/diagnostics.
- 2026-08-28 — реализована weighted cycle projection
  `argmin_t (tau-Bt)^T W (tau-Bt)` с zero-mean gauge для TOA. Возвращаются
  consistent TDOA, residual/cost и нормы cycle-компоненты до/после. Проверены:
  неизменность идеальных TDOA, точные циклы после проекции, минимальность
  weighted residual, инвариантность к ориентации пар, diagonal/full PSD
  weights и явное обнаружение disconnected graph. Результат: **6 passed in
  0.56s**.
- 2026-08-28 — добавлены три раздельно маркированных signal model:
  `deterministic_multisine`, новый независимый для каждой реализации
  `random_broadband` и `harmonic_stress`. Последний явно является только
  тестом неоднозначных пиков, не реалистичной моделью БПЛА. Реализован единый
  all-6-pair calibration/evaluation engine с независимыми SeedSequence,
  pair bias/covariance/correlation/confidence thresholds только из calibration
  и четырьмя DOA-вариантами только на evaluation. Добавлена exact spherical
  WLS при известном centroid-range и раздельные GCC measurement, plane-model
  и DOA bias. Целевые statistical/spherical тесты: **11 passed in 1.61s**;
  полный промежуточный регрессионный результат: **159 passed in 6.79s**.

- 2026-08-28 — выполнен первый полный массовый прогон статистической валидации GCC-PHAT:
  **70 конфигураций**, в каждой независимые `1000` calibration и `2000` evaluation
  реализаций, всего **210000 испытаний** с базовым seed `20260828`. Сохранены
  `840` строк pair-метрик, `264` строки DOA-метрик и `70` строк ковариационных
  диагностик. Все шесть пар присутствуют для каждого split; calibration/evaluation
  seeds не совпадают. Это промежуточный численный результат: итоговая приёмка будет
  только после выполнения notebook и полного повторного контроля.
- 2026-08-28 — численно локализована граница срыва на fine SNR-сетке. Для всех четырёх
  DOA-вариантов обеих решёток первый исследованный уровень, где `P95 < 5°`, равен
  `-6 dB`; при `-8 dB` P95 ещё превышает `10°`. Для варианта cycle projection +
  calibration на `-10 dB` получены RMSE/P95 `22.50°/35.34°` (square) и
  `17.96°/31.62°` (tetrahedral); на `0 dB` — `0.680°/1.229°` и
  `0.429°/0.751°`; на `10 dB` — `0.259°/0.478°` и `0.157°/0.275°`.
  Доля pair-ошибок больше одного отсчёта падает в среднем с `0.279` на `-10 dB`
  до `0.0227` на `-6 dB`, `2.08e-4` на `-2 dB` и нуля на `0 dB` и выше.
- 2026-08-28 — проверены три класса сигналов. Новый случайный broadband существенно
  устойчивее искусственной фиксированной multisine и harmonic stress: при `-10 dB`
  calibrated RMSE/P95 равны `4.91°/3.06°` (square) и `2.86°/1.25°`
  (tetrahedral), тогда как harmonic stress даёт `28.39°/43.08°` и
  `25.63°/43.78°`. Harmonic stress остаётся только стресс-тестом неоднозначных
  пиков и **не** называется реалистичной моделью БПЛА. Связь peak-to-second ratio
  с абсолютной pair-ошибкой оказалась слабой: средняя Spearman correlation
  `-0.110` для multisine, `-0.044` для random broadband и `-0.168` для harmonic
  stress; confidence-отбраковка поэтому считается эвристикой, а не вероятностной
  калибровкой.
- 2026-08-28 — weighted cycle projection уменьшает наблюдавшийся в том
  промежуточном прогоне maximum of record means после проекции до
  `1.255e-13 мкс`. При одинаковых весах all-6 raw и projected дают
  одинаковый DOA: удаляемая cycle-компонента лежит вне пространства физически
  согласованных TDOA. Преимущество появляется при calibration covariance/confidence
  weighting; на очень низком SNR остаются редкие disconnected/rejected случаи.
- 2026-08-28 — сферический эксперимент при `20 dB` разделяет три источника ошибки.
  GCC measurement RMSE составляет `0.79–0.80 мкс` для square и `0.58–0.59 мкс`
  для tetrahedral. Направленно-зависимый plane mismatch при `R=5→50 м` падает
  `1.785→0.179 мкс` (square) и `2.705→0.272 мкс` (tetrahedral), согласуясь с
  `O(1/R)`. Exact known-range WLS устраняет noiseless model bias до численного нуля;
  у plane WLS для tetrahedral он падает `0.0876°→0.00877°`. Для симметричной square
  в выбранном направлении time-domain mismatch заметен, но его noiseless DOA bias
  мал (`0.00496°` при 5 м) из-за геометрической компенсации — это не означает
  отсутствия сферической ошибки.
- 2026-08-28 — часть A завершена после массового пересчёта: **840 pair / 574
  DOA / 70 covariance** строк, 372 risk–coverage строки в 62 независимых
  конфигурациях. Для каждой DOA-строки `success+failure=2000`, conditional
  P95≤P99≤P99.9; hard P05/P10/P25/P50 rejection и disconnected fractions
  монотонны по percentile во всех 62 группах. Финальный pytest:
  **160 passed in 7.05s**.
- 2026-08-28 — sampled P95 statement сохранено отдельно: первый проверенный
  fine-SNR уровень с conditional P95 ниже 5° для обеих решёток — `-6 dB`.
  **Operational threshold не выбран**: для него пока не утверждены минимальный
  coverage, P99/P99.9, catastrophic fraction и допустимая disconnected/failure
  probability. Условный P95 при низком coverage не используется как основание
  operational решения.
- 2026-08-28 — stale cycle-residual число разъяснено. `1.255e-13 мкс` было
  промежуточным maximum of per-record means на более раннем наборе вариантов.
  После финального Part-A CSV-аудита global maximum поля
  `mean_cycle_residual_after_us` равен `2.146390709e-13 мкс`; maximum before
  projection — `374.2993049 мкс`. Поле является средним по успешно
  спроецированным trials от евклидовой нормы координат cycle-space, а затем
  берётся максимум этих средних по CSV-строкам. Оба after-значения имеют
  порядок `1e-19 s` и отражают roundoff SVD/lstsq и изменившийся набор
  selected-pair подзадач, а не физический остаток.
- 2026-08-28 — реализован детерминированный equal-weight far-field SRP-PHAT.
  Direct reference явно суммирует пары/частоты; независимая vectorized версия
  блоками вычисляет те же steering phases. Основной estimator выполняет
  глобальную полуоткрытую azimuth grid, два локальных grid-уровня и непрерывный
  L-BFGS-B refinement; возвращает score, boundary/invalid, energy и runtime
  diagnostics. Peak-ratio confidence в SRP не используется. Исправлена одна
  setup-ошибка теста (`frequency_domain` → фактическое API-имя `frequency`).
  Проверены знак, direct/vector agreement, grid/refinement, translation,
  permutation/orientation, square mirror ambiguity, tetrahedral upper/lower,
  silence и unit sphere: **9 passed in 2.19s**. Полный pytest:
  **169 passed in 8.22s**. Детерминированный gate пройден; следующий gate —
  небольшой paired smoke Monte Carlo до полного исследования.
- 2026-08-28 — paired smoke Monte Carlo gate пройден. Четыре метода получили
  один и тот же evaluation channel/noise set; calibration/evaluation seeds
  различны. При `12/20` trials, tetrahedral, `0 dB`, conditional RMSE:
  reference-3 `0.787°`, all-6 equal `0.438°`, all-6 calibrated `0.532°`,
  equal-weight SRP `0.425°`; это smoke, не итоговое сравнение. Для массового
  запуска разрешён correlation-interpolated SRP backend: он использует те же
  all-pair PHAT correlations и cubic delay interpolation, а первые trials
  каждой конфигурации проверяются независимой exact vectorized формулой.
  В smoke maximum exact/fast disagreement `0.00212°`; mean runtimes
  fast search/exact reference `0.00332/0.0541 s`. SRP-тесты:
  **11 passed in 2.61s**, полный pytest: **171 passed in 8.63s**.
- 2026-08-28 — первый полный SRP-прогон (**198 конфигураций**, `500/1000`
  calibration/evaluation, **297000 реализаций**) прошёл структурный аудит, но
  численный анализ не был принят: на `az=20°, el=10°` при высоком SNR поиск
  устойчиво выбирал боковой лепесток с ошибкой `27–38°`. Причина — слишком
  редкая начальная сетка `15°`: локальное уточнение улучшало неверный локальный
  максимум, хотя score в истинном направлении был существенно выше. Начальный
  шаг заменён на `5°`, последующие — на `1°/0.25°`; также исправлена metadata
  fast backend, где `valid_region.stop` ошибочно обозначал длину GCC-корреляции,
  а не число входных отсчётов. Добавлен отдельный regression-тест низкого угла
  места. После исправления noiseless ошибки для random broadband/multisine
  составили `0.015°/0.034°` вместо `27.4°`; harmonic stress — `0.361°`.
  Целевой результат: **13 passed in 5.80s**. CSV первого прогона признаны
  промежуточными и должны быть перезаписаны исправленным массовым прогоном.
- 2026-08-28 — исправленный полный SRP Monte Carlo завершён и численно принят:
  **198 конфигураций**, `500/1000` calibration/evaluation, **297000**
  реализаций, base seed `20260829`; сохранены `792` DOA и `594` runtime строк.
  Все группы содержат четыре метода и common evaluation seed, calibration seed
  в каждой группе отличен; `success+failure=1000`, P95≤P99≤P99.9. Выполнены
  **594** уникальные sampled exact-vectorized контрольные оценки (первые 3
  evaluation trials/config, не все evaluation trials): mean/max расхождение с
  correlation-interpolated search `0.002243°/0.031365°` (< regression-порога
  `0.05°`). После исправления worst high-SNR SRP P95 равен `4.544°`, прежних
  систематических `27–38°` максимумов нет. SRP имеет coverage `1.0` во всех
  строках; calibrated GCC coverage опускается до `0.923` из-за hard P10
  rejection/disconnected graph. Среднее total runtime на оценку:
  reference-3/all-6/calibrated/SRP `2.386/1.835/2.680/5.122 ms`; отдельно
  accelerated SRP search `4.127 ms`, exact vectorized reference `253.485 ms`.
  SRP даёт меньший RMSE, чем reference-3/all-6/calibrated соответственно в
  `191/166/168` из `198` парных конфигураций; это наблюдение, не критерий
  обязательного превосходства. На этой промежуточной записи следующей стадией
  оставались выполнение всех notebook и финальный pytest.
- 2026-08-28 — итоговая приёмка этапа пройдена. Полный pytest после всех
  notebook: **173 passed in 14.43s**; `pip check`: `No broken requirements
  found`. Все **7/7** notebook выполнены через `nbconvert`, проходят
  `nbformat.validate`, имеют уникальные непустые cell ID, **0** error-output и
  **0** невыполненных непустых code-ячеек. CSV-аудит подтвердил строки:
  far-field/fractional-delay/benchmark `80/80/4`, GCC deterministic/MC
  `12/36`, GCC pair/DOA/covariance `840/574/70`, WLS-CRLB `42`, SRP
  DOA/runtime `792/594`. Исправлен notebook path bug: kernel запускается из
  `notebooks/`, поэтому GCC statistical notebook теперь явно пишет в корневой
  `results/`; три stale CSV и каталог `notebooks/results/` удалены. Первый
  запуск SRP-notebook также исправлен после `FileNotFoundError` относительного
  пути и успешно повторён. Нефатальные Windows ZMQ/TCP warnings остались,
  error-output отсутствует.
- 2026-08-28 — создан воспроизводимый `notebooks/gcc_statistical_validation.ipynb`
  из 17 ячеек. Он заново запускает полный Monte Carlo, проверяет 840/264/70 строк,
  независимость ролей calibration/evaluation, все шесть пар и терминологию benchmark,
  затем визуализирует fine-SNR boundary, catastrophic/boundary/invalid fractions,
  signal/frame dependence, empirical covariance и spherical model mismatch.
  До длительного выполнения notebook полный pytest повторно прошёл:
  **159 passed in 6.27s**; notebook прошёл `nbformat.validate`.
- 2026-08-28 — полный `gcc_statistical_validation.ipynb` успешно выполнен через
  `nbconvert` с повторной генерацией всех **210000** реализаций и сохранёнными
  графическими/численными outputs. В процессе приёмки были обнаружены и исправлены
  только ошибки notebook-аналитики: фактические имена pair-полей и отсутствие
  cycle-поля у сферических in-memory записей; сами Monte Carlo CSV при повторных
  запусках воспроизводились. После этого успешно программно выполнены остальные
  пять notebooks. Финальный структурный аудит и полный pytest ещё не объявлены
  завершёнными на этой стадии журнала.
- 2026-08-28 — финальная приёмка этапа завершена. Все **6 notebooks** прошли
  `nbconvert` с exit code 0; для каждого `nbformat` valid, число пропущенных
  cell ID, error-output и невыполненных code-ячеек равно **0**. CSV-аудит:
  `gcc_pair_error_summary.csv` — **840** строк, `gcc_doa_summary.csv` — **264**,
  `gcc_covariance_summary.csv` — **70**; все 140 configuration/split-групп
  содержат шесть пар. Полный pytest: **159 passed in 6.20s**. `pip check`:
  `No broken requirements found`. Этап объявлен завершённым только после этого
  численного и структурного контроля.

- 2026-08-27 — начаты три приёмочные поправки: точная область действия
  amplitude/phase допуска, полуоткрытая сетка azimuth `[0°,360°)` и
  повторный расчёт 25 границ на шагах `0.5°/0.25°`.
- 2026-08-27 — сетка azimuth исправлена на полуоткрытую `[0°,360°)`:
  при шагах `0.5°/0.25°` используются соответственно 720/1440 уникальных
  азимутов и 115920/462240 направлений при elevation `0…80°`.
- 2026-08-27 — все 25 численно найденных `R_min` повторно вычислены на обеих
  сетках. Максимальное `relative_grid_difference=2.79747e-5`, что меньше
  порога `1e-4`; локальное непрерывное уточнение по условию не запускалось.
  На refined-сетке нет нарушений целевых sample/phase-порогов, а наибольший
  сохранённый worst-case azimuth равен `324.25°` и не дублирует `360°`.
  В `results/far_field_boundary.csv` для каждой границы сохранены coarse/refined
  maximum error, их относительное расхождение и refined направление/пара.
- 2026-08-27 — приёмочный регрессионный контроль завершён: **107 passed in
  4.75s**; все три notebook выполнены через `nbconvert` с exit code 0,
  `nbformat` valid, error-output 0, невыполненных code-ячеек 0. Приёмочные
  поправки завершены; дальнейшие изменения относятся к GCC-PHAT.
- 2026-08-27 — реализован базовый детерминированный GCC-PHAT с ориентацией
  пары `tau_ij=T_i-T_j`, zero-padding до суммарной длины записей,
  PHAT-нормировкой, настраиваемой полосой, 32-кратной сеткой задержек и
  параболическим sub-sample уточнением. Целевые тесты: **22 passed in 1.51s**.
- 2026-08-28 — финальный notebook benchmark четырёх каналов показал для
  2400 отсчётов `2.673/0.935 ms` (frequency/FIR, ускорение `2.86x`), для
  12000 отсчётов `25.639/1.314 ms` (`19.51x`). Frequency-domain остаётся высокоточным
  эталоном; Kaiser FIR разрешён для будущего массового синтеза.
- 2026-08-27 — детерминированное сравнение GCC на 12 конфигурациях
  (square/tetrahedral, plane/spherical, три направления) прошло: maximum
  TDOA error `5.031e-7` отсчёта для frequency-domain и `5.868e-7` для FIR;
  maximum расхождение GCC между генераторами `1.340e-7` отсчёта.
- 2026-08-27 — финальная приёмка GCC: **129 passed in 5.14s**, `pip check`
  без нарушений; четыре notebook выполнены через `nbconvert`, у всех
  `nbformat` valid, ID присутствуют, error-output 0 и невыполненных
  code-ячеек 0. CSV содержат 80/80/4/12/42 строк соответственно.
- 2026-08-27 — после подтверждения ускорения FIR и cross-generator согласия
  начат массовый Monte Carlo GCC. Принятое диагностическое допущение:
  независимый по каналам и отсчётам аддитивный Gaussian noise; SNR каждого
  канала определяется по RMS чистого сигнала в общем valid region. Это
  численная noise-модель, а не измеренная модель БПЛА.
- 2026-08-27 — выполнены **7200** signal-level GCC/WLS испытаний: 2 решётки ×
  3 направления × 6 SNR × 200 реализаций, seed `20260827`. При `20 dB`
  TDOA RMSE во всех конфигурациях `0.505…0.905 мкс`, geodesic RMSE
  `0.090…0.609°`, ошибок больше 10° нет. При `0 dB` geodesic RMSE
  `0.577…2.536°`; при `-10/-20 dB` возникает выраженный срыв GCC с RMSE
  до `46.74/91.14°`. WLS формально завершился в 100% реализаций, но это не
  означает корректность ошибочного корреляционного максимума.
- 2026-08-28 — полный финальный контроль после Monte Carlo: **131 passed in
  5.45s**, `pip check` без нарушений; все пять notebook имеют валидный
  `nbformat`, ID всех ячеек, 0 error-output и 0 невыполненных code-ячеек.

- 2026-08-27 — проверены действующие соглашения: `tau_ij=T_i-T_j`,
  плосковолновой знак `(r_j-r_i)^T u/c`, единицы SI и запрет
  целочисленного округления дробных задержек.
- Принято: положение источника для направленной сферической модели задаётся
  относительно centroid решётки; для синтеза каналов общий минимальный TOA
  вычитается, но физические сферические TOA сохраняются в метаданных.
- Запланированы отдельные модули `simulation/fractional_delay.py`,
  `simulation/propagation.py` и численный аудит в `validation/far_field.py`.
- Реализованы centroid-инвариантные точные сферические TOA, плоские
  относительные TOA, отдельное разложение второго порядка и численный поиск
  границы дальней зоны на настраиваемой угловой сетке.
- Проверено: знак плоской TDOA совпадает со сферическим пределом, общий
  перенос не меняет TDOA, максимальная plane error убывает как `O(1/R)`,
  ошибка второго порядка — как `O(1/R^2)`. Промежуточно:
  **23 passed in 0.62s** для новых и базовых TDOA-тестов.
- Реализованы две независимые дробные задержки с единым знаком
  `delay>0 => y[n]=x[n-delay]`: zero-padded frequency-domain phase ramp и
  Kaiser-windowed sinc FIR.
- FIR по умолчанию имеет 129 коэффициентов (`half_length=64`,
  `beta=8.6`). Его causal group delay равен `64+fractional_part` отсчёта;
  API компенсирует фиксированные 64 отсчёта и оставляет запрошенную задержку.
- Частотная версия помещает сигнал между нулевыми guard-интервалами не менее
  `max(1024, 8*N)` отсчётов, включает в FFT длину задержки/выходного
  расширения и извлекает центральный линейный интервал без circular wrap.
- Консервативный общий valid region требует отступ 64 отсчёта исходного
  сигнала для каждой задержки; для спектральных измерений синусоид использован
  дополнительный диагностический отступ 512 отсчётов.
- Приёмочная полоса amplitude/phase — до `12 кГц` при `fs=48 кГц`: на
  задержках `0.1, 0.25, 0.5, 0.75, 0.9` ошибки обеих реализаций меньше
  `1e-5`. В расширенной исследовательской полосе до `19 кГц` FIR amplitude
  error меньше `2e-5`. Ошибка broadband group delay меньше `5e-5` отсчёта,
  расхождение реализаций меньше `4e-6` в valid region. Промежуточно:
  **34 passed in 1.16s**.
- Реализован `simulation/propagation.py`. Результат содержит матрицу
  `channels x samples`, истинные модельные TOA, применённые TOA после
  вычитания общего минимума, TDOA для выбранных пар и полную TDOA-матрицу,
  точные задержки в секундах/отсчётах, amplitude factors и общий valid region.
- Для spherical propagation метаданные TOA физические (`||q-r_m||/c`);
  для plane propagation TOA centroid-relative. Синтез в обоих случаях
  использует `T_m-min(T)`, не меняя ни одной TDOA.
- Опциональное ослабление равно `1/||q-r_m||` для spherical и общему `1/R`
  для plane. По умолчанию ослабление выключено, шума и отражений нет.
- Сквозные тесты подтвердили совпадение TDOA с `model/tdoa.py`, сходимость
  spherical к plane, согласованную перестановку каналов, переносы,
  отсутствие целочисленного округления дробных задержек и битовую
  воспроизводимость. Промежуточно: **11 passed in 1.11s**.
- Первый программный запуск нового notebook остановился на `SyntaxError` в
  строковом литерале Markdown-таблицы. Численные модули и CSV отработали,
  но стадия не принята; литерал исправлен, требуется полный повторный запуск.
- Повторный запуск нового notebook успешен: 16/16 ячеек имеют ID,
  `nbformat` valid, error-outputs — 0, невыполненных code-ячеек — 0.
  Сохранены `results/far_field_boundary.csv` и
  `results/fractional_delay_accuracy.csv`, по 80 строк результатов каждый.
- Предварительная одиночная проверка при `R=10 м` заменена полным аудитом
  всех 25 границ: полуоткрытые сетки `0.5°` (115920 направлений) и `0.25°`
  (462240 направлений) дали максимальное относительное расхождение
  `2.79747e-5`, что меньше принятого диагностического порога `1e-4`.
- Промежуточный полный результат после добавления notebook/CSV-тестов:
  **105 passed in 3.19s**.
- При финальном повторе Monte Carlo notebook все 84000 оценок были
  вычислены, но открытый в IDE `monte_carlo_crlb_summary.csv` дал Windows
  `PermissionError` на повторное открытие в режиме `w`. Стадия не принята.
  CSV-writer изменён на идемпотентный: если рассчитанное содержимое уже
  байт-в-байт совпадает с существующим файлом, лишняя перезапись не делается.
- Финальная приёмка после исправления writer: **106 passed in 3.19s**;
  все три notebook выполнены с exit code 0; у всех ячеек есть ID,
  `nbformat` valid, error-outputs — 0, невыполненных code-ячеек — 0;
  CSV имеют 42/80/80 строк; `pip check` и импорт установленного пакета
  `simulation` успешны.

### Формулы текущего этапа

Для `rho_m = r_m - centroid(r)` и
`q = centroid(r) + R u(phi,elevation)`:

\[
T_m^{\rm sph}=\frac{\|R\mathbf u-\boldsymbol\rho_m\|}{c},
\qquad
T_m^{\rm plane}=-\frac{\mathbf u^\mathsf T\boldsymbol\rho_m}{c}.
\]

Поэтому

\[
T_i^{\rm plane}-T_j^{\rm plane}
=\frac{(\mathbf r_j-\mathbf r_i)^\mathsf T\mathbf u}{c},
\]

то есть знак точно совпадает с существующим соглашением. Для синтеза
разрешено вычесть `min_m T_m`, не меняя TDOA.

Второй порядок реализован отдельно от точной нормы:

\[
T_m^{(2)}=\frac{1}{c}\left[
R-\mathbf u^\mathsf T\boldsymbol\rho_m+
\frac{\|\boldsymbol\rho_m\|^2-
(\mathbf u^\mathsf T\boldsymbol\rho_m)^2}{2R}\right].
\]

Количественная ошибка дальней зоны:

\[
E_\tau(R)=\max_{\varphi,\varepsilon,(i,j)}
|\tau_{ij}^{\rm sph}(R)-\tau_{ij}^{\rm plane}|,
\]

а её диагностические представления — `fs*E_tau` отсчётов и
`2*pi*f_max*E_tau` радиан. Минимальное расстояние ищется расширением
интервала и логарифмической бисекцией, а не выбирается из заданного списка.

Для GCC-PHAT ориентированной пары `(i,j)` используется

\[
G_{ij}[k]=\frac{X_i[k]X_j^*[k]}{|X_i[k]X_j^*[k]|},\qquad
\widehat\tau_{ij}=\arg\max_{\tau\in[-\tau_{\max},\tau_{\max}]}
\operatorname{IFFT}\{G_{ij}\}(\tau).
\]

Именно порядок `X_i X_j^*` обеспечивает соглашение
`\widehat{\tau}_{ij}=T_i-T_j`: положительный результат означает, что канал `i`
приходит позже канала `j`. Корреляция вычисляется после zero-padding как
минимум до суммарной длины записей; задержка не округляется до отсчёта.

Независимый прямой эталон вычисляет ту же PHAT-корреляцию без IFFT:

\[
R_{ij}(\tau)=\sum_{k\in\mathcal K}
\Psi_{ij}[k]\exp(j2\pi f_k\tau),\qquad
\Psi_{ij}[k]=\frac{X_i[k]X_j^*[k]}{|X_i[k]X_j^*[k]|}.
\]

Для полного набора пар incidence matrix `B` связывает микрофонные TOA `t`
и TDOA как `tau=B t`. Weighted cycle projection решает

\[
\widehat t=\arg\min_{\mathbf 1^\mathsf Tt=0}
(\widehat\tau-Bt)^\mathsf T W(\widehat\tau-Bt),
\qquad \widehat\tau_{\rm consistent}=B\widehat t.
\]

Zero-mean gauge устраняет общий ненаблюдаемый TOA; disconnected pair graph
отклоняется явно. При одинаковых весах cycle-компонента ортогональна
физическому pair-пространству, поэтому raw all-pair и projected all-pair WLS
дают одинаковый DOA, хотя только второй набор точно выполняет циклы.

Для signal-level Monte Carlo в каждом канале независимо используется

\[
n_m[k]\sim\mathcal N(0,\sigma_{n,m}^2),\qquad
\mathrm{SNR}_m=20\log_{10}\frac{
\operatorname{RMS}(x_m[k])}{\sigma_{n,m}},
\]

где RMS чистого канала вычисляется в общем valid region. После GCC три
опорных TDOA подаются в WLS с равными весами. Это не TOA/TDOA Gaussian-модель
предыдущего CRLB-этапа, поэтому signal-level CRLB здесь не вычисляется.

### Численные результаты текущего этапа

Диагностические настройки (не окончательная модель БПЛА): `fs=48000 Гц`,
`f_max={2000,4000,8000,12000} Гц`, пределы 0.1 отсчёта и 0.1 рад,
azimuth `[0°,360°)`, elevation `0…80°`, coarse/refined шаги `0.5°/0.25°`.

Минимальные расстояния по численному поиску, м:

| Геометрия | 0.1 sample | phase 2 kHz | 4 kHz | 8 kHz | 12 kHz |
|---|---:|---:|---:|---:|---:|
| linear | 6.220 | 1.629 | 3.257 | 6.513 | 9.770 |
| L-shaped | 9.809 | 2.570 | 5.137 | 10.272 | 15.406 |
| rectangle 3:1 | 4.285 | 1.180 | 2.283 | 4.484 | 6.682 |
| square | 6.997 | 1.831 | 3.663 | 7.327 | 10.991 |
| tetrahedral | 9.920 | 2.614 | 5.205 | 10.387 | 15.568 |

При `R=10 м` максимальные plane/second-order ошибки, мкс:

| Геометрия | plane | second order | worst direction/pair для plane |
|---|---:|---:|---|
| linear | 1.29576 | 0.01122 | около 270.5°/48°, 1-3 |
| L-shaped | 2.04352 | 0.01422 | 28.5°/0°, 0-3 |
| rectangle 3:1 | 0.88244 | 0.01122 | 225°/0°, 2-3 |
| square | 1.45769 | 0.01122 | 225°/0°, 2-3 |
| tetrahedral | 2.06661 | 0.02062 | 305.5°/30°, 0-2 |

На последних пяти расстояниях sweep fitted slope plane error находится в
`[-1.00053,-0.9999997]`, second-order error — в
`[-2.00002,-1.99977]`. Граница по фазе почти линейно растёт с `f_max`,
как следует из `E_tau~1/R` и порога `E_tau<=0.1/(2*pi*f_max)`.

Фактические максимумы fractional-delay исследования на задержках
`0.1,0.25,0.5,0.75,0.9` отсчёта:

| Реализация | max amplitude error | max phase error, rad | max tone waveform error |
|---|---:|---:|---:|
| frequency-domain | 3.344e-7 | 2.175e-7 | 9.266e-4 |
| windowed-sinc FIR | 1.382e-5 | 8.351e-6 | 1.458e-5 |

Эти maxima относятся к расширенной полосе `1.5…19 кГц` при `fs=48 кГц`
и tone-guard 512 samples. В диагностической полосе до 12 кГц maxima
amplitude/phase равны `2.445e-8/4.623e-8` для frequency-domain и
`4.257e-6/4.588e-6` для FIR. Максимальная ошибка broadband group delay:
`2.956e-12` и `1.659e-6` отсчёта соответственно; максимальное расхождение
методов в общем valid region — `2.023e-6`.

Signal-level GCC/WLS Monte Carlo, усреднение по трём направлениям:

| SNR, dB | square TDOA RMSE, мкс | square geo RMSE, ° | tetra TDOA RMSE, мкс | tetra geo RMSE, ° |
|---:|---:|---:|---:|---:|
| -20 | 396.738 | 71.329 | 516.366 | 87.563 |
| -10 | 158.991 | 27.845 | 223.062 | 39.346 |
| 0 | 4.072 | 1.427 | 4.102 | 0.701 |
| 5 | 2.334 | 0.823 | 2.431 | 0.410 |
| 10 | 1.438 | 0.511 | 1.456 | 0.239 |
| 20 | 0.735 | 0.306 | 0.761 | 0.116 |

Для SNR `>=0 dB` тетраэдр преобразует близкие TDOA-ошибки в меньшую
угловую ошибку. На square в направлении `20°/10°` при `20 dB` остаётся
elevation bias `0.486°`; это наблюдаемый signal/GCC/geometry bias, а не
случайный провал теста. При `-10/-20 dB` ложные корреляционные максимумы
образуют тяжёлые хвосты, поэтому эти режимы характеризуют границу срыва.

Финальная all-pair статистика использует **70 конфигураций ×
(1000 calibration + 2000 evaluation) = 210000** независимых реализаций,
seed `20260828`.
На fine-SNR-сетке первый исследованный SNR, при котором P95 DOA ниже и `5°`,
и `10°`, равен `-6 dB` для обеих решёток и всех четырёх DOA-вариантов; при
`-8 dB` P95 ещё выше `10°`. Для calibrated-варианта RMSE/P95:

| SNR | square | tetrahedral |
|---:|---:|---:|
| -10 dB | 23.063° / 34.417° | 17.486° / 32.619° |
| -6 dB | 1.965° / 3.550° | 1.114° / 1.891° |
| 0 dB | 0.729° / 1.292° | 0.455° / 0.803° |
| 10 dB | 0.262° / 0.490° | 0.160° / 0.277° |
| 30 dB | 0.047° / 0.089° | 0.024° / 0.041° |

Эти числа относятся к hard-P10 и условны по успешным оценкам; coverage для
square/tetrahedral равен `0.9775/0.9735` на `-10 dB`, `0.9965/0.9960` на
`-6 dB` и `0.9965/0.9910` на `0 dB`. Средняя по парам вероятность
catastrophic error `>1 sample` равна `0.2791, 0.1039, 0.0227, 0.00279,
0.000208, 0` для SNR `-10,-8,-6,-4,-2,0 dB`; выше 0 dB также ноль.
Максимальная по парам boundary-hit fraction на тех же уровнях равна
`0.008, 0.0035, 0.0015, 0.0005, 0, 0`; invalid fraction равна нулю на всей
fine-SNR-сетке. DOA failure calibrated-варианта остаётся `0.4…3.2%` из-за
confidence-отбраковки, способной временно разъединить accepted pair graph;
это отдельно от GCC invalid flag.

По всем 840 pair-строкам invalid fraction строго равна нулю; invalid semantics
для тишины/малой энергии отдельно покрыта строгими unit-тестами. Ненулевой
catastrophic fraction имеется в 249 строках, boundary-hit — в 115. Глобальные
максимумы равны `0.728` (harmonic stress, square, calibration, `-10 dB`, пара
2-3) и `0.020` (harmonic stress, tetrahedral, calibration, `-10 dB`, пара
0-1). На evaluation ненулевые случаи ограничены: fine SNR `-10…-2 dB`;
frame study `-5 dB` и единичная доля `0.0005` на `0 dB`; signal comparison
на `-10 dB` для всех сигналов и дополнительно harmonic stress на `0 dB`.
Для deterministic/random signal comparison при `0 dB` и выше, fine study
при `0 dB` и выше, frame study при `5/10 dB`, direction и spherical studies
catastrophic/boundary/invalid fractions равны нулю.

Все шесть пар улучшают низко-/среднешумовую оценку по сравнению с тремя
опорными. Например, при `0 dB` RMSE/P95 square меняется
`0.876°/1.567° → 0.679°/1.239°`, tetrahedral —
`0.780°/1.382° → 0.429°/0.761°`. Equal-weight cycle projection не меняет
DOA относительно raw all-6, но финальный global maximum поля
`mean_cycle_residual_after_us` равен `2.146e-13 мкс`. Hard-P10 confidence
rejection меняет тяжёлый хвост на `-10 dB`: P95 square `51.84°→34.42°`,
tetrahedral `57.56°→32.62°`, ценой `2.25%/2.65%` disconnected failures.

Результат сильно зависит от искусственного сигнала. При `-10 dB` hard-P10
conditional RMSE/P95 random broadband равны `1.55°/2.71°` (square) и
`0.68°/1.18°` (tetrahedral), deterministic multisine — `21.91°/33.53°` и
`18.12°/33.60°`, harmonic stress — `29.89°/46.13°` и `28.47°/50.11°`.
Harmonic stress — только стресс-тест неоднозначных пиков. Средняя Spearman
correlation peak-ratio с абсолютной pair-ошибкой равна соответственно
`-0.044`, `-0.110`, `-0.168`: confidence связан с ошибкой в ожидаемом
направлении, но слабо и не является калиброванной вероятностью.

Risk–coverage на `-10 dB` (coverage / conditional P95):

| Режим | square | tetrahedral |
|---|---:|---:|
| covariance, no rejection | 1.000 / 37.58° | 1.000 / 33.90° |
| hard P05 | 0.996 / 34.60° | 0.993 / 33.01° |
| hard P10 | 0.978 / 34.42° | 0.974 / 32.62° |
| hard P25 | 0.847 / 31.15° | 0.822 / 31.09° |
| hard P50 | 0.444 / 30.54° | 0.435 / 28.20° |
| soft, no deletion | 1.000 / 35.64° | 1.000 / 32.16° |

P50 демонстрирует, почему conditional risk нельзя читать без coverage: modest
P95 improvement покупается потерей более половины реализаций. На `-6 dB`
hard-P10 даёт coverage `0.9965/0.9960`, P95 `3.55°/1.89°`, P99
`5.33°/2.66°`; no-rejection сохраняет coverage 1, но P99 равен
`11.75°/4.93°`, а у tetrahedral P99.9 достигает `121.53°` из-за очень редких
peak swaps. P99.9 при 2000 trials определяется лишь крайними несколькими
наблюдениями и остаётся диагностическим tail-показателем с высокой
статистической неопределённостью.

Frame-length study не показывает монотонного улучшения для фиксированной
периодической multisine. При `0 dB` tetrahedral P95 для `1024/2048/4096/8192`
равны `0.709°/0.742°/0.752°/0.782°`; при `-5 dB` редкие peak swaps дают
RMSE `4.744°/2.152°/1.948°/3.405°`. Этот вывод signal-dependent и не
переносится на реальный БПЛА без измеренного спектра.

Empirical covariance сохраняется с correlation matrix, eigenvalues, rank и
condition number и сравнивается по относительной Frobenius-норме с моделями
independent TDOA и independent TOA. В выбросном режиме её использование в
квадратичном benchmark не объявляется точной CRLB; во всех CSV стоит
`benchmark_name=Gaussian covariance benchmark` и `exact_crlb_claimed=False`.

Для spherical signals при `20 dB` GCC measurement RMSE равна `0.79…0.80 мкс`
(square) и `0.58…0.59 мкс` (tetrahedral). Direction-specific plane mismatch
при `R=5→50 м` падает `1.785→0.179 мкс` и `2.705→0.272 мкс`; global
`E_tau` на сетке выше или равна этой направленной ошибке. Exact known-range
WLS устраняет noiseless model bias до численного нуля. Plane tetrahedral bias
падает `0.0876°→0.00877°`; у square выбранная симметрия компенсирует большую
часть углового смещения, хотя временной mismatch остаётся ненулевым.

Большая pointwise tone error частотной версии по сравнению с её
амплитудно-фазовой ошибкой вызвана остаточным sinc-рингингом конечного
zero-extended тона около границ; кругового переноса нет. Для основного
тестового tapered broadband сигнала согласие двух методов оценивается
отдельной broadband-метрикой.

### Ослабленные допуски и неуспешные проверки текущего этапа

- Финальных проваленных критериев нет. Были два промежуточных сбоя:
  `SyntaxError` в первой версии Markdown-таблицы нового notebook и
  `PermissionError` при лишней перезаписи открытого Monte Carlo CSV. Оба
  исправлены и полностью повторно проверены.
- При сборке статистического notebook были три дополнительные промежуточные
  ошибки только аналитического слоя: неверное имя импортируемой trial-count
  константы, устаревшие имена pair-полей и прямой доступ к cycle-полю,
  отсутствующему у spherical records. Каждый случай исправлен; полный
  210000-trial notebook после исправлений повторён до exit code 0.
- Быстрый pytest использует угловой шаг `20°` для локального `E_tau`-теста
  и `30°` для smoke-study; итоговый notebook использует `0.5°` и отдельно
  сравнивает его с `0.25°` на всех 25 границах. Допуск grid convergence —
  `1e-4`, фактическое максимальное расхождение — `2.79747e-5`.
- Относительный допуск численного поиска расстояния — `2e-6`.
- Pytest-порог amplitude/phase равен `1e-5` на пяти нормированных частотах;
  notebook допускает amplitude error до `2e-5` в расширенной полосе до
  19 кГц, потому что фактический FIR maximum там `1.382e-5`. В основной
  диагностической полосе до 12 кГц фактический maximum `4.257e-6`.
- Допуск broadband group delay — `2e-6` отсчёта; фактический maximum
  `1.659e-6`. Допуск согласия двух реализаций — `3e-6`; фактически
  `2.023e-6`.
- Сквозной допуск сходимости spherical channels к plane при `R=1000 м` —
  `3e-4` по максимальной амплитуде в valid region. Он проверяет генератор,
  а не задаёт физическую границу дальней зоны.
- Pointwise error конечного zero-extended тона записывается в CSV, но не
  используется как самостоятельный pass/fail-критерий из-за sinc-рингинга.
  Контролируемые критерии — phase, amplitude, broadband group delay,
  отсутствие circular wrap и cross-method error.
- Monte Carlo notebook применяет диагностические критерии только к режимам
  `0/20 dB`: при `20 dB` TDOA RMSE < `1 мкс`, geodesic RMSE < `1°` и нет
  ошибок > `10°`; при `0 dB` geodesic RMSE < `3°`. Режимы `-10/-20 dB`
  намеренно не объявляются успешными — они показывают срыв GCC.
- Детерминированный GCC использует interpolation factor 32; массовый Monte
  Carlo — 8, быстрый pytest — 4. Это явное вычислительное ослабление, при
  котором сохраняется параболическое sub-sample уточнение; фактические
  ошибки приведены в CSV. Monte Carlo-сигнал длится `50 ms` и занимает
  диагностическую полосу `300…10000 Hz`.
- Финальный statistical Monte Carlo использует interpolation factor `2` ради
  стоимости 210000 реализаций, но сохраняет параболическое sub-grid уточнение.
  Это явное вычислительное ослабление; знак, прямой reference и высокая точность
  interpolation 32 независимо покрыты детерминированными тестами.
- Для статистических результатов не задан искусственный pass/fail-порог по
  RMSE: режим срыва должен быть измерен, а не «пройден». Приёмочные утверждения
  основаны на фиксированных counts/seeds, покомпонентных tail-метриках и
  численном сравнении вариантов. P95-порог сообщается только на исследованной
  SNR-сетке, поэтому `-6 dB` — первый sampled уровень, а не непрерывно найденная
  физическая граница.
- Confidence rejection использует 10-й percentile calibration peak-ratio и
  эвристическое масштабирование precision. Из-за слабой Spearman-связи порог
  не интерпретируется как вероятность корректной оценки; все вызванные им
  disconnected failures включены в итоговые fractions.

### Изменённые и созданные файлы текущего этапа

- `model/geometry.py`, `model/tdoa.py` — centroid, точные/плоские/второго
  порядка TOA и TDOA.
- `simulation/__init__.py`, `simulation/fractional_delay.py`,
  `simulation/propagation.py`, `simulation/signals.py` — новый пакет
  детерминированного синтеза.
- `validation/far_field.py`, `validation/propagation_study.py`,
  `validation/__init__.py` — сеточный анализ, численный поиск и CSV.
- `estimators/gcc_phat.py` — ориентированный sub-sample GCC-PHAT и пакетная
  оценка выбранных пар.
- `validation/gcc_study.py` — benchmark задержек и детерминированное
  cross-generator исследование GCC.
- `validation/gcc_monte_carlo.py` — фиксированные seed, явное SNR и 7200
  signal-level GCC/WLS испытаний.
- `validation/study.py` — идемпотентная запись воспроизводимого Monte Carlo CSV.
- `tests/test_far_field.py`, `tests/test_fractional_delay.py`,
  `tests/test_propagation.py`, `tests/test_propagation_study.py` — новые тесты.
- `tests/test_gcc_phat.py` — знак, дробные задержки, перестановки и совпадение
  GCC на frequency-domain/Kaiser FIR генераторах.
- `tests/test_gcc_monte_carlo.py` — быстрые проверки воспроизводимости,
  CSV и улучшения TDOA/DOA при росте SNR.
- `tests/test_monte_carlo.py` — тест идемпотентного CSV-writer.
- `notebooks/far_field_fractional_delay_validation.ipynb` — новый notebook;
  два существующих notebook повторно выполнены и сохранены.
- `results/far_field_boundary.csv`,
  `results/fractional_delay_accuracy.csv` — новые численные результаты.
- `results/fractional_delay_benchmark.csv`, `results/gcc_phat_validation.csv`
  — временные измерения и 12 детерминированных GCC-конфигураций.
- `notebooks/gcc_phat_monte_carlo.ipynb`,
  `results/gcc_phat_monte_carlo.csv` — графики и 36 статистических агрегатов.
- `README.md`, `AGENTS.md`, `pyproject.toml`, `PROJECT_STATUS.md` — API,
  соглашения, пакет `simulation` и журнал проверок.
- `estimators/gcc_phat.py`, `estimators/__init__.py` — полная peak/energy
  диагностика, invalid semantics и прямой arbitrary-grid reference.
- `estimators/cycle_projection.py` — weighted cycle projection и явное
  обнаружение disconnected graph.
- `estimators/wls_doa.py` — exact spherical WLS при известном centroid-range.
- `simulation/signals.py`, `simulation/__init__.py` — независимый random
  band-limited broadband и harmonic stress-test.
- `validation/gcc_statistical.py` — all-six-pair calibration/evaluation engine,
  fine-SNR, signal/frame, covariance и spherical model studies.
- `tests/test_gcc_diagnostics.py`, `tests/test_cycle_projection.py`,
  `tests/test_signal_classes.py`, `tests/test_gcc_statistical.py`,
  `tests/test_spherical_wls.py` — новые строгие и быстрые проверки.
- `notebooks/gcc_statistical_validation.ipynb` — полный 210000-trial анализ;
  остальные пять notebooks повторно выполнены и сохранены.
- `results/gcc_pair_error_summary.csv`, `results/gcc_doa_summary.csv`,
  `results/gcc_covariance_summary.csv` — 840/264/70 итоговых агрегатов.

## Предыдущий завершённый этап: ковариация и Monte Carlo WLS/CRLB

### Журнал предыдущего этапа

- 2026-08-27 — начат аудит терминологии и ковариационной алгебры.
- API разделён на `sigma_toa` и `sigma_tdoa`; опорные TDOA названы
  линейно независимыми, но не статистически независимыми в TOA-модели.
- Добавлены тесты структуры ковариации, циклического нулевого пространства и
  инвариантности Fisher/WLS.
- Реализована координатно осмысленная angular CRLB и явная диагностика
  вырожденной информации.
- Реализован воспроизводимый Monte Carlo-движок и быстрые статистические
  pytest-тесты с фиксированными seed.
- Выполнен подробный эксперимент: 42 конфигурации по 2000 реализаций,
  всего **84 000 WLS-оценок**. Результаты сохранены в
  `results/monte_carlo_crlb_summary.csv`.
- Промежуточный полный прогон после последней правки:
  **50 passed in 2.32s**.
- В проектном `.venv` обнаружен отсутствующий build-пакет `wheel`;
  установлен `wheel==0.48.0`, версия закреплена в `requirements.txt`, а
  editable-install после этого успешно повторён. Устаревший игнорируемый
  каталог `uav_acoustic_model.egg-info` удалён.
- Финальная проверка: **50 passed in 2.21s**; оба notebook выполнены с
  кодом возврата 0; у каждого валидная схема `nbformat`, `0` error-outputs
  и `0` невыполненных непустых code-ячеек; CSV содержит 42 конфигурации,
  84 000 испытаний и конечные итоговые метрики во всех строках;
  `pip check` сообщает `No broken requirements found`.

## Реализованные математические формулы

Направление от центра решётки к источнику:

\[
\mathbf u(\varphi,\varepsilon)=
\begin{bmatrix}
\cos\varepsilon\cos\varphi\\
\cos\varepsilon\sin\varphi\\
\sin\varepsilon
\end{bmatrix},\qquad \|\mathbf u\|=1.
\]

Соглашение TDOA, сферическая и дальнепольная модели:

\[
\tau_{ij}=T_i-T_j,
\qquad
\tau_{ij}(\mathbf q)=
\frac{\|\mathbf q-\mathbf r_i\|-\|\mathbf q-\mathbf r_j\|}{c},
\qquad
\tau_{ij}(\varphi,\varepsilon)=
\frac{(\mathbf r_j-\mathbf r_i)^\mathsf T\mathbf u}{c}.
\]

Аналитический Якобиан:

\[
\mathbf u_\varphi=
\begin{bmatrix}
-\cos\varepsilon\sin\varphi\\
\cos\varepsilon\cos\varphi\\
0
\end{bmatrix},
\quad
\mathbf u_\varepsilon=
\begin{bmatrix}
-\sin\varepsilon\cos\varphi\\
-\sin\varepsilon\sin\varphi\\
\cos\varepsilon
\end{bmatrix},
\]

\[
H_{ij,:}=\frac{1}{c}
\begin{bmatrix}
(\mathbf r_j-\mathbf r_i)^\mathsf T\mathbf u_\varphi &
(\mathbf r_j-\mathbf r_i)^\mathsf T\mathbf u_\varepsilon
\end{bmatrix}.
\]

### Две разные шумовые модели

**A. Независимые ошибки TOA**

\[
e_m\sim\mathcal N(0,\sigma_{\rm toa}^2),\qquad
\boldsymbol\eta_{\rm tdoa}=\mathbf B\mathbf e,qquad
\mathbf\Sigma_{\rm tdoa}
=\mathbf B\mathbf\Sigma_{\rm toa}\mathbf B^\mathsf T.
\]

Здесь `sigma_toa` — стандартное отклонение ошибки одного времени прихода.
При \(\mathbf\Sigma_{\rm toa}=\sigma_{\rm toa}^2\mathbf I\) и
опорных парах относительно одного микрофона:

\[
(\Sigma_{\rm tdoa})_{kk}=2\sigma_{\rm toa}^2,qquad
(\Sigma_{\rm tdoa})_{k\ell}=\sigma_{\rm toa}^2\;(k\ne\ell),
\qquad \rho_{k\ell}=\tfrac12.
\]

Опорные \(M-1\) TDOA линейно независимы, но статистически коррелированы.
Для полного набора пар
\(\operatorname{rank}(\Sigma_{\rm tdoa})=M-1\). Если строки
\(\mathbf C\) кодируют циклы, то
\(\mathbf C\mathbf B=0\),
\(\mathbf C\boldsymbol\tau=0\) и
\(\Sigma_{\rm tdoa}\mathbf C^\mathsf T=0\).

**B. Независимые непосредственно измеренные TDOA**

\[
\boldsymbol\eta\sim
\mathcal N(0,\sigma_{\rm tdoa}^2\mathbf I).
\]

Здесь `sigma_tdoa` — стандартное отклонение ошибки одной непосредственно
измеренной разности. Это абстрактная модель, не равная TOA-индуцированной
ковариации. Значение по умолчанию `sigma_tdoa = 50 мкс` относится только
к этой модели.

### Fisher, CRLB и WLS

\[
\mathbf F=\mathbf H^\mathsf T
\mathbf\Sigma_{\rm tdoa}^{+}\mathbf H,
\qquad
\operatorname{Cov}(\widehat{\boldsymbol\theta})
\succeq \mathbf F^{-1}
\]

для полного ранга `F`; знак \(+\) нужен для избыточного полного набора
пар с сингулярной TOA-индуцированной ковариацией. При вырождении конечная
общая angular CRLB не возвращается: выдаются собственные значения,
координатное нулевое направление и то же направление в локальном
ортонормированном касательном базисе.

Координатно осмысленная локальная метрика:

\[
\operatorname{angular\_crlb\_rms}
=\sqrt{\cos^2\varepsilon\,C_{\varphi\varphi}
+C_{\varepsilon\varepsilon}}.
\]

Результат возвращается в радианах и градусах. Обычное матричное обращение
для вырожденной `F` не используется: ранг проверяется до
`numpy.linalg.solve`.

Ограниченный WLS:

\[
\widehat{\mathbf u}=
\arg\min_{\|\mathbf u\|=1,\,u_z\ge0}
(\widehat{\boldsymbol\tau}-\mathbf G\mathbf u)^\mathsf T
\mathbf W
(\widehat{\boldsymbol\tau}-\mathbf G\mathbf u).
\]

## Проверка ковариации и инвариантности

Автоматически подтверждено:

- диагональ опорной TOA-индуцированной ковариации равна
  \(2\sigma_{\rm toa}^2\), внедиагональные элементы —
  \(\sigma_{\rm toa}^2\), корреляции — \(1/2\);
- ранг полной попарной ковариации при `M=4` равен `3`;
- её численное нулевое пространство совпадает с пространством циклических
  ограничений;
- Fisher information совпадает для любого опорного микрофона, полного
  набора пар с псевдообратной, изменения ориентации пар, перестановки
  микрофонов и общего переноса решётки;
- максимальная относительная ошибка Fisher среди этих преобразований:
  `3.723e-15` для квадрата и `3.712e-15` для тетраэдра
  (тестовый допуск `2e-12`);
- WLS инвариантен для тех же представлений при одной и той же реализации
  исходных TOA-ошибок; максимальное покомпонентное расхождение направления:
  `6.630e-10` для квадрата и `1.226e-10` для тетраэдра
  (допуск `2e-9`).

## Monte Carlo-валидация

Фиксированный базовый seed: `20260827`; независимые воспроизводимые
подпотоки создаются через `numpy.random.SeedSequence`.

- Решётки: квадратная и тетраэдрическая.
- Направления: `(20°, 10°)`, `(45°, 30°)`, `(120°, 50°)`.
- Независимые TDOA: `sigma_tdoa = 1, 5, 10, 20, 50, 100 мкс`.
- Независимые TOA:
  `sigma_toa = 50/sqrt(2) = 35.355339 мкс`, поэтому стандартное
  отклонение каждой отдельной разности равно `50 мкс`.
- Подробный notebook: `N=2000` на конфигурацию, 42 конфигурации,
  **84 000 реализаций**.
- Быстрый pytest: `N=256` для шести малошумовых конфигураций.
- Elevation `0°` не используется для прямого сравнения с классической
  CRLB.

Для каждой конфигурации сохранены bias по двум координатам, эмпирическая
ковариация, geodesic RMSE/median/P95, доля ошибок больше 10°, angular CRLB
RMS, отношение RMSE/CRLB, нормированная ковариация, доля успешных
оптимизаций и доля решений на границе `elevation=0`.

### Малый шум: численное соответствие CRLB

При `sigma_tdoa=1 мкс` все 6 конфигураций прошли подробный критерий:

- `RMSE/CRLB = 0.9774…1.0096`;
- собственные значения
  \(C_{\rm CRLB}^{-1/2}C_{\rm emp}C_{\rm CRLB}^{-1/2}\):
  `0.9348…1.0387`;
- максимальный модуль координатного bias, нормированного на CRLB-std:
  `0.0382`;
- все оценки оптимизатора успешны, попаданий на границу нет.

| Решётка | Направление | RMSE, ° | CRLB RMS, ° | RMSE/CRLB |
|---|---:|---:|---:|---:|
| квадрат | 20°/10° | 0.5518 | 0.5538 | 0.9963 |
| квадрат | 45°/30° | 0.2138 | 0.2123 | 1.0070 |
| квадрат | 120°/50° | 0.1964 | 0.1968 | 0.9978 |
| тетраэдр | 20°/10° | 0.1708 | 0.1711 | 0.9983 |
| тетраэдр | 45°/30° | 0.1909 | 0.1953 | 0.9774 |
| тетраэдр | 120°/50° | 0.1620 | 0.1605 | 1.0096 |

### Нелинейный и граничный режим

- Квадрат `20°/10°` наиболее чувствителен: при `10 мкс` bias уже
  `1.449°` и 15.25% решений лежат на границе; при `50 мкс`
  RMSE `12.999°`, CRLB `27.692°`, отношение `0.469`, bias `1.498°`,
  граница 34.3%; при `100 мкс` RMSE `19.732°`, bias `3.214°`,
  граница 44.05%.
- Для квадрата `45°/30°` и `120°/50°` WLS близок к CRLB примерно до
  `20 мкс`; при `100 мкс` bias составляет соответственно
  `6.553°` и `4.755°`.
- Тетраэдр остаётся значительно равномернее. При `50 мкс` отношения
  RMSE/CRLB равны `0.975…1.028`; при `100 мкс` — `0.900…1.039`.
  Самый большой bias тетраэдра — `2.204°` для `20°/10°` при
  `100 мкс`.
- Отношение меньше единицы в сильном шуме не является нарушением CRLB:
  ограниченный WLS становится смещённым, а решения на границе параметров
  нарушают регулярные условия классической границы.

Для TOA-модели с маргинальным std каждой разности `50 мкс`:

| Решётка | Направление | RMSE, ° | CRLB RMS, ° | RMSE/CRLB | Bias, ° |
|---|---:|---:|---:|---:|---:|
| квадрат | 20°/10° | 12.8346 | 28.7169 | 0.4469 | 1.0841 |
| квадрат | 45°/30° | 12.7515 | 10.9861 | 1.1607 | 2.3119 |
| квадрат | 120°/50° | 8.3007 | 8.0792 | 1.0274 | 0.8975 |
| тетраэдр | 20°/10° | 6.9224 | 6.9482 | 0.9963 | 0.1147 |
| тетраэдр | 45°/30° | 6.9341 | 6.9482 | 0.9980 | 0.3053 |
| тетраэдр | 120°/50° | 7.0717 | 6.9482 | 1.0178 | 0.2620 |

Тетраэдрическая решётка даёт почти изотропную CRLB и гораздо более
стабильное соответствие WLS границе. Плоский квадрат сохраняет зеркальную
неоднозначность относительно своей плоскости; ограничение верхней
полусферы выбирает одну ветвь, но не устраняет ухудшение около горизонта.

## Критерии и их статус

Подробный малошумовой критерий notebook:

- RMSE/CRLB в `[0.90, 1.10]`;
- собственные значения нормированной ковариации в `[0.80, 1.20]`;
- модуль каждого координатного bias не больше `0.15` CRLB-std.

Все 6 проверок прошли. Проваленных малошумовых критериев нет.

Быстрый pytest намеренно использует менее жёсткие интервалы из-за
`N=256`: `[0.80, 1.20]`, `[0.65, 1.45]` и bias не больше
`0.30` CRLB-std. Это единственные ослабленные статистические критерии.
Высокошумовые случаи не маркируются как «провал CRLB», потому что там
наблюдаются смещение и активная граница параметрического пространства.

## Геометрические вырождения

Все сравниваемые решётки имеют `M=4`, максимальную апертуру `D=0.20 м`.

| Геометрия | Аффинный ранг | Наблюдаемость |
|---|---:|---|
| линейная | 1 | два угла не наблюдаются на всей сетке |
| L-образная | 2 | локальное вырождение на горизонте; зеркальная неоднозначность |
| прямоугольная 3:1 | 2 | локальное вырождение на горизонте; зеркальная неоднозначность |
| квадратная | 2 | локальное вырождение на горизонте; зеркальная неоднозначность |
| тетраэдрическая | 3 | полный локальный ранг на сетке elevation 0–60° |

## Изменённые и созданные файлы предыдущего этапа

- `model/geometry.py` — точная терминология опорных пар.
- `model/tdoa.py` — incidence/cycle-алгебра и обе ковариационные модели.
- `model/statistics.py` — Fisher с полной/сингулярной ковариацией,
  диагностика и angular CRLB.
- `estimators/wls_doa.py` — явные модели веса и инвариантный WLS.
- `validation/__init__.py`, `validation/monte_carlo.py`,
  `validation/study.py` — воспроизводимый Monte Carlo и CSV.
- `tests/test_geometry_tdoa.py`, `tests/test_jacobian_statistics.py`,
  `tests/test_wls_doa.py` — миграция API и дополнительные проверки.
- `tests/test_covariance_invariance.py` — ковариация, циклы,
  Fisher/WLS-инвариантность.
- `tests/test_monte_carlo.py` — быстрые статистические проверки.
- `notebooks/array_comparison.ipynb` — новая терминология и новая
  angular CRLB.
- `notebooks/monte_carlo_crlb_validation.ipynb` — подробный эксперимент
  и графики.
- `results/monte_carlo_crlb_summary.csv` — 42 строки агрегатов.
- `README.md`, `AGENTS.md`, `pyproject.toml`, `requirements.txt`,
  `PROJECT_STATUS.md` — документация и конфигурация.

## Команды проверки

Из каталога `uav_acoustic_model`:

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
& ".\.venv\Scripts\python.exe" -m pip install -e . --no-build-isolation
& ".\.venv\Scripts\python.exe" -m pytest -q

& ".\.venv\Scripts\python.exe" -c `
  "from validation.srp_statistical import run_srp_statistical_validation; run_srp_statistical_validation()"

& ".\.venv\Scripts\python.exe" -c `
  "from validation.moving_source_study import run_moving_source_study; run_moving_source_study()"

& ".\.venv\Scripts\python.exe" -m jupyter nbconvert `
  --to notebook --execute --inplace `
  --ExecutePreprocessor.timeout=300 `
  ".\notebooks\array_comparison.ipynb"

& ".\.venv\Scripts\python.exe" -m jupyter nbconvert `
  --to notebook --execute --inplace `
  --ExecutePreprocessor.timeout=1200 `
  ".\notebooks\monte_carlo_crlb_validation.ipynb"

& ".\.venv\Scripts\python.exe" -m jupyter nbconvert `
  --to notebook --execute --inplace `
  --ExecutePreprocessor.timeout=1200 `
  ".\notebooks\far_field_fractional_delay_validation.ipynb"

& ".\.venv\Scripts\python.exe" -m jupyter nbconvert `
  --to notebook --execute --inplace `
  --ExecutePreprocessor.timeout=600 `
  ".\notebooks\gcc_phat_validation.ipynb"

& ".\.venv\Scripts\python.exe" -m jupyter nbconvert `
  --to notebook --execute --inplace `
  --ExecutePreprocessor.timeout=1200 `
  ".\notebooks\gcc_phat_monte_carlo.ipynb"

& ".\.venv\Scripts\python.exe" -m jupyter nbconvert `
  --to notebook --execute --inplace `
  --ExecutePreprocessor.timeout=3600 `
  ".\notebooks\gcc_statistical_validation.ipynb"

& ".\.venv\Scripts\python.exe" -m jupyter nbconvert `
  --to notebook --execute --inplace `
  --ExecutePreprocessor.timeout=3600 `
  ".\notebooks\srp_phat_validation.ipynb"

& ".\.venv\Scripts\python.exe" -m jupyter nbconvert `
  --to notebook --execute --inplace `
  --ExecutePreprocessor.timeout=1200 `
  ".\notebooks\moving_source_validation.ipynb"

& ".\.venv\Scripts\python.exe" -m jupyter nbconvert `
  --to notebook --execute --inplace `
  --ExecutePreprocessor.timeout=1200 `
  ".\notebooks\moving_source_3d.ipynb"

& ".\.venv\Scripts\python.exe" -m pip check
```

## Итоговая проверка артефактов

- `pytest`: **196 passed in 14.07s** после выполнения всех notebook.
- `notebooks/array_comparison.ipynb`: `nbconvert` exit code 0,
  `nbformat` schema valid, error-outputs — **0**, невыполненных непустых
  code-ячеек — **0**.
- `notebooks/monte_carlo_crlb_validation.ipynb`: `nbconvert` exit code 0,
  `nbformat` schema valid, error-outputs — **0**, невыполненных непустых
  code-ячеек — **0**.
- `notebooks/far_field_fractional_delay_validation.ipynb`: `nbconvert`
  exit code 0, 16/16 ячеек имеют ID, `nbformat` schema valid,
  error-outputs — **0**, невыполненных непустых code-ячеек — **0**.
- `notebooks/gcc_phat_validation.ipynb`: `nbconvert` exit code 0,
  11/11 ячеек имеют ID, `nbformat` schema valid, error-outputs — **0**,
  невыполненных code-ячеек — **0**.
- `notebooks/gcc_phat_monte_carlo.ipynb`: `nbconvert` exit code 0,
  9/9 ячеек имеют ID, `nbformat` schema valid, error-outputs — **0**,
  невыполненных code-ячеек — **0**.
- `notebooks/gcc_statistical_validation.ipynb`: `nbconvert` exit code 0,
  17/17 ячеек имеют ID, `nbformat` schema valid, error-outputs — **0**,
  невыполненных code-ячеек — **0**; полный Monte Carlo выполнен внутри notebook.
- `notebooks/srp_phat_validation.ipynb`: `nbconvert` exit code 0, 10/10 ячеек
  имеют уникальные ID, `nbformat` schema valid, error-outputs — **0**,
  невыполненных непустых code-ячеек — **0**; проверяет принятые полные CSV.
- `notebooks/moving_source_validation.ipynb`: `nbconvert` exit code 0,
  `nbformat` schema valid, error-outputs — **0**, невыполненных непустых
  code-ячеек — **0**; проверяет 6480 moving-source aggregate rows.
- `notebooks/moving_source_3d.ipynb`: `nbconvert` exit code 0, `nbformat`
  schema valid, error-outputs — **0**, невыполненных непустых code-ячеек —
  **0**; 13 независимых кадров и интерактивные bearing-ray controls.
- `results/monte_carlo_crlb_summary.csv`: **42** строки конфигураций,
  **84 000** испытаний, конечные RMSE/CRLB-метрики в **42/42** строках,
  подробный малошумовой критерий пройден в **6/6** конфигурациях.
- `results/far_field_boundary.csv`: **80** строк; 55 точек sweep и 25
  refined-границ, каждая удовлетворяет своему порогу; максимальное
  coarse/refined расхождение `2.79747e-5 < 1e-4`.
- `results/fractional_delay_accuracy.csv`: **80** строк; 70 tone и 10
  broadband измерений, все контролируемые метрики конечны и проходят пороги.
- `results/fractional_delay_benchmark.csv`: **4** строки; FIR быстрее
  frequency-domain в `2.86x` на 2400 и `19.51x` на 12000 отсчётах.
- `results/gcc_phat_validation.csv`: **12** строк; максимальные ошибки
  frequency/FIR `5.031e-7/5.868e-7` отсчёта, cross-generator maximum
  `1.340e-7` отсчёта.
- `results/gcc_phat_monte_carlo.csv`: **36** строк, **7200** испытаний,
  фиксированные seed и конечные TDOA/DOA/tail-метрики во всех строках.
- `results/gcc_pair_error_summary.csv`: **840** строк, все шесть пар для
  каждой из 140 configuration/split-групп.
- `results/gcc_doa_summary.csv`: **574** строки evaluation-метрик, включая
  **372** risk–coverage строки и отдельное spherical exact/plane сравнение.
- `results/gcc_covariance_summary.csv`: **70** calibration-ковариаций;
  **210000** суммарных calibration/evaluation испытаний, seeds выборок раздельны.
- `results/srp_doa_summary.csv`: **792** строки, четыре метода в каждой из
  **198** конфигураций, **297000** calibration/evaluation реализаций.
- `results/srp_runtime_summary.csv`: **594** component-строки; сумма
  unique-count contribution равна **594 sampled exact-reference trials**
  (3 из 1000 evaluation trials/config), maximum sampled exact/fast
  disagreement `0.031365°`.
- `results/moving_source_summary.csv`: **6480** строк, **2160** конфигураций,
  **43200** paired trials; все обязательные RMSE/P95/P99/diagnostics конечны.
- Все **9/9** notebook: schema valid, error-output **0**, невыполненных
  непустых code-ячеек **0**; `notebooks/results/` отсутствует.
- Окружение: editable-install успешен; `pip check` —
  `No broken requirements found`.
- Во время Jupyter на Windows остаются только нефатальные предупреждения
  ZMQ о selector thread и локальном TCP-транспорте ядра; output типа
  `error` в notebook отсутствует.

### Формула и численный итог SRP-PHAT

Для ориентированной пары `(i,j)` используется
`Psi_ij[k] = X_i[k] conj(X_j[k]) / |X_i[k] conj(X_j[k])|` и
`tau_ij(u) = (r_j-r_i)^T u/c`. Равновесный six-pair score:

`S(u) = (1/6) sum_(i,j) Re sum_k w_k Psi_ij[k] exp(+j 2 pi f_k tau_ij(u))`.

Знак `+` в steering компенсирует фазу `-2 pi f tau_ij` cross-spectrum и
согласован с `tau_ij=T_i-T_j`. Вес каждой пары одинаков; `w_k` учитывает
одностороннее rFFT-представление, а GCC peak-ratio confidence не используется.

При усреднении по трём сигналам и трём направлениям SRP RMSE/P95 для
square/tetrahedral равны: на `-10 dB` `10.500°/20.404°` и
`9.745°/18.346°`; на `-6 dB` `3.308°/6.015°` и `1.971°/3.245°`; на
`0 dB` `1.580°/2.880°` и `0.648°/1.123°`; на `30 dB`
`0.916°/1.402°` и `0.234°/0.432°`. Тетраэдр в среднем точнее, особенно
в среднем/высоком SNR, но отдельные signal/direction cases не подчиняются
универсальному порядку. Worst SRP boundary fraction `0.302` наблюдается для
square/harmonic-stress/20°/10° при `-8 dB`; worst fraction `>30°` — `0.155`
для той же геометрии/сигнала/направления при `-10 dB`. Это диагностический
AWGN/synthetic-signal результат, не полевая характеристика БПЛА.

Ослабления и оговорки текущего этапа: статистический тест не требует, чтобы
SRP всегда превосходил GCC+WLS; exact/fast regression threshold установлен
`0.05°` и фактический максимум `0.031365°`; regression-тест finite-window
low-elevation main-lobe допускает `0.25°` после измеренного смещения `0.183°`
и проверяет устранение прежней `27°` ошибки, а не нулевой finite-crop bias.
Operational confidence/SNR threshold не выбран. Monte Carlo использует
диагностические AWGN и синтетические сигналы, не отражения/ветер/движение.

### Изменённые файлы текущего этапа

- `estimators/srp_phat.py`, `estimators/__init__.py` — direct/vectorized SRP,
  coarse-to-fine/local refinement, invalid/score/boundary/runtime API.
- `validation/gcc_statistical.py` — conditional tails, counts и risk–coverage;
  `validation/srp_statistical.py`, `validation/__init__.py` — paired study,
  common random numbers, раздельные seeds и exact/fast audit.
- `tests/test_gcc_statistical.py`, `tests/test_srp_phat.py` — отчётная схема,
  risk–coverage, SRP invariances, ambiguity, invalid и statistical smoke tests.
- `notebooks/gcc_statistical_validation.ipynb`,
  `notebooks/srp_phat_validation.ipynb` — выполненные отчёты Part A/B.
- `results/gcc_pair_error_summary.csv`, `results/gcc_doa_summary.csv`,
  `results/gcc_covariance_summary.csv`, `results/srp_doa_summary.csv`,
  `results/srp_runtime_summary.csv` — итоговые таблицы.
- `README.md`, `AGENTS.md`, `PROJECT_STATUS.md` — API, ограничения, команды и
  численная приёмка. Stale-каталог `notebooks/results/` с тремя дублями удалён.

## Изменённые и созданные файлы этапа движения

- `simulation/trajectory.py`, `simulation/moving_source.py`,
  `simulation/__init__.py` — кинематика, retarded-time solver и синтез.
- `validation/moving_source_study.py`, `validation/__init__.py` — gates,
  paired Monte Carlo и independent frame-wise GCC/WLS/SRP.
- `visualization/moving_scene.py`, `visualization/__init__.py` — 3D bearing
  rays и интерактивные controls.
- `tests/test_trajectory.py`, `tests/test_moving_source.py`,
  `tests/test_moving_source_study.py`, `tests/test_moving_scene.py` — новые
  deterministic/statistical/visualization tests.
- `validation/srp_statistical.py`, `tests/test_srp_phat.py`,
  `results/srp_runtime_summary.csv`, `notebooks/srp_phat_validation.ipynb` —
  однозначный unique exact-trial reporting.
- `notebooks/moving_source_validation.ipynb`,
  `notebooks/moving_source_3d.ipynb`, `results/moving_source_summary.csv` —
  новые выполненные отчёты и 6480 агрегатов.
- Все остальные семь notebook программно перевыполнены; их outputs
  обновлены. GCC/far-field/Monte Carlo CSV были воспроизводимо пересчитаны
  соответствующими notebook и сохранили ожидаемые размеры.
- `README.md`, `AGENTS.md`, `PROJECT_STATUS.md`, `pyproject.toml`,
  `requirements.txt` — API, ограничения, зависимости и команды.

## Нерешённые вопросы и допущения

- Экспериментально выбрать реальный `f_max` и допустимую временную/фазовую
  ошибку БПЛА; текущие границы являются настраиваемой диагностикой.
- Оценить `sigma_toa`, `sigma_tdoa` и полную TDOA-ковариацию по
  реальным данным; гауссовость и независимость пока модельные допущения.
- Проверить влияние ошибок координат, синхронизации, температуры, ветра и
  пространственно неоднородной скорости звука.
- Расширить реализованную exact retarded-time kinematic model in a homogeneous
  stationary medium на неизвестное время
  излучения, отражения, многолучёвость, коррелированный шум и дополнительные
  источники.
- Сопоставить условную TDOA-CRLB с полной границей, включающей неизвестный
  сигнал и мешающие параметры.
- Провести лабораторную и полевую проверку с независимым ground truth.
- Signal-level Monte Carlo использует независимый белый Gaussian noise и три
  искусственных класса сигналов: deterministic multisine, независимый random
  broadband и harmonic stress-test. Ни один из них не является измеренной
  моделью БПЛА. Нужны измеренные спектры, цветной/коррелированный фон,
  SNR-калибровка и независимый ground truth.
- Реализована exact retarded-time kinematic model in a homogeneous stationary
  medium и независимые покадровые
  equal-weight far-field SRP/GCC bearings. SRP-Harmonics, отражения, ветер,
  коррелированный фон и EKF/UKF tracking остаются вне этапа.
