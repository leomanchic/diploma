# Модель акустического определения направления на БПЛА

Воспроизводимая геометрическая TDOA-модель для четырёхмикрофонных решёток. Реализация охватывает точную сферическую модель, приближение плоской волны, аналитический Якобиан, условную CRLB согласованной гауссовской TDOA-модели и ограниченный взвешенный МНК для направления прихода.

## Соглашения

- координаты и апертура — в метрах, TDOA — в секундах, внутренние углы — в радианах;
- `phi` — азимут, `elevation` — угол места над плоскостью `xy`;
- вектор `u` направлен от центра решётки к источнику;
- ориентированная пара `(i, j)` означает `tau_ij = T_i - T_j`;
- `c = 343 м/с`;
- `sigma_toa` — стандартное отклонение ошибки одного TOA `e_m`;
- `sigma_tdoa` — стандартное отклонение абстрактной независимой ошибки непосредственно измеренного TDOA; для этой модели значение по умолчанию равно `50 мкс`;
- основной набор — `M-1` линейно независимых пар относительно микрофона 0; в TOA-индуцированной модели они статистически коррелированы.

Используемые формулы:

```text
u(phi, elevation) = [cos(elevation) cos(phi),
                     cos(elevation) sin(phi),
                     sin(elevation)]^T

tau_ij(q) = (||q-r_i|| - ||q-r_j||) / c
tau_ij(phi, elevation) = (r_j-r_i)^T u / c
F = H^T Sigma_tau^{-1} H
```

Для независимых TOA `e ~ N(0, sigma_toa² I)` и `Sigma_tdoa = B Sigma_toa B^T`. Для опорных пар диагональ равна `2 sigma_toa²`, внедиагональные элементы — `sigma_toa²`, а корреляция — `1/2`. Для полного набора пар эта ковариация имеет ранг `M-1`; библиотека применяет спектральную псевдообратную матрицу только при явном разрешении. Это отдельная модель от `eta ~ N(0, sigma_tdoa² I)`.

Для направленного источника расстояние отсчитывается от centroid решётки:
`q = centroid(r) + R u`. Плоские относительные TOA определены как
`T_m = -(r_m-centroid)^T u/c`, поэтому `T_i-T_j` точно совпадает с принятой
формулой TDOA. Отдельно реализовано разложение второго порядка
`R-u^T rho_m + (||rho_m||²-(u^T rho_m)²)/(2R)`.

Положительная дробная задержка означает `y(t)=x(t-delay)` и сдвигает сигнал
вправо. Высокоточный frequency-domain метод с большими нулевыми guard-
интервалами остаётся независимым эталоном; 129-tap Kaiser-windowed sinc FIR
после cross-generator проверки разрешён для массового синтеза. Фиксированная
FIR latency 64 samples компенсируется, а valid region и применённые
неокруглённые задержки возвращаются явно.

GCC-PHAT для ориентированной пары формирует `X_i*conj(X_j)`, поэтому
положительная оценка означает `tau_ij=T_i-T_j>0`. Корреляция zero-padded,
TDOA оценивается на 32-кратно интерполированной сетке и уточняется локальной
параболой; целочисленного квантования задержки нет. Оценка возвращает значение
и кривизну пика, отношение первого пика ко второму, boundary/invalid-признаки,
причину invalid и использованную спектральную энергию. Тишина и недостаточная
энергия дают `delay=NaN`, а не произвольную задержку. Независимый медленный
эталон напрямую вычисляет `sum_k Psi[k] exp(j 2 pi f_k tau)` на произвольной
сетке задержек.

Отдельный signal-level Monte Carlo добавляет только независимый Gaussian
noise по каналам/отсчётам. SNR задаётся как
`20 log10(valid-region signal RMS / noise std)` отдельно для каждого канала.
Это диагностическое допущение, не измеренная шумовая модель БПЛА. После
подтверждённого ускорения и детерминированного согласия массовый синтез
использует Kaiser FIR; frequency-domain метод остаётся эталоном.
Полная статистическая проверка GCC использует все шесть пар, раздельные наборы
`1000` calibration и `2000` evaluation реализаций, weighted cycle projection
и девять DOA/risk–coverage вариантов: без отбрасывания, hard P05/P10/P25/P50
и soft weighting. Метрики успешных оценок явно называются `conditional_*` и
всегда сопровождаются coverage, числом failures, P99/P99.9 и долями ошибок
больше 5°/10°/30°. Эмпирическая covariance в режимах с выбросами называется
`Gaussian covariance benchmark`, а не точной CRLB. Найденный на дискретной
SNR-сетке P95-порог −6 dB не объявляется operational threshold без отдельно
утверждённых критериев coverage/tails/failures.

Far-field SRP-PHAT использует тот же ориентированный cross-spectrum, valid
region, frequency mask и шесть пар. Прямая reference-формула и независимая
векторизованная реализация проверяют друг друга; основной поиск использует
полуоткрытую azimuth-сетку 5°→1°→0.25° и непрерывное локальное уточнение.
Веса всех шести пар равны: GCC peak-ratio confidence в SRP не переносится.
Тишина/недостаточная энергия дают `invalid`, а результат содержит score,
boundary flag и runtime diagnostics.

Итоговый paired SRP study использует base seed `20260829`, по `500`
calibration и `1000` evaluation реализаций в каждой из 198 конфигураций
(297000 реализаций). Первые три evaluation trials каждой конфигурации
проверяются exact vectorized SRP; максимум по этим **594 sampled exact trials**
составил `0.031365°`. Он не относится ко всем 198000 evaluation trials.
Runtime CSV хранит unique-count contribution только в exact-component строке,
поэтому сумма `exact_reference_trial_count` равна 594, а не тройному счёту.
Эти counts относятся к SRP study; полный GCC reporting
study отдельно использует seed `20260828` и `1000/2000` trials.

Для движущегося источника реализована **exact retarded-time kinematic model
in a homogeneous stationary medium** с запаздывающим временем
`t = t_e + ||q(t_e)-r_m||/c`. Общий Newton/Brent solver и независимое
аналитическое constant-velocity решение проверяют друг друга. Поскольку
`d/dt_e = 1+v_r/c>0` при `|v|<c`, корень единственен и причинен; естественный
Doppler появляется через `dt_e/dt=1/(1+v_r/c)`. Fractional Kaiser-sinc time
warp вычисляет `s(t_e)` без округления. Отдельный `frozen_delay` служит только
диагностическим baseline.

Покадровый paired study использует base seed `20260830`, 2160 конфигураций и
20 moving/static пар на конфигурацию (43200 пар, 86400 кадров). Истинный DOA
кадра берётся в centroid emission time, а matched static получает тот же
исходный сигнал и тот же массив шума. Это последовательность независимых
GCC/WLS и equal-weight SRP-PHAT bearings, **не tracking**. P99 при 20 trials
сохраняется как диагностический sampled tail и не считается устойчивым
operational quantile. Clean-signal seed общий для одинаковых факторов при
разных SNR, noise seeds раздельны. CSV различает nominal, expected-effective
и mean realized effective moving/static SNR. Общий frontend всегда вычисляет
все шесть GCC-пар; runtime отдельно хранит его стоимость и backend оценивателя,
а reference-3 boundary и backend используют только три опорные пары.

Continuous-stream этап синтезирует один source waveform, один непрерывный
набор микрофонных каналов и одну noise matrix на всю последовательность.
Перекрывающиеся frames являются views этого общего массива: общие 768
отсчётов при `frame_length=1024`, `hop_length=256` совпадают точно и не
синтезируются повторно. Chunked Kaiser-sinc режим с блоком 4096 ограничивает
интерполяционный working set величиной `4096*129=528384` коэффициента и
совпадает с monolithic режимом до `3e-12` по channels и `2e-15 s` по
emission times/delays.

Основной sequential study использует `fs=48000 Hz`, duration `0.25 s`, 12000
reception samples и 43 frame на каждую из семи последовательностей. Истина
каждого frame вычисляется в centroid emission time, соответствующем
геометрическому центру reception frame. Отдельно сохраняются physical
propagation delay, center-to-end acquisition latency, shared GCC frontend,
estimator backend, available timestamp и total latency. Estimator получает
только текущий frame, геометрию и `fs`; truth и будущие bearing-оценки ему не
передаются. Это **sequential independent bearings, not tracking**. Число
перекрывающихся frames не называется числом независимых trials.

Вырожденная матрица Фишера не обращается: `conditional_crlb` поднимает `DegenerateInformationError`, а `conditional_angular_crlb` возвращает собственные значения и ненаблюдаемые локальные направления без конечной общей angular CRLB. Для полного ранга используется метрически корректная величина `sqrt(cos(elevation)² C_phi_phi + C_elevation_elevation)` одновременно в радианах и градусах.

## Сравниваемые решётки

Все конфигурации содержат `M=4` микрофона и имеют максимальную апертуру ровно `D=0.20 м`:

| Решётка | Аффинный ранг | Глобальная идентифицируемость |
|---|---:|---|
| линейная | 1 | два угла не восстанавливаются |
| L-образная | 2 | зеркальная неоднозначность относительно `xy`; на горизонте локально вырождена |
| прямоугольная 3:1 | 2 | зеркальная неоднозначность относительно `xy`; на горизонте локально вырождена |
| квадратная | 2 | зеркальная неоднозначность относительно `xy`; на горизонте локально вырождена |
| тетраэдрическая | 3 | верх/низ различимы; на рассматриваемой сетке 0–60° полный локальный ранг |

Конечная локальная CRLB плоской решётки при ненулевом угле места не устраняет глобальную пару решений `+elevation`/`-elevation`. WLS по умолчанию выбирает известную для наземного устройства верхнюю полусферу `u_z >= 0` и сообщает `mirror_ambiguous=True`.

## Структура и API

- `model/geometry.py` — направления, пары, апертура и пять геометрий;
- `model/tdoa.py` — времена прихода, сферические и плосковолновые TDOA, распространение ковариации;
- `model/jacobian.py` — аналитический и центрально-разностный Якобианы;
- `model/statistics.py` — матрица Фишера, ранг, обусловленность и условная CRLB;
- `estimators/wls_doa.py` — WLS на единичной верхней полусфере;
- `estimators/gcc_phat.py` — ориентированный sub-sample GCC-PHAT;
- `estimators/srp_phat.py` — direct/vectorized equal-pair far-field SRP-PHAT;
- `estimators/cycle_projection.py` — weighted projection всех пар на пространство
  физически согласованных TDOA;
- `simulation/fractional_delay.py` — frequency-domain и windowed-sinc дробные задержки;
- `simulation/propagation.py` — детерминированный plane/spherical многоканальный генератор;
- `simulation/signals.py` — deterministic multisine, независимый random broadband
  и отдельно маркированный harmonic stress-test;
- `simulation/trajectory.py` — stationary, constant-velocity, circular и
  piecewise-linear строго дозвуковые траектории;
- `simulation/moving_source.py` — exact retarded-time kinematics in a
  homogeneous stationary medium, analytic/frozen cross-checks,
  Doppler time warp, causality и moving multi-channel synthesis;
- `simulation/continuous_stream.py` — chunked continuous source/channel/noise
  synthesis и точные overlap frame views;
- `validation/far_field.py` — сеточный `E_tau(R)` и численный поиск границы;
- `validation/propagation_study.py` — итоговые CSV-исследования;
- `validation/gcc_study.py` — benchmark задержек и cross-generator GCC-аудит;
- `validation/gcc_monte_carlo.py` — воспроизводимый signal-level GCC/WLS Monte Carlo;
- `validation/gcc_statistical.py` — all-pair calibration/evaluation engine,
  covariance/confidence diagnostics и spherical/plane study;
- `validation/srp_statistical.py` — paired common-random GCC/WLS–SRP study с
  раздельными calibration/evaluation seeds и exact/accelerated SRP-аудитом;
- `validation/moving_source_study.py` — paired frame-wise moving/static
  GCC/WLS/SRP study с emission-time truth;
- `validation/sequential_doa_study.py` — хронологические независимые
  frame-wise bearings, latency metadata и frame/sequence CSV;
- `visualization/moving_scene.py` — интерактивная 3D-сцена bearing rays без
  фиктивной оценённой дальности;
- `notebooks/array_comparison.ipynb` — карты CRLB, ранга, обусловленности и вырождения;
- `notebooks/monte_carlo_crlb_validation.ipynb` — статистическая проверка WLS относительно CRLB;
- `notebooks/far_field_fractional_delay_validation.ipynb` — дальняя зона, задержки и каналы;
- `notebooks/gcc_phat_validation.ipynb` — скорость генераторов, знак и точность GCC;
- `notebooks/gcc_phat_monte_carlo.ipynb` — SNR-зависимость TDOA/DOA и tail-ошибки;
- `notebooks/gcc_statistical_validation.ipynb` — 210000 испытаний, fine-SNR,
  signal/frame/covariance/spherical сравнения;
- `notebooks/srp_phat_validation.ipynb` — 297000 парных реализаций, RMSE/tails,
  coverage, square/tetrahedral и runtime-сравнение четырёх оценивателей;
- `notebooks/moving_source_validation.ipynb` — 43200 moving/static пар,
  motion excess, within-frame DOA/TDOA change и Doppler;
- `notebooks/moving_source_3d.ipynb` — Play/Pause и вращаемая сцена истинной
  траектории и независимых GCC/SRP bearing rays;
- `notebooks/sequential_doa_validation.ipynb` — continuous-stream overlap,
  causality, invalid, azimuth-wrap, error и latency validation;
- `validation/monte_carlo.py` — воспроизводимый Monte Carlo-движок и CSV-метрики;
- `tests/` — автоматические проверки соглашений и обратной задачи.

Минимальный пример:

```python
import numpy as np

from estimators.wls_doa import estimate_doa_wls
from model.geometry import comparison_arrays
from model.tdoa import far_field_tdoa

microphones = comparison_arrays(0.20)["tetrahedral"]
phi, elevation = np.deg2rad([35.0, 25.0])
tau = far_field_tdoa(phi, elevation, microphones)
estimate = estimate_doa_wls(tau, microphones)
print(np.rad2deg([estimate.phi, estimate.elevation]))
```

## Воспроизведение

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -c "from validation.srp_statistical import run_srp_statistical_validation; run_srp_statistical_validation()"
.\.venv\Scripts\python.exe -c "from validation.moving_source_study import run_moving_source_study; run_moving_source_study()"
.\.venv\Scripts\python.exe -c "from validation.sequential_doa_study import run_sequential_doa_study; run_sequential_doa_study()"
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 notebooks\array_comparison.ipynb
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 notebooks\monte_carlo_crlb_validation.ipynb
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 notebooks\far_field_fractional_delay_validation.ipynb
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks\gcc_phat_validation.ipynb
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 notebooks\gcc_phat_monte_carlo.ipynb
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks\gcc_statistical_validation.ipynb
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks\srp_phat_validation.ipynb
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 notebooks\moving_source_validation.ipynb
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 notebooks\moving_source_3d.ipynb
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 notebooks\sequential_doa_validation.ipynb
.\.venv\Scripts\python.exe -m pip check
```

Notebook выполняется из корня проекта и сохраняет рассчитанные таблицы и рисунки в самом `.ipynb`.

## Границы модели

CRLB здесь условна относительно уже полученных гауссовских TDOA. Она не является границей по исходным микрофонным отсчётам и пока не включает неизвестный акустический сигнал, зависимость ошибок TDOA от SNR/спектра, калибровочные ошибки, ветер/температурный профиль, отражения, коррелированный акустический шум или дополнительные источники. Реализована **exact retarded-time kinematic model in a homogeneous stationary medium**; это точная кинематическая модель запаздывающего времени при её допущениях, а не полная модель акустической среды, и её paired AWGN study не является signal-level CRLB. Дальнепольность количественно вычисляется для заданных `fs`, `f_max`, временного/фазового допуска и угловой сетки; диагностические значения 48 кГц, 2–12 кГц, 0.1 sample и 0.1 rad не являются измеренными характеристиками реального БПЛА.

Целочисленные сдвиги отсчётов в проекте не используются. Реализованы только
независимые покадровые far-field equal-weight SRP-PHAT/GCC bearings.
SRP-Harmonics, отражения, ветер, коррелированный фон и EKF/UKF tracking пока
не реализованы.
