# Диагностика источников ошибки локализации

Дата анализа: 2026-09-26. Протокол:
`LOCALIZATION_ERROR_ATTRIBUTION_PROTOCOL.md`. Анализ выполнен на восьми
заранее выбранных известных случаях опубликованного range study. Это
диагностика механизмов отказа, а не независимое подтверждение качества.
Эффекты вариантов взаимодействуют; разности ошибок не интерпретируются как
аддитивное разложение.

## Воспроизводимость и объём

Исходные `results/localization_range_study`, manifest, frozen-протокол и 120
`run_id` не изменены. Их SHA остались:

- manifest: `00f3e2e259885d34abd9e2247e1aa1ed0f74c201ad63d4981dfd6fe8b5d194e6`;
- полный набор run IDs: `6c7704f02e499976c25028c9a328cc1fcc52d851384ca989275cacf3f76bbd4c`.

Выполнено ровно **8** восстановлений аудиопотока. Для каждого сохранены
полные GCC/SRP направления, evaluator-only truth, station-specific emission
time, reception/availability timestamps, quality metadata и event IDs. Затем
один и тот же event schedule повторно использован в трёх вариантах каждого
метода, всего **48** method-variant результатов. Новые полёты, остальные 112
range cases, Monte Carlo и старые notebooks не запускались.

Все 16 исходных method cases воспроизвели опубликованные bearing errors,
координаты, статусы, причины, confirmation time, counts, RMSE и P95. Для всех
проверенных чисел максимальная разница равна `0`, runtime исключён из критерия.

## Сравниваемые варианты

- `original`: исходные GCC/SRP bearings, сохранённые bias и R;
- `ideal_bearing`: точное направление для отдельного времени излучения каждой
  станции, bias=0; прежняя R используется только как вес и не является
  истинной covariance идеального входа;
- `zero_bias`: исходные bearings и R, но bias=0.

Во всех вариантах сохранены frame stride 32, event delivery, Qc, history,
gates и budget 128. Truth не входит в `BearingMeasurement`, и трек не
инициализируется истинным состоянием.

## Ошибка состояния и согласованность ковариации при подтверждении

`first_confirmation_summary.csv` содержит все 48 комбинаций
`case × method × variant`. Для 44 подтверждённых треков в нём явно сохранены
ошибки положения и скорости, радиальная/поперечная компоненты, position NEES,
максимальный локальный P95 scale ковариации и признак покрытия. Четыре строки
без подтверждения — GCC original и zero-bias в fixed-background случаях 400 и
1000 м; их значения состояния корректно оставлены пустыми.

| Случай | Вариант | Ошибка положения при confirmation | Ошибка скорости | Position NEES / covariance P95 scale |
|---|---|---:|---:|---:|
| single-turn 700 м, GCC/SRP | original | 34,44 / 34,54 м | 12,11 / 12,09 м/с | 0,152 / 0,150; 308,52 / 307,66 м |
| opposite-turns 700 м, GCC/SRP | original | 43,23 / 42,78 м | 17,94 / 17,85 м/с | 0,219 / 0,216; 364,24 / 362,96 м |
| fixed 400 м, SRP | original | 21,58 м | 4,68 м/с | 31,12; 32,97 м, truth вне P95 region |
| fixed 400 м, SRP | ideal-bearing | 0,309 м | 0,243 м/с | 0,0092; 72,01 м, truth покрыт |
| fixed 1000 м, SRP | original | 224,70 м | 46,70 м/с | 29,07; 153,73 м, truth вне P95 region |
| fixed 1000 м, SRP | ideal-bearing | 1,267 м | 0,588 м/с | 0,0066; 914,33 м, truth покрыт |

`uncertainty_error_summary.csv` сопоставляет фактическую ошибку с posterior
NEES, coverage и локальным масштабом covariance по всей допустимой части
каждого трека. Например, для исходного SRP на 1000 м P95 фактической ошибки
равен 2147,65 м, median covariance maximum-axis P95 scale — 5366,62 м,
position coverage — 0,71. При подтверждении covariance слишком уверенная и
не покрывает truth; после потери полезных updates она сильно расширяется.
Поэтому позднее покрытие большой ковариацией не является признаком точной
локализации.

## Главные численные результаты

### Контроль постоянного принятого SNR +10 dB

Angular P95 остаётся примерно `0,13–0,14°` на 200–700 м. Одновременно локальный
статический geometry benchmark с фиксированной угловой sigma 1° показывает:

| Дальность | median condition | median radial sigma | median transverse RSS | weak-axis radial alignment |
|---:|---:|---:|---:|---:|
| 200 м | 12–13 | 7,45–7,82 м | 3,08–3,15 м | ≥0,9999 |
| 400 м | 51–52 | 29,49–30,24 м | 5,86–5,94 м | ≥0,99999 |
| 700 м | 164–166 | 91,40–92,73 м | 10,12–10,19 м | ≈1,0 |
| 1000 м | 342 | 187,93 м | 14,39 м | ≈1,0 |

Это отдельная статическая Gaussian-линеаризация одной общей позиции источника.
Она не является точной границей ошибки динамического трекера.

На 700 м исходная ошибка уже в момент первого подтверждения равна
`34,44/34,54 м` для single-turn GCC/SRP и `43,23/42,78 м` для opposite-turns.
Почти вся она направлена вдоль луча от центра станций. С идеальными bearings
ошибка первого подтверждения уменьшается до `0,63–0,65 м`. Исходный P95
`47,21–68,08 м`, ideal-bearing P95 `7,70–9,11 м`.

Следовательно, небольшая акустическая угловая ошибка сильно усиливается слабым
радиальным направлением. При этом ideal-bearing P95 не равен нулю: остаются
инициализация, strict-CV/Qc динамика, причинные задержки и последовательные
updates. Геометрический и tracking-вклады этим опытом не разделяются точно.

### Fixed background, 400 м, SNRref=0 dB, replicate 0

GCC original и zero-bias исчерпывают 128 batch fits в `47.7019791667 s` и ни
разу не инициализируются. Ideal-bearing подтверждается после 2 fits через
`1.0553125 s`, P95 равен `4,7568 м`. Максимальный station angular P95 исходного
GCC достигает `136,22°`. Это подтверждает, что отказ вызывается акустически
повреждёнными bearings, а не самим budget или calibration bias.

SRP original подтверждается поздно, через `5.1513125 s`, уже с ошибкой
`21,58 м`, затем имеет два reset и заканчивает invalid. Его conditional P95
`73,25 м`; ideal-bearing P95 `4,76 м`. В исходном replay отклонены 14 updates
по NIS и 21 по `emission_outside_history`.

### Fixed background, 1000 м, SNRref=10 dB, replicate 1

GCC original и zero-bias исчерпывают 128 fits в `56.9179791667 s`;
ideal-bearing подтверждается после 2 fits и даёт P95 `11,58 м`.

SRP original подтверждается через `4.4836458333 s` уже с ошибкой `224,70 м`,
из которой `224,19 м` приходится на радиальную компоненту. Ошибка затем
возрастает до `2222,83 м`; P95 `2147,65 м`. Принято только 3 updates,
отклонено 96: 90 `emission_outside_history` и 6 NIS gate. Три принятых updates
имеют NIS `0,77–7,65`, хотя текущая ошибка координат составляет `280–339 м`.
Это прямой пример того, что геометрически слабое радиальное смещение может
иметь небольшой bearing residual и пройти update gate при сильно ошибочной
позиции.

Ideal-bearing подтверждается через `1.0553125 s` с ошибкой `1,27 м`, принимает
113 updates и даёт P95 `11,58 м`. Значит большая исходная ошибка присутствует
уже при подтверждении, после чего растёт из-за слабого радиального наблюдения
и почти полного прекращения полезных коррекций.

### Calibration bias

`zero_bias` не устраняет ни один GCC budget failure и не исправляет сильно
ошибочные SRP tracks. На 700 м его изменение P95 смешанное: оно слегка
улучшает single-turn и ухудшает opposite-turns. Сохранённый bias не является
главной причиной наблюдаемых отказов; автоматически выбирать bias=0 рабочим
вариантом нельзя.

## Итоговая таблица причин

| Случай | Наблюдение | Подтверждённая причина или гипотеза | Поддерживающее сравнение |
|---|---|---|---|
| control 200 м | P95 original 2,46–2,70 м | Геометрия ещё умеренная; acoustic вклад меньше | ideal P95 2,22–2,63 м, condition ≈12 |
| control 400 м | P95 original 5,13–7,32 м | Геометрия и tracking уже ограничивают точность; bearings усиливаются | ideal P95 4,30–4,76 м, condition ≈51–52 |
| control 700 м | 34–43 м ошибки уже при confirmation, почти радиально | **Подтверждено:** слабая радиальная геометрия усиливает малые bearing errors | ideal first error 0,63–0,65 м; condition ≈164–166 |
| fixed 400 м GCC | 128 fits, initialization отсутствует | **Подтверждено:** gross acoustic bearings разрушают initialization | ideal: 2 fits; zero-bias также 128 |
| fixed 400 м SRP | late bad confirmation, resets, final invalid | **Подтверждено:** acoustic error создаёт плохую initialization и дальнейшие потери | original P95 73,25 м; ideal 4,76 м |
| fixed 1000 м GCC | 128 fits, initialization отсутствует | **Подтверждено:** acoustic corruption при слабой геометрии блокирует initialization | ideal: 2 fits, P95 11,58 м |
| fixed 1000 м SRP | 224,70 м при confirmation → 2222,83 м | **Подтверждено:** bad initialization взаимодействует с радиальной неоднозначностью и update starvation | ideal first 1,27 м; только 3/99 updates приняты |
| zero-bias | статусы отказов не меняются | **Подтверждено:** bias не является основной причиной | original против zero-bias |

## Следующее изменение

Приоритетное направление — **observable-data-only robust gate согласованности
пеленгов до bounded candidate search инициализации**. Он должен исключать
грубые angular outliers до расходования 128 nonlinear fits, сохранять event IDs
и причины исключения и не использовать truth, range или coordinate error.

Текущая диагностика показывает, почему это направление приоритетнее изменения
bias или EKF update: ideal bearings устраняют оба budget failure, а zero-bias
нет. Она ещё не доказывает, что существующие quality metadata достаточно
надёжно распознают выбросы. До реализации порог и score нужно заморозить на
отдельных development cases и проверить на held-out streams. Само улучшение в
этой ветке не реализовано.

## Артефакты и повторный просмотр

Основные таблицы находятся в `results/localization_error_attribution`:

- `variant_summary.csv`, `phase_summary.csv`, `update_summary.csv`;
- `first_confirmation_summary.csv`, `uncertainty_error_summary.csv`;
- `initialization_summary.csv`, `bearing_station_summary.csv`;
- `geometry_summary.csv`, `geometry_case_summary.csv`;
- `diagnosis_table.csv`, `analysis_summary.json`;
- полные case-specific bearings, tracking, updates, batch fits, hypotheses и
  lifecycle logs в `cases/`;
- четыре графика в `figures/`.

Выполненный notebook: `notebooks/localization_error_attribution.ipynb`.

```bash
cd ~/projects/diploma-gazebo/uav_acoustic_model
source .venv/bin/activate

# Проверка готовых восьми случаев: аудио повторно не синтезируется.
python -m analysis.localization_error_attribution restore-all \
  --output results/localization_error_attribution
python -m analysis.localization_error_attribution aggregate \
  --output results/localization_error_attribution

# Только новый notebook.
python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=300 \
  notebooks/localization_error_attribution.ipynb
```
