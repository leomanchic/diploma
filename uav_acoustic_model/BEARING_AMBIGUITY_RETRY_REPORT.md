# Observable bearing ambiguity gate: повторное исследование

## Итог

После явного разрешения пользователя повтор выполнен в новом каталоге
`results/bearing_ambiguity_retry_001`. Старый отказ, его исходный код, SHA,
manifest и ledger сохранены без изменения. Протокол и правила выбора не
переписаны под результаты. Замороженный совместный критерий **не выполнен**.

Завершены **16/16 audio streams** и **8/8 replay**.
Время внешних workers вместе с первоначальным preflight-отказом:
**1852.936 с** из 21600 с. Предыдущая попытка: 1 worker,
0 синтезов/0 replay, 1.287 с. В повторе фактически
24 worker attempts. Технический stop: False.
Ни старую 15/24 матрицу, ни старые исследования, ни полёты не повторяли.

SRP margin частично выделил ненадёжные направления: >10° passed/all valid
**24,88%→18,99%**, относительное снижение **23,67%** при сохранении **92,66%**
измерений и **2,91%** good false rejection (macro по cells). GCC не изменился.
**Сопровождение не улучшилось**: broadband off/on одинаковы, harmonic
по-прежнему не подтверждается. Заранее требовалось ≥25% для обоих методов;
критерий не пройден. Это завершённый отрицательный результат заданного gate,
а не подтверждение operational пригодности или бесполезности всех observables.

## Протокол и неизменные настройки

- Исходный SHA `4ef0fb1`; protocol commit `07ad77b`; код и разрешение
  повторить в отдельном каталоге `4c23999`; gate frozen до evaluation
  коммитом `1437cea`. SHA сохранены в protocol/study/gate/authorization manifests.
- Development: существующая spatial запись; evaluation: radial. Каждый
  split: broadband / nonstationary harmonic ×200/700 m×2 реализации.
  Source/noise seeds disjoint между split. Harmonic параметры разных split
  заранее разные; точные числа и алгоритм — в исходном протоколе.
- Прежние ENU station/microphone geometry, retarded time, 1/r, фиксированный
  фон, SNR_ref=+10 dB на 100 m, fs=48 kHz, frame/hop/stride=32, calibrated
  bias/R, Qc/history, construction/confirmation и computational budgets
  не изменены. Three-station confirmation — единственный вариант replay.
- Максимум 16 новых потоков/8 replay, 1200 с/audio, 600 с/replay, 6 часов
  heavy wall; один внешний worker. Parent timeout, exclusive lock,
  SHA/completion проверка полного расписания. Нет автоматических повторов.

## Наблюдаемые признаки и development

Два признака на метод, вычисленные только из текущего полного кадра и raw
outputs: spectral participation p=(ΣP)²/(BΣP²) и нормированный competitor
margin m=(peak−competitor)/(max−min). GCC: minimum по шести парам;
competitor ≥2 samples от maxima. SRP: прежняя equal-pair cubic score-map,
coarse 5° и competitor ≥20° от refined estimate. Undefined/silence/spectrum
failures получают причины; scores не вероятности. Endpoint не скрывается.

Cartesian threshold grid и selection rule заданы до расчётов: сохранять
≥70% original valid и ≥95% good ≤5° по каждому development source class,
затем минимизировать >10° passed/all valid. Матрица и полные tradeoffs:
`development_tradeoff.csv.gz`. Выбраны и заморожены:

- GCC: p_min=0,
  m_min=0.
- SRP: p_min=0,
  m_min=0.05.

GCC выбор без порога показывает, что development критерий не обнаружил
полезного отсечения с требуемым сохранением good measurements. Это не
проверка всех возможных наблюдаемых признаков. Набор и grid не расширяли.

## Evaluation: bearing-level результат

Macro по stream/station cells, каждый cell с исходным valid denominator.
Повторные дальности/source banks зависимы; это descriptive transfer,
не независимые UAV sessions и не iid-frame статистика.

| Метод | Сохранено / original valid | >10° passed / all valid, off→on | Относительное снижение | Good ≤5° falsely rejected / good | Frozen bearing criterion |
|---|---:|---:|---:|---:|---|
| GCC | 100.00% | 26.42% → 26.42% | 0.00% | 0.00% | нет |
| SRP | 92.66% | 24.88% → 18.99% | 23.67% | 2.91% | нет |

`bearing_metrics.csv.gz` содержит каждый метод/поток/станцию, отдельно
all frames и stride: accepted/rejected counts/fractions, conditional median/P95,
>5/10/30° conditional fractions и passed/all original valid, good loss с
двумя знаменателями, duration каждого rejected run, max/total duration,
reason counts. `station_availability.csv.gz`: instantaneous groups и
causal trailing 3 s окна с ≥2/3 stations; будущая доставка не учитывается.
`gate_decisions.csv.gz` сохраняет каждый raw ID и причину решения.
Не наблюдавшиеся условные метрики обозначены отсутствующими, не нулём.

## Tracking: paired off/on на заранее выбранных случаях

Evaluation 200 m / replicate0, оба source class, GCC/SRP, gate off/on.
Общие raw events и publication schedule. Invalid gate events остаются
в `gate_events.csv.gz`, не допускаются ни в construction, ни в confirmation,
ни в update. Event-use journals проверены на отсутствие их IDs.
Всего fits — по **всем поколениям**, не только final generation.

| Source | Метод | Gate | Confirmation (s) | Valid publications | Conditional RMSE / P95 (m) | All-generation fits | Final status |
|---|---|---|---:|---:|---:|---:|---|
| broadband | GCC | off | 1.055 | 94.55% | 2.006 / 3.828 | 2 | confirmed |
| broadband | GCC | on | 1.055 | 94.55% | 2.006 / 3.828 | 2 | confirmed |
| broadband | SRP | off | 1.055 | 94.55% | 2.000 / 3.815 | 2 | confirmed |
| broadband | SRP | on | 1.055 | 94.55% | 2.000 / 3.815 | 2 | confirmed |
| nonstationary_harmonic | GCC | off | — | 0.00% | — / — | 128 | computational_budget_exceeded |
| nonstationary_harmonic | GCC | on | — | 0.00% | — / — | 128 | computational_budget_exceeded |
| nonstationary_harmonic | SRP | off | — | 0.00% | — / — | 128 | computational_budget_exceeded |
| nonstationary_harmonic | SRP | on | — | 0.00% | — / — | 128 | computational_budget_exceeded |

`paired_tracking_review.csv.gz` дополнительно содержит first-confirmation
error, accepted/rejected updates, >10/50 m tails /all publication epochs,
потерю availability, confirmation loss и реальные причины отказов.
`tracking_metrics.csv.gz` и шесть журналов каждого replay сохраняют
полные метрики/гипотезы/fit/lifecycle/uses. Conditional accuracy учитывает
только valid публикации; availability и полные знаменатели обязательны.
В `tracking_errors.png` отсутствия оценки остаются NaN-разрывами линий.

## Механизм и ограничения

По 9 заранее выбранных frame/station combinations на поток: frames 0,32,
last, станции S0–S2. Сохранены реальные шесть correlation curves, выбранные
lag maxima/competitors, SRP grid и refined estimate/competitor — `.npz`.
Notebook показывает один заранее выбранный harmonic evaluation frame 0.
Competitor определён как максимальный score вне исключённой области:
он может быть плечом широкого пика, не отдельным локальным экстремумом.
Modelled availability задержки сохранены; измеренная wall runtime включает
диагностику и не является заявлением о real-time. Наличие далёкого
конкурента — наблюдаемый сигнал, но не доказательство
истинности выбранного maxima. Спектральный p может расти при доминировании
шума, а coarse grid не ловит близкие competing peaks. Эти ограничения
объявлены до evaluation. Нет claims о вероятностной калибровке score.

Перенос synthetic harmonic структуры — ограниченный стресс-тест, не
независимое акустическое исследование реальных БПЛА. Ошибки/корреляции/R
могут взаимодействовать с geometry, construction и gating. Эти эффекты
здесь не разделены причинно; R/Qc/filter/budgets не менялись для компенсации.
Два realization и общие banks не дают популяционной квалификации.

## Проверки и просмотр

Целевые тесты проверяют truth-free/opt-out, raw estimate invariance,
rejected исключение всех стадий, undefined reasons, seed split,
worker serialization/completion на короткой fixture без нового synthesis,
запрет evaluation до freeze и causal window без future delivery.
Целевые тесты **22 passed**, включая настоящий byte-exact checkout с
`core.autocrlf=true`; полный pytest **619 passed** за 188,77 с.
`pip check` и `git diff --check` пройдены. Новый notebook: **7/7** code cells
выполнены без ошибок, только saved-artifact analysis. Фактический
Linux/Windows CI сообщается с итоговым SHA.

```bash
cd uav_acoustic_model
python -m analysis.bearing_ambiguity_study verify --output results/bearing_ambiguity_retry_001
python -m analysis.bearing_ambiguity_study report --output results/bearing_ambiguity_retry_001
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/bearing_ambiguity_retry.ipynb
```

Gate **остаётся выключенным по умолчанию**; S8 In progress. Рабочая
модель звука, calibration и tracker не изменены. Совместный протокольный
критерий не выполнен; новые признаки или параметры требуют следующего отдельного
frozen development/evaluation, не подбора по этой evaluation.
