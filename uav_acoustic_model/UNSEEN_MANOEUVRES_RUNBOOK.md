# Повторение проверки новых манёвров и спектра

Протокол и все параметры предварительно зафиксированы в
`UNSEEN_MANOEUVRES_PROTOCOL.md` и
`UNSEEN_MANOEUVRES_PROTOCOL_MANIFEST.json` (коммит `186aebb847bc5db2824205bb87b82448ed33c68e`).
Существующие каталоги `unseen_spatial_001`, `unseen_radial_001` и
`results/unseen_manoeuvres_and_sources` не перезаписывать. Сохранённый частичный результат проверяется без повторной обработки;
повторный физический полёт не обещает побайтового совпадения.

## Проверка сохранённого технического стопа без повторного синтеза

Матрица остановлена на индексе 11 после превышения замороженного лимита
30 минут на поток. Из 24 потоков завершены 15, из 96 треков — 60.
`run-all` здесь **не** является командой проверки: execution-v2 запрещает запуск
при наличии `technical_stop.json`. Прежний runner мог начать обработку
незавершённых потоков и нарушил бы правило остановки.

```bash
cd ~/projects/diploma-gazebo/uav_acoustic_model
.venv/bin/python -m analysis.unseen_partial_audit verify
.venv/bin/python -m analysis.unseen_partial_audit audit
.venv/bin/python -m jupyter nbconvert --execute --to notebook --inplace \
  notebooks/unseen_manoeuvres_and_sources.ipynb
```

`verify` проверяет SHA сохранённых артефактов и метки остановки; `audit`
повторно строит описательные таблицы только по 15 полностью завершённым
потокам. Прерванные потоки и ещё не начатые потоки явно перечислены в
`partial_audit.json`.

## Новый эксперимент с новыми физическими записями

Нужны PX4 v1.17.0, Gazebo Harmonic, уже собранные PX4 SITL и
`build/px4-observer/libgazebo_px4_observer.so`, отдельная PX4 Python среда.
Не включать PX4-зависимости в основную `.venv`. Для измерения RSS в
основной `.venv` дополнительно установить:

```bash
cd ~/projects/diploma-gazebo/uav_acoustic_model
.venv/bin/python -m pip install -r analysis/requirements-unseen.txt
```

Для каждого **нового пустого** каталога (`RUN_DIR` меняется и никогда не
совпадает с принятой записью) создать сцену. Команды для двух планов:

```bash
cd ~/projects/diploma-gazebo/uav_acoustic_model
export PX4_ROOT="$HOME/projects/PX4-Autopilot"
export RUN_DIR="$PWD/results/px4_flight/unseen_spatial_002"
.venv/bin/python -m px4.create_scene "$RUN_DIR" \
  --plan px4/flight_plan_spatial_manoeuvre.json --px4-root "$PX4_ROOT"
# Для второго каталога используйте unseen_radial_002 и:
# --plan px4/flight_plan_radial_approach_depart.json
```

Терминал A:

```bash
cd ~/projects/diploma-gazebo/uav_acoustic_model
export RUN_DIR="$PWD/results/px4_flight/unseen_spatial_002"
.venv/bin/python -m px4.launch_gazebo "$RUN_DIR" \
  --px4-root "$HOME/projects/PX4-Autopilot" --headless
```

Терминал B после появления мира Gazebo:

```bash
cd ~/projects/PX4-Autopilot
GZ_IP=127.0.0.1 PX4_SIM_MODEL=gz_x500 PX4_GZ_STANDALONE=1 \
PX4_GZ_WORLD=px4_acoustic PX4_GZ_MODEL_POSE=35,35,0,0,0,0 \
build/px4_sitl_default/bin/px4 -d
```

Терминал C после готовности PX4:

```bash
cd ~/projects/diploma-gazebo/uav_acoustic_model
export RUN_DIR="$PWD/results/px4_flight/unseen_spatial_002"
~/projects/PX4-Autopilot/.venv-px4/bin/python -m px4.run_flight "$RUN_DIR"
```

После посадки остановить PX4 и Gazebo в терминалах B и A. Затем:

```bash
cd ~/projects/diploma-gazebo/uav_acoustic_model
export RUN_DIR="$PWD/results/px4_flight/unseen_spatial_002"
.venv/bin/python -m px4.finalize_recording "$RUN_DIR"
.venv/bin/python -m px4.assess_recording "$RUN_DIR"
```

Повторить с новым каталогом и радиальным планом. До нового исследования
создать **новый** snapshot протокола, содержащий SHA этих двух записей и
новый каталог результата. Принятый frozen manifest нельзя менять под новую
запись. Принятый исследовательский manifest привязан к SHA именно этих двух записей
и к замороженному плану 24 потоков. Новые полёты требуют отдельного нового
протокола, нового каталога результата и новой технической проверки стоимости;
их нельзя подменить в принятом manifest. Execution-v2 блокирует обычный
`run-all` после технического стопа. Для воспроизведения доступен SHA-аудит выше.

В журналах обработки время локальное относительно начала приёма;
`time_origin_gazebo_s` в `bearing_manifest.json` переводит его обратно в
абсолютное время Gazebo. Это представление устраняет потерю точности шага
48 кГц при больших абсолютных метках, не меняя физическую временную шкалу.


## Execution-v2 и диагностический S8

Обычные `run-one`, `smoke` и `run-all` отказываются до вычислений, если есть
`technical_stop.json` либо `execution_v2/technical_stop.json`. Историческую
метку не удалять. В этой задаче продолжение остановленной матрицы не разрешено.

Новые версии запуска: `analysis/unseen_execution_v2.py` и
`analysis/study_execution.py`. Завершение каждого tracker variant сохраняется
атомарным `completion_{method}_{variant}.json` в отдельном `execution_v2/runs`.
Сверяются шесть журналов, их схемы/хеши, exact publication schedule,
event IDs и происхождение bearing/settings. Наличие одного tracking CSV не
доказывает завершение; проверяемые полные варианты пропускаются. Если
исторический stream прерван, full variant импортируется только по сохранённым
SHA partial audit и полному набору/расписанию. Исторические outputs не
перезаписываются. Parent supervisor соблюдает frozen external timeout и
сохраняет новые supervision/stop артефакты, worker имеет отдельный watchdog.

Будущее продолжение возможно только с отдельно согласованным frozen protocol
и authorization JSON: `schema_version=1`, `action=authorized_continuation`,
`execution_version=2`, SHA evaluation/stop/protocol, `allowed_indices` и явное
`approved_by_user`. Разрешение на продолжение основной матрицы сейчас не
создано. Изолированный вход для такого будущего разрешения — `continue-one`
в `analysis.unseen_execution_v2`, не обход проверки в обычном runner.
Этот вход сохраняет отдельные variant completions и stream summary в
`execution_v2`; прежний `aggregate` предназначен для полного исторического
формата и не объединяет их с неполной остановленной матрицей. Продолжение
и его отдельная итоговая агрегация требуют нового согласованного протокола.

Исторический runner побайтно сохранён в `analysis/frozen`; опубликованный
manifest сверяется с его SHA, новые manifests — с новым runner. Processing
code/settings SHA и исторические run IDs остаются обязательными.

Текущая read-only диагностика и просмотр notebook:

```bash
.venv/bin/python -m analysis.harmonic_tracking_failure verify
.venv/bin/python -m jupyter nbconvert --execute --to notebook --inplace notebooks/harmonic_tracking_failure.ipynb
```

Эти команды не запускают audio synthesis, tracker replay или полёты.
Полный разбор: `HARMONIC_TRACKING_FAILURE_REPORT.md`.
