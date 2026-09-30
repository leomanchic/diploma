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
`run-all` здесь **не** является командой проверки: он начал бы обработку
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
их нельзя подменить в принятом manifest. Текущий `run-all` автоматически
обрабатывает незавершённые индексы и потому **не запускается** после
технического стопа. Для воспроизведения доступен SHA-аудит выше.

В журналах обработки время локальное относительно начала приёма;
`time_origin_gazebo_s` в `bearing_manifest.json` переводит его обратно в
абсолютное время Gazebo. Это представление устраняет потерю точности шага
48 кГц при больших абсолютных метках, не меняя физическую временную шкалу.
