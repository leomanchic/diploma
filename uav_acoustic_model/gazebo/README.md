# Gazebo → офлайн-акустика: три станции и один источник

Проверено на Ubuntu 24.04.5, Gazebo Harmonic `gz sim 8.15.0`,
`libgz-sim8-dev 8.15.0`, CMake 3.28.3 и g++ 13.3.0. Интеграция требует
системные пакеты `gz-harmonic`, `libgz-sim8-dev`, `cmake`, `g++`; они **не**
добавлены в зависимости основного Python-пакета. Python использует уже
объявленные NumPy/SciPy. На другой версии Gazebo плагин нужно пересобрать.

Сцена создаётся из [общего JSON](../simulation/gazebo_scene.json), который
поставляется вместе с Python-пакетом. Те же станции читает
`simulation.gazebo_offline.shared_stations()` и прежний `pilot_stations()`:
центры S0 `(0,0,0)`, S1 `(100,0,5)`, S2 `(10,90,-2)` м; все ориентации
`roll=pitch=yaw=0` рад. Это существующий проверочный сценарий проекта.
Земля находится ниже самой низкой станции, станции неподвижны. Оранжевая
сфера `sound_source` — видимый источник. Плагин задаёт две кинематические
команды: постоянная скорость и гладкий поворот в `[2.5,3.5]` с. **Динамика
полёта и автопилот здесь ещё не моделируются.**

## Точные команды из корня `uav_acoustic_model`

```bash
cmake -S gazebo -B build/gazebo
cmake --build build/gazebo -j2
mkdir -p build/gazebo-xdg build/gazebo-logs

.venv/bin/python gazebo/create_scene.py constant_velocity results/gazebo_offline/constant_velocity
XDG_CONFIG_HOME="$PWD/build/gazebo-xdg" XDG_DATA_HOME="$PWD/build/gazebo-xdg" \
GZ_LOG_PATH="$PWD/build/gazebo-logs" GZ_SIM_SYSTEM_PLUGIN_PATH="$PWD/build/gazebo" \
gz sim -s -r --iterations 6000 --seed 20260920 -v 2 results/gazebo_offline/constant_velocity/scene.sdf

.venv/bin/python gazebo/create_scene.py smooth_turn results/gazebo_offline/smooth_turn
XDG_CONFIG_HOME="$PWD/build/gazebo-xdg" XDG_DATA_HOME="$PWD/build/gazebo-xdg" \
GZ_LOG_PATH="$PWD/build/gazebo-logs" GZ_SIM_SYSTEM_PLUGIN_PATH="$PWD/build/gazebo" \
gz sim -s -r --iterations 6000 --seed 20260920 -v 2 results/gazebo_offline/smooth_turn/scene.sdf

.venv/bin/python gazebo/create_scene.py smooth_turn results/gazebo_offline/smooth_turn_100hz --export-period-s 0.01
XDG_CONFIG_HOME="$PWD/build/gazebo-xdg" XDG_DATA_HOME="$PWD/build/gazebo-xdg" \
GZ_LOG_PATH="$PWD/build/gazebo-logs" GZ_SIM_SYSTEM_PLUGIN_PATH="$PWD/build/gazebo" \
gz sim -s -r --iterations 6000 --seed 20260920 -v 2 results/gazebo_offline/smooth_turn_100hz/scene.sdf

.venv/bin/python -m validation.gazebo_offline_run validate results/gazebo_offline
.venv/bin/python -m validation.gazebo_offline_run process results/gazebo_offline/constant_velocity
.venv/bin/python -m validation.gazebo_offline_run process results/gazebo_offline/smooth_turn
.venv/bin/python -m visualization.gazebo_offline_view results/gazebo_offline/constant_velocity
.venv/bin/python -m visualization.gazebo_offline_view results/gazebo_offline/smooth_turn
xdg-open results/gazebo_offline/constant_velocity/viewer.html
xdg-open results/gazebo_offline/smooth_turn/viewer.html
```

Для просмотра сцены с GUI вместо `-s` можно запустить `gz sim -r ...` с тем
же SDF и переменными окружения. Команда `--iterations 6000` завершает запись
после 6 с **симуляционного** времени при шаге физики 1 мс. Паузы и изменение
скорости воспроизведения меняют лишь время ожидания: плагин пропускает
`PostUpdate` при паузе и пишет `UpdateInfo.simTime`. Увеличение частоты
экспорта требует только `--export-period-s`, не изменения частоты физики.

## Формат и проверки

Плагин читает **фактическую позу `sound_source` из Gazebo ECM после шага
физики** и пишет `gazebo_state.csv`. Он не пишет заданную траекторию и не
пересчитывает её в Python для экспорта. Столбцы: `sim_time_s`, `x_m,y_m,z_m`,
`qw,qx,qy,qz`. Мировая система правосторонняя ENU: `x=East`, `y=North`,
`z=Up`; координаты — метры, время — секунды, углы внутри модели — радианы,
кватернион имеет порядок **w,x,y,z**. Компонент мировой линейной скорости
для модели с pose-командой надёжно не предоставляется, поэтому скорость
в CSV не заявлена. `SampledTrajectory.v()` и `.a()` являются производными
того же кубического сплайна положения, что и `.q()`.

Импорт отклоняет нечисловые/пропущенные данные, неправильные кватернионы,
повтор времени, сброс времени, разрыв больше `1.5 ×` заданного периода,
недостаточную длительность и дозвуковой предел. Экстраполяция запрещена.
Запись охватывает `[0.001,6.0]` с; рецепция аудио `[0.5,5.0)` с, а
минимальное время излучения около `0.213` с, поэтому у retarded-time solver
есть предыстория. Генератор сам создаёт единый исходный сигнал достаточной
длины (около 221–222 тыс. отсчётов) и непрерывный шум каждого канала.

Аудио синтезируется при 48 кГц **после** записи Gazebo. Это не частота
физики и не частота экспорта. Шум AWGN `+10 dB` и seed `20260920` фиксированы
в конфигурации. GCC-PHAT, SRP-PHAT, stride `32`, `Qc` и пороги остаются из
существующего пилота. Калибровка bias/R читается из
`results/three_station_audio_calibration.csv` и зафиксирована до этих двух
запусков; её SHA записан в `summary.json`. Истина применяется только в
генераторе и офлайн-проверке, а не как вход estimator/tracker.

На выходе каждого основного запуска: записанный CSV, `manifest.json`,
`bearing_results.csv`, `tracking_*.csv`, `updates_*.csv`, `summary.json` и
самостоятельный `viewer.html`. В последнем можно вращать 3D-сцену, двигать
время и выбирать GCC/SRP. Невалидные публикации разрывают линию оценки и
выделяются на графике ошибки; причины отказов доступны в панели и при
наведении. `validation.json` содержит сравнение прямой с аналитикой,
одинакового исходного аудио/задержек и сходимость поворота при 50/100 Гц.
`manifest.json` сохраняет версии, хеши конфигурации/плагина/SDF, шаг физики,
частоту экспорта, геометрию, начальные условия и seeds; `summary.json`
содержит хеш фактически записанного CSV.

Допуски интеграционной проверки: `1e-6 м` для прямой, `1e-8 с` для задержки,
`1e-5` RMS-единиц для общего аудио и `1e-5 м` для поворота. Они значительно
шире измеренных ошибок: численный интеграл предписанной кинематики,
20-мс кубический сплайн и точность FIR/retarded-time генератора входят в
сравнение. На густой сетке 50→100 Гц ошибка положения поворота уменьшается
примерно в 4.4 раза, скорости — в 8 раз; аудио и задержки прямой совпадают
существенно точнее допусков. Эти проверки не являются оценкой полевой
точности акустической системы.
