# Lunar_project
Решение для gym.openai.com/envs/LunarLander-v2

![Image alt](https://github.com/sunny3/Lunar_project/raw/master/img/LunarLander.png)

Запуск демонстрации в консоли
```
pip install gym
pip install box2d
pip install tensorflow
pip install keras
git clone https://github.com/sunny3/Lunar_project
cd Lunar_project
python demonstration.py
```
demonstration.py по умолчанию запускает модель с длинной памятью в 13 эпизодов. Если хочется переключиться на короткую в 2 эпизода, то следует запустить
```
python demonstration.py --mode short
```
Процесс обучения сети (с короткой памятью) с классом агента и подробными комментариями представлен в файле юпитер ноутбука, также там есть график и возможность запустить анимацию 1 эпизода

Все основные комментарии к классу также представлены в файле юпитер ноутбука
