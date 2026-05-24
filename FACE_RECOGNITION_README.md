# Система распознавания лиц с возможностью пополнения библиотеки

## 📋 Описание

Современная система распознавания лиц на основе библиотеки `face_recognition` (dlib + ResNet), 
которая позволяет:
- Добавлять новые лица в базу данных
- Распознавать лица на изображениях
- Распознавать лица в реальном времени с веб-камеры
- Обрабатывать несколько лиц одновременно

## 🎯 Ключевые особенности

1. **Современная модель**: Используется ResNet architecture с точностью 99.38% на LFW benchmark
2. **128-мерные эмбеддинги**: Каждое лицо представляется как вектор из 128 признаков
3. **Гибкий порог распознавания**: Настраиваемый параметр tolerance (по умолчанию 0.6)
4. **Две модели обнаружения**: 
   - HOG (быстрее, работает на CPU)
   - CNN (точнее, но медленнее, требует GPU для лучшей производительности)
5. **Кэширование**: Сохранение кодирований для ускорения повторной загрузки

## 📁 Структура проекта

```
/workspace/
├── face_recognition_system.py  # Основной модуль с классом FaceRecognizer
├── add_face.py                 # Скрипт для добавления новых лиц
├── recognize_faces.py          # Скрипт для распознавания лиц
├── requirements.txt            # Зависимости Python
├── README.md                   # Этот файл
└── faces_database/             # Папка для хранения эталонных изображений
    ├── Ivan_Petrov/
    │   ├── photo1.jpg
    │   └── photo2.jpg
    └── Maria_Sidorova/
        └── image.png
```

## 🚀 Установка

### 1. Установка системных зависимостей

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install build-essential cmake
sudo apt-get install libopenblas-dev liblapack-dev
sudo apt-get install libx11-dev libgtk-3-dev
```

**macOS:**
```bash
brew install cmake
```

**Windows:**
Установите Visual Studio Build Tools или используйте conda:
```bash
conda install -c conda-forge dlib
```

### 2. Установка Python-зависимостей

```bash
pip install -r requirements.txt
```

### 3. Проверка установки

```bash
python -c "import face_recognition; print('✅ face_recognition установлен')"
python -c "import cv2; print('✅ opencv-python установлен')"
```

## 📖 Использование

### Шаг 1: Добавление лиц в базу данных

```bash
# Добавить одного человека с несколькими фото
python add_face.py --name "Ivan Petrov" --images ivan1.jpg ivan2.jpg ivan3.jpg

# Добавить другого человека
python add_face.py --name "Maria Sidorova" --images maria.png

# С указанием параметров
python add_face.py --name "John Doe" --images john.jpg --tolerance 0.5 --model cnn
```

**Параметры:**
- `--name, -n`: Имя человека (обязательно)
- `--images, -i`: Пути к изображениям (обязательно, можно несколько)
- `--database, -d`: Путь к базе данных (по умолчанию: faces_database)
- `--tolerance, -t`: Порог распознавания (по умолчанию: 0.6)
- `--model, -m`: Модель обнаружения hog/cnn (по умолчанию: hog)

### Шаг 2: Распознавание на изображении

```bash
# Распознать лица на изображении
python recognize_faces.py --image test_photo.jpg

# С сохранением результата
python recognize_faces.py --image test_photo.jpg --output result.jpg

# С более строгим порогом
python recognize_faces.py --image test_photo.jpg --tolerance 0.5
```

### Шаг 3: Распознавание с веб-камеры

```bash
# Запустить распознавание с камеры
python recognize_faces.py --camera

# С использованием второй камеры
python recognize_faces.py --camera --camera-id 1

# С сохранением скриншотов
python recognize_faces.py --camera --save-results

# С использованием кэша для ускорения
python recognize_faces.py --camera --use-cache
```

**Нажатие 'q' останавливает распознавание с камеры.**

### Программное использование

```python
from face_recognition_system import FaceRecognizer

# Инициализация
recognizer = FaceRecognizer(
    database_path="faces_database",
    tolerance=0.6,
    model='hog'
)

# Добавление нового человека
recognizer.add_person(
    person_name="Anna Smith",
    image_paths=["anna1.jpg", "anna2.jpg"]
)

# Распознавание на изображении
results = recognizer.recognize_faces_in_image("group_photo.jpg")

for face in results:
    if face['is_known']:
        print(f"Распознан: {face['name']} (уверенность: {face['confidence']:.2f})")
    else:
        print("Неизвестное лицо")

# Распознавание с камеры
recognizer.recognize_from_camera(camera_id=0)

# Статистика базы данных
stats = recognizer.get_database_stats()
print(f"Людей в базе: {stats['total_people']}")
print(f"Изображений: {stats['total_images']}")

# Сохранение/загрузка кэша
recognizer.save_database_cache()
recognizer.load_database_cache()
```

## 🔧 Настройка параметров

### Порог распознавания (tolerance)

Меньшее значение = более строгое распознавание:
- `0.6` (по умолчанию) - баланс между точностью и полнотой
- `0.5` - более строгое, меньше ложных срабатываний
- `0.7` - более мягкое, больше совпадений

### Модель обнаружения (model)

- `hog` (Histogram of Oriented Gradients):
  - ✅ Быстро, работает на CPU
  - ✅ Хорошо для лиц анфас
  - ❌ Может пропускать повернутые лица
  
- `cnn` (Convolutional Neural Network):
  - ✅ Точнее, особенно для сложных ракурсов
  - ✅ Лучше при плохом освещении
  - ❌ Медленнее без GPU

## 📊 Как это работает

1. **Обнаружение лица**: Нахождение координат лиц на изображении (HOG или CNN)
2. **Выделение признаков**: Преобразование лица в 128-мерный вектор (face encoding)
3. **Сравнение**: Вычисление евклидова расстояния до известных лиц
4. **Классификация**: Если расстояние < tolerance → лицо распознано

## 💡 Рекомендации

### Для лучших результатов:

1. **Качество фото**:
   - Используйте четкие фотографии
   - Хорошее освещение
   - Лицо должно быть хорошо видно

2. **Количество фото**:
   - Минимум 2-3 фото на человека
   - Разные ракурсы и выражения лица
   - Разное освещение

3. **Настройка порога**:
   - Начните с 0.6
   - Если много ложных срабатываний → уменьшите до 0.5
   - Если не распознает известные лица → увеличьте до 0.7

## ⚠️ Возможные проблемы

### "Лица не найдены на изображении"
- Убедитесь, что лицо хорошо видно
- Попробуйте модель CNN вместо HOG
- Проверьте освещение и угол

### "ModuleNotFoundError: No module named 'face_recognition'"
```bash
pip install -r requirements.txt
```

### Ошибки компиляции dlib
Используйте conda:
```bash
conda install -c conda-forge dlib
pip install face_recognition
```

## 📝 Лицензия

Проект использует библиотеку face_recognition с лицензией MIT.

## 🔗 Полезные ссылки

- [face_recognition на GitHub](https://github.com/ageitgey/face_recognition)
- [dlib библиотека](http://dlib.net/)
- [LFW Benchmark](http://vis-www.cs.umass.edu/lfw/)
