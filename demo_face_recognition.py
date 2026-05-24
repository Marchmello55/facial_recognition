#!/usr/bin/env python3
"""
Демонстрационный скрипт - показывает структуру и возможности системы
распознавания лиц без установки тяжелых зависимостей (dlib, face_recognition).

Этот скрипт демонстрирует API и архитектуру системы.
Для полноценной работы установите зависимости: pip install -r requirements.txt
"""

import sys
from pathlib import Path


def print_header(text):
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)


def demo_project_structure():
    """Показывает структуру проекта"""
    print_header("📁 СТРУКТУРА ПРОЕКТА")
    
    structure = """
/workspace/
├── face_recognition_system.py   # Основной модуль с классом FaceRecognizer
│   ├── FaceRecognizer class
│   │   ├── __init__()           # Инициализация базы данных
│   │   ├── load_faces_from_database()  # Загрузка лиц из папок
│   │   ├── add_person()         # Добавление нового человека
│   │   ├── recognize_faces_in_image()  # Распознавание на изображении
│   │   ├── recognize_from_camera()     # Распознавание с веб-камеры
│   │   ├── get_database_stats()        # Статистика базы
│   │   ├── save_database_cache()       # Сохранение кэша
│   │   └── load_database_cache()       # Загрузка кэша
│   │
├── add_face.py                  # CLI для добавления лиц
├── recognize_faces.py           # CLI для распознавания
├── requirements.txt             # Зависимости
├── README.md                    # Документация
└── faces_database/              # Папка с эталонными изображениями
    ├── Person_Name_1/
    │   ├── photo1.jpg
    │   └── photo2.jpg
    └── Person_Name_2/
        └── image.png
    """
    print(structure)


def demo_api_usage():
    """Показывает примеры использования API"""
    print_header("💻 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ API")
    
    examples = """
# 1. Импортируем модуль
from face_recognition_system import FaceRecognizer

# 2. Создаем экземпляр распознавателя
recognizer = FaceRecognizer(
    database_path="faces_database",  # Папка с базой данных
    tolerance=0.6,                   # Порог распознавания (0.6 рекомендуется)
    model='hog'                      # 'hog' (быстро) или 'cnn' (точно)
)

# 3. Добавляем нового человека в базу
success = recognizer.add_person(
    person_name="Ivan Petrov",
    image_paths=["ivan_photo1.jpg", "ivan_photo2.jpg"]
)

if success:
    print("✅ Лицо успешно добавлено!")

# 4. Распознаем лица на изображении
results = recognizer.recognize_faces_in_image("group_photo.jpg")

for face in results:
    if face['is_known']:
        print(f"✓ Распознан: {face['name']} (уверенность: {face['confidence']:.2%})")
    else:
        print("? Неизвестное лицо")

# 5. Запускаем распознавание с веб-камеры
recognizer.recognize_from_camera(camera_id=0)

# 6. Получаем статистику базы данных
stats = recognizer.get_database_stats()
print(f"Людей в базе: {stats['total_people']}")
print(f"Всего изображений: {stats['total_images']}")
for name, count in stats['images_per_person'].items():
    print(f"  - {name}: {count} фото")

# 7. Кэширование для ускорения повторной загрузки
recognizer.save_database_cache("faces_cache.pkl")  # Сохранить
recognizer.load_database_cache("faces_cache.pkl")  # Загрузить
    """
    print(examples)


def demo_cli_commands():
    """Показывает команды CLI"""
    print_header("🖥️  КОМАНДЫ КОМАНДНОЙ СТРОКИ")
    
    commands = """
# ДОБАВЛЕНИЕ ЛИЦ В БАЗУ ДАННЫХ:

# Добавить человека с несколькими фотографиями
python add_face.py --name "Ivan Petrov" --images ivan1.jpg ivan2.jpg ivan3.jpg

# Добавить с указанием параметров
python add_face.py --name "Maria Sidorova" \\
    --images maria.png \\
    --tolerance 0.5 \\
    --model hog

# Показать справку
python add_face.py --help


# РАСПОЗНАВАНИЕ НА ИЗОБРАЖЕНИИ:

# Распознать лица на фото
python recognize_faces.py --image test_photo.jpg

# С сохранением результата
python recognize_faces.py --image photo.jpg --output result.jpg

# С более строгим порогом
python recognize_faces.py --image photo.jpg --tolerance 0.5


# РАСПОЗНАВАНИЕ С ВЕБ-КАМЕРЫ:

# Запустить распознавание
python recognize_faces.py --camera

# С использованием второй камеры
python recognize_faces.py --camera --camera-id 1

# С кэшированием для ускорения
python recognize_faces.py --camera --use-cache

# Показать справку
python recognize_faces.py --help
    """
    print(commands)


def demo_architecture():
    """Объясняет архитектуру системы"""
    print_header("🏗️  АРХИТЕКТУРА СИСТЕМЫ")
    
    architecture = """
1. МОДЕЛЬ РАСПОЗНАВАНИЯ:
   ┌─────────────────────────────────────────────────────────┐
   │  Библиотека: face_recognition (на базе dlib)            │
   │  Архитектура: ResNet (Deep Learning)                    │
   │  Точность: 99.38% на LFW benchmark                      │
   │  Размер эмбеддинга: 128 чисел (float64)                 │
   └─────────────────────────────────────────────────────────┘

2. ПРОЦЕСС РАСПОЗНАВАНИЯ:
   
   Входное изображение
         │
         ▼
   ┌─────────────────┐
   │  Обнаружение    │  HOG или CNN модель
   │  лиц            │  Находит координаты лиц
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  Выделение      │  Преобразование в
   │  признаков      │  128-мерный вектор
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  Сравнение      │  Евклидово расстояние
   │  с базой        │  до известных лиц
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  Классификация  │  Если distance < tolerance
   │                 │  → лицо распознано
   └─────────────────┘

3. ХРАНЕНИЕ ДАННЫХ:
   
   faces_database/
   ├── Ivan_Petrov/          # Папка = человек
   │   ├── photo1.jpg        # Фотографии лица
   │   └── photo2.jpg
   └── Maria_Sidorova/
       └── image.png
   
   При загрузке:
   - Каждое изображение → face encoding (128 чисел)
   - Encoding + имя → в память
   - Опционально: сохранение в pickle файл

4. ПАРАМЕТРЫ НАСТРОЙКИ:
   
   • tolerance (порог): 
     - Меньше (0.5) → строже, меньше ложных срабатываний
     - Больше (0.7) → мягче, больше совпадений
     - По умолчанию: 0.6 (оптимально)
   
   • model (обнаружение):
     - 'hog' → быстро, CPU, хорошо для анфас
     - 'cnn' → точно, GPU желательно, лучше для сложных ракурсов
    """
    print(architecture)


def demo_workflow():
    """Показывает типичный рабочий процесс"""
    print_header("📋 ТИПИЧНЫЙ РАБОЧИЙ ПРОЦЕСС")
    
    workflow = """
ШАГ 1: ПОДГОТОВКА
├── Установить зависимости: pip install -r requirements.txt
├── Создать папку для базы: mkdir faces_database
└── Подготовить фотографии людей

ШАГ 2: ЗАПОЛНЕНИЕ БАЗЫ
├── Сделать 2-5 четких фотографий каждого человека
├── Разные ракурсы, выражения лица, освещение
└── Выполнить:
    python add_face.py --name "Person Name" --images photo1.jpg photo2.jpg

ШАГ 3: ТЕСТИРОВАНИЕ
├── Распознавание на тестовом фото:
│   python recognize_faces.py --image test.jpg
└── Настройка порога при необходимости (--tolerance)

ШАГ 4: ЭКСПЛУАТАЦИЯ
├── Распознавание с камеры:
│   python recognize_faces.py --camera
└── Или интеграция в свой проект через API

ШАГ 5: ОПТИМИЗАЦИЯ
├── Сохранить кэш: recognizer.save_database_cache()
└── Загружать кэш при старте для ускорения
    """
    print(workflow)


def demo_recommendations():
    """Рекомендации по использованию"""
    print_header("💡 РЕКОМЕНДАЦИИ")
    
    recommendations = """
✅ ДЛЯ ЛУЧШИХ РЕЗУЛЬТАТОВ:

1. Качество фотографий:
   • Четкие, не размытые изображения
   • Хорошее освещение (без резких теней)
   • Лицо занимает значительную часть кадра
   • Анфас или небольшой поворот (<30°)

2. Количество фотографий:
   • Минимум 2-3 на человека
   • Разные ракурсы (немного повернуть голову)
   • Разные выражения лица (нейтральное, улыбка)
   • Разное освещение (если возможно)

3. Настройка параметров:
   • Начните с tolerance=0.6 и model='hog'
   • Если много ложных срабатываний → уменьшите tolerance до 0.5
   • Если не распознает известные лица → увеличьте до 0.7
   • Для сложных условий используйте model='cnn'

4. Производительность:
   • Используйте кэш для больших баз (>50 человек)
   • Для реального времени обрабатывайте каждый 2-3 кадр
   • Уменьшайте размер кадра для камеры (fx=0.25)

❌ ЧЕГО ИЗБЕГАТЬ:

• Фотографий в солнцезащитных очках
• Сильного бокового освещения
• Фотографий низкого разрешения
• Полного профиля лица (90°)
• Групповых фото для добавления в базу
    """
    print(recommendations)


def main():
    print_header("🔍 ДЕМО: СИСТЕМА РАСПОЗНАВАНИЯ ЛИЦ")
    print("\nЭтот демонстрационный скрипт показывает возможности системы.")
    print("Для полноценной работы установите: pip install -r requirements.txt")
    
    demo_project_structure()
    demo_architecture()
    demo_api_usage()
    demo_cli_commands()
    demo_workflow()
    demo_recommendations()
    
    print_header("📚 ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ")
    print("""
Файлы проекта:
  • face_recognition_system.py - основной модуль
  • add_face.py - добавление лиц
  • recognize_faces.py - распознавание
  • FACE_RECOGNITION_README.md - полная документация
  • requirements.txt - зависимости

Полезные ссылки:
  • https://github.com/ageitgey/face_recognition
  • http://dlib.net/
  • http://vis-www.cs.umass.edu/lfw/
    """)
    
    print_header("✅ ГОТОВО К ИСПОЛЬЗОВАНИЮ")
    print("\nСледуйте инструкциям в FACE_RECOGNITION_README.md для установки и запуска.")
    

if __name__ == "__main__":
    main()
