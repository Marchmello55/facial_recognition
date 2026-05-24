#!/usr/bin/env python3
"""
Скрипт для добавления новых лиц в базу данных распознавания.

Использование:
    python add_face.py --name "Имя Фамилия" --images image1.jpg image2.jpg
    
Примеры:
    python add_face.py --name "Ivan Petrov" --images ivan1.jpg ivan2.jpg
    python add_face.py --name "Maria Sidorova" --images maria.png
"""

import argparse
import sys
from pathlib import Path

# Импортируем наш модуль
from face_recognition_system import FaceRecognizer


def main():
    parser = argparse.ArgumentParser(
        description='Добавление нового человека в базу данных распознавания лиц',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s --name "Ivan Petrov" --images photo1.jpg photo2.jpg
  %(prog)s --name "Maria Sidorova" --images maria.png --tolerance 0.5
        """
    )
    
    parser.add_argument(
        '--name', '-n',
        type=str,
        required=True,
        help='Имя человека (будет использовано как название папки)'
    )
    
    parser.add_argument(
        '--images', '-i',
        type=str,
        nargs='+',
        required=True,
        help='Пути к изображениям лица (можно несколько)'
    )
    
    parser.add_argument(
        '--database', '-d',
        type=str,
        default='faces_database',
        help='Путь к папке базы данных (по умолчанию: faces_database)'
    )
    
    parser.add_argument(
        '--tolerance', '-t',
        type=float,
        default=0.6,
        help='Порог распознавания (по умолчанию: 0.6, меньше = строже)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=['hog', 'cnn'],
        default='hog',
        help='Модель обнаружения лиц: hog (быстрее) или cnn (точнее, но медленнее)'
    )
    
    args = parser.parse_args()
    
    # Проверка существования файлов
    valid_images = []
    for image_path in args.images:
        if Path(image_path).exists():
            valid_images.append(image_path)
        else:
            print(f"⚠️  Файл не найден: {image_path}")
    
    if not valid_images:
        print("❌ Нет доступных изображений для обработки")
        sys.exit(1)
    
    print("=" * 60)
    print(f"ДОБАВЛЕНИЕ ЛИЦА: {args.name}")
    print("=" * 60)
    print(f"📁 База данных: {args.database}")
    print(f"🎯 Порог распознавания: {args.tolerance}")
    print(f"⚙️  Модель: {args.model}")
    print(f"📷 Изображений: {len(valid_images)}")
    print("-" * 60)
    
    # Создаем распознаватель
    recognizer = FaceRecognizer(
        database_path=args.database,
        tolerance=args.tolerance,
        model=args.model
    )
    
    # Добавляем человека
    success = recognizer.add_person(args.name, valid_images)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ УСПЕШНО ДОБАВЛЕНО!")
        print("=" * 60)
        
        # Показываем обновленную статистику
        stats = recognizer.get_database_stats()
        print(f"\n📊 Обновленная статистика:")
        print(f"   Всего людей: {stats['total_people']}")
        print(f"   Всего изображений: {stats['total_images']}")
        
        if stats['people_names']:
            print(f"   Люди в базе: {', '.join(stats['people_names'])}")
        
        # Сохраняем кэш для ускорения следующей загрузки
        recognizer.save_database_cache()
        
    else:
        print("\n" + "=" * 60)
        print("❌ НЕ УДАЛОСЬ ДОБАВИТЬ ЛИЦО")
        print("=" * 60)
        print("\nВозможные причины:")
        print("  • На изображениях не найдено лиц")
        print("  • Лица слишком маленькие или размытые")
        print("  • Плохое освещение на фото")
        print("  • Лицо повернуто в профиль (>90 градусов)")
        print("\nРекомендации:")
        print("  • Используйте четкие фотографии анфас")
        print("  • Хорошее освещение")
        print("  • Несколько разных ракурсов одного человека")
        sys.exit(1)


if __name__ == "__main__":
    main()
