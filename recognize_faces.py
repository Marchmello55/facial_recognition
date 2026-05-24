#!/usr/bin/env python3
"""
Скрипт для распознавания лиц на изображениях или с веб-камеры.

Использование:
    # Распознавание на изображении:
    python recognize_faces.py --image photo.jpg
    
    # Распознавание с веб-камеры:
    python recognize_faces.py --camera
    
    # Распознавание с камеры и сохранением результатов:
    python recognize_faces.py --camera --save-results
"""

import argparse
import sys
import cv2
from pathlib import Path

# Импортируем наш модуль
from face_recognition_system import FaceRecognizer


def recognize_from_image(recognizer, image_path, output_path=None):
    """Распознавание лиц на статичном изображении"""
    
    if not Path(image_path).exists():
        print(f"❌ Файл не найден: {image_path}")
        return False
    
    print("=" * 60)
    print(f"РАСПОЗНАВАНИЕ ЛИЦ НА ИЗОБРАЖЕНИИ")
    print("=" * 60)
    print(f"📁 Файл: {image_path}")
    print("-" * 60)
    
    results = recognizer.recognize_faces_in_image(image_path)
    
    if not results:
        print("❌ Лица не найдены на изображении")
        return False
    
    # Загружаем изображение для отрисовки
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Не удалось загрузить изображение")
        return False
    
    # Отрисовка результатов
    for i, result in enumerate(results):
        top, right, bottom, left = result['location']
        
        color = (0, 255, 0) if result['is_known'] else (0, 0, 255)
        
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        
        label = f"{result['name']} ({result['confidence']:.2f})"
        cv2.putText(image, label, (left + 6, bottom - 6), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1)
    
    # Сохранение или отображение результата
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"\n💾 Результат сохранен в: {output_path}")
    else:
        # Отображение в окне
        cv2.imshow('Результат распознавания', image)
        print("\n🖼️  Нажмите любую клавишу для закрытия окна...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Вывод статистики
    known_count = sum(1 for r in results if r['is_known'])
    unknown_count = len(results) - known_count
    
    print("\n" + "=" * 60)
    print("СТАТИСТИКА:")
    print(f"   Всего лиц найдено: {len(results)}")
    print(f"   Распознано (известные): {known_count}")
    print(f"   Не распознано (неизвестные): {unknown_count}")
    print("=" * 60)
    
    return True


def recognize_from_camera(recognizer, camera_id=0, save_results=False):
    """Распознавание лиц с веб-камеры"""
    
    print("=" * 60)
    print("РАСПОЗНАВАНИЕ ЛИЦ С ВЕБ-КАМЕРЫ")
    print("=" * 60)
    print(f"📷 Камера: {camera_id}")
    print(f"💾 Сохранение результатов: {'Да' if save_results else 'Нет'}")
    print("-" * 60)
    
    recognizer.recognize_from_camera(camera_id=camera_id, save_results=save_results)


def main():
    parser = argparse.ArgumentParser(
        description='Распознавание лиц на изображениях или с веб-камеры',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s --image photo.jpg                    # Распознать на изображении
  %(prog)s --image photo.jpg --output out.jpg   # С сохранением результата
  %(prog)s --camera                             # Распознавание с веб-камеры
  %(prog)s --camera --camera-id 1               # Использовать вторую камеру
  %(prog)s --image test.jpg --tolerance 0.5     # Более строгий порог
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    
    group.add_argument(
        '--image', '-i',
        type=str,
        help='Путь к изображению для распознавания'
    )
    
    group.add_argument(
        '--camera', '-c',
        action='store_true',
        help='Запустить распознавание с веб-камеры'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Путь для сохранения результата (для изображений)'
    )
    
    parser.add_argument(
        '--camera-id',
        type=int,
        default=0,
        help='ID камеры (по умолчанию: 0)'
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
    
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Сохранять скриншоты распознанных лиц (для камеры)'
    )
    
    parser.add_argument(
        '--use-cache',
        action='store_true',
        help='Использовать кэш кодирований для ускорения загрузки'
    )
    
    args = parser.parse_args()
    
    print("\n" + "🔍" * 30)
    print("СИСТЕМА РАСПОЗНАВАНИЯ ЛИЦ")
    print("🔍" * 30 + "\n")
    
    # Создаем распознаватель
    recognizer = FaceRecognizer(
        database_path=args.database,
        tolerance=args.tolerance,
        model=args.model
    )
    
    # Пытаемся загрузить кэш если указано
    if args.use_cache:
        recognizer.load_database_cache()
    
    # Проверяем, есть ли лица в базе
    stats = recognizer.get_database_stats()
    
    if stats['total_people'] == 0:
        print("❌ База данных пуста!")
        print("\nСначала добавьте лица используя скрипт add_face.py:")
        print("   python add_face.py --name \"Имя\" --images photo1.jpg photo2.jpg")
        sys.exit(1)
    
    print(f"✅ Загружено {stats['total_images']} изображений {stats['total_people']} человек(а)")
    print(f"   Люди в базе: {', '.join(stats['people_names'])}\n")
    
    # Выполняем распознавание
    if args.image:
        success = recognize_from_image(
            recognizer, 
            args.image, 
            output_path=args.output
        )
        if not success:
            sys.exit(1)
    
    elif args.camera:
        recognize_from_camera(
            recognizer,
            camera_id=args.camera_id,
            save_results=args.save_results
        )
    
    print("\n✅ Готово!")


if __name__ == "__main__":
    main()
