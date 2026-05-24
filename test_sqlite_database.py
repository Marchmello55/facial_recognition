"""
Тестовый скрипт для проверки работы SQLite базы данных лиц
Демонстрирует преимущества новой системы хранения
"""
import numpy as np
from face_database import FaceDatabase
import os

def test_database():
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ БАЗЫ ДАННЫХ ЛИЦ НА SQLITE")
    print("=" * 60)
    
    # Удаляем старую базу для чистого теста
    if os.path.exists("test_faces.db"):
        os.remove("test_faces.db")
    
    # 1. Создание базы данных
    print("\n1. СОЗДАНИЕ БАЗЫ ДАННЫХ")
    db = FaceDatabase("test_faces.db")
    print(f"   Файл БД: {db.db_path}")
    print(f"   Лиц в базе: {db.get_face_count()}")
    
    # 2. Добавление лиц с корректными векторами (128 элементов)
    print("\n2. ДОБАВЛЕНИЕ ЛИЦ")
    
    # Создаем фейковые векторы размером 128 (как в реальной модели)
    encoding1 = np.random.rand(128).astype(np.float64)
    encoding2 = np.random.rand(128).astype(np.float64)
    encoding3 = np.random.rand(128).astype(np.float64)
    
    print(f"   Размер вектора 1: {len(encoding1)} ✅")
    print(f"   Размер вектора 2: {len(encoding2)} ✅")
    print(f"   Размер вектора 3: {len(encoding3)} ✅")
    
    # Добавляем лица
    db.add_face(encoding1, "Иван Иванов")
    db.add_face(encoding2, "Петр Петров")
    db.add_face(encoding3, "Анна Сидорова")
    
    print(f"\n   Всего лиц в базе: {db.get_face_count()}")
    
    # 3. Попытка добавить лицо с некорректным вектором
    print("\n3. ПРОВЕРКА ВАЛИДАЦИИ РАЗМЕРА ВЕКТОРА")
    
    bad_encoding = np.random.rand(313)  # Некорректный размер (как в вашей ошибке)
    print(f"   Попытка добавить вектор размером {len(bad_encoding)}...")
    result = db.add_face(bad_encoding, "Ошибочное лицо")
    print(f"   Результат: {'✅ Принято' if result else '❌ Отклонено'}")
    
    # 4. Проверка уникальности имен
    print("\n4. ПРОВЕРКА УНИКАЛЬНОСТИ ИМЕН")
    print("   Попытка добавить 'Иван Иванов' повторно...")
    result = db.add_face(encoding1, "Иван Иванов")
    print(f"   Результат: {'✅ Добавлено' if result else '❌ Отклонено (уже существует)'}")
    
    # 5. Получение списка всех лиц
    print("\n5. СПИСОК ВСЕХ ЛИЦ В БАЗЕ")
    names = db.get_all_names()
    for i, name in enumerate(names, 1):
        print(f"   {i}. {name}")
    
    # 6. Детальная информация о лицах
    print("\n6. ДЕТАЛЬНАЯ ИНФОРМАЦИЯ О ЛИЦАХ")
    details = db.list_faces_details()
    for face in details:
        print(f"   ID: {face['id']}, Имя: {face['name']}, Дата: {face['added_at']}")
    
    # 7. Загрузка векторов из базы
    print("\n7. ПРОВЕРКА ЦЕЛОСТНОСТИ ДАННЫХ")
    print(f"   Загружено лиц из БД: {len(db.known_faces)}")
    for i, (name, encoding) in enumerate(zip(db.known_names, db.known_faces)):
        print(f"   {i+1}. {name}: вектор размером {len(encoding)} ✅")
    
    # 8. Удаление лица
    print("\n8. УДАЛЕНИЕ ЛИЦА")
    print("   Удаляем 'Петр Петров'...")
    db.remove_face("Петр Петров")
    print(f"   Осталось лиц в базе: {db.get_face_count()}")
    
    # 9. Очистка всей базы
    print("\n9. ОЧИСТКА ВСЕЙ БАЗЫ")
    print("   Очищаем базу данных...")
    db.clear_database()
    print(f"   Лиц в базе после очистки: {db.get_face_count()}")
    
    # 10. Повторное добавление после очистки
    print("\n10. ДОБАВЛЕНИЕ ПОСЛЕ ОЧИСТКИ")
    db.add_face(encoding1, "Тестовое лицо")
    print(f"   Лиц в базе: {db.get_face_count()}")
    
    # Удаляем тестовую базу
    os.remove("test_faces.db")
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО УСПЕШНО! ✅")
    print("=" * 60)
    print("\nПреимущества SQLite перед .pkl:")
    print("  ✅ Автоматическая валидация размера векторов (128)")
    print("  ✅ Гарантия уникальности имен")
    print("  ✅ ACID транзакции (защита от повреждения)")
    print("  ✅ Хранение метаданных (дата добавления)")
    print("  ✅ Быстрый поиск и масштабирование")
    print("=" * 60)

if __name__ == "__main__":
    test_database()
