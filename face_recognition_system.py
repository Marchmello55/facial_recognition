"""
Проект: Система распознавания лиц с возможностью пополнения библиотеки
Автор: AI Assistant
Дата: 2024

ОПИСАНИЕ РЕШЕНИЯ:

1. ВЫБОР БИБЛИОТЕКИ:
   - Используем `face_recognition` на базе dlib - это одна из самых точных 
     и популярных библиотек для распознавания лиц
   - Модель основана на deep learning (ResNet architecture)
   - Точность: 99.38% на benchmark LFW

2. АРХИТЕКТУРА ПРОЕКТА:
   - face_recognition_system.py - основной модуль с классом FaceRecognizer
   - add_face.py - скрипт для добавления новых лиц в базу
   - recognize_faces.py - скрипт для распознавания лиц на изображениях/веб-камере
   - faces_database/ - папка для хранения эталонных изображений лиц
   - requirements.txt - зависимости проекта

3. КЛЮЧЕВЫЕ РЕШЕНИЯ:
   - Хранение эталонов: изображения лиц + их имена в структуре папок
   - Кодирование лиц: используем 128-мерные векторы признаков (face encodings)
   - Метрика сходства: евклидово расстояние между кодированиями
   - Порог распознавания: 0.6 (оптимальное значение по документации)
   - Поддержка множественных лиц на одном изображении

4. ВОЗМОЖНОСТИ:
   - Добавление новых лиц в базу данных
   - Распознавание на статичных изображениях
   - Распознавание в реальном времени с веб-камеры
   - Сохранение результатов распознавания
   - Обработка нескольких лиц одновременно

5. ТРЕБОВАНИЯ:
   - Python 3.7+
   - dlib (требует компилятор C++)
   - face_recognition
   - opencv-python
   - numpy
"""

import face_recognition
import cv2
import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pickle
from datetime import datetime


class FaceRecognizer:
    """
    Класс для распознавания лиц с возможностью пополнения базы данных.
    
    Attributes:
        database_path: Путь к папке с базой данных лиц
        tolerance: Порог сходства для распознавания (меньше = строже)
        model: Модель для обнаружения лиц ('hog' или 'cnn')
    """
    
    def __init__(self, database_path: str = "faces_database", tolerance: float = 0.6, 
                 model: str = 'hog'):
        """
        Инициализация распознавателя лиц.
        
        Args:
            database_path: Путь к папке с изображениями лиц
            tolerance: Порог евклидова расстояния для совпадения (0.6 рекомендуется)
            model: Модель обнаружения - 'hog' (быстрее) или 'cnn' (точнее, но медленнее)
        """
        self.database_path = Path(database_path)
        self.tolerance = tolerance
        self.model = model
        
        # Создаем базу данных если не существует
        self.database_path.mkdir(parents=True, exist_ok=True)
        
        # Кэш для хранений кодирований лиц
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Загружаем существующие лица при инициализации
        self.load_faces_from_database()
    
    def load_faces_from_database(self) -> int:
        """
        Загрузка всех лиц из базы данных.
        
        Returns:
            Количество загруженных лиц
        """
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Проходим по всем подпапкам (каждая папка - человек)
        for person_folder in self.database_path.iterdir():
            if person_folder.is_dir():
                person_name = person_folder.name
                
                # Загружаем все изображения человека
                for image_path in person_folder.glob("*"):
                    if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        encoding = self._encode_face_from_image(str(image_path))
                        
                        if encoding is not None:
                            self.known_face_encodings.append(encoding)
                            self.known_face_names.append(person_name)
                            print(f"Загружено лицо: {person_name} из {image_path.name}")
        
        print(f"\nВсего загружено {len(self.known_face_names)} лиц в базу")
        return len(self.known_face_names)
    
    def _encode_face_from_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Получение кодирования лица из изображения.
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Кодирование лица или None если лицо не найдено
        """
        try:
            image = face_recognition.load_image_file(image_path)
            
            # Находим все лица на изображении
            face_locations = face_recognition.face_locations(image, model=self.model)
            
            if len(face_locations) == 0:
                print(f"⚠️  Лица не найдены на изображении: {image_path}")
                return None
            
            if len(face_locations) > 1:
                print(f"⚠️  Найдено {len(face_locations)} лиц. Используем первое.")
            
            # Получаем кодирование первого лица
            face_encodings = face_recognition.face_encodings(image, [face_locations[0]])
            
            if len(face_encodings) > 0:
                return face_encodings[0]
            else:
                return None
                
        except Exception as e:
            print(f"❌ Ошибка при обработке изображения {image_path}: {e}")
            return None
    
    def add_person(self, person_name: str, image_paths: List[str]) -> bool:
        """
        Добавление нового человека в базу данных.
        
        Args:
            person_name: Имя человека (будет использовано как название папки)
            image_paths: Список путей к изображениям лица
            
        Returns:
            True если успешно добавлено, False иначе
        """
        if not image_paths:
            print("❌ Не предоставлены изображения для добавления")
            return False
        
        # Создаем папку для человека
        person_folder = self.database_path / person_name
        person_folder.mkdir(exist_ok=True)
        
        added_count = 0
        
        for image_path in image_paths:
            if not os.path.exists(image_path):
                print(f"⚠️  Файл не найден: {image_path}")
                continue
            
            # Копируем изображение в базу
            dest_path = person_folder / os.path.basename(image_path)
            
            try:
                # Пробуем получить кодирование перед копированием
                encoding = self._encode_face_from_image(image_path)
                
                if encoding is not None:
                    # Копируем файл
                    import shutil
                    shutil.copy2(image_path, dest_path)
                    
                    # Добавляем в кэш
                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(person_name)
                    
                    added_count += 1
                    print(f"✅ Добавлено лицо: {person_name} из {os.path.basename(image_path)}")
                else:
                    print(f"⚠️  Лицо не найдено на изображении: {image_path}")
                    
            except Exception as e:
                print(f"❌ Ошибка при добавлении {image_path}: {e}")
        
        if added_count > 0:
            print(f"🎉 Успешно добавлено {added_count} изображений для {person_name}")
            return True
        else:
            print(f"❌ Не удалось добавить ни одного изображения для {person_name}")
            return False
    
    def recognize_faces_in_image(self, image_path: str) -> List[Dict]:
        """
        Распознавание лиц на изображении.
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Список словарей с информацией о распознанных лицах
        """
        try:
            image = face_recognition.load_image_file(image_path)
            
            # Находим все лица
            face_locations = face_recognition.face_locations(image, model=self.model)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            results = []
            
            for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
                result = {
                    'location': face_location,  # (top, right, bottom, left)
                    'name': 'Unknown',
                    'confidence': 0.0,
                    'is_known': False
                }
                
                # Сравниваем с известными лицами
                if len(self.known_face_encodings) > 0:
                    distances = face_recognition.face_distance(
                        self.known_face_encodings, 
                        face_encoding
                    )
                    
                    # Находим ближайшее совпадение
                    best_match_index = np.argmin(distances)
                    
                    if distances[best_match_index] < self.tolerance:
                        result['name'] = self.known_face_names[best_match_index]
                        result['confidence'] = 1 - distances[best_match_index]
                        result['is_known'] = True
                        result['match_distance'] = distances[best_match_index]
                
                results.append(result)
                
                status = "✓" if result['is_known'] else "?"
                print(f"{status} Лицо {i+1}: {result['name']} (уверенность: {result['confidence']:.2f})")
            
            return results
            
        except Exception as e:
            print(f"❌ Ошибка при распознавании: {e}")
            return []
    
    def recognize_from_camera(self, camera_id: int = 0, save_results: bool = False):
        """
        Распознавание лиц в реальном времени с веб-камеры.
        
        Args:
            camera_id: ID камеры (обычно 0)
            save_results: Сохранять ли скриншоты распознанных лиц
        """
        if len(self.known_face_encodings) == 0:
            print("❌ База данных пуста. Добавьте лица перед распознаванием.")
            return
        
        video_capture = cv2.VideoCapture(camera_id)
        
        if not video_capture.isOpened():
            print("❌ Не удалось открыть камеру")
            return
        
        print("\n🎥 Запуск распознавания с камеры...")
        print("Нажмите 'q' для выхода")
        
        frame_number = 0
        
        while True:
            ret, frame = video_capture.read()
            
            if not ret:
                break
            
            # Уменьшаем размер кадра для ускорения обработки
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Обрабатываем каждый 3-й кадр для производительности
            if frame_number % 3 == 0:
                face_locations = face_recognition.face_locations(
                    rgb_small_frame, 
                    model=self.model
                )
                face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, 
                    face_locations
                )
                
                current_faces = []
                
                for face_encoding, face_location in zip(face_encodings, face_locations):
                    name = "Unknown"
                    confidence = 0.0
                    
                    if len(self.known_face_encodings) > 0:
                        distances = face_recognition.face_distance(
                            self.known_face_encodings, 
                            face_encoding
                        )
                        best_match_index = np.argmin(distances)
                        
                        if distances[best_match_index] < self.tolerance:
                            name = self.known_face_names[best_match_index]
                            confidence = 1 - distances[best_match_index]
                    
                    current_faces.append({
                        'name': name,
                        'location': face_location,
                        'confidence': confidence
                    })
            else:
                current_faces = []
            
            # Отрисовка результатов
            for face in current_faces:
                top, right, bottom, left = face['location']
                
                # Масштабируем координаты обратно
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Цвет рамки: зеленый для известных, красный для неизвестных
                color = (0, 255, 0) if face['name'] != "Unknown" else (0, 0, 255)
                
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                # Подпись с именем
                label = f"{face['name']} ({face['confidence']:.2f})"
                cv2.putText(frame, label, (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1)
            
            # Отображение
            cv2.imshow('Face Recognition', frame)
            
            # Выход по 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_number += 1
        
        video_capture.release()
        cv2.destroyAllWindows()
        print("\n📹 Распознавание остановлено")
    
    def get_database_stats(self) -> Dict:
        """
        Получение статистики базы данных.
        
        Returns:
            Словарь со статистикой
        """
        stats = {
            'total_people': 0,
            'total_images': 0,
            'people_names': [],
            'images_per_person': {}
        }
        
        people_count = {}
        
        for name in self.known_face_names:
            people_count[name] = people_count.get(name, 0) + 1
        
        stats['total_people'] = len(people_count)
        stats['total_images'] = len(self.known_face_names)
        stats['people_names'] = list(people_count.keys())
        stats['images_per_person'] = people_count
        
        return stats
    
    def save_database_cache(self, cache_path: str = "faces_cache.pkl"):
        """Сохранение кэша кодирований для быстрой загрузки"""
        cache_data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"💾 Кэш сохранен в {cache_path}")
    
    def load_database_cache(self, cache_path: str = "faces_cache.pkl") -> bool:
        """Загрузка кэша кодирований"""
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.known_face_encodings = cache_data['encodings']
                self.known_face_names = cache_data['names']
                
                print(f"💾 Кэш загружен из {cache_path}")
                print(f"   Лиц в кэше: {len(self.known_face_names)}")
                return True
        except Exception as e:
            print(f"⚠️  Не удалось загрузить кэш: {e}")
        
        return False


# Пример использования
if __name__ == "__main__":
    print("=" * 60)
    print("СИСТЕМА РАСПОЗНАВАНИЯ ЛИЦ")
    print("=" * 60)
    
    # Инициализация
    recognizer = FaceRecognizer(
        database_path="faces_database",
        tolerance=0.6,
        model='hog'  # 'hog' для скорости, 'cnn' для точности
    )
    
    # Показываем статистику
    stats = recognizer.get_database_stats()
    print(f"\n📊 Статистика базы данных:")
    print(f"   Всего людей: {stats['total_people']}")
    print(f"   Всего изображений: {stats['total_images']}")
    
    if stats['people_names']:
        print(f"   Люди в базе: {', '.join(stats['people_names'])}")
    
    print("\n" + "=" * 60)
    print("Для добавления лиц используйте скрипт add_face.py")
    print("Для распознавания используйте recognize_faces.py")
    print("=" * 60)
