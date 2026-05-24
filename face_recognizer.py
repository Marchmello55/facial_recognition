"""
Модуль для распознавания лиц
Использует библиотеку face_recognition для детекции и распознавания
"""
import cv2
import numpy as np
import face_recognition
from typing import List, Tuple, Optional
from face_database import FaceDatabase


class FaceRecognizer:
    """Класс для распознавания лиц в реальном времени"""
    
    def __init__(self, tolerance: float = 0.6):
        """
        Инициализация распознавателя лиц
        
        Args:
            tolerance: Порог схожести для распознавания (меньше = строже)
        """
        self.tolerance = tolerance
        self.database = FaceDatabase()
        self.process_every_n_frames = 3  # Обрабатывать каждый N-й кадр для производительности
        self.frame_count = 0
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
    
    def add_face_from_image(self, image_path: str, name: str) -> bool:
        """
        Добавление лица из изображения в базу данных
        
        Args:
            image_path: Путь к изображению
            name: Имя человека
            
        Returns:
            True если добавление успешно
        """
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) == 0:
                print(f"Лица не найдены на изображении {image_path}")
                return False
            
            if len(encodings) > 1:
                print(f"Найдено {len(encodings)} лиц. Будет использовано первое.")
            
            encoding = encodings[0]
            return self.database.add_face(encoding, name)
        
        except Exception as e:
            print(f"Ошибка при добавлении лица: {e}")
            return False
    
    def add_face_from_frame(self, frame: np.ndarray, name: str) -> bool:
        """
        Добавление лица из кадра видео в базу данных
        
        Args:
            frame: Кадр изображения (BGR формат от OpenCV)
            name: Имя человека
            
        Returns:
            True если добавление успешно
        """
        try:
            # Конвертация BGR в RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_frame)
            
            if len(encodings) == 0:
                print("Лица не найдены в кадре")
                return False
            
            if len(encodings) > 1:
                print(f"Найдено {len(encodings)} лиц. Будет использовано первое.")
            
            encoding = encodings[0]
            return self.database.add_face(encoding, name)
        
        except Exception as e:
            print(f"Ошибка при добавлении лица из кадра: {e}")
            return False
    
    def recognize_faces(self, frame: np.ndarray) -> Tuple[List[Tuple], List[str]]:
        """
        Распознавание лиц в кадре
        
        Args:
            frame: Кадр изображения (BGR формат)
            
        Returns:
            Кортеж из (locations, names) - координаты лиц и их имена
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Обрабатываем только каждый N-й кадр для производительности
        if self.frame_count % self.process_every_n_frames == 0:
            # Поиск лиц и их энкодингов
            self.face_locations = face_recognition.face_locations(rgb_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_frame, self.face_locations)
            
            self.face_names = []
            
            for face_encoding in self.face_encodings:
                name = "Unknown"
                
                if len(self.database.known_faces) > 0:
                    # Сравнение с известными лицами
                    matches = face_recognition.compare_faces(
                        self.database.known_faces, 
                        face_encoding, 
                        self.tolerance
                    )
                    
                    # Вычисление расстояний до известных лиц
                    face_distances = face_recognition.face_distance(
                        self.database.known_faces, 
                        face_encoding
                    )
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        
                        if matches[best_match_index]:
                            name = self.database.known_names[best_match_index]
                
                self.face_names.append(name)
        
        self.frame_count += 1
        
        return self.face_locations, self.face_names
    
    def draw_results(self, frame: np.ndarray, locations: List[Tuple], names: List[str]) -> np.ndarray:
        """
        Отрисовка результатов распознавания на кадре
        
        Args:
            frame: Кадр изображения
            locations: Координаты лиц
            names: Имена распознанных лиц
            
        Returns:
            Кадр с отрисованными результатами
        """
        for (top, right, bottom, left), name in zip(locations, names):
            # Цвет рамки: зеленый для известных, красный для неизвестных
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            # Рисуем рамку вокруг лица
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Рисуем прямоугольник для имени
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            
            # Пишем имя
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)
        
        return frame
    
    def get_database_info(self) -> dict:
        """Получение информации о базе данных"""
        return {
            'count': self.database.get_face_count(),
            'names': self.database.get_all_names()
        }
    
    def clear_cache(self):
        """Очистка кэша кадров"""
        self.frame_count = 0
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
