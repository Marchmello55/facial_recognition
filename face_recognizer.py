"""
Модуль для распознавания лиц с использованием OpenCV
Легковесная альтернатива, не требующая тяжелых зависимостей
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from face_database import FaceDatabase


class FaceRecognizer:
    """Класс для распознавания лиц в реальном времени с использованием OpenCV"""
    
    def __init__(self, tolerance: float = 0.35):
        """
        Инициализация распознавателя лиц
        
        Args:
            tolerance: Порог схожести для распознавания (меньше = строже)
        """
        self.tolerance = tolerance
        self.database = FaceDatabase()
        
        # Загрузка каскада Хаара для детекции лиц
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            print("Предупреждение: Не удалось загрузить каскад Хаара")
        
        self.process_every_n_frames = 2
        self.frame_count = 0
        self.face_locations = []
        self.face_names = []
        self.face_embeddings = []
    
    def _extract_embedding(self, image: np.ndarray, face_rect: tuple) -> Optional[np.ndarray]:
        """
        Извлечение эмбеддинга лица из изображения
        
        Args:
            image: BGR изображение
            face_rect: Координаты лица (x, y, width, height)
            
        Returns:
            Вектор признаков лица или None
        """
        x, y, w, h = face_rect
        
        # Вырезаем область лица с небольшим запасом
        margin = int(w * 0.15)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image.shape[1], x + w + margin)
        y2 = min(image.shape[0], y + h + margin)
        
        face_crop = image[y1:y2, x1:x2]
        
        if face_crop.size == 0 or face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            return None
        
        # Ресайз к фиксированному размеру
        face_crop = cv2.resize(face_crop, (112, 112))
        
        # Преобразуем в grayscale
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
        # Применяем эквалайзинг гистограммы для улучшения контраста
        normalized = cv2.equalizeHist(gray).astype(np.float32) / 255.0
        
        # Создаем эмбеддинг на основе гистограмм ориентированных градиентов (упрощенно)
        # Гистограмма интенсивностей
        hist_full = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist_full = cv2.normalize(hist_full, hist_full).flatten()
        
        # Разбиваем изображение на регионы и считаем гистограмму для каждого
        h, w = gray.shape
        region_hist = []
        for i in range(0, h, h//4):
            for j in range(0, w, w//4):
                region = gray[i:i+h//4, j:j+w//4]
                if region.size > 0:
                    hist_region = cv2.calcHist([region], [0], None, [16], [0, 256])
                    hist_region = cv2.normalize(hist_region, hist_region).flatten()
                    region_hist.extend(hist_region)
        
        # Статистики изображения
        stats = np.array([
            np.mean(normalized),
            np.std(normalized),
            np.min(normalized),
            np.max(normalized),
            np.median(normalized),
            np.percentile(normalized, 25),
            np.percentile(normalized, 75)
        ])
        
        # Объединяем все признаки
        embedding = np.concatenate([hist_full, np.array(region_hist), stats])
        
        # Нормализуем итоговый вектор
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
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
            image = cv2.imread(image_path)
            if image is None:
                print(f"Не удалось загрузить изображение {image_path}")
                return False
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                print(f"Лица не найдены на изображении {image_path}")
                return False
            
            if len(faces) > 1:
                print(f"Найдено {len(faces)} лиц. Будет использовано первое.")
            
            x, y, w, h = faces[0]
            embedding = self._extract_embedding(image, (x, y, w, h))
            
            if embedding is None:
                print("Не удалось извлечь признаки лица")
                return False
            
            return self.database.add_face(embedding, name)
        
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
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                print("Лица не найдены в кадре")
                return False
            
            if len(faces) > 1:
                print(f"Найдено {len(faces)} лиц. Будет использовано первое.")
            
            x, y, w, h = faces[0]
            embedding = self._extract_embedding(frame, (x, y, w, h))
            
            if embedding is None:
                print("Не удалось извлечь признаки лица")
                return False
            
            return self.database.add_face(embedding, name)
        
        except Exception as e:
            print(f"Ошибка при добавлении лица из кадра: {e}")
            return False
    
    def _calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Вычисление косинусного сходства между двумя эмбеддингами"""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = np.dot(emb1, emb2) / (norm1 * norm2)
        return cosine_sim
    
    def recognize_faces(self, frame: np.ndarray) -> Tuple[List[Tuple], List[str]]:
        """
        Распознавание лиц в кадре
        
        Args:
            frame: Кадр изображения (BGR формат)
            
        Returns:
            Кортеж из (locations, names) - координаты лиц и их имена
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Обрабатываем только каждый N-й кадр для производительности
        if self.frame_count % self.process_every_n_frames == 0:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            self.face_locations = []
            self.face_names = []
            self.face_embeddings = []
            
            for (x, y, w, h) in faces:
                self.face_locations.append((x, y, x + w, y + h))
                
                # Извлекаем эмбеддинг для текущего лица
                embedding = self._extract_embedding(frame, (x, y, w, h))
                self.face_embeddings.append(embedding)
                
                name = "Unknown"
                
                if embedding is not None and len(self.database.known_faces) > 0:
                    best_similarity = 0
                    best_name = ""
                    
                    for i, known_emb in enumerate(self.database.known_faces):
                        similarity = self._calculate_similarity(embedding, known_emb)
                        
                        if similarity > best_similarity and similarity > (1 - self.tolerance):
                            best_similarity = similarity
                            best_name = self.database.known_names[i]
                    
                    if best_name:
                        name = best_name
                
                self.face_names.append(name)
        
        self.frame_count += 1
        
        return self.face_locations, self.face_names
    
    def draw_results(self, frame: np.ndarray, locations: List[Tuple], names: List[str]) -> np.ndarray:
        """
        Отрисовка результатов распознавания на кадре
        
        Args:
            frame: Кадр изображения
            locations: Координаты лиц (x1, y1, x2, y2)
            names: Имена распознанных лиц
            
        Returns:
            Кадр с отрисованными результатами
        """
        for (x1, y1, x2, y2), name in zip(locations, names):
            # Цвет рамки: зеленый для известных, красный для неизвестных
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            # Рисуем рамку вокруг лица
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Рисуем прямоугольник для имени
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
            
            # Пишем имя
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (x1 + 6, y2 - 6), font, 0.7, (255, 255, 255), 1)
        
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
        self.face_names = []
        self.face_embeddings = []
    
    def release(self):
        """Освобождение ресурсов"""
        pass  # Для OpenCV Cascade не требуется явное освобождение
