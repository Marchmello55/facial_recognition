"""
Модуль для работы с базой данных лиц
Хранит энкодинги лиц и имена пользователей
"""
import pickle
import os
from typing import List, Tuple, Optional
import numpy as np


class FaceDatabase:
    """Класс для управления базой данных лиц"""
    
    def __init__(self, db_path: str = "face_database.pkl"):
        """
        Инициализация базы данных
        
        Args:
            db_path: Путь к файлу базы данных
        """
        self.db_path = db_path
        self.known_faces: List[np.ndarray] = []  # Энкодинги известных лиц
        self.known_names: List[str] = []  # Имена соответствующих лиц
        self.load()
    
    def load(self) -> bool:
        """
        Загрузка базы данных из файла
        
        Returns:
            True если загрузка успешна, False если файл не существует
        """
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_faces = data.get('faces', [])
                    self.known_names = data.get('names', [])
                print(f"База данных загружена: {len(self.known_names)} лиц")
                return True
            except Exception as e:
                print(f"Ошибка загрузки базы данных: {e}")
                return False
        else:
            print("База данных не найдена, создана новая")
            return False
    
    def save(self) -> bool:
        """
        Сохранение базы данных в файл
        
        Returns:
            True если сохранение успешно
        """
        try:
            with open(self.db_path, 'wb') as f:
                data = {
                    'faces': self.known_faces,
                    'names': self.known_names
                }
                pickle.dump(data, f)
            print(f"База данных сохранена: {len(self.known_names)} лиц")
            return True
        except Exception as e:
            print(f"Ошибка сохранения базы данных: {e}")
            return False
    
    def add_face(self, encoding: np.ndarray, name: str) -> bool:
        """
        Добавление нового лица в базу
        
        Args:
            encoding: Энкодинг лица (вектор признаков)
            name: Имя человека
            
        Returns:
            True если добавление успешно
        """
        if encoding is None or len(encoding) == 0:
            print("Ошибка: пустой энкодинг")
            return False
        
        if not name or name.strip() == "":
            print("Ошибка: пустое имя")
            return False
        
        self.known_faces.append(encoding)
        self.known_names.append(name.strip())
        self.save()
        print(f"Добавлено лицо: {name}")
        return True
    
    def remove_face(self, name: str) -> bool:
        """
        Удаление лица из базы по имени
        
        Args:
            name: Имя человека для удаления
            
        Returns:
            True если удаление успешно
        """
        if name in self.known_names:
            index = self.known_names.index(name)
            self.known_names.pop(index)
            self.known_faces.pop(index)
            self.save()
            print(f"Удалено лицо: {name}")
            return True
        else:
            print(f"Лицо '{name}' не найдено в базе")
            return False
    
    def get_all_names(self) -> List[str]:
        """Получить список всех имен в базе"""
        return self.known_names.copy()
    
    def get_face_count(self) -> int:
        """Получить количество лиц в базе"""
        return len(self.known_names)
    
    def clear_database(self) -> bool:
        """
        Очистка всей базы данных
        
        Returns:
            True если очистка успешна
        """
        self.known_faces = []
        self.known_names = []
        self.save()
        print("База данных очищена")
        return True
