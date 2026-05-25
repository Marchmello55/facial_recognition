"""
Модуль для работы с базой данных лиц на SQLite
Хранит энкодинги лиц и имена пользователей в SQL базе данных
"""
import sqlite3
import pickle
import os
from typing import List, Tuple, Optional
import numpy as np


class FaceDatabase:
    """Класс для управления базой данных лиц на SQLite"""
    
    def __init__(self, db_path: str = "faces.db"):
        """
        Инициализация базы данных SQLite
        
        Args:
            db_path: Путь к файлу базы данных SQLite
        """
        self.db_path = db_path
        self.known_faces: List[np.ndarray] = []  # Энкодинги известных лиц (кэш)
        self.known_names: List[str] = []  # Имена соответствующих лиц (кэш)
        self._init_db()
        self._load_to_cache()
    
    def _init_db(self):
        """Инициализация таблицы в базе данных SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Создание таблицы faces
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                encoding BLOB NOT NULL,
                image_data BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"База данных SQLite инициализирована: {self.db_path}")
    
    def _load_to_cache(self):
        """Загрузка данных из SQLite в кэш памяти"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT name, encoding FROM faces')
        rows = cursor.fetchall()
        
        self.known_names = []
        self.known_faces = []
        
        for name, encoding_blob in rows:
            try:
                encoding = pickle.loads(encoding_blob)
                self.known_names.append(name)
                self.known_faces.append(encoding)
            except Exception as e:
                print(f"Ошибка загрузки эмбеддинга для {name}: {e}")
        
        conn.close()
        print(f"Загружено {len(self.known_names)} лиц из базы данных в кэш")
    
    def load(self) -> bool:
        """
        Загрузка базы данных (для совместимости интерфейса)
        
        Returns:
            True если загрузка успешна
        """
        self._load_to_cache()
        return True
    
    def save(self) -> bool:
        """
        Сохранение базы данных (для совместимости интерфейса)
        
        Returns:
            True если сохранение успешно
        """
        return True
    
    def add_face(self, encoding: np.ndarray, name: str) -> bool:
        """
        Добавление нового лица в базу SQLite
        
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
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Сериализуем эмбеддинг в байты
            encoding_blob = pickle.dumps(encoding)
            
            # Вставляем или обновляем запись
            cursor.execute('''
                INSERT OR REPLACE INTO faces (name, encoding)
                VALUES (?, ?)
            ''', (name.strip(), encoding_blob))
            
            conn.commit()
            conn.close()
            
            # Обновляем кэш
            self._load_to_cache()
            
            print(f"Добавлено лицо в SQLite: {name}")
            return True
        except Exception as e:
            print(f"Ошибка при добавлении лица в SQLite: {e}")
            return False
    
    def remove_face(self, name: str) -> bool:
        """
        Удаление лица из базы по имени
        
        Args:
            name: Имя человека для удаления
            
        Returns:
            True если удаление успешно
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM faces WHERE name = ?', (name,))
            
            if cursor.rowcount > 0:
                conn.commit()
                conn.close()
                
                # Обновляем кэш
                self._load_to_cache()
                
                print(f"Удалено лицо из SQLite: {name}")
                return True
            else:
                conn.close()
                print(f"Лицо '{name}' не найдено в базе SQLite")
                return False
        except Exception as e:
            print(f"Ошибка при удалении лица: {e}")
            return False
    
    def get_all_names(self) -> List[str]:
        """Получить список всех имен в базе"""
        return self.known_names.copy()
    
    def get_face_count(self) -> int:
        """Получить количество лиц в базе"""
        return len(self.known_names)
    
    def clear_database(self) -> bool:
        """
        Очистка всей базы данных SQLite
        
        Returns:
            True если очистка успешна
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM faces')
            
            conn.commit()
            conn.close()
            
            # Обновляем кэш
            self._load_to_cache()
            
            print("База данных SQLite очищена")
            return True
        except Exception as e:
            print(f"Ошибка при очистке базы данных: {e}")
            return False
