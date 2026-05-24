"""
Модуль для работы с базой данных лиц на SQLite
Хранит энкодинги лиц и имена пользователей
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
        Инициализация базы данных
        
        Args:
            db_path: Путь к файлу базы данных SQLite
        """
        self.db_path = db_path
        self.known_faces: List[np.ndarray] = []
        self.known_names: List[str] = []
        self._init_db()
        self.load()
    
    def _init_db(self):
        """Создание таблицы если она не существует"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                encoding BLOB NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load(self) -> bool:
        """
        Загрузка базы данных из SQLite
        
        Returns:
            True если загрузка успешна
        """
        self.known_faces = []
        self.known_names = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name, encoding FROM faces")
            rows = cursor.fetchall()
            conn.close()
            
            for name, encoding_blob in rows:
                try:
                    encoding = pickle.loads(encoding_blob)
                    # Проверка целостности: вектор должен быть размером 128
                    if isinstance(encoding, np.ndarray) and len(encoding) == 128:
                        self.known_faces.append(encoding)
                        self.known_names.append(name)
                    else:
                        print(f"⚠️ Пропущено поврежденное лицо '{name}' (неверный размер вектора: {len(encoding) if hasattr(encoding, '__len__') else 'N/A'})")
                except Exception as e:
                    print(f"⚠️ Ошибка при загрузке лица '{name}': {e}")
            
            print(f"База данных загружена: {len(self.known_names)} лиц")
            return True
            
        except Exception as e:
            print(f"Ошибка загрузки базы данных: {e}")
            return False
    
    def save(self) -> bool:
        """
        Сохранение базы данных в SQLite
        Примечание: В SQLite сохранение происходит при каждом добавлении/удалении
        
        Returns:
            True
        """
        return True
    
    def add_face(self, encoding: np.ndarray, name: str) -> bool:
        """
        Добавление нового лица в базу
        
        Args:
            encoding: Энкодинг лица (вектор признаков, 128 элементов)
            name: Имя человека
            
        Returns:
            True если добавление успешно
        """
        if encoding is None or len(encoding) == 0:
            print("❌ Ошибка: пустой энкодинг")
            return False
        
        if len(encoding) != 128:
            print(f"❌ Ошибка: неверный размер вектора ({len(encoding)}). Ожидается 128.")
            return False
        
        if not name or name.strip() == "":
            print("❌ Ошибка: пустое имя")
            return False
        
        name = name.strip()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Конвертируем numpy массив в бинарные данные
            encoding_blob = pickle.dumps(encoding)
            
            cursor.execute(
                "INSERT OR REPLACE INTO faces (name, encoding) VALUES (?, ?)",
                (name, encoding_blob)
            )
            conn.commit()
            conn.close()
            
            # Обновляем кэш в памяти
            self.load()
            
            print(f"✅ Добавлено лицо: {name}")
            return True
            
        except sqlite3.IntegrityError:
            print(f"❌ Лицо с именем '{name}' уже существует (ошибка уникальности)")
            return False
        except Exception as e:
            print(f"❌ Ошибка добавления лица: {e}")
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
            
            cursor.execute("DELETE FROM faces WHERE name = ?", (name,))
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            
            if deleted > 0:
                # Обновляем кэш в памяти
                self.load()
                print(f"✅ Удалено лицо: {name}")
                return True
            else:
                print(f"❌ Лицо '{name}' не найдено в базе")
                return False
                
        except Exception as e:
            print(f"❌ Ошибка удаления лица: {e}")
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
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM faces")
            conn.commit()
            conn.close()
            
            # Обновляем кэш в памяти
            self.known_faces = []
            self.known_names = []
            
            print("✅ База данных очищена")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка очистки базы данных: {e}")
            return False
    
    def list_faces_details(self) -> List[dict]:
        """
        Получить подробную информацию о всех лицах
        
        Returns:
            Список словарей с информацией о лицах
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT id, name, added_at FROM faces ORDER BY name")
            rows = cursor.fetchall()
            conn.close()
            
            faces = []
            for row in rows:
                faces.append({
                    'id': row[0],
                    'name': row[1],
                    'added_at': row[2]
                })
            
            return faces
            
        except Exception as e:
            print(f"Ошибка получения списка лиц: {e}")
            return []
