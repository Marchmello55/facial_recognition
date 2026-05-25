import sqlite3
import pickle
import numpy as np
from typing import List, Optional, Tuple

class FaceDatabase:
    def __init__(self, db_name: str = "faces.db"):
        """Инициализация подключения к БД и создание таблицы"""
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        """Создание таблицы, если она не существует"""
        # id: уникальный номер
        # name: имя человека
        # encoding: бинарные данные вектора (BLOB)
        # image_data: опционально фото лица (BLOB) для отображения в UI
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                encoding BLOB NOT NULL,
                image_data BLOB
            )
        """)
        self.conn.commit()

    def save_face(self, name: str, encoding: List[float], image_bytes: Optional[bytes] = None) -> bool:
        """
        Сохранение лица в БД.
        encoding: список из 128 чисел.
        """
        try:
            # Превращаем список float в байты (сериализация)
            encoding_blob = pickle.dumps(encoding)
            
            self.cursor.execute("""
                INSERT OR REPLACE INTO faces (name, encoding, image_data)
                VALUES (?, ?, ?)
            """, (name, encoding_blob, image_bytes))
            
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            print(f"⚠️ Лицо с именем '{name}' уже существует (обновлено).")
            return False
        except Exception as e:
            print(f"❌ Ошибка сохранения в БД: {e}")
            return False

    def load_all_faces(self) -> List[Tuple[str, List[float]]]:
        """
        Загрузка всех лиц из БД.
        Возвращает список кортежей: [(name, [128 floats]), ...]
        """
        self.cursor.execute("SELECT name, encoding FROM faces")
        rows = self.cursor.fetchall()
        
        faces = []
        for name, encoding_blob in rows:
            try:
                # Превращаем байты обратно в список float
                encoding = pickle.loads(encoding_blob)
                
                # Проверка на целостность данных (на всякий случай)
                if len(encoding) != 128:
                    print(f"⚠️ Пропуск записи '{name}': неверный размер вектора ({len(encoding)})")
                    continue
                    
                faces.append((name, encoding))
            except Exception as e:
                print(f"❌ Ошибка чтения вектора для {name}: {e}")
        
        return faces

    def delete_face(self, name: str) -> bool:
        """Удаление лица по имени"""
        self.cursor.execute("DELETE FROM faces WHERE name = ?", (name,))
        self.conn.commit()
        return self.cursor.rowcount > 0

    def get_image(self, name: str) -> Optional[bytes]:
        """Получение сохраненного изображения лица"""
        self.cursor.execute("SELECT image_data FROM faces WHERE name = ?", (name,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def close(self):
        """Закрытие соединения"""
        self.conn.close()

# Пример использования (можно запустить отдельно для теста)
if __name__ == "__main__":
    db = FaceDatabase()
    
    # Фейковый вектор для теста
    fake_encoding = [0.1] * 128 
    
    print("💾 Сохранение тестового пользователя...")
    db.save_face("TestUser", fake_encoding)
    
    print("📂 Загрузка всех пользователей...")
    loaded = db.load_all_faces()
    for name, enc in loaded:
        print(f" - Имя: {name}, Размер вектора: {len(enc)}, Первые 3 значения: {enc[:3]}")
    
    db.close()
    print("✅ Тест завершен.")
