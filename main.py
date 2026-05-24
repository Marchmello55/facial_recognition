"""
Основной скрипт для распознавания лиц с возможностью пополнения библиотеки
Запускает приложение с веб-камерой и предоставляет меню для управления базой лиц
"""
import cv2
import sys
from face_recognizer import FaceRecognizer


def print_menu():
    """Вывод меню управления"""
    print("\n" + "="*50)
    print("СИСТЕМА РАСПОЗНАВАНИЯ ЛИЦ")
    print("="*50)
    print("1. Запустить распознавание с веб-камеры")
    print("2. Добавить лицо из файла изображения")
    print("3. Показать список всех лиц в базе")
    print("4. Удалить лицо из базы")
    print("5. Очистить всю базу данных")
    print("6. Выход")
    print("="*50)


def add_face_from_file(recognizer: FaceRecognizer):
    """Добавление лица из файла"""
    name = input("Введите имя человека: ").strip()
    if not name:
        print("Ошибка: имя не может быть пустым")
        return
    
    image_path = input("Введите путь к изображению: ").strip()
    if not image_path:
        print("Ошибка: путь не может быть пустым")
        return
    
    if recognizer.add_face_from_image(image_path, name):
        print(f"Лицо '{name}' успешно добавлено в базу")
    else:
        print("Не удалось добавить лицо")


def show_all_faces(recognizer: FaceRecognizer):
    """Показать все лица в базе"""
    info = recognizer.get_database_info()
    print(f"\nВсего лиц в базе: {info['count']}")
    if info['count'] > 0:
        print("Список имен:")
        for i, name in enumerate(info['names'], 1):
            print(f"  {i}. {name}")
    else:
        print("База пуста")


def remove_face(recognizer: FaceRecognizer):
    """Удаление лица из базы"""
    show_all_faces(recognizer)
    if recognizer.database.get_face_count() == 0:
        return
    
    name = input("\nВведите имя для удаления: ").strip()
    if recognizer.database.remove_face(name):
        print(f"Лицо '{name}' удалено")
    else:
        print(f"Не удалось удалить лицо '{name}'")


def run_camera_recognition(recognizer: FaceRecognizer):
    """Запуск распознавания с веб-камеры"""
    print("\nЗапуск камеры... Нажмите 'q' для выхода, 'a' для добавления текущего лица")
    
    # Попытка открыть камеру (0 - основная камера)
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Ошибка: не удалось открыть камеру")
        return
    
    # Установка разрешения камеры
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    add_mode = False
    add_name = ""
    
    while True:
        ret, frame = video_capture.read()
        
        if not ret:
            print("Ошибка: не удалось получить кадр")
            break
        
        # Если режим добавления активен
        if add_mode:
            if recognizer.add_face_from_frame(frame, add_name):
                print(f"Лицо '{add_name}' добавлено в базу!")
                add_mode = False
                add_name = ""
            else:
                print("Не удалось добавить лицо. Убедитесь, что лицо видно в кадре.")
                add_mode = False
        
        # Распознавание лиц
        face_locations, face_names = recognizer.recognize_faces(frame)
        
        # Отрисовка результатов
        frame = recognizer.draw_results(frame, face_locations, face_names)
        
        # Добавление информации о количестве лиц в базе
        db_info = recognizer.get_database_info()
        cv2.putText(
            frame, 
            f"Faces in DB: {db_info['count']}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        # Показ кадра
        cv2.imshow('Face Recognition', frame)
        
        # Обработка нажатий клавиш
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('a'):
            # Режим добавления лица из камеры
            add_name = input("Введите имя для добавления: ").strip()
            if add_name:
                add_mode = True
                print("Нажмите любую клавишу для захвата лица...")
            else:
                print("Имя не может быть пустым")
    
    # Освобождение ресурсов
    video_capture.release()
    cv2.destroyAllWindows()
    recognizer.clear_cache()
    print("\nРаспознавание остановлено")


def main():
    """Основная функция программы"""
    print("Инициализация системы распознавания лиц...")
    
    # Создание экземпляра распознавателя
    recognizer = FaceRecognizer(tolerance=0.6)
    
    # Информация о текущей базе
    info = recognizer.get_database_info()
    print(f"Загружено {info['count']} лиц из базы данных")
    
    while True:
        print_menu()
        choice = input("\nВыберите действие (1-6): ").strip()
        
        if choice == '1':
            run_camera_recognition(recognizer)
        elif choice == '2':
            add_face_from_file(recognizer)
        elif choice == '3':
            show_all_faces(recognizer)
        elif choice == '4':
            remove_face(recognizer)
        elif choice == '5':
            confirm = input("Вы уверены? Это удалит все лица из базы (y/n): ").strip().lower()
            if confirm == 'y':
                recognizer.database.clear_database()
        elif choice == '6':
            print("\nВыход из программы. До свидания!")
            break
        else:
            print("Неверный выбор. Пожалуйста, выберите от 1 до 6")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nПрограмма прервана пользователем")
    except Exception as e:
        print(f"\nОшибка: {e}")
        sys.exit(1)
