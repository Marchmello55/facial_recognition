"""
GUI приложение для распознавания лиц с использованием tkinter
Заменяет консольный интерфейс на графический
"""
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import numpy as np
from face_recognizer import FaceRecognizer


class FaceRecognitionApp:
    """Графическое приложение для распознавания лиц"""
    
    def __init__(self, root):
        """Инициализация приложения"""
        self.root = root
        self.root.title("Система Распознавания Лиц")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Создание распознавателя
        self.recognizer = FaceRecognizer(tolerance=0.6)
        
        # Переменные для управления камерой
        self.camera_running = False
        self.video_capture = None
        self.add_mode = False
        self.add_name = ""
        
        # Настройка стилей
        self.style = ttk.Style()
        self.style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
        self.style.configure('Status.TLabel', font=('Helvetica', 10))
        
        # Создание основного интерфейса
        self._create_widgets()
        
        # Обновление информации о базе
        self._update_db_info()
        
        # Обработчик закрытия окна
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _create_widgets(self):
        """Создание виджетов интерфейса"""
        # Основной контейнер
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Настройка расширения
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Заголовок
        title_label = ttk.Label(
            main_frame, 
            text="СИСТЕМА РАСПОЗНАВАНИЯ ЛИЦ", 
            style='Title.TLabel'
        )
        title_label.grid(row=0, column=0, pady=(0, 10))
        
        # Фрейм для видео
        video_frame = ttk.LabelFrame(main_frame, text="Веб-камера", padding="5")
        video_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        
        # Метка для отображения видео
        self.video_label = ttk.Label(video_frame)
        self.video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Фрейм для кнопок управления
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, pady=10)
        
        # Кнопки управления камерой
        self.start_camera_btn = ttk.Button(
            control_frame, 
            text="Запустить камеру", 
            command=self._toggle_camera
        )
        self.start_camera_btn.grid(row=0, column=0, padx=5)
        
        self.add_face_btn = ttk.Button(
            control_frame, 
            text="Добавить лицо из камеры", 
            command=self._add_face_from_camera,
            state=tk.DISABLED
        )
        self.add_face_btn.grid(row=0, column=1, padx=5)
        
        # Кнопки управления базой
        db_frame = ttk.Frame(main_frame)
        db_frame.grid(row=3, column=0, pady=5)
        
        ttk.Button(
            db_frame, 
            text="Добавить лицо из файла", 
            command=self._add_face_from_file
        ).grid(row=0, column=0, padx=5)
        
        ttk.Button(
            db_frame, 
            text="Показать список лиц", 
            command=self._show_faces_list
        ).grid(row=0, column=1, padx=5)
        
        ttk.Button(
            db_frame, 
            text="Удалить лицо", 
            command=self._remove_face
        ).grid(row=0, column=2, padx=5)
        
        ttk.Button(
            db_frame, 
            text="Очистить базу", 
            command=self._clear_database
        ).grid(row=0, column=3, padx=5)
        
        # Фрейм информации о базе
        info_frame = ttk.LabelFrame(main_frame, text="Информация о базе данных", padding="5")
        info_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=5)
        info_frame.columnconfigure(1, weight=1)
        
        ttk.Label(info_frame, text="Всего лиц в базе:").grid(row=0, column=0, sticky=tk.W)
        self.db_count_label = ttk.Label(info_frame, text="0", font=('Helvetica', 10, 'bold'))
        self.db_count_label.grid(row=0, column=1, sticky=tk.W, padx=10)
        
        ttk.Label(info_frame, text="Список имен:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        
        # Текстовое поле со списком имен
        self.names_text = tk.Text(info_frame, height=6, width=50, state=tk.DISABLED)
        self.names_text.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=10, pady=(10, 0))
        
        # Статус бар
        self.status_label = ttk.Label(
            main_frame, 
            text="Готов к работе", 
            style='Status.TLabel',
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def _update_db_info(self):
        """Обновление информации о базе данных"""
        info = self.recognizer.get_database_info()
        self.db_count_label.config(text=str(info['count']))
        
        self.names_text.config(state=tk.NORMAL)
        self.names_text.delete(1.0, tk.END)
        if info['names']:
            for i, name in enumerate(info['names'], 1):
                self.names_text.insert(tk.END, f"{i}. {name}\n")
        else:
            self.names_text.insert(tk.END, "База пуста")
        self.names_text.config(state=tk.DISABLED)
    
    def _toggle_camera(self):
        """Переключение состояния камеры"""
        if not self.camera_running:
            self._start_camera()
        else:
            self._stop_camera()
    
    def _start_camera(self):
        """Запуск веб-камеры"""
        self.video_capture = cv2.VideoCapture(0)
        
        if not self.video_capture.isOpened():
            messagebox.showerror("Ошибка", "Не удалось открыть камеру")
            return
        
        # Установка разрешения
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.camera_running = True
        self.start_camera_btn.config(text="Остановить камеру")
        self.add_face_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Камера запущена. Нажмите 'q' на клавиатуре для остановки.")
        
        # Запуск потока для обработки видео
        self._update_frame()
    
    def _stop_camera(self):
        """Остановка веб-камеры"""
        self.camera_running = False
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        
        self.start_camera_btn.config(text="Запустить камеру")
        self.add_face_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Камера остановлена")
        self.video_label.config(image='')
    
    def _update_frame(self):
        """Обновление кадра видео"""
        if not self.camera_running or not self.video_capture:
            return
        
        ret, frame = self.video_capture.read()
        
        if not ret:
            self.status_label.config(text="Ошибка получения кадра")
            return
        
        # Если режим добавления активен
        if self.add_mode:
            if self.recognizer.add_face_from_frame(frame, self.add_name):
                self.status_label.config(text=f"Лицо '{self.add_name}' добавлено в базу!")
                self._update_db_info()
                self.add_mode = False
                self.add_name = ""
                messagebox.showinfo("Успех", f"Лицо '{self.add_name}' добавлено в базу!")
            else:
                self.status_label.config(text="Не удалось добавить лицо. Убедитесь, что лицо видно в кадре.")
                self.add_mode = False
        
        # Распознавание лиц
        face_locations, face_names = self.recognizer.recognize_faces(frame)
        
        # Отрисовка результатов
        frame = self.recognizer.draw_results(frame, face_locations, face_names)
        
        # Добавление информации о количестве лиц в базе
        db_info = self.recognizer.get_database_info()
        cv2.putText(
            frame, 
            f"Faces in DB: {db_info['count']}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        # Вывод статуса распознавания в статус бар
        if face_names:
            recognized_users = []
            unknown_count = 0
            for name in face_names:
                if name != "Unknown":
                    recognized_users.append(name)
                else:
                    unknown_count += 1
            
            if recognized_users:
                status_text = f"Распознаны: {', '.join(recognized_users)}"
                if unknown_count > 0:
                    status_text += f" | Неизвестных: {unknown_count}"
            elif unknown_count > 0:
                status_text = f"Неизвестный пользователь ({unknown_count})"
            else:
                status_text = "Лица обнаружены, но не распознаны"
            
            self.status_label.config(text=status_text)
        else:
            self.status_label.config(text="Лица не обнаружены")
        
        # Конвертация BGR в RGB для отображения в tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Изменение размера для отображения
        display_height = 480
        display_width = 640
        frame_resized = cv2.resize(frame_rgb, (display_width, display_height))
        
        # Конвертация в формат PIL
        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Обновление метки
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)
        
        # Планирование следующего обновления
        if self.camera_running:
            self.root.after(30, self._update_frame)
    
    def _add_face_from_camera(self):
        """Добавление лица из текущего кадра камеры"""
        if not self.camera_running:
            messagebox.showwarning("Предупреждение", "Сначала запустите камеру")
            return
        
        # Диалог ввода имени
        dialog = tk.Toplevel(self.root)
        dialog.title("Добавление лица")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Введите имя человека:").pack(pady=10)
        
        name_entry = ttk.Entry(dialog, width=30)
        name_entry.pack(pady=5)
        name_entry.focus()
        
        def on_submit():
            name = name_entry.get().strip()
            if name:
                self.add_name = name
                self.add_mode = True
                self.status_label.config(text="Нажмите любую клавишу для захвата лица...")
                dialog.destroy()
            else:
                messagebox.showwarning("Предупреждение", "Имя не может быть пустым")
        
        ttk.Button(dialog, text="OK", command=on_submit).pack(pady=5)
        
        # Привязка клавиши Enter
        dialog.bind('<Return>', lambda e: on_submit())
    
    def _add_face_from_file(self):
        """Добавление лица из файла изображения"""
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[
                ("Изображения", "*.jpg *.jpeg *.png *.bmp"),
                ("Все файлы", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        # Диалог ввода имени
        dialog = tk.Toplevel(self.root)
        dialog.title("Добавление лица из файла")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Введите имя человека:").pack(pady=10)
        
        name_entry = ttk.Entry(dialog, width=30)
        name_entry.pack(pady=5)
        name_entry.focus()
        
        def on_submit():
            name = name_entry.get().strip()
            if name:
                if self.recognizer.add_face_from_image(file_path, name):
                    self.status_label.config(text=f"Лицо '{name}' успешно добавлено в базу")
                    self._update_db_info()
                    messagebox.showinfo("Успех", f"Лицо '{name}' успешно добавлено в базу")
                else:
                    messagebox.showerror("Ошибка", "Не удалось добавить лицо")
                dialog.destroy()
            else:
                messagebox.showwarning("Предупреждение", "Имя не может быть пустым")
        
        ttk.Button(dialog, text="OK", command=on_submit).pack(pady=5)
        dialog.bind('<Return>', lambda e: on_submit())
    
    def _show_faces_list(self):
        """Показ списка всех лиц в базе"""
        info = self.recognizer.get_database_info()
        
        list_window = tk.Toplevel(self.root)
        list_window.title("Список лиц в базе")
        list_window.geometry("400x400")
        list_window.transient(self.root)
        
        ttk.Label(
            list_window, 
            text=f"Всего лиц в базе: {info['count']}", 
            font=('Helvetica', 12, 'bold')
        ).pack(pady=10)
        
        # Список с прокруткой
        list_frame = ttk.Frame(list_window)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        faces_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, font=('Helvetica', 11))
        faces_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=faces_listbox.yview)
        
        if info['names']:
            for i, name in enumerate(info['names'], 1):
                faces_listbox.insert(tk.END, f"{i}. {name}")
        else:
            faces_listbox.insert(tk.END, "База пуста")
        
        ttk.Button(
            list_window, 
            text="Закрыть", 
            command=list_window.destroy
        ).pack(pady=10)
    
    def _remove_face(self):
        """Удаление лица из базы"""
        info = self.recognizer.get_database_info()
        
        if info['count'] == 0:
            messagebox.showinfo("Информация", "База данных пуста")
            return
        
        # Диалог выбора лица для удаления
        dialog = tk.Toplevel(self.root)
        dialog.title("Удаление лица")
        dialog.geometry("350x250")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Выберите лицо для удаления:").pack(pady=10)
        
        # Список с прокруткой
        list_frame = ttk.Frame(dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        faces_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, font=('Helvetica', 11))
        faces_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=faces_listbox.yview)
        
        for i, name in enumerate(info['names'], 1):
            faces_listbox.insert(tk.END, f"{i}. {name}")
        
        selected_index = [None]
        
        def on_select(event):
            selection = faces_listbox.curselection()
            if selection:
                selected_index[0] = selection[0]
        
        faces_listbox.bind('<<ListboxSelect>>', on_select)
        
        def on_delete():
            if selected_index[0] is not None:
                name_to_remove = info['names'][selected_index[0]]
                if self.recognizer.database.remove_face(name_to_remove):
                    self.status_label.config(text=f"Лицо '{name_to_remove}' удалено")
                    self._update_db_info()
                    messagebox.showinfo("Успех", f"Лицо '{name_to_remove}' удалено")
                    dialog.destroy()
                else:
                    messagebox.showerror("Ошибка", f"Не удалось удалить лицо '{name_to_remove}'")
            else:
                messagebox.showwarning("Предупреждение", "Выберите лицо для удаления")
        
        ttk.Button(dialog, text="Удалить", command=on_delete).pack(pady=5)
        ttk.Button(dialog, text="Отмена", command=dialog.destroy).pack(pady=5)
    
    def _clear_database(self):
        """Очистка всей базы данных"""
        info = self.recognizer.get_database_info()
        
        if info['count'] == 0:
            messagebox.showinfo("Информация", "База данных уже пуста")
            return
        
        if messagebox.askyesno("Подтверждение", 
                               "Вы уверены? Это удалит все лица из базы данных.",
                               icon='warning'):
            self.recognizer.database.clear_database()
            self.status_label.config(text="База данных очищена")
            self._update_db_info()
            messagebox.showinfo("Успех", "База данных очищена")
    
    def _on_closing(self):
        """Обработчик закрытия окна"""
        self._stop_camera()
        self.recognizer.clear_cache()
        self.root.destroy()


def main():
    """Основная функция запуска приложения"""
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПриложение прервано пользователем")
    except Exception as e:
        print(f"\nОшибка: {e}")
