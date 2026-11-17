import sys
import os
import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QTabWidget,
    QProgressBar, QComboBox, QCheckBox, QGroupBox, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QStatusBar,
    QMessageBox, QFrame, QScrollArea, QSizePolicy, QSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon, QPalette, QColor
from PIL import Image
import json
import csv
from datetime import datetime

# Импорт OCR движков (из предыдущего кода)
from ocr_engines import OCREngineManager, ImagePreprocessor, FontAnalyzer, TableExtractor


# ============================================================
# ПОТОК ОБРАБОТКИ (чтобы не блокировать GUI)
# ============================================================

class ProcessingThread(QThread):
    """Поток для обработки документа без блокировки интерфейса"""
    
    progress = pyqtSignal(int, str)          # прогресс, сообщение
    finished = pyqtSignal(dict)              # результат
    error = pyqtSignal(str)                  # ошибка
    page_processed = pyqtSignal(int, int)    # текущая страница, всего
    
    def __init__(self, file_path, ocr_manager, preprocess=True, dpi=300):
        super().__init__()
        self.file_path = file_path
        self.ocr_manager = ocr_manager
        self.preprocess = preprocess
        self.dpi = dpi
        self.preprocessor = ImagePreprocessor()
        self.font_analyzer = FontAnalyzer()
        self.table_extractor = TableExtractor()
        self._is_running = True
    
    def stop(self):
        self._is_running = False
    
    def run(self):
        try:
            ext = Path(self.file_path).suffix.lower()
            
            if ext == '.pdf':
                result = self._process_pdf()
            elif ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp']:
                result = self._process_image()
            else:
                raise ValueError(f"Неподдерживаемый формат: {ext}")
            
            if self._is_running:
                self.finished.emit(result)
                
        except Exception as e:
            self.error.emit(str(e))
    
    def _process_image(self):
        self.progress.emit(10, "Загрузка изображения...")
        image = cv2.imread(self.file_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить: {self.file_path}")
        
        if self.preprocess:
            self.progress.emit(30, "Предобработка...")
            processed = self.preprocessor.full_pipeline(image)
        else:
            processed = image
        
        self.progress.emit(50, "Распознавание текста...")
        ocr_result = self.ocr_manager.recognize(processed, with_details=True)
        
        self.progress.emit(80, "Анализ шрифтов...")
        font_analysis = self.font_analyzer.analyze_blocks(ocr_result.blocks)
        
        self.progress.emit(90, "Поиск таблиц...")
        tables = self.table_extractor.extract_from_image(image)
        
        self.progress.emit(100, "Готово")
        
        return {
            'file': self.file_path,
            'type': 'image',
            'engine_used': ocr_result.engine,
            'text': ocr_result.text,
            'confidence': ocr_result.confidence,
            'language': ocr_result.language,
            'font_analysis': font_analysis,
            'tables': tables,
            'blocks_count': len(ocr_result.blocks),
            'total_pages': 1,
            'pages': [{'text': ocr_result.text, 'confidence': ocr_result.confidence}]
        }
    
    def _process_pdf(self):
        from pdf2image import convert_from_path
        
        self.progress.emit(5, "Конвертация PDF...")
        images = convert_from_path(self.file_path, dpi=self.dpi)
        total = len(images)
        
        result = {
            'file': self.file_path,
            'type': 'pdf',
            'total_pages': total,
            'pages': [],
            'full_text': '',
            'tables': [],
            'overall_confidence': 0,
            'font_analysis': {}
        }
        
        all_blocks = []
        confidences = []
        
        for i, image in enumerate(images, 1):
            if not self._is_running:
                break
            
            progress = 10 + int(80 * i / total)
            self.progress.emit(progress, f"Страница {i}/{total}...")
            self.page_processed.emit(i, total)
            
            img_array = np.array(image)
            
            if self.preprocess:
                processed = self.preprocessor.full_pipeline(img_array)
            else:
                processed = img_array
            
            ocr_result = self.ocr_manager.recognize(processed, with_details=True)
            
            page_data = {
                'page_number': i,
                'text': ocr_result.text,
                'confidence': ocr_result.confidence,
                'blocks_count': len(ocr_result.blocks)
            }
            
            result['pages'].append(page_data)
            result['full_text'] += f"\n--- Страница {i} ---\n{ocr_result.text}\n"
            all_blocks.extend(ocr_result.blocks)
            confidences.append(ocr_result.confidence)
        
        result['overall_confidence'] = float(np.mean(confidences)) if confidences else 0
        result['font_analysis'] = self.font_analyzer.analyze_blocks(all_blocks)
        result['engine_used'] = self.ocr_manager.current_engine.name
        result['text'] = result['full_text']
        result['confidence'] = result['overall_confidence']
        
        self.progress.emit(95, "Извлечение таблиц...")
        result['tables'] = self.table_extractor.extract_from_pdf(self.file_path)
        
        self.progress.emit(100, "Готово")
        return result


# ============================================================
# ГЛАВНОЕ ОКНО
# ============================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DocumentOCR — Распознавание документов")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        # OCR менеджер
        self.ocr_manager = OCREngineManager(languages=['ru', 'en'])
        self.current_file = None
        self.current_result = None
        self.processing_thread = None
        
        self._init_ui()
        self._apply_styles()
        self._update_engine_info()
    
    def _init_ui(self):
        """Инициализация интерфейса"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)
        
        # Верхняя панель с настройками
        main_layout.addWidget(self._create_top_panel())
        
        # Основной разделитель (левая/правая части)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Левая панель - загрузка и превью
        splitter.addWidget(self._create_left_panel())
        
        # Правая панель - результаты
        splitter.addWidget(self._create_right_panel())
        
        splitter.setSizes([450, 750])
        main_layout.addWidget(splitter, 1)
        
        # Нижняя панель - прогресс
        main_layout.addWidget(self._create_bottom_panel())
        
        # Статусная строка
        self.statusBar().showMessage("Готов к работе")
    
    # ========== ВЕРХНЯЯ ПАНЕЛЬ ==========
    def _create_top_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("topPanel")
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(15, 10, 15, 10)
        
        # Заголовок
        title = QLabel("📄 DocumentOCR")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        layout.addWidget(title)
        
        layout.addSpacing(30)
        
        # Выбор OCR движка
        layout.addWidget(QLabel("Движок:"))
        self.engine_combo = QComboBox()
        self.engine_combo.setMinimumWidth(150)
        self._populate_engines()
        self.engine_combo.currentTextChanged.connect(self._on_engine_changed)
        layout.addWidget(self.engine_combo)
        
        layout.addSpacing(15)
        
        # Языки
        self.chk_russian = QCheckBox("Русский")
        self.chk_russian.setChecked(True)
        layout.addWidget(self.chk_russian)
        
        self.chk_english = QCheckBox("Английский")
        self.chk_english.setChecked(True)
        layout.addWidget(self.chk_english)
        
        layout.addSpacing(15)
        
        # Предобработка
        self.chk_preprocess = QCheckBox("Предобработка")
        self.chk_preprocess.setChecked(True)
        self.chk_preprocess.setToolTip("Улучшение качества для лучшего OCR")
        layout.addWidget(self.chk_preprocess)
        
        layout.addSpacing(15)
        
        # DPI для PDF
        layout.addWidget(QLabel("DPI:"))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(150, 600)
        self.dpi_spin.setValue(300)
        self.dpi_spin.setSingleStep(50)
        layout.addWidget(self.dpi_spin)
        
        layout.addStretch()
        
        # Кнопка помощи
        help_btn = QPushButton("❓")
        help_btn.setFixedSize(35, 35)
        help_btn.clicked.connect(self._show_help)
        layout.addWidget(help_btn)
        
        return panel
    
    # ========== ЛЕВАЯ ПАНЕЛЬ ==========
    def _create_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Область загрузки (drag & drop)
        self.drop_area = DropArea()
        self.drop_area.file_dropped.connect(self._on_file_dropped)
        layout.addWidget(self.drop_area, 2)
        
        # Кнопки действий
        btn_layout = QHBoxLayout()
        
        self.btn_open = QPushButton("📂 Открыть файл")
        self.btn_open.clicked.connect(self._open_file)
        btn_layout.addWidget(self.btn_open)
        
        self.btn_process = QPushButton("▶ Распознать")
        self.btn_process.setObjectName("primaryButton")
        self.btn_process.clicked.connect(self._start_processing)
        self.btn_process.setEnabled(False)
        btn_layout.addWidget(self.btn_process)
        
        layout.addLayout(btn_layout)
        
        # Информация о файле
        info_group = QGroupBox("Информация о документе")
        info_layout = QVBoxLayout(info_group)
        self.lbl_filename = QLabel("Файл: не выбран")
        self.lbl_filesize = QLabel("Размер: —")
        self.lbl_filetype = QLabel("Тип: —")
        self.lbl_dimensions = QLabel("Разрешение: —")
        
        for lbl in [self.lbl_filename, self.lbl_filesize, 
                    self.lbl_filetype, self.lbl_dimensions]:
            lbl.setWordWrap(True)
            info_layout.addWidget(lbl)
        
        layout.addWidget(info_group)
        
        return panel
    
    # ========== ПРАВАЯ ПАНЕЛЬ ==========
    def _create_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Вкладки с результатами
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_text_tab(), "📝 Распознанный текст")
        self.tabs.addTab(self._create_tables_tab(), "📊 Таблицы")
        self.tabs.addTab(self._create_analysis_tab(), "🔍 Анализ")
        self.tabs.addTab(self._create_pages_tab(), "📄 Страницы")
        
        layout.addWidget(self.tabs)
        
        # Кнопки экспорта
        export_layout = QHBoxLayout()
        
        self.btn_save_txt = QPushButton("💾 Сохранить TXT")
        self.btn_save_txt.clicked.connect(lambda: self._save_result('txt'))
        self.btn_save_txt.setEnabled(False)
        export_layout.addWidget(self.btn_save_txt)
        
        self.btn_save_json = QPushButton("💾 Сохранить JSON")
        self.btn_save_json.clicked.connect(lambda: self._save_result('json'))
        self.btn_save_json.setEnabled(False)
        export_layout.addWidget(self.btn_save_json)
        
        self.btn_copy = QPushButton("📋 Копировать текст")
        self.btn_copy.clicked.connect(self._copy_text)
        self.btn_copy.setEnabled(False)
        export_layout.addWidget(self.btn_copy)
        
        export_layout.addStretch()
        layout.addLayout(export_layout)
        
        return panel
    
    def _create_text_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.text_result = QTextEdit()
        self.text_result.setReadOnly(True)
        self.text_result.setFont(QFont("Consolas", 11))
        self.text_result.setPlaceholderText(
            "Здесь появится распознанный текст после обработки документа..."
        )
        layout.addWidget(self.text_result)
        
        return tab
    
    def _create_tables_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.tables_container = QVBoxLayout()
        self.tables_container.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_widget.setLayout(self.tables_container)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        
        layout.addWidget(scroll)
        
        self.lbl_no_tables = QLabel("📊 Таблицы не обнаружены")
        self.lbl_no_tables.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_no_tables.setStyleSheet("color: gray; font-size: 14px; padding: 40px;")
        self.tables_container.addWidget(self.lbl_no_tables)
        
        return tab
    
    def _create_analysis_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Общая статистика
        stats_group = QGroupBox("📈 Общая статистика")
        stats_layout = QVBoxLayout(stats_group)
        
        self.lbl_confidence = QLabel("Уверенность распознавания: —")
        self.lbl_engine = QLabel("Использованный движок: —")
        self.lbl_blocks = QLabel("Количество блоков: —")
        self.lbl_language = QLabel("Язык: —")
        
        for lbl in [self.lbl_confidence, self.lbl_engine, 
                    self.lbl_blocks, self.lbl_language]:
            stats_layout.addWidget(lbl)
        
        layout.addWidget(stats_group)
        
        # Анализ шрифтов
        font_group = QGroupBox("🔤 Анализ шрифтов")
        font_layout = QVBoxLayout(font_group)
        
        self.lbl_font_type = QLabel("Тип шрифта: —")
        self.lbl_font_size = QLabel("Средний размер: —")
        self.lbl_font_range = QLabel("Диапазон размеров: —")
        self.lbl_bold = QLabel("Жирный текст: —")
        self.lbl_italic = QLabel("Курсив: —")
        
        for lbl in [self.lbl_font_type, self.lbl_font_size, 
                    self.lbl_font_range, self.lbl_bold, self.lbl_italic]:
            font_layout.addWidget(lbl)
        
        layout.addWidget(font_group)
        layout.addStretch()
        
        return tab
    
    def _create_pages_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.pages_list = QComboBox()
        self.pages_list.currentIndexChanged.connect(self._on_page_changed)
        layout.addWidget(QLabel("Выберите страницу:"))
        layout.addWidget(self.pages_list)
        
        self.page_text = QTextEdit()
        self.page_text.setReadOnly(True)
        self.page_text.setFont(QFont("Consolas", 11))
        layout.addWidget(self.page_text)
        
        self.pages_data = []
        
        return tab
    
    # ========== НИЖНЯЯ ПАНЕЛЬ ==========
    def _create_bottom_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("bottomPanel")
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(15, 10, 15, 10)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar, 1)
        
        self.lbl_status = QLabel("Ожидание...")
        self.lbl_status.setMinimumWidth(200)
        layout.addWidget(self.lbl_status)
        
        self.btn_stop = QPushButton("⏹ Остановить")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_processing)
        layout.addWidget(self.btn_stop)
        
        return panel
    
    # ========== СТИЛИ ==========
    def _apply_styles(self):
        """Применение современного стиля"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f7;
            }
            #topPanel {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
            }
            #bottomPanel {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #d0d0d0;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #c0c0c0;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
                border-color: #0078d4;
            }
            QPushButton:pressed {
                background-color: #e0e0e0;
            }
            QPushButton:disabled {
                background-color: #f5f5f5;
                color: #a0a0a0;
            }
            #primaryButton {
                background-color: #0078d4;
                color: white;
                border: none;
                font-weight: bold;
            }
            #primaryButton:hover {
                background-color: #106ebe;
            }
            #primaryButton:disabled {
                background-color: #c0c0c0;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 6px;
                padding: 10px;
            }
            QComboBox, QSpinBox {
                padding: 5px 10px;
                border: 1px solid #c0c0c0;
                border-radius: 4px;
                background-color: white;
            }
            QTabWidget::pane {
                border: 1px solid #d0d0d0;
                border-radius: 6px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                border: 1px solid #d0d0d0;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom-color: white;
                font-weight: bold;
            }
            QProgressBar {
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                text-align: center;
                height: 22px;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 3px;
            }
            QLabel {
                color: #333333;
            }
        """)
    
    # ========== МЕТОДЫ ==========
    def _populate_engines(self):
        """Заполнение списка доступных движков"""
        self.engine_combo.clear()
        available = self.ocr_manager.get_available_engines()
        if available:
            self.engine_combo.addItems(available)
            current = self.ocr_manager.current_engine.name
            idx = self.engine_combo.findText(current)
            if idx >= 0:
                self.engine_combo.setCurrentIndex(idx)
        else:
            self.engine_combo.addItem("Нет доступных движков")
    
    def _update_engine_info(self):
        """Обновление информации о доступных движках"""
        available = self.ocr_manager.get_available_engines()
        if not available:
            QMessageBox.warning(
                self, "Внимание",
                "Не найдено ни одного OCR движка!\n\n"
                "Установите хотя бы один:\n"
                "• pip install easyocr\n"
                "• pip install rapidocr-onnxruntime\n"
                "• pip install paddlepaddle paddleocr"
            )
    
    def _on_engine_changed(self, engine_name):
        """Смена OCR движка"""
        if self.ocr_manager.switch_engine(engine_name):
            self.statusBar().showMessage(f"Переключено на {engine_name}", 3000)
    
    def _open_file(self):
        """Открытие файла через диалог"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите документ",
            "",
            "Документы (*.pdf *.jpg *.jpeg *.png *.tiff *.bmp *.webp);;Все файлы (*.*)"
        )
        if file_path:
            self._load_file(file_path)
    
    def _on_file_dropped(self, file_path):
        """Обработка файла из drag&drop"""
        self._load_file(file_path)
    
    def _load_file(self, file_path):
        """Загрузка информации о файле"""
        if not os.path.exists(file_path):
            QMessageBox.critical(self, "Ошибка", f"Файл не найден:\n{file_path}")
            return
        
        self.current_file = file_path
        filename = os.path.basename(file_path)
        size = os.path.getsize(file_path)
        ext = Path(file_path).suffix.lower()
        
        self.lbl_filename.setText(f"Файл: {filename}")
        self.lbl_filesize.setText(f"Размер: {self._format_size(size)}")
        self.lbl_filetype.setText(f"Тип: {ext.upper().replace('.', '')}")
        
        # Получение размеров для изображений
        if ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp']:
            try:
                img = Image.open(file_path)
                self.lbl_dimensions.setText(f"Разрешение: {img.size[0]}×{img.size[1]}")
            except:
                self.lbl_dimensions.setText("Разрешение: —")
        else:
            self.lbl_dimensions.setText("Разрешение: — (PDF)")
        
        self.drop_area.set_file(filename)
        self.btn_process.setEnabled(True)
        self.statusBar().showMessage(f"Загружен: {filename}", 3000)
    
    def _format_size(self, size):
        """Форматирование размера файла"""
        for unit in ['Б', 'КБ', 'МБ', 'ГБ']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} ТБ"
    
    def _start_processing(self):
        """Запуск обработки документа"""
        if not self.current_file:
            return
        
        # Обновление языков
        langs = []
        if self.chk_russian.isChecked():
            langs.append('ru')
        if self.chk_english.isChecked():
            langs.append('en')
        if not langs:
            langs = ['ru', 'en']
        self.ocr_manager.languages = langs
        
        # Запуск потока
        self.processing_thread = ProcessingThread(
            self.current_file,
            self.ocr_manager,
            self.chk_preprocess.isChecked(),
            self.dpi_spin.value()
        )
        self.processing_thread.progress.connect(self._on_progress)
        self.processing_thread.finished.connect(self._on_finished)
        self.processing_thread.error.connect(self._on_error)
        self.processing_thread.page_processed.connect(self._on_page_processed)
        
        # Блокировка интерфейса
        self.btn_process.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Запуск обработки...")
        
        self.processing_thread.start()
    
    def _stop_processing(self):
        """Остановка обработки"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.lbl_status.setText("Остановка...")
    
    def _on_progress(self, value, message):
        """Обновление прогресса"""
        self.progress_bar.setValue(value)
        self.lbl_status.setText(message)
    
    def _on_page_processed(self, current, total):
        """Обработка страницы"""
        self.statusBar().showMessage(f"Страница {current} из {total}", 2000)
    
    def _on_finished(self, result):
        """Обработка завершена"""
        self.current_result = result
        self._display_result(result)
        
        self.btn_process.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_save_txt.setEnabled(True)
        self.btn_save_json.setEnabled(True)
        self.btn_copy.setEnabled(True)
        
        self.progress_bar.setValue(100)
        self.lbl_status.setText("✓ Обработка завершена")
        self.statusBar().showMessage("Готово", 5000)
    
    def _on_error(self, error_msg):
        """Обработка ошибки"""
        QMessageBox.critical(self, "Ошибка обработки", error_msg)
        
        self.btn_process.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_status.setText(f"✗ Ошибка: {error_msg[:50]}")
    
    def _display_result(self, result):
        """Отображение результатов"""
        # Текст
        text = result.get('full_text', result.get('text', ''))
        self.text_result.setPlainText(text)
        
        # Статистика
        conf = result.get('overall_confidence', result.get('confidence', 0))
        self.lbl_confidence.setText(f"Уверенность распознавания: {conf:.1%}")
        self.lbl_engine.setText(f"Использованный движок: {result.get('engine_used', '—')}")
        self.lbl_blocks.setText(f"Количество блоков: {result.get('blocks_count', '—')}")
        self.lbl_language.setText(f"Язык: {result.get('language', '—')}")
        
        # Анализ шрифтов
        font = result.get('font_analysis', {})
        if font:
            self.lbl_font_type.setText(f"Тип шрифта: {font.get('estimated_font', '—')}")
            self.lbl_font_size.setText(f"Средний размер: {font.get('average_size', 0):.1f} px")
            sr = font.get('size_range', [0, 0])
            self.lbl_font_range.setText(f"Диапазон размеров: {sr[0]:.1f} — {sr[1]:.1f} px")
            self.lbl_bold.setText(f"Жирный текст: {font.get('bold_percentage', 0):.1f}%")
            self.lbl_italic.setText(f"Курсив: {font.get('italic_percentage', 0):.1f}%")
        
        # Таблицы
        self._display_tables(result.get('tables', []))
        
        # Страницы
        self._display_pages(result.get('pages', []))
    
    def _display_tables(self, tables):
        """Отображение таблиц"""
        # Очистка контейнера
        while self.tables_container.count():
            item = self.tables_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if not tables:
            lbl = QLabel("📊 Таблицы не обнаружены")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color: gray; font-size: 14px; padding: 40px;")
            self.tables_container.addWidget(lbl)
            return
        
        for i, table in enumerate(tables, 1):
            group = QGroupBox(f"Таблица {i}")
            if 'accuracy' in table:
                group.setTitle(f"Таблица {i} (точность: {table['accuracy']:.1f}%)")
            
            layout = QVBoxLayout(group)
            
            if 'data' in table:
                data = table['data']
                if data:
                    rows = len(data)
                    cols = max(len(row) for row in data) if data else 0
                    
                    table_widget = QTableWidget(rows, cols)
                    table_widget.horizontalHeader().setSectionResizeMode(
                        QHeaderView.ResizeMode.Stretch
                    )
                    
                    for r, row in enumerate(data):
                        for c, cell in enumerate(row):
                            item = QTableWidgetItem(str(cell) if cell else "")
                            table_widget.setItem(r, c, item)
                    
                    table_widget.setMaximumHeight(min(400, 50 + rows * 30))
                    layout.addWidget(table_widget)
                    
                    # Кнопка экспорта таблицы
                    btn_export = QPushButton(f"💾 Экспорт таблицы {i} в CSV")
                    btn_export.clicked.connect(
                        lambda checked, t=table, idx=i: self._export_table(t, idx)
                    )
                    layout.addWidget(btn_export)
            else:
                layout.addWidget(QLabel(table.get('note', 'Нет данных')))
            
            self.tables_container.addWidget(group)
    
    def _display_pages(self, pages):
        """Отображение страниц"""
        self.pages_data = pages
        self.pages_list.clear()
        
        if not pages:
            self.page_text.setPlainText("Нет данных о страницах")
            return
        
        for i, page in enumerate(pages, 1):
            conf = page.get('confidence', 0)
            self.pages_list.addItem(f"Страница {i} ({conf:.1%})")
        
        self.pages_list.setCurrentIndex(0)
    
    def _on_page_changed(self, index):
        """Смена страницы"""
        if 0 <= index < len(self.pages_data):
            page = self.pages_data[index]
            self.page_text.setPlainText(page.get('text', ''))
    
    def _save_result(self, format_type):
        """Сохранение результата"""
        if not self.current_result:
            return
        
        if format_type == 'txt':
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить текст", "result.txt", "Текстовые файлы (*.txt)"
            )
            if file_path:
                text = self.current_result.get('full_text', 
                                               self.current_result.get('text', ''))
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                self.statusBar().showMessage(f"Сохранено: {file_path}", 5000)
        
        elif format_type == 'json':
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить JSON", "result.json", "JSON файлы (*.json)"
            )
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.current_result, f, ensure_ascii=False, 
                             indent=2, default=str)
                self.statusBar().showMessage(f"Сохранено: {file_path}", 5000)
    
    def _export_table(self, table, idx):
        """Экспорт таблицы в CSV"""
        if 'data' not in table:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, f"Экспорт таблицы {idx}", f"table_{idx}.csv", "CSV файлы (*.csv)"
        )
        if file_path:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(table['data'])
            self.statusBar().showMessage(f"Таблица сохранена: {file_path}", 5000)
    
    def _copy_text(self):
        """Копирование текста в буфер"""
        if self.current_result:
            text = self.current_result.get('full_text', 
                                           self.current_result.get('text', ''))
            QApplication.clipboard().setText(text)
            self.statusBar().showMessage("✓ Текст скопирован в буфер обмена", 3000)
    
    def _show_help(self):
        """Показать справку"""
        QMessageBox.information(
            self, "Справка",
            "<h3>DocumentOCR — Распознавание документов</h3>"
            "<p><b>Возможности:</b></p>"
            "<ul>"
            "<li>Распознавание текста из PDF и изображений</li>"
            "<li>Поддержка русского и английского языков</li>"
            "<li>Извлечение таблиц</li>"
            "<li>Анализ шрифтов</li>"
            "<li>Несколько OCR движков на выбор</li>"
            "</ul>"
            "<p><b>Использование:</b></p>"
            "<ol>"
            "<li>Перетащите файл в область загрузки или нажмите 'Открыть файл'</li>"
            "<li>Выберите OCR движок и настройки</li>"
            "<li>Нажмите 'Распознать'</li>"
            "<li>Сохраните результат в TXT или JSON</li>"
            "</ol>"
        )


# ============================================================
# ОБЛАСТЬ DRAG & DROP
# ============================================================

class DropArea(QFrame):
    """Область для перетаскивания файлов"""
    
    file_dropped = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumHeight(200)
        self.setObjectName("dropArea")
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.icon_label = QLabel("📁")
        self.icon_label.setFont(QFont("Segoe UI Emoji", 48))
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.icon_label)
        
        self.text_label = QLabel("Перетащите файл сюда\nили нажмите «Открыть файл»")
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label.setStyleSheet("color: #666; font-size: 14px;")
        layout.addWidget(self.text_label)
        
        self._apply_styles()
    
    def _apply_styles(self):
        self.setStyleSheet("""
            #dropArea {
                background-color: #fafafa;
                border: 2px dashed #c0c0c0;
                border-radius: 10px;
            }
            #dropArea:hover {
                background-color: #f0f7ff;
                border-color: #0078d4;
            }
        """)
    
    def set_file(self, filename):
        self.text_label.setText(f"✓ {filename}")
        self.icon_label.setText("✓")
    
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
            self.setStyleSheet("""
                #dropArea {
                    background-color: #e3f2fd;
                    border: 2px dashed #0078d4;
                    border-radius: 10px;
                }
            """)
        else:
            event.ignore()
    
    def dragLeaveEvent(self, event):
        self._apply_styles()
    
    def dropEvent(self, event):
        self._apply_styles()
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.file_dropped.emit(file_path)


# ============================================================
# ТОЧКА ВХОДА
# ============================================================

def main():
    app = QApplication(sys.argv)
    
    # Установка шрифта
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
