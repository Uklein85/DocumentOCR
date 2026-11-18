"""
DocumentOCR - Программа для распознавания сканированных документов

Разработчик: Уклеин И.В.
Организация: АО "СТНГ"
Город: г. Новосибирск
Год: 2026
"""

import sys
import os

# ВАЖНО: установить TESSDATA_PREFIX ДО импорта ocr_engines
tesseract_paths = [
    r'C:\Program Files\Tesseract-OCR\tessdata',
    r'C:\Program Files (x86)\Tesseract-OCR\tessdata',
]
for path in tesseract_paths:
    if os.path.exists(path):
        os.environ['TESSDATA_PREFIX'] = path
        break

import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QTabWidget,
    QProgressBar, QComboBox, QCheckBox, QGroupBox, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QStatusBar,
    QMessageBox, QFrame, QScrollArea, QSpinBox, QDialog,
    QRadioButton, QDialogButtonBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PIL import Image
import json
import csv

from ocr_engines import OCREngineManager, ImagePreprocessor, FontAnalyzer, TableExtractor


# ============================================================
# ПОТОК ОБРАБОТКИ
# ============================================================

class ProcessingThread(QThread):
    """Поток для обработки документа без блокировки интерфейса"""
    
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    page_processed = pyqtSignal(int, int)
    
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
        """Обработка PDF через PyMuPDF (не требует Poppler)"""
        try:
            import pymupdf as fitz
        except ImportError:
            try:
                import fitz
            except ImportError:
                raise ImportError("Установите PyMuPDF: pip install pymupdf")
        
        self.progress.emit(5, "Открытие PDF...")
        
        doc = fitz.open(self.file_path)
        total = len(doc)
        
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
        
        for i in range(total):
            if not self._is_running:
                break
            
            progress = 10 + int(80 * (i + 1) / total)
            self.progress.emit(progress, f"Страница {i+1}/{total}...")
            self.page_processed.emit(i + 1, total)
            
            page = doc[i]
            mat = fitz.Matrix(300 / 72, 300 / 72)
            pix = page.get_pixmap(matrix=mat)
            
            img_array = np.frombuffer(pix.samples, dtype=np.uint8)
            img_array = img_array.reshape(pix.height, pix.width, pix.n)
            
            if pix.n == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            if self.preprocess:
                processed = self.preprocessor.full_pipeline(img_array)
            else:
                processed = img_array
            
            ocr_result = self.ocr_manager.recognize(processed, with_details=True)
            
            page_data = {
                'page_number': i + 1,
                'text': ocr_result.text,
                'confidence': ocr_result.confidence,
                'blocks_count': len(ocr_result.blocks)
            }
            
            result['pages'].append(page_data)
            result['full_text'] += f"\n--- Страница {i+1} ---\n{ocr_result.text}\n"
            all_blocks.extend(ocr_result.blocks)
            confidences.append(ocr_result.confidence)
        
        doc.close()
        
        result['overall_confidence'] = float(np.mean(confidences)) if confidences else 0
        result['font_analysis'] = self.font_analyzer.analyze_blocks(all_blocks)
        result['engine_used'] = self.ocr_manager.current_engine.name if self.ocr_manager.current_engine else 'Неизвестно'
        result['text'] = result['full_text']
        result['confidence'] = result['overall_confidence']
        
        self.progress.emit(95, "Извлечение таблиц...")
        result['tables'] = self.table_extractor.extract_from_pdf(self.file_path)
        
        self.progress.emit(100, "Готово")
        return result


# ============================================================
# ОБЛАСТЬ DRAG & DROP (КОМПАКТНАЯ)
# ============================================================

class DropArea(QFrame):
    """Компактная область для перетаскивания файлов"""
    
    file_dropped = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumHeight(100)
        self.setMaximumHeight(120)
        self.setObjectName("dropArea")
        
        layout = QHBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(10, 8, 10, 8)
        
        self.icon_label = QLabel("📁")
        self.icon_label.setFont(QFont("Segoe UI Emoji", 28))
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.icon_label)
        
        self.text_label = QLabel("Перетащите файл сюда\nили нажмите «Открыть»")
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.text_label)
        
        self._apply_styles()
    
    def _apply_styles(self):
        self.setStyleSheet("""
            #dropArea {
                background-color: #fafafa;
                border: 2px dashed #c0c0c0;
                border-radius: 8px;
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
                    border-radius: 8px;
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
# ДИАЛОГ ВЫВОДА ГОТОВОГО ФАЙЛА
# ============================================================

class ExportDialog(QDialog):
    """Диалог выбора формата экспорта"""
    
    def __init__(self, parent, file_info: dict):
        super().__init__(parent)
        self.setWindowTitle("Вывод готового файла")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        
        # Информация о документе
        info_label = QLabel(
            f"<b>Документ:</b> {file_info.get('filename', 'Неизвестно')}<br>"
            f"<b>Страниц:</b> {file_info.get('pages', 1)}<br>"
            f"<b>Уверенность:</b> {file_info.get('confidence', 0):.1%}<br>"
            f"<b>Движок:</b> {file_info.get('engine', '—')}"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("padding: 10px; background-color: #f5f5f7; border-radius: 6px;")
        layout.addWidget(info_label)
        
        layout.addWidget(QLabel("<b>Выберите формат сохранения:</b>"))
        
        # Радиокнопки форматов
        self.format_group = QRadioButton()
        self.radio_txt = QRadioButton("📄 TXT - только текст")
        self.radio_txt.setChecked(True)
        self.radio_json = QRadioButton("📋 JSON - полные данные (текст, таблицы, анализ)")
        self.radio_csv = QRadioButton("📊 CSV - таблицы")
        self.radio_all = QRadioButton("📦 Все форматы (TXT + JSON + CSV)")
        
        for radio in [self.radio_txt, self.radio_json, self.radio_csv, self.radio_all]:
            layout.addWidget(radio)
        
        # Кнопки
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def get_selected_format(self) -> str:
        if self.radio_txt.isChecked():
            return 'txt'
        elif self.radio_json.isChecked():
            return 'json'
        elif self.radio_csv.isChecked():
            return 'csv'
        elif self.radio_all.isChecked():
            return 'all'
        return 'txt'


# ============================================================
# ГЛАВНОЕ ОКНО
# ============================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DocumentOCR — Распознавание документов")
        self.setMinimumSize(1100, 700)
        self.resize(1300, 800)
        
        self.ocr_manager = OCREngineManager(languages=['ru', 'en'])
        self.current_file = None
        self.current_result = None
        self.processing_thread = None
        
        self._init_ui()
        self._apply_styles()
        self._update_engine_info()
    
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)
        
        # Верхняя панель
        main_layout.addWidget(self._create_top_panel())
        
        # Основной разделитель
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._create_left_panel())
        splitter.addWidget(self._create_right_panel())
        splitter.setSizes([380, 720])
        main_layout.addWidget(splitter, 1)
        
        # Нижняя панель
        main_layout.addWidget(self._create_bottom_panel())
        
        self.statusBar().showMessage("Готов к работе")
    
    # ========== ВЕРХНЯЯ ПАНЕЛЬ ==========
    def _create_top_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("topPanel")
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(12, 8, 12, 8)
        
        title = QLabel("📄 DocumentOCR")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        layout.addWidget(title)
        
        layout.addSpacing(20)
        
        layout.addWidget(QLabel("Движок:"))
        self.engine_combo = QComboBox()
        self.engine_combo.setMinimumWidth(140)
        self._populate_engines()
        self.engine_combo.currentTextChanged.connect(self._on_engine_changed)
        layout.addWidget(self.engine_combo)
        
        layout.addSpacing(10)
        
        self.chk_russian = QCheckBox("Русский")
        self.chk_russian.setChecked(True)
        layout.addWidget(self.chk_russian)
        
        self.chk_english = QCheckBox("Английский")
        self.chk_english.setChecked(True)
        layout.addWidget(self.chk_english)
        
        layout.addSpacing(10)
        
        self.chk_preprocess = QCheckBox("Предобработка")
        self.chk_preprocess.setChecked(True)
        layout.addWidget(self.chk_preprocess)
        
        layout.addSpacing(10)
        
        layout.addWidget(QLabel("DPI:"))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(150, 600)
        self.dpi_spin.setValue(300)
        self.dpi_spin.setSingleStep(50)
        layout.addWidget(self.dpi_spin)
        
        layout.addStretch()
        
        help_btn = QPushButton("❓")
        help_btn.setFixedSize(32, 32)
        help_btn.clicked.connect(self._show_help)
        layout.addWidget(help_btn)
        
        return panel
    
    # ========== ЛЕВАЯ ПАНЕЛЬ ==========
    def _create_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Компактная область загрузки
        self.drop_area = DropArea()
        self.drop_area.file_dropped.connect(self._on_file_dropped)
        layout.addWidget(self.drop_area)
        
        # Кнопки действий
        btn_layout = QHBoxLayout()
        
        self.btn_open = QPushButton("📂 Открыть")
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
        
        # Информация о разработчике
        dev_group = QGroupBox("О программе")
        dev_layout = QVBoxLayout(dev_group)
        
        dev_label = QLabel(
            "<b>Разработчик:</b> Уклеин И.В.<br>"
            "<b>Организация:</b> АО «СТНГ»<br>"
            "<b>Город:</b> г. Новосибирск<br>"
            "<b>Год:</b> 2026"
        )
        dev_label.setWordWrap(True)
        dev_label.setStyleSheet("color: #333; font-size: 11px; padding: 5px;")
        dev_layout.addWidget(dev_label)
        
        layout.addWidget(dev_group)
        layout.addStretch()
        
        return panel
    
    # ========== ПРАВАЯ ПАНЕЛЬ ==========
    def _create_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_text_tab(), "📝 Распознанный текст")
        self.tabs.addTab(self._create_tables_tab(), "📊 Таблицы")
        self.tabs.addTab(self._create_analysis_tab(), "🔍 Анализ")
        self.tabs.addTab(self._create_pages_tab(), "📄 Страницы")
        
        layout.addWidget(self.tabs)
        
        # Кнопки экспорта
        export_layout = QHBoxLayout()
        
        self.btn_save_txt = QPushButton("💾 TXT")
        self.btn_save_txt.clicked.connect(lambda: self._save_result('txt'))
        self.btn_save_txt.setEnabled(False)
        export_layout.addWidget(self.btn_save_txt)
        
        self.btn_save_json = QPushButton("💾 JSON")
        self.btn_save_json.clicked.connect(lambda: self._save_result('json'))
        self.btn_save_json.setEnabled(False)
        export_layout.addWidget(self.btn_save_json)
        
        self.btn_copy = QPushButton("📋 Копировать")
        self.btn_copy.clicked.connect(self._copy_text)
        self.btn_copy.setEnabled(False)
        export_layout.addWidget(self.btn_copy)
        
        # КНОПКА ВЫВОДА ГОТОВОГО ФАЙЛА
        self.btn_export_file = QPushButton("📤 Вывести готовый файл")
        self.btn_export_file.setObjectName("primaryButton")
        self.btn_export_file.clicked.connect(self._export_ready_file)
        self.btn_export_file.setEnabled(False)
        self.btn_export_file.setToolTip("Сохранить результат в выбранном формате")
        export_layout.addWidget(self.btn_export_file)
        
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
        layout.setContentsMargins(12, 8, 12, 8)
        
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
                padding: 6px 12px;
                font-size: 12px;
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
                padding: 8px 16px;
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
        available = self.ocr_manager.get_available_engines()
        if not available:
            QMessageBox.warning(
                self, "Внимание",
                "Не найдено ни одного OCR движка!\n\n"
                "Установите хотя бы один:\n"
                "• pip install rapidocr-onnxruntime (рекомендуется)\n"
                "• pip install pytesseract (нужен Tesseract OCR)\n"
                "• pip install easyocr"
            )
    
    def _on_engine_changed(self, engine_name):
        if self.ocr_manager.switch_engine(engine_name):
            self.statusBar().showMessage(f"Переключено на {engine_name}", 3000)
    
    def _open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите документ",
            "",
            "Документы (*.pdf *.jpg *.jpeg *.png *.tiff *.bmp *.webp);;Все файлы (*.*)"
        )
        if file_path:
            self._load_file(file_path)
    
    def _on_file_dropped(self, file_path):
        self._load_file(file_path)
    
    def _load_file(self, file_path):
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
        for unit in ['Б', 'КБ', 'МБ', 'ГБ']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} ТБ"
    
    def _start_processing(self):
        if not self.current_file:
            return
        
        langs = []
        if self.chk_russian.isChecked():
            langs.append('ru')
        if self.chk_english.isChecked():
            langs.append('en')
        if not langs:
            langs = ['ru', 'en']
        self.ocr_manager.languages = langs
        
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
        
        self.btn_process.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Запуск обработки...")
        
        self.processing_thread.start()
    
    def _stop_processing(self):
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.lbl_status.setText("Остановка...")
    
    def _on_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.lbl_status.setText(message)
    
    def _on_page_processed(self, current, total):
        self.statusBar().showMessage(f"Страница {current} из {total}", 2000)
    
    def _on_finished(self, result):
        self.current_result = result
        self._display_result(result)
        
        self.btn_process.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_save_txt.setEnabled(True)
        self.btn_save_json.setEnabled(True)
        self.btn_copy.setEnabled(True)
        self.btn_export_file.setEnabled(True)
        
        self.progress_bar.setValue(100)
        self.lbl_status.setText("✓ Обработка завершена")
        self.statusBar().showMessage("Готово", 5000)
    
    def _on_error(self, error_msg):
        QMessageBox.critical(self, "Ошибка обработки", error_msg)
        
        self.btn_process.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_status.setText(f"✗ Ошибка: {error_msg[:50]}")
    
    def _display_result(self, result):
        text = result.get('full_text', result.get('text', ''))
        self.text_result.setPlainText(text)
        
        conf = result.get('overall_confidence', result.get('confidence', 0))
        self.lbl_confidence.setText(f"Уверенность распознавания: {conf:.1%}")
        self.lbl_engine.setText(f"Использованный движок: {result.get('engine_used', '—')}")
        self.lbl_blocks.setText(f"Количество блоков: {result.get('blocks_count', '—')}")
        self.lbl_language.setText(f"Язык: {result.get('language', '—')}")
        
        font = result.get('font_analysis', {})
        if font:
            self.lbl_font_type.setText(f"Тип шрифта: {font.get('estimated_font', '—')}")
            self.lbl_font_size.setText(f"Средний размер: {font.get('average_size', 0):.1f} px")
            sr = font.get('size_range', [0, 0])
            self.lbl_font_range.setText(f"Диапазон размеров: {sr[0]:.1f} — {sr[1]:.1f} px")
            self.lbl_bold.setText(f"Жирный текст: {font.get('bold_percentage', 0):.1f}%")
            self.lbl_italic.setText(f"Курсив: {font.get('italic_percentage', 0):.1f}%")
        
        self._display_tables(result.get('tables', []))
        self._display_pages(result.get('pages', []))
    
    def _display_tables(self, tables):
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
                    
                    btn_export = QPushButton(f"💾 Экспорт таблицы {i} в CSV")
                    btn_export.clicked.connect(
                        lambda checked, t=table, idx=i: self._export_table(t, idx)
                    )
                    layout.addWidget(btn_export)
            else:
                layout.addWidget(QLabel(table.get('note', 'Нет данных')))
            
            self.tables_container.addWidget(group)
    
    def _display_pages(self, pages):
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
        if 0 <= index < len(self.pages_data):
            page = self.pages_data[index]
            self.page_text.setPlainText(page.get('text', ''))
    
    def _save_result(self, format_type):
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
    
    def _export_ready_file(self):
        """Кнопка вывода готового файла - открывает диалог выбора формата"""
        if not self.current_result:
            QMessageBox.warning(self, "Внимание", "Сначала обработайте документ")
            return
        
        # Информация о файле для диалога
        file_info = {
            'filename': os.path.basename(self.current_file) if self.current_file else 'Неизвестно',
            'pages': self.current_result.get('total_pages', 1),
            'confidence': self.current_result.get('confidence', 0),
            'engine': self.current_result.get('engine_used', '—')
        }
        
        dialog = ExportDialog(self, file_info)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        
        selected_format = dialog.get_selected_format()
        
        # Базовое имя файла
        base_name = os.path.splitext(self.current_file)[0] if self.current_file else "result"
        
        if selected_format == 'txt':
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить TXT", f"{base_name}.txt", "Текстовые файлы (*.txt)"
            )
            if file_path:
                text = self.current_result.get('full_text', self.current_result.get('text', ''))
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                self.statusBar().showMessage(f"✓ Сохранено: {file_path}", 5000)
        
        elif selected_format == 'json':
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить JSON", f"{base_name}.json", "JSON файлы (*.json)"
            )
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.current_result, f, ensure_ascii=False,
                             indent=2, default=str)
                self.statusBar().showMessage(f"✓ Сохранено: {file_path}", 5000)
        
        elif selected_format == 'csv':
            self._export_all_tables(f"{base_name}_tables.csv")
        
        elif selected_format == 'all':
            # Сохранить все форматы
            txt_path = f"{base_name}.txt"
            json_path = f"{base_name}.json"
            csv_path = f"{base_name}_tables.csv"
            
            # TXT
            text = self.current_result.get('full_text', self.current_result.get('text', ''))
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.current_result, f, ensure_ascii=False,
                         indent=2, default=str)
            
            # CSV
            self._export_all_tables(csv_path, force=True)
            
            QMessageBox.information(
                self, "Готово",
                f"Файлы сохранены:\n"
                f"• {txt_path}\n"
                f"• {json_path}\n"
                f"• {csv_path}"
            )
            self.statusBar().showMessage("✓ Все файлы сохранены", 5000)
    
    def _export_all_tables(self, default_path: str, force: bool = False):
        """Экспорт всех таблиц в один CSV"""
        if not self.current_result or not self.current_result.get('tables'):
            if not force:
                QMessageBox.warning(self, "Внимание", "Таблицы не найдены")
            return
        
        if force:
            file_path = default_path
        else:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Экспорт всех таблиц", default_path, "CSV файлы (*.csv)"
            )
        
        if file_path:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for i, table in enumerate(self.current_result['tables'], 1):
                    if 'data' in table:
                        writer.writerow([f"=== ТАБЛИЦА {i} ==="])
                        writer.writerows(table['data'])
                        writer.writerow([])
            if not force:
                self.statusBar().showMessage(f"✓ Все таблицы сохранены: {file_path}", 5000)
    
    def _copy_text(self):
        if self.current_result:
            text = self.current_result.get('full_text',
                                           self.current_result.get('text', ''))
            QApplication.clipboard().setText(text)
            self.statusBar().showMessage("✓ Текст скопирован в буфер обмена", 3000)
    
    def _show_help(self):
        QMessageBox.information(
            self, "Справка о программе",
            "<h3>DocumentOCR — Распознавание документов</h3>"
            "<p><b>Разработчик:</b> Уклеин И.В.</p>"
            "<p><b>Организация:</b> АО «СТНГ»</p>"
            "<p><b>Город:</b> г. Новосибирск</p>"
            "<p><b>Год:</b> 2026</p>"
            "<hr>"
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
            "<li>Перетащите файл в область загрузки или нажмите «Открыть»</li>"
            "<li>Выберите OCR движок и настройки</li>"
            "<li>Нажмите «Распознать»</li>"
            "<li>Нажмите «📤 Вывести готовый файл» для сохранения</li>"
            "</ol>"
        )


# ============================================================
# ТОЧКА ВХОДА
# ============================================================

def main():
    app = QApplication(sys.argv)
    
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
