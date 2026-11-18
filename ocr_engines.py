"""
OCR Engines Module
Модуль с поддержкой нескольких OCR движков для распознавания текста
Использует PyMuPDF для работы с PDF (не требует Poppler)
"""

import os
import cv2
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import csv
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# СТРУКТУРЫ ДАННЫХ
# ============================================================

@dataclass
class TextBlock:
    """Блок распознанного текста"""
    text: str
    confidence: float
    bbox: List[List[float]] = field(default_factory=list)
    font_size: float = 0.0
    is_bold: bool = False
    is_italic: bool = False


@dataclass
class OCRResult:
    """Результат OCR обработки"""
    text: str
    blocks: List[TextBlock]
    confidence: float
    engine: str
    language: str
    metadata: Dict = field(default_factory=dict)


# ============================================================
# АБСТРАКТНЫЙ БАЗОВЫЙ КЛАСС OCR ДВИЖКА
# ============================================================

class OCREngine(ABC):
    """Абстрактный базовый класс для OCR движков"""
    
    def __init__(self, languages: List[str] = None):
        self.languages = languages or ['ru', 'en']
        self.name = "BaseEngine"
        self._is_available = False
        
    @property
    def is_available(self) -> bool:
        return self._is_available
    
    @abstractmethod
    def recognize(self, image: np.ndarray) -> OCRResult:
        pass
    
    @abstractmethod
    def recognize_with_details(self, image: np.ndarray) -> OCRResult:
        pass


# ============================================================
# ДВИЖОК 1: EASYOCR
# ============================================================

class EasyOCREngine(OCREngine):
    """EasyOCR - открытый код, PyTorch-based"""
    
    def __init__(self, languages: List[str] = None, gpu: bool = False):
        super().__init__(languages)
        self.name = "EasyOCR"
        self.gpu = gpu
        
        try:
            import easyocr
            lang_map = {'ru': 'ru', 'en': 'en', 'rus': 'ru', 'eng': 'en'}
            self.easyocr_langs = [lang_map.get(l, l) for l in self.languages]
            
            logger.info(f"Инициализация EasyOCR с языками: {self.easyocr_langs}")
            self.reader = easyocr.Reader(
                self.easyocr_langs, 
                gpu=self.gpu,
                verbose=False
            )
            self._is_available = True
            logger.info("✅ EasyOCR успешно инициализирован")
        except ImportError as e:
            logger.warning(f"❌ EasyOCR не установлен: {e}")
            logger.warning("   Решение: pip install easyocr")
            self._is_available = False
        except Exception as e:
            logger.error(f"❌ Ошибка EasyOCR: {type(e).__name__}: {e}")
            self._is_available = False
    
    def recognize(self, image: np.ndarray) -> OCRResult:
        if not self._is_available:
            raise RuntimeError("EasyOCR недоступен")
        
        results = self.reader.readtext(image)
        
        blocks = []
        texts = []
        confidences = []
        
        for detection in results:
            bbox, text, confidence = detection
            blocks.append(TextBlock(
                text=text,
                confidence=confidence,
                bbox=bbox
            ))
            texts.append(text)
            confidences.append(confidence)
        
        return OCRResult(
            text='\n'.join(texts),
            blocks=blocks,
            confidence=float(np.mean(confidences)) if confidences else 0.0,
            engine=self.name,
            language='+'.join(self.languages)
        )
    
    def recognize_with_details(self, image: np.ndarray) -> OCRResult:
        result = self.recognize(image)
        
        for block in result.blocks:
            if block.bbox and len(block.bbox) >= 2:
                height = abs(block.bbox[2][1] - block.bbox[0][1])
                block.font_size = height
                block.is_bold = self._estimate_bold(image, block.bbox)
        
        return result
    
    def _estimate_bold(self, image: np.ndarray, bbox: List) -> bool:
        try:
            x_coords = [int(p[0]) for p in bbox]
            y_coords = [int(p[1]) for p in bbox]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            if x_max <= x_min or y_max <= y_min:
                return False
            
            roi = image[y_min:y_max, x_min:x_max]
            if roi.size == 0:
                return False
            
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi
            
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            density = np.sum(binary > 0) / binary.size
            return density > 0.35
        except Exception:
            return False


# ============================================================
# ДВИЖОК 2: RAPIDOCR
# ============================================================

class RapidOCREngine(OCREngine):
    """RapidOCR - легковесная обертка над PaddleOCR через ONNX"""
    
    def __init__(self, languages: List[str] = None):
        super().__init__(languages)
        self.name = "RapidOCR"
        
        try:
            from rapidocr_onnxruntime import RapidOCR
            self.engine = RapidOCR()
            self._is_available = True
            logger.info("✅ RapidOCR успешно инициализирован")
        except ImportError as e:
            logger.warning(f"❌ RapidOCR не установлен: {e}")
            logger.warning("   Решение: pip install rapidocr-onnxruntime")
            self._is_available = False
        except Exception as e:
            logger.error(f"❌ Ошибка RapidOCR: {type(e).__name__}: {e}")
            self._is_available = False
    
    def recognize(self, image: np.ndarray) -> OCRResult:
        if not self._is_available:
            raise RuntimeError("RapidOCR недоступен")
        
        result, _ = self.engine(image)
        
        if not result:
            return OCRResult(
                text='',
                blocks=[],
                confidence=0.0,
                engine=self.name,
                language='+'.join(self.languages)
            )
        
        blocks = []
        texts = []
        confidences = []
        
        for detection in result:
            bbox, text, confidence = detection
            blocks.append(TextBlock(
                text=text,
                confidence=confidence,
                bbox=bbox
            ))
            texts.append(text)
            confidences.append(confidence)
        
        return OCRResult(
            text='\n'.join(texts),
            blocks=blocks,
            confidence=float(np.mean(confidences)) if confidences else 0.0,
            engine=self.name,
            language='+'.join(self.languages)
        )
    
    def recognize_with_details(self, image: np.ndarray) -> OCRResult:
        return self.recognize(image)


# ============================================================
# ДВИЖОК 3: PADDLEOCR
# ============================================================

class PaddleOCREngine(OCREngine):
    """PaddleOCR от Baidu - открытый код"""
    
    def __init__(self, languages: List[str] = None, use_gpu: bool = False):
        super().__init__(languages)
        self.name = "PaddleOCR"
        
        try:
            from paddleocr import PaddleOCR
            
            lang_map = {'ru': 'ru', 'en': 'en', 'rus': 'ru', 'eng': 'en'}
            primary_lang = lang_map.get(self.languages[0], 'ru')
            
            logger.info(f"Инициализация PaddleOCR с языком: {primary_lang}")
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=primary_lang,
                use_gpu=use_gpu,
                show_log=False
            )
            self._is_available = True
            logger.info("✅ PaddleOCR успешно инициализирован")
        except ImportError as e:
            logger.warning(f"❌ PaddleOCR не установлен: {e}")
            logger.warning("   Решение: pip install paddlepaddle paddleocr")
            self._is_available = False
        except Exception as e:
            logger.error(f"❌ Ошибка PaddleOCR: {type(e).__name__}: {e}")
            self._is_available = False
    
    def recognize(self, image: np.ndarray) -> OCRResult:
        if not self._is_available:
            raise RuntimeError("PaddleOCR недоступен")
        
        result = self.ocr.ocr(image, cls=True)
        
        if not result or not result[0]:
            return OCRResult(
                text='',
                blocks=[],
                confidence=0.0,
                engine=self.name,
                language='+'.join(self.languages)
            )
        
        blocks = []
        texts = []
        confidences = []
        
        for line in result[0]:
            bbox, (text, confidence) = line
            blocks.append(TextBlock(
                text=text,
                confidence=confidence,
                bbox=bbox
            ))
            texts.append(text)
            confidences.append(confidence)
        
        return OCRResult(
            text='\n'.join(texts),
            blocks=blocks,
            confidence=float(np.mean(confidences)) if confidences else 0.0,
            engine=self.name,
            language='+'.join(self.languages)
        )
    
    def recognize_with_details(self, image: np.ndarray) -> OCRResult:
        return self.recognize(image)


# ============================================================
# ДВИЖОК 4: TESSERACT
# ============================================================

class TesseractEngine(OCREngine):
    """Классический Tesseract OCR"""
    
    def __init__(self, languages: List[str] = None, tesseract_path: str = None):
        super().__init__(languages)
        self.name = "Tesseract"
        
        try:
            import pytesseract
            
            if tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
            elif os.name == 'nt':
                for path in [
                    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                ]:
                    if os.path.exists(path):
                        pytesseract.pytesseract.tesseract_cmd = path
                        break
            
            lang_map = {'ru': 'rus', 'en': 'eng', 'rus': 'rus', 'eng': 'eng'}
            self.tess_langs = '+'.join([lang_map.get(l, l) for l in self.languages])
            
            pytesseract.get_tesseract_version()
            self._is_available = True
            logger.info(f"✅ Tesseract инициализирован с языками: {self.tess_langs}")
        except ImportError as e:
            logger.warning(f"❌ pytesseract не установлен: {e}")
            self._is_available = False
        except Exception as e:
            logger.warning(f"❌ Tesseract недоступен: {e}")
            self._is_available = False
    
    def recognize(self, image: np.ndarray) -> OCRResult:
        if not self._is_available:
            raise RuntimeError("Tesseract недоступен")
        
        import pytesseract
        
        text = pytesseract.image_to_string(
            image, 
            lang=self.tess_langs,
            config='--oem 3 --psm 6'
        )
        
        return OCRResult(
            text=text.strip(),
            blocks=[],
            confidence=0.0,
            engine=self.name,
            language=self.tess_langs
        )
    
    def recognize_with_details(self, image: np.ndarray) -> OCRResult:
        if not self._is_available:
            raise RuntimeError("Tesseract недоступен")
        
        import pytesseract
        
        data = pytesseract.image_to_data(
            image,
            lang=self.tess_langs,
            config='--oem 3 --psm 6',
            output_type=pytesseract.Output.DICT
        )
        
        blocks = []
        texts = []
        confidences = []
        
        for i in range(len(data['text'])):
            if data['text'][i].strip():
                block = TextBlock(
                    text=data['text'][i],
                    confidence=float(data['conf'][i]) / 100.0,
                    bbox=[
                        [data['left'][i], data['top'][i]],
                        [data['left'][i] + data['width'][i], data['top'][i]],
                        [data['left'][i] + data['width'][i], data['top'][i] + data['height'][i]],
                        [data['left'][i], data['top'][i] + data['height'][i]]
                    ],
                    font_size=float(data['height'][i])
                )
                blocks.append(block)
                texts.append(data['text'][i])
                confidences.append(block.confidence)
        
        return OCRResult(
            text=' '.join(texts),
            blocks=blocks,
            confidence=float(np.mean(confidences)) if confidences else 0.0,
            engine=self.name,
            language=self.tess_langs
        )


# ============================================================
# МЕНЕДЖЕР OCR ДВИЖКОВ
# ============================================================

class OCREngineManager:
    """Менеджер для выбора и переключения OCR движков"""
    
    PRIORITY = ['PaddleOCR', 'EasyOCR', 'RapidOCR', 'Tesseract']
    
    def __init__(self, preferred_engine: str = None, languages: List[str] = None):
        self.languages = languages or ['ru', 'en']
        self.engines: Dict[str, OCREngine] = {}
        self.current_engine: Optional[OCREngine] = None
        
        self._init_all_engines(preferred_engine)
    
    def _init_all_engines(self, preferred: str = None):
        logger.info("Инициализация OCR движков...")
        
        engine_classes = [
            ('EasyOCR', lambda: EasyOCREngine(self.languages)),
            ('RapidOCR', lambda: RapidOCREngine(self.languages)),
            ('PaddleOCR', lambda: PaddleOCREngine(self.languages)),
            ('Tesseract', lambda: TesseractEngine(self.languages)),
        ]
        
        for name, factory in engine_classes:
            try:
                engine = factory()
                self.engines[name] = engine
                status = "✓ доступен" if engine.is_available else "✗ недоступен"
                logger.info(f"  {name}: {status}")
            except Exception as e:
                logger.warning(f"Ошибка инициализации {name}: {e}")
        
        if preferred and preferred in self.engines and self.engines[preferred].is_available:
            self.current_engine = self.engines[preferred]
        else:
            for engine_name in self.PRIORITY:
                if engine_name in self.engines and self.engines[engine_name].is_available:
                    self.current_engine = self.engines[engine_name]
                    break
        
        if self.current_engine:
            logger.info(f"Выбран движок: {self.current_engine.name}")
        else:
            logger.error("Ни один OCR движок не доступен!")
    
    def get_available_engines(self) -> List[str]:
        return [name for name, engine in self.engines.items() if engine.is_available]
    
    def switch_engine(self, engine_name: str) -> bool:
        if engine_name in self.engines and self.engines[engine_name].is_available:
            self.current_engine = self.engines[engine_name]
            logger.info(f"Переключено на: {engine_name}")
            return True
        return False
    
    def recognize(self, image: np.ndarray, with_details: bool = True) -> OCRResult:
        if not self.current_engine:
            raise RuntimeError("Нет доступных OCR движков")
        
        try:
            if with_details:
                return self.current_engine.recognize_with_details(image)
            else:
                return self.current_engine.recognize(image)
        except Exception as e:
            logger.error(f"Ошибка распознавания: {e}")
            return self._try_fallback(image, with_details)
    
    def _try_fallback(self, image: np.ndarray, with_details: bool) -> OCRResult:
        for name, engine in self.engines.items():
            if engine.is_available and engine != self.current_engine:
                logger.info(f"Попытка использовать fallback: {name}")
                try:
                    if with_details:
                        return engine.recognize_with_details(image)
                    else:
                        return engine.recognize(image)
                except Exception:
                    continue
        raise RuntimeError("Все OCR движки не смогли обработать изображение")


# ============================================================
# ПРЕДОБРАБОТКА ИЗОБРАЖЕНИЙ
# ============================================================

class ImagePreprocessor:
    """Предварительная обработка изображений для OCR"""
    
    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    @staticmethod
    def denoise(image: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    
    @staticmethod
    def binarize(image: np.ndarray) -> np.ndarray:
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    
    @staticmethod
    def deskew(image: np.ndarray) -> np.ndarray:
        gray = ImagePreprocessor.to_grayscale(image)
        coords = np.column_stack(np.where(gray > 0))
        if len(coords) < 100:
            return image
        
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        if abs(angle) < 0.5:
            return image
        
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), 
                              flags=cv2.INTER_CUBIC, 
                              borderMode=cv2.BORDER_REPLICATE)
    
    @classmethod
    def full_pipeline(cls, image: np.ndarray, 
                      deskew: bool = True,
                      denoise: bool = True,
                      enhance: bool = True,
                      binarize: bool = False) -> np.ndarray:
        result = cls.to_grayscale(image)
        
        if deskew:
            result = cls.deskew(result)
        if denoise:
            result = cls.denoise(result)
        if enhance:
            result = cls.enhance_contrast(result)
        if binarize:
            result = cls.binarize(result)
        
        return result


# ============================================================
# АНАЛИЗАТОР ШРИФТОВ
# ============================================================

class FontAnalyzer:
    """Анализ шрифтов в документе"""
    
    @staticmethod
    def analyze_blocks(blocks: List[TextBlock]) -> Dict:
        if not blocks:
            return {'estimated_font': 'Unknown', 'average_size': 0}
        
        sizes = [b.font_size for b in blocks if b.font_size > 0]
        bold_count = sum(1 for b in blocks if b.is_bold)
        italic_count = sum(1 for b in blocks if b.is_italic)
        
        avg_size = float(np.mean(sizes)) if sizes else 0
        
        if avg_size < 12:
            font_name = "Мелкий шрифт"
        elif avg_size < 16:
            font_name = "Стандартный (12-14pt)"
        elif avg_size < 20:
            font_name = "Крупный (14-18pt)"
        else:
            font_name = "Заголовочный"
        
        return {
            'estimated_font': font_name,
            'font_type': 'Sans-Serif',
            'average_size': avg_size,
            'size_range': [float(min(sizes)), float(max(sizes))] if sizes else [0, 0],
            'bold_percentage': (bold_count / len(blocks) * 100) if blocks else 0,
            'italic_percentage': (italic_count / len(blocks) * 100) if blocks else 0,
            'total_blocks': len(blocks)
        }


# ============================================================
# ИЗВЛЕЧЕНИЕ ТАБЛИЦ
# ============================================================

class TableExtractor:
    """Извлечение таблиц из документов"""
    
    @staticmethod
    def extract_from_pdf(pdf_path: str) -> List[Dict]:
        tables = []
        try:
            import camelot
            table_list = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
            
            for i, table in enumerate(table_list):
                tables.append({
                    'table_id': i + 1,
                    'data': table.df.values.tolist(),
                    'accuracy': table.accuracy,
                    'page': table.page
                })
        except ImportError:
            logger.warning("Camelot не установлен. Установите: pip install camelot-py[cv]")
        except Exception as e:
            logger.error(f"Ошибка извлечения таблиц: {e}")
        
        return tables
    
    @staticmethod
    def extract_from_image(image: np.ndarray) -> List[Dict]:
        tables = []
        
        try:
            gray = ImagePreprocessor.to_grayscale(image)
            _, binary = cv2.threshold(gray, 0, 255, 
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
            
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
            vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
            
            grid = cv2.add(horizontal, vertical)
            contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            table_id = 1
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 100 and h > 50:
                    tables.append({
                        'table_id': table_id,
                        'bbox': [x, y, w, h],
                        'note': 'Таблица обнаружена, используйте PDF для точного извлечения'
                    })
                    table_id += 1
        except Exception as e:
            logger.error(f"Ошибка детекции таблиц: {e}")
        
        return tables
