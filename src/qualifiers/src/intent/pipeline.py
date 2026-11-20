#!/usr/bin/env python3
"""
EDGE AI CHALLENGE 2026 - UNIFIED INTENT & VQA PIPELINE
======================================================
100% BTC Invariant Compliance:
  "Được sử dụng từ điển và logic để chuẩn hóa câu hỏi, trích xuất lớp vật
   thể và định dạng đáp án; không được thay thế hoàn toàn TextCNN bằng
   luật từ khóa"

Pipeline Workflow:
  1. Text Normalization: Diacritics stripping, punctuation cleaning, polite filler removal.
  2. Neural Intent Classification: Multi-Scale TextCNN model (TFLM Full INT8, ~38 KB, 25k params).
  3. Object Class Extraction: High-precision dictionary lookup for 7 competition classes.
  4. Answer Synthesis: Combines (Intent, Target Class, Detection BBoxes) into standard answer.
"""

import os
import json
import re
import unicodedata
import numpy as np

from .entity_extractor import extract_target_class, CLASS_ID_TO_NAME
from .answer_formatter import format_answer

# 5 Intent Classes
INTENTS = ["COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE"]
LABEL_MAP = {name: idx for idx, name in enumerate(INTENTS)}
INV_LABEL_MAP = {idx: name for idx, name in enumerate(INTENTS)}

# BTC-Approved Polite Social Filler Particles
STOPWORDS = {
    'ban', 'oi', 'cho', 'toi', 'hoi', 'xin', 'minh',
    'vay', 'nhi', 'the', 'a', 'gium', 'giup'
}


def remove_accents(text):
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'[\u0300-\u036f]', '', text)
    text = text.replace('đ', 'd').replace('Đ', 'd')
    return unicodedata.normalize('NFC', text)


def clean_and_tokenize(text):
    text = remove_accents(text)
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in text.split() if w and w not in STOPWORDS]


class IntentVQAPipeline:
    """
    Unified End-to-End Edge AI Intent & Question Answering Pipeline.
    Supports both Keras model (for development/training) and
    TensorFlow Lite INT8 model (for microcontroller parity verification).
    """

    def __init__(self, model_path=None, vocab_path=None, tflite_path=None, max_len=18):
        self.max_len = max_len
        self.vocab = {}
        self.keras_model = None
        self.interpreter = None

        current_file = os.path.abspath(__file__)
        intent_dir = os.path.dirname(current_file)
        src_dir = os.path.dirname(intent_dir)
        qual_dir = os.path.dirname(src_dir)

        if vocab_path is None:
            vocab_path = os.path.join(qual_dir, "weights/text_cnn_vocab.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)

        # 1. Try loading TFLite model first if provided
        if tflite_path and os.path.exists(tflite_path):
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

        # 2. Or load Keras model
        elif model_path and os.path.exists(model_path):
            import tensorflow as tf
            self.keras_model = tf.keras.models.load_model(model_path)
        else:
            default_keras = os.path.join(qual_dir, "weights/text_cnn.keras")
            if os.path.exists(default_keras):
                import tensorflow as tf
                self.keras_model = tf.keras.models.load_model(default_keras)

    def encode(self, text):
        """Encodes question string into fixed-length integer token IDs."""
        tokens = [self.vocab.get(w, 1) for w in clean_and_tokenize(text)][:self.max_len]
        if len(tokens) < self.max_len:
            tokens = tokens + [0] * (self.max_len - len(tokens))
        return np.array([tokens], dtype=np.int32)

    def predict_intent(self, question_text):
        """
        Predicts intent strictly via neural network inference.
        Returns: str ("COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE")
        """
        x = self.encode(question_text)

        if self.interpreter is not None:
            # TFLite inference
            in_tensor = self.input_details[0]
            in_dtype = in_tensor['dtype']
            in_scale, in_zp = in_tensor.get('quantization', (0.0, 0))

            if in_dtype == np.int8:
                x_quant = (x / in_scale + in_zp).astype(np.int8) if in_scale > 0 else x.astype(np.int8)
                self.interpreter.set_tensor(in_tensor['index'], x_quant)
            else:
                self.interpreter.set_tensor(in_tensor['index'], x.astype(in_dtype))

            self.interpreter.invoke()
            out_tensor = self.output_details[0]
            out = self.interpreter.get_tensor(out_tensor['index'])[0]
            best_idx = int(np.argmax(out))
            return INV_LABEL_MAP[best_idx]

        elif self.keras_model is not None:
            preds = self.keras_model.predict(x, verbose=0)[0]
            best_idx = int(np.argmax(preds))
            return INV_LABEL_MAP[best_idx]

        else:
            raise RuntimeError("No TextCNN model loaded in pipeline.")

    def extract_entity(self, question_text):
        """Extracts target object class (0..6) or -1 via dictionary logic."""
        return extract_target_class(question_text)

    def answer(self, question_text, detections):
        """
        Full Question Answering pipeline:
        1. TextCNN -> intent
        2. Entity extractor -> target_class
        3. Formatter + detections -> standardized contest answer string
        """
        intent = self.predict_intent(question_text)
        target_class = self.extract_entity(question_text)
        ans = format_answer(intent, target_class, detections)
        return {
            "intent": intent,
            "target_class": target_class,
            "target_name": CLASS_ID_TO_NAME.get(target_class, "None"),
            "answer": ans
        }
