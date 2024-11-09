import cv2
from typing import Optional
from PIL import Image
import re
import easyocr
import numpy as np
from pyzbar.pyzbar import decode

class DetectTransactionIdResult:
    def __init__(self, qr_transaction_id: Optional[str], text_transaction_id: Optional[str], payer: Optional[str], receiver: Optional[str], date: Optional[str], amount: Optional[str]):
        self.qr_transaction_id = qr_transaction_id          
        self.text_transaction_id = text_transaction_id      
        self.payer = payer
        self.receiver = receiver
        self.date = date
        self.amount = amount


class TransactionIDDetector:
    def __init__(self):
        # Initialize EasyOCR reader once
        self.reader = easyocr.Reader(['en'], gpu=False)

    def detect_transaction_id(self, image_path: str) -> DetectTransactionIdResult:
        # Detect transaction ID from the QR code using the image path
        qr_transaction_id = self.detect_from_image_qr(image_path)  # Use image_path for QR code detection

        # Load the image file as bytes for text detection
        with open(image_path, "rb") as img_file:
            image_data = img_file.read()

        # Perform text detection to extract other details and a text-based transaction ID
        detection_result = self.detect_from_image_text(image_data)

        # Return both QR and text transaction IDs along with other details
        return DetectTransactionIdResult(
            qr_transaction_id=qr_transaction_id,
            text_transaction_id=detection_result.get("transaction_id"),
            payer=detection_result.get("payer"),
            receiver=detection_result.get("receiver"),
            date=detection_result.get("date"),
            amount=detection_result.get("amount")
        )


    def detect_from_image_qr(self, image_path: str) -> Optional[str]:
        # Open and crop image centrally using PIL
        image = Image.open(image_path)
        width, height = image.size
        target_width, target_height = 477, 381
        left, top = (width - target_width) // 2, (height - target_height) // 2
        cropped_image = image.crop((left, top, left + target_width, top + target_height))

        # Decode QR code from cropped image
        for obj in decode(cropped_image):
            data = obj.data.decode("utf-8")
            match = re.search(r"FT\w{10}", data)
            if match:
                return match.group(0)

        return None

    def detect_from_image_text(self, buffer: bytes) -> dict:
        # Decode image from bytes to grayscale for OCR
        image = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_GRAYSCALE)
        full_text = " ".join(self.reader.readtext(image, detail=0))

        # Regex pattern matching for transaction details
        txn_id_match = re.search(r"FT\w{10}", full_text)
        payer_match = re.search(r"debited from\s+([A-Za-z\s]+?)\s+for", full_text)
        receiver_match = re.search(r"for\s+([A-Za-z\s]+?)-ETB-", full_text)
        date_match = re.search(r"on\s+(\d{2}-[A-Za-z]{3}-\d{4})", full_text)
        amount_match = re.search(r"ETB\s+([0-9,]+\.00)", full_text)

        return {
            "transaction_id": txn_id_match.group(0) if txn_id_match else None,
            "payer": payer_match.group(1) if payer_match else None,
            "receiver": receiver_match.group(1) if receiver_match else None,
            "date": date_match.group(1) if date_match else None,
            "amount": amount_match.group(1) if amount_match else None
        }
