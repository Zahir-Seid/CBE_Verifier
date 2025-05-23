import io
import re
import cv2
import numpy as np
import easyocr
import logging
import pdfplumber
from typing import Optional, Dict, Union
from datetime import datetime
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class DetectTransactionIdResult:
    text_transaction_id: Optional[str]
    payer: Optional[str]
    receiver: Optional[str]
    date: Optional[str]
    amount: Optional[str]
    suffix: Optional[str]


@dataclass
class VerifyResult:
    success: bool
    details: Union[Dict[str, Union[str, float, datetime]], Dict[str, str]]

    def __repr__(self):
        return f"<VerifyResult success={self.success} details={self.details}>"


def title_case(name: str) -> str:
    return ' '.join(word.capitalize() for word in name.strip().split())


def extract_common_fields(text: str) -> Dict[str, Optional[str]]:
    text = re.sub(r"\s+", " ", text).strip()

    return {
        "transaction_id": re.search(r"FT\w{10}", text).group(0) if re.search(r"FT\w{10}", text) else None,
        "payer": re.search(r"debited from\s+([A-Z\s]+?)\s+for", text, re.IGNORECASE).group(1).strip()
        if re.search(r"debited from\s+([A-Z\s]+?)\s+for", text, re.IGNORECASE) else None,
        "receiver": re.search(r"for\s+([A-Z\s]+?)-ETB-", text, re.IGNORECASE).group(1).strip()
        if re.search(r"for\s+([A-Z\s]+?)-ETB-", text, re.IGNORECASE) else None,
        "date": re.search(r"on\s+(\d{2}-[A-Za-z]{3}-\d{4})", text).group(1)
        if re.search(r"on\s+(\d{2}-[A-Za-z]{3}-\d{4})", text) else None,
        "amount": re.search(r"ETB\s+([0-9,]+\.\d{2})", text).group(1)
        if re.search(r"ETB\s+([0-9,]+\.\d{2})", text) else None
    }


class TransactionIDDetector:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)

    def detect_transaction_id(self, image_path: str, manual_suffix: Optional[str] = None) -> DetectTransactionIdResult:
        with open(image_path, "rb") as img_file:
            buffer = img_file.read()
        text_data = self.detect_from_image_text(buffer)

        suffix = manual_suffix if manual_suffix is not None else text_data.get("suffix")

        return DetectTransactionIdResult(
            text_transaction_id=text_data.get("transaction_id"),
            payer=text_data.get("payer"),
            receiver=text_data.get("receiver"),
            date=text_data.get("date"),
            amount=text_data.get("amount"),
            suffix=suffix
        )

    def detect_from_image_text(self, buffer: bytes) -> Dict[str, Optional[str]]:
        image = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_GRAYSCALE)
        raw_text = " ".join(self.reader.readtext(image, detail=0))
        return extract_common_fields(raw_text)

    def extract_data_from_pdf(self, pdf_path: str) -> Dict[str, Optional[str]]:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages_text = [page.extract_text() for page in pdf.pages if page.extract_text()]
            raw_text = " ".join(pages_text)
            raw_text = re.sub(r"\s+", " ", raw_text).strip()

            payer_match = re.search(r"Payer\s*:?\s*(.*?)\s+Account", raw_text, re.IGNORECASE)
            receiver_match = re.search(r"Receiver\s*:?\s*(.*?)\s+Account", raw_text, re.IGNORECASE)
            accounts = list(re.finditer(r"Account\s*:?\s*([A-Z0-9]?\*{4}\d{4})", raw_text, re.IGNORECASE))
            reason_match = re.search(r"Reason\s*/\s*Type of service\s*:?\s*(.*?)\s+Transferred Amount", raw_text, re.IGNORECASE)
            amount_match = re.search(r"Transferred Amount\s*:?\s*([\d,]+\.\d{2})\s*ETB", raw_text, re.IGNORECASE)
            reference_match = re.search(r"Reference No\.?\s*\(VAT Invoice No\)\s*:?\s*([A-Z0-9]+)", raw_text, re.IGNORECASE)
            date_match = re.search(
                r"Payment Date & Time\s*:?\s*(\d{1,2}/\d{1,2}/\d{4}, \d{1,2}:\d{2}:\d{2} [AP]M)",
                raw_text, re.IGNORECASE
            )

            payer = title_case(payer_match.group(1)) if payer_match else None
            receiver = title_case(receiver_match.group(1)) if receiver_match else None
            payer_account = accounts[0].group(1) if len(accounts) > 0 else None
            receiver_account = accounts[1].group(1) if len(accounts) > 1 else None
            reason = reason_match.group(1).strip() if reason_match else None
            amount = float(amount_match.group(1).replace(",", "")) if amount_match else None
            transaction_id = reference_match.group(1).strip() if reference_match else None

            date = None
            if date_match:
                date_raw = date_match.group(1).strip()
                for fmt in ("%m/%d/%Y, %I:%M:%S %p", "%d-%b-%Y"):
                    try:
                        date = datetime.strptime(date_raw, fmt)
                        break
                    except ValueError:
                        continue

            return {
                "payer": payer,
                "payerAccount": payer_account,
                "receiver": receiver,
                "receiverAccount": receiver_account,
                "amount": amount,
                "date": date,
                "transaction_id": transaction_id,
                "reason": reason
            }

        except Exception as e:
            logger.error(f"Failed to extract data from CBE PDF: {e}")
            return {}



def parse_cbe_receipt(pdf_bytes: bytes) -> VerifyResult:
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages_text = [page.extract_text() for page in pdf.pages if page.extract_text()]
        raw_text = " ".join(pages_text)
        raw_text = re.sub(r"\s+", " ", raw_text).strip()
        payer_match = re.search(r"Payer\s*:?\s*(.*?)\s+Account", raw_text, re.IGNORECASE)
        receiver_match = re.search(r"Receiver\s*:?\s*(.*?)\s+Account", raw_text, re.IGNORECASE)
        accounts = list(re.finditer(r"Account\s*:?\s*([A-Z0-9]?\*{4}\d{4})", raw_text, re.IGNORECASE))
        reason_match = re.search(r"Reason\s*/\s*Type of service\s*:?\s*(.*?)\s+Transferred Amount", raw_text, re.IGNORECASE)
        amount_match = re.search(r"Transferred Amount\s*:?\s*([\d,]+\.\d{2})\s*ETB", raw_text, re.IGNORECASE)
        reference_match = re.search(r"Reference No\.?\s*\(VAT Invoice No\)\s*:?\s*([A-Z0-9]+)", raw_text, re.IGNORECASE)

        date_match = re.search(
            r"Payment Date & Time\s*:?\s*(\d{1,2}/\d{1,2}/\d{4}, \d{1,2}:\d{2}:\d{2} [AP]M)", 
            raw_text, 
            re.IGNORECASE
        )

        payer = title_case(payer_match.group(1)) if payer_match else None
        receiver = title_case(receiver_match.group(1)) if receiver_match else None
        payer_account = accounts[0].group(1) if len(accounts) > 0 else None
        receiver_account = accounts[1].group(1) if len(accounts) > 1 else None
        reason = reason_match.group(1).strip() if reason_match else None
        amount = float(amount_match.group(1).replace(",", "")) if amount_match else None
        transaction_id = reference_match.group(1).strip() if reference_match else None

        # Parse date with fallback formats
        date = None
        if date_match:
            date_raw = date_match.group(1).strip()
            for fmt in ("%m/%d/%Y, %I:%M:%S %p", "%d-%b-%Y"):
                try:
                    date = datetime.strptime(date_raw, fmt)
                    break
                except ValueError:
                    continue
            if date is None:
                logger.warning(f"Date found but could not parse: '{date_raw}'")
        else:
            logger.warning("No date found in PDF text")

        if all([payer, payer_account, receiver, receiver_account, amount, date, transaction_id]):
            return VerifyResult(success=True, details={
                "payer": payer,
                "payerAccount": payer_account,
                "receiver": receiver,
                "receiverAccount": receiver_account,
                "amount": amount,
                "date": date,
                "transaction_id": transaction_id,
                "reason": reason
            })
        else:
            missing = [k for k, v in {
                "payer": payer,
                "payerAccount": payer_account,
                "receiver": receiver,
                "receiverAccount": receiver_account,
                "amount": amount,
                "date": date,
                "transaction_id": transaction_id
            }.items() if not v]
            logger.warning(f"Could not extract all required fields from PDF: missing {missing}")
            return VerifyResult(success=False, details={"error": "Could not extract all required fields", "missing": missing})

    except Exception as e:
        logger.error(f"PDF parsing failed: {e}")
        return VerifyResult(success=False, details={"error": "Error parsing PDF data"})
