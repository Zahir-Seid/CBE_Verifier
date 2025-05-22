from typing import Union, Optional, Dict
import logging
import httpx
from detector import parse_cbe_receipt, VerifyResult

logger = logging.getLogger(__name__)


class VerifyFailure:
    def __init__(self, error_type: str, mismatches: Optional[dict] = None):
        self.type = error_type
        self.mismatches = mismatches or {}

    def __repr__(self):
        return f"<VerifyFailure type={self.type}, mismatches={self.mismatches}>"


class VerifySuccess:
    def __init__(self, **kwargs):
        self.verified_details = kwargs

    def __repr__(self):
        return f"<VerifySuccess verified_details={self.verified_details}>"


class TransactionVerifier:

    @staticmethod
    async def verify_cbe(reference: str, account_suffix: str) -> VerifyResult:
        """
        Async version: Fetch the official CBE PDF receipt using the transaction ID + account suffix.
        """
        full_id = f"{reference}{account_suffix}"
        url = f"https://apps.cbe.com.et:100/?id={full_id}"
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/pdf'
        }

        logger.info(f"Fetching CBE receipt from: {url}")
        try:
            async with httpx.AsyncClient(verify=False, timeout=30) as client:
                response = await client.get(url, headers=headers)

            content_type = response.headers.get('Content-Type', '').lower()
            if response.status_code == 200 and 'application/pdf' in content_type:
                logger.info("Successfully fetched official CBE PDF receipt.")
                # parse_cbe_receipt is assumed async or sync depending on your code
                result = await parse_cbe_receipt(response.content) if callable(getattr(parse_cbe_receipt, "__await__", None)) else parse_cbe_receipt(response.content)
                return result
            else:
                logger.error(f"Invalid response. Status: {response.status_code}, Content-Type: {content_type}")
                raise ValueError("Could not fetch a valid PDF receipt from CBE.")
        except httpx.RequestError as e:
            logger.exception("Network error while fetching the receipt.")
            raise ValueError("Network error while requesting CBE receipt.") from e

    @staticmethod
    def verify_transaction(provided_data: dict, extracted_data: dict) -> Union[VerifyFailure, VerifySuccess]:
        """
        Compare only the transaction_id and amount from provided data with official receipt.

        Args:
            provided_data (dict): The original values (manual input or from image).
            extracted_data (dict): Parsed values from the official CBE receipt.

        Returns:
            Union[VerifyFailure, VerifySuccess]: Verification result.
        """
        mismatches = {}

        # Check transaction ID
        provided_tx_id = str(provided_data.get("transaction_id", "")).strip()
        official_tx_id = str(extracted_data.get("transaction_id", "")).strip()
        if provided_tx_id != official_tx_id:
            mismatches["transaction_id"] = {
                "provided": provided_tx_id,
                "official": official_tx_id
            }

        # Check amount, normalizing both sides
        try:
            provided_amt = float(str(provided_data.get("amount", "0")).replace(",", ""))
            official_amt = float(str(extracted_data.get("amount", "0")).replace(",", ""))
            if provided_amt != official_amt:
                mismatches["amount"] = {
                    "provided": provided_data.get("amount"),
                    "official": extracted_data.get("amount")
                }
        except Exception as e:
            mismatches["amount"] = {
                "provided": provided_data.get("amount"),
                "official": extracted_data.get("amount"),
                "error": f"Amount parsing error: {e}"
            }

        if mismatches:
            logger.warning("Verification failed. Mismatches found.")
            return VerifyFailure("VERIFICATION_FAILED", mismatches)

        logger.info("Verification passed. Data matches official receipt.")
        return VerifySuccess(**extracted_data)

    @classmethod
    async def verify_against_official(cls, provided_data: dict) -> Union[VerifyFailure, VerifySuccess]:
        """
        Async version: full verification
        """
        reference = provided_data.get("transaction_id")
        suffix = provided_data.get("suffix")

        if not reference or not suffix:
            logger.error("Missing transaction_id or suffix in provided data.")
            return VerifyFailure("MISSING_FIELDS", {"required": ["transaction_id", "suffix"]})

        try:
            result = await cls.verify_cbe(reference, suffix)

            if getattr(result, 'success', False):
                # assuming parsed data is in result.details
                extracted_data = getattr(result, 'details', {}) or {}
                return cls.verify_transaction(provided_data, extracted_data)
            else:
                logger.error("Receipt fetch or parse failed.")
                error_details = getattr(result, 'details', {}).get("error") if hasattr(result, 'details') else None
                return VerifyFailure("RECEIPT_PARSE_ERROR", {"error": error_details})
        except Exception as e:
            logger.exception("Failed to verify against official receipt.")
            return VerifyFailure("EXCEPTION", {"error": str(e)})
