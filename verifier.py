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
                result = await parse_cbe_receipt(response.content) if callable(getattr(parse_cbe_receipt, "__await__", None)) else parse_cbe_receipt(response.content)
                return result
            else:
                logger.error(f"Invalid response. Status: {response.status_code}, Content-Type: {content_type}")
                raise ValueError("Could not fetch a valid PDF receipt from CBE.")
        except httpx.RequestError as e:
            logger.exception("Network error while fetching the receipt.")
            raise ValueError("Network error while requesting CBE receipt.") from e

    @staticmethod
    def verify_transaction(provided_data: dict, extracted_data: dict, include_data: bool = False) -> tuple[bool, Optional[dict]]:
        """
        Verifies transaction ID and amount match. 
        By default, only returns success boolean. Optionally returns parsed data.

        Args:
            provided_data (dict): Developer-provided transaction ID and expected amount.
            extracted_data (dict): Parsed values from official receipt.
            include_data (bool): If True, return extracted data on success.

        Returns:
            tuple: 
                (True, ) if success,
                (True, extracted_data) if include_data=True,
                (False, {"reason": ..., "mismatches": {...}}) if failed
        """
        mismatches = {}

        # Match transaction_id
        provided_txn_id = provided_data.get("transaction_id")
        extracted_txn_id = extracted_data.get("transaction_id")
        if not provided_txn_id or not extracted_txn_id or str(provided_txn_id).strip() != str(extracted_txn_id).strip():
            mismatches["transaction_id"] = {
                "provided": provided_txn_id,
                "official": extracted_txn_id
            }

        # Match amount
        provided_amount_raw = provided_data.get("amount")
        extracted_amount = extracted_data.get("amount")

        try:
            provided_amount = float(str(provided_amount_raw).replace(",", "").strip())
        except (ValueError, TypeError):
            provided_amount = None

        if provided_amount is None or extracted_amount is None or round(provided_amount, 2) != round(float(extracted_amount), 2):
            mismatches["amount"] = {
                "provided": provided_amount_raw,
                "official": extracted_amount
            }

        # Final result
        if mismatches:
            return False, {
                "reason": "VERIFICATION_FAILED",
                "mismatches": mismatches
            }

        return (True, extracted_data) if include_data else (True, )



    @classmethod
    async def verify_against_official(cls, provided_data: dict, include_data: bool = False) -> Union[bool, tuple[bool, dict], VerifyFailure]:
        reference = provided_data.get("transaction_id")
        suffix = provided_data.get("suffix")

        if not reference or not suffix:
            logger.error("Missing transaction_id or suffix in provided data.")
            return VerifyFailure("MISSING_FIELDS", {"required": ["transaction_id", "suffix"]})

        try:
            result = await cls.verify_cbe(reference, suffix)

            if getattr(result, 'success', False):
                extracted_data = getattr(result, 'details', {}) or {}

                verified = cls.verify_transaction(provided_data, extracted_data, include_data=include_data)

                if verified is True:
                    return True
                elif isinstance(verified, tuple) and verified[0] is True:
                    return verified
                else:
                    logger.warning("Verification failed. Data mismatch.")
                    return VerifyFailure("VERIFICATION_FAILED", {"provided": provided_data, "extracted": extracted_data})
            else:
                logger.error("Receipt fetch or parse failed.")
                error_details = getattr(result, 'details', {}).get("error") if hasattr(result, 'details') else None
                return VerifyFailure("RECEIPT_PARSE_ERROR", {"error": error_details})
        except Exception as e:
            logger.exception("Failed to verify against official receipt.")
            return VerifyFailure("EXCEPTION", {"error": str(e)})
