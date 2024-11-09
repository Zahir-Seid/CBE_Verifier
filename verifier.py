from typing import Union, Optional, Dict

class VerifyFailure:
    def __init__(self, error_type: str, mismatches: Optional[dict] = None):
        self.type = error_type
        self.mismatches = mismatches

class VerifySuccess:
    def __init__(self, **kwargs):
        self.verified_details = kwargs

class TransactionVerifier:
    def verify_transaction(self, provided_data: dict, extracted_data: dict) -> Union[VerifyFailure, VerifySuccess]:
        mismatches = {}

        # Check transaction ID from both QR and text detection
        provided_txn_id = provided_data.get("transaction_id")
        extracted_txn_id = extracted_data.get("transaction_id")
        
        if provided_txn_id:
            # Check if extracted QR or text transaction ID matches the provided ID
            if extracted_txn_id:
                if provided_txn_id != extracted_txn_id:
                    mismatches["transaction_id"] = {
                        "provided": provided_txn_id,
                        "extracted": extracted_txn_id
                    }
        
        # Compare other transaction details (payer, receiver, date, amount)
        for key in ["payer", "receiver", "date", "amount"]:
            provided_value = provided_data.get(key)
            extracted_value = extracted_data.get(key)
            if provided_value != extracted_value:
                mismatches[key] = {
                    "provided": provided_value,
                    "extracted": extracted_value
                }

        if mismatches:
            return VerifyFailure("VERIFICATION_FAILED", mismatches)
        
        # Return success if all checks pass
        return VerifySuccess(**extracted_data)
