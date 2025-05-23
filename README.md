# CBE_Verifier

## Overview

`CBE_Verifier` is a Python library designed to validate transaction data by extracting information from transaction screenshots or official receipts (PDF) and comparing it with provided reference data. It offers streamlined verification with clear result formats, ideal for applications requiring reliable, quick validation of transaction details from the Commercial Bank of Ethiopia (CBE).

### Key Features
- **Transaction Data Extraction**: Extracts essential transaction details such as transaction ID, payer, receiver, date, and amount from images or PDF receipts using OCR and PDF parsing.
- **Official Receipt Verification**: Fetches and parses official CBE PDF receipts online based on transaction ID and account suffix for authoritative verification.
- **Automated Verification**: Compares extracted data against user-provided reference data, identifying any mismatches.
- **Concise Results**: Returns simple verification results indicating either success or specifying mismatched fields with detailed reasons.

## Installation

Install `CBE_Verifier` via pip:

```bash
pip install CBE_Verifier

```

## Usage

To use `CBE_verifier`, follow these steps:

### 1. Import the Library

```python
from cbe_verifier.detector import TransactionIDDetector, parse_cbe_receipt, VerifyResult
from cbe_verifier.verifier import TransactionVerifier, VerifySuccess, VerifyFailure

```

### 2. Initialize and Run Verification

1. **Prepare Data**: Define a dictionary of reference transaction details (`provided_data`) and specify the path to the transaction screenshot (`image_path`).
2. **Verify**: Use `TransactionIDDetector` to extract data from the image, then pass the extracted and provided data to `TransactionVerifier`.

### Example Usage

```python
from cbe_verifier.detector import TransactionIDDetector
from cbe_verifier.verifier import TransactionVerifier

# Initialize detector and verifier
detector = TransactionIDDetector()
verifier = TransactionVerifier()

# Reference data to verify against
provided_data = {
    "transaction_id": "FTxxxxxxxxxx",
    "payer": "xxx xxx xxx",
    "receiver": "xxx xxx xxx",
    "date": "05-Nov-2024",
    "amount": "xxx.00"
}

# Path to the transaction screenshot
image_path = "image.png"

# Step 1: Detect transaction details from the image
detection_result = detector.detect_transaction_id(image_path)

# Step 2: Prepare extracted data
extracted_data = {
    "transaction_id": detection_result.text_transaction_id,
    "payer": detection_result.payer,
    "receiver": detection_result.receiver,
    "date": detection_result.date,
    "amount": detection_result.amount
}

# Step 3: Verify extracted data against provided data
verification_result = verifier.verify_transaction(provided_data, extracted_data)

# Step 4: Check verification outcome
if verification_result[0] is True:
    print("Verification Success: Details match.")
else:
    print("Verification Failed. Mismatches found:")
    for key, mismatch in verification_result[1]["mismatches"].items():
        print(f"{key}: Provided - {mismatch['provided']}, Extracted - {mismatch['official']}")
```

### Result Structure

- **Local Verification**:
  - Returns `(True,)` if all provided data matches extracted data.
  - Returns `(True, extracted_data)` if `include_data=True`.
  - Returns `(False, {"reason": "VERIFICATION_FAILED", "mismatches": {...}})` if any mismatch occurs.

- **Official Verification**:
  - Returns `True` or `(True, extracted_data)` on success.
  - Returns `VerifyFailure` instance on failure, with `.type` and `.mismatches` explaining the error.

## Classes and Functions

### `TransactionVerifier`

- `verify_cbe(reference: str, account_suffix: str)`: Async method that fetches and parses the official CBE PDF receipt.
- `verify_transaction(provided_data: dict, extracted_data: dict, include_data: bool = False)`: Compares provided and extracted data for local verification.
- `verify_against_official(provided_data: dict, include_data: bool = False)`: Async method to verify provided data against the official online receipt.

### `VerifyFailure`

Represents a verification failure with error type and mismatch details.

### `VerifySuccess`

Represents a successful verification with verified details.

### `TransactionIDDetector`

Extracts transaction details from images or PDFs using OCR (`easyocr`) and PDF parsing (`pdfplumber`).

### `parse_cbe_receipt`

Parse the official CBE PDF receipt from raw bytes, returning a `VerifyResult`.


### Utility Functions (Optional, in `utils.py`)
Provides validation functions:
- **validate_txn_id**: Validates the format of a transaction ID.
- **validate_acc_no**: Validates account number format.

## Error Handling

- **Network or Parsing Issues**: Raises detailed exceptions or returns failure objects explaining the problem.
- **Data Extraction Issues**: Logs warnings and returns partial or failure results when unable to extract required fields.
- **Invalid or Missing Fields**: Returns `VerifyFailure` with `MISSING_FIELDS` if required fields are missing.

## Example Test Code

To test, create a script with `provided_data` and an `image_path` as shown in the usage example. This allows you to test both successful and failed verification cases.

## License

This library is open-source under the MIT license.

## Contributions

Contributions are welcome! Please submit a pull request with any improvements, features, or bug fixes.