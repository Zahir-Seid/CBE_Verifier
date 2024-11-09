# CBE_Verifier

## Overview

`CBE_verifier` is a Python library for validating transaction data by extracting information from transaction screenshots and comparing it with provided reference data. It offers streamlined verification with a clear result format, ideal for applications needing reliable, quick validation of transaction details.

### Key Features
- **Transaction Data Extraction**: Extracts essential transaction details such as transaction ID, payer, receiver, date, and amount from an image.
- **Automated Verification**: Compares extracted data against user-provided reference data to identify any mismatches.
- **Concise Results**: Returns a simple verification result indicating either "verified" or specifying any mismatched fields.

## Installation

Install `CBE_verifier` via pip:

```bash
pip install CBE_verifier
```

## Usage

To use `CBE_verifier`, follow these steps:

### 1. Import the Library

```python
from cbe_verifier.detector import TransactionIDDetector
from cbe_verifier.verifier import TransactionVerifier
```

### 2. Initialize and Run Verification

1. **Prepare Data**: Define a dictionary of reference transaction details (`provided_data`) and specify the path to the transaction screenshot (`image_path`).
2. **Verify**: Use `TransactionIDDetector` to extract data from the image, then pass the extracted and provided data to `TransactionVerifier`.

### Example Usage

```python
from cbe_verifier.detector import TransactionIDDetector
from cbe_verifier.verifier import TransactionVerifier, VerifySuccess

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
    "transaction_id": detection_result.qr_transaction_id or detection_result.text_transaction_id,
    "payer": detection_result.payer,
    "receiver": detection_result.receiver,
    "date": detection_result.date,
    "amount": detection_result.amount
}

# Step 3: Verify extracted data against provided data
verification_result = verifier.verify_transaction(provided_data, extracted_data)

# Step 4: Check verification outcome
if isinstance(verification_result, VerifySuccess):
    print("Verification Success: All details match!")
else:
    print("Verification Failed. Mismatches found:")
    for key, details in verification_result.mismatches.items():
        print(f"{key}: Provided - {details['provided']}, Extracted - {details['extracted']}")
```

### Result Structure
The verification result will be one of the following:
- `Verification Success`: All provided data matches extracted data.
- `Verification Failure`: A dictionary listing mismatched fields, showing both expected and extracted values.

## Classes and Functions

### 1. `TransactionVerifier`
The main interface for performing verification. Contains:
- **verify_transaction(provided_data, extracted_data)**: Compares provided data with extracted data, returning verification status.

### 2. `VerifyFailure` and `VerifySuccess`
- **VerifyFailure**: Contains mismatched details if any fields don’t match.
- **VerifySuccess**: Returned if all fields match, confirming verification.

### 3. Utility Functions (Optional, in `utils.py`)
Provides validation functions:
- **validate_txn_id**: Validates the format of a transaction ID.
- **validate_acc_no**: Validates account number format.

## Error Handling

- **Invalid Data**: Raises `ValueError` for missing or incorrectly formatted required fields.
- **Image File Issues**: Provides an error if the image file is invalid or unreadable.

## Example Test Code

To test, create a script with `provided_data` and an `image_path` as shown in the usage example. This allows you to test both successful and failed verification cases.

## License

This library is open-source under the MIT license.

## Contributions

Contributions are welcome! Please submit a pull request with any improvements, features, or bug fixes.