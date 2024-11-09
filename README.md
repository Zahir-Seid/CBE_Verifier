# CBE_Verifier

## Overview

`CBE_verifier` is a Python library for validating transaction data against a digital transaction record. This library is designed to simplify the verification of transaction details by extracting information from uploaded screenshots and comparing it with provided data. It is built with ease of use and accessibility in mind, making it ideal for scenarios where quick and reliable verification of transaction details is needed.

### Key Features
- **Transaction Data Extraction**: Extracts key transaction details such as transaction ID, payer name, receiver name, date, and amount from a screenshot.
- **Automated Verification**: Compares the extracted transaction data against provided reference data to detect mismatches.
- **Clear Result Feedback**: Returns a clean and concise verification result, indicating either "verified" or specifying any mismatched fields.

## Installation

Install `CBE_verifier` via pip:

```bash
pip install CBE_verifier
```

## Usage

To use `CBE_verifier`, follow these steps to initialize the verification process and receive results:

### 1. Import the Library

```python
from CBE_verifier import TransactionVerifier
```

### 2. Initialize and Run Verification

1. **Prepare Data**: Prepare a dictionary of reference transaction details (`provided_data`) and a path to the transaction screenshot image.
2. **Verification**: Instantiate `TransactionVerifier` and call the `verify_transaction` method with the provided data and image path.

### Example Usage

```python
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

# Instantiate and verify
verifier = TransactionVerifier()
result = verifier.verify_transaction(provided_data, image_path)

# Check result
if result.is_verified:
    print("Verification Success")
else:
    print("Verification Failed. Mismatches found:")
    print(result.details)
```

### Result Structure
The verification result will be returned as either:
- `Verification Success`: All provided data matches extracted data.
- `Verification Failure`: A dictionary specifying any mismatched fields, including expected and extracted values.

## Functions and Classes

### 1. `TransactionVerifier`
The main interface for conducting verification. Contains:
- **verify_transaction(provided_data, image_path)**: Conducts verification using provided reference data and a screenshot image.

### 2. `VerifyFailure` and `VerifySuccess` Classes
These classes encapsulate verification results:
- **VerifyFailure**: Contains details of mismatched fields.
- **VerifySuccess**: Indicates a successful verification when all fields match.

### 3. Utility Functions (`utils.py`)
Provides validation utilities:
- **validate_txn_id**: Checks if a transaction ID is valid.
- **validate_acc_no**: Validates account number format.

## Error Handling

- **Invalid Data**: Raises a `ValueError` if required fields are missing or incorrectly formatted.
- **Image File Issues**: Provides errors if the image is invalid or unreadable.

## Example Test Code

To test the library, create a test file with `provided_data` and an `image_path` as in the example usage. Use various scenarios to validate both successful and failed verification cases.

## License

This library is open-source under the MIT license.

## Contributions

We welcome contributions! Please submit a pull request with any improvements, features, or bug fixes.
