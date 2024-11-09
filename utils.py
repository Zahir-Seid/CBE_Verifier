import re

def validate_txn_id(txn_id: str) -> str:
    if not re.match(r"^FT\w{10}$", txn_id):
        raise ValueError("Invalid transaction ID")
    return txn_id

def validate_acc_no(acc_no: str) -> str:
    if not re.match(r"^1000\d{9}$", acc_no):
        raise ValueError("Invalid account number")
    return acc_no
