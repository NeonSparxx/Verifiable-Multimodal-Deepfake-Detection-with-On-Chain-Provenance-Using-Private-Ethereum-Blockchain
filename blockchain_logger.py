from web3 import Web3
import os

class BlockchainLogger:
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
        if not self.w3.is_connected():
            raise ConnectionError("Ganache is not running!")

        self.contract_address = "0x80bC5B9b11c0853e2bCF7dCBb50fd3b5A0F44a77"

        self.account = "0x79148E4F3F10ea6a890Cc4D48104bFB57552AD30"
        self.private_key = "0x65c26d2c8431aa139db9439a6d9ea57141f1dd39721bdb08c0b38283201fc482"
        self.contract = self.w3.eth.contract(
            address=self.contract_address,
            abi=[
                {
                    "inputs": [
                        {"internalType": "string", "name": "fileName", "type": "string"},
                        {"internalType": "string", "name": "prediction", "type": "string"},
                        {"internalType": "string", "name": "confidence", "type": "string"}
                    ],
                    "name": "logPrediction",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                }, 
                {
                    "anonymous": False,
                    "inputs": [
                        {"indexed": True, "internalType": "string", "name": "fileName", "type": "string"},
                        {"indexed": True, "internalType": "string", "name": "prediction", "type": "string"},
                        {"indexed": False, "internalType": "string", "name": "confidence", "type": "string"},
                        {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"},
                        {"indexed": True, "internalType": "address", "name": "detector", "type": "address"}
                    ],
                    "name": "PredictionLogged",
                    "type": "event"
                }
            ]
        )

        print(f"Blockchain connected → {self.contract_address}")

    def log_prediction(self, file_path: str, prediction: str, confidence: float) -> str:
        try:
            tx = self.contract.functions.logPrediction(
                os.path.basename(file_path),
                prediction,
                f"{confidence:.6f}"
            ).build_transaction({
                "chainId": 1337,
                "gas": 200000,
                "gasPrice": self.w3.to_wei(2, "gwei"),
                "nonce": self.w3.eth.get_transaction_count(self.account),
            })

            signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)

            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)

            print(f"LOGGED ON BLOCKCHAIN → {tx_hash.hex()}")
            return tx_hash.hex()

        except Exception as e:
            print(f"Blockchain failed: {e}")
            return "failed"