"""Secure HAL: boot hash chain, SE, TPM quote, authenticated TinyML, AES-GCM.

Protocol simulation, not a silicon TPM model. Keys never leave the SE object.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass, field

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class TamperError(RuntimeError):
    pass


@dataclass
class SecureElement:
    """Stores the model signing key; never exports it."""

    _sk: ed25519.Ed25519PrivateKey = field(default_factory=ed25519.Ed25519PrivateKey.generate)

    def public_bytes(self) -> bytes:
        return self._sk.public_key().public_bytes_raw()

    def sign(self, message: bytes) -> bytes:
        return self._sk.sign(message)

    def verify(self, message: bytes, signature: bytes, public: bytes | None = None) -> bool:
        pk = (
            ed25519.Ed25519PublicKey.from_public_bytes(public)
            if public is not None
            else self._sk.public_key()
        )
        try:
            pk.verify(signature, message)
            return True
        except Exception:
            return False


@dataclass
class SimulatedTPM:
    pcrs: dict[int, bytes] = field(default_factory=dict)
    _sk: ed25519.Ed25519PrivateKey = field(default_factory=ed25519.Ed25519PrivateKey.generate)

    def extend(self, pcr: int, digest: bytes) -> None:
        prev = self.pcrs.get(pcr, b"\x00" * 32)
        self.pcrs[pcr] = hashlib.sha256(prev + digest).digest()

    def quote(self, nonce: bytes, pcr: int = 0) -> bytes:
        body = self.pcrs.get(pcr, b"\x00" * 32) + nonce
        return self._sk.sign(hashlib.sha256(body).digest())

    def public_bytes(self) -> bytes:
        return self._sk.public_key().public_bytes_raw()


class SecureHAL:
    def __init__(self, firmware: bytes, model_id: bytes) -> None:
        self.se = SecureElement()
        self.tpm = SimulatedTPM()
        self.root_hash = hashlib.sha256(firmware).digest()
        self.firmware_sig = self.se.sign(self.root_hash)
        self.model_id = model_id
        self.tpm.extend(0, hashlib.sha256(model_id).digest())
        self._session = os.urandom(32)
        self.booted = False

    def secure_boot(self, firmware: bytes) -> None:
        h = hashlib.sha256(firmware).digest()
        if h != self.root_hash or not self.se.verify(h, self.firmware_sig):
            raise TamperError("secure boot failed: firmware hash mismatch")
        self.booted = True

    def authenticate_inference(self, window: bytes, output: bytes) -> bytes:
        if not self.booted:
            raise TamperError("inference before secure boot")
        msg = hashlib.sha256(self.model_id + window + output).digest()
        return hmac.new(self._session, msg, hashlib.sha256).digest()

    def verify_inference(self, window: bytes, output: bytes, tag: bytes) -> bool:
        expected = self.authenticate_inference(window, output)
        return hmac.compare_digest(expected, tag)

    def encrypt(self, plaintext: bytes) -> tuple[bytes, bytes]:
        aes = AESGCM(self._session)
        nonce = os.urandom(12)
        return nonce, aes.encrypt(nonce, plaintext, None)

    def decrypt(self, nonce: bytes, blob: bytes) -> bytes:
        aes = AESGCM(self._session)
        return aes.decrypt(nonce, blob, None)

    def attestation(self, nonce: bytes) -> bytes:
        return self.tpm.quote(nonce)
