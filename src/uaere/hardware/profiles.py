"""Software-defined hardware emulation. Datasheet-calibrated cost tables.

Sources (documented, not fetched):
- STM32L476: 100 µA/MHz typical, CMSIS-NN INT8 ~ few nJ/MAC order.
- ESP32-S3: Xtensa LX7 240 MHz, published TinyML mJ/inference.
- GD32VF103: RV32IMC 108 MHz, no SE (negative control).
- iCE40UP5K: SPRAM + DSP toggle energy (Lattice order-of-magnitude).
- Ethos-U55: published MAC/s and mJ/MAC class numbers.
"""

from __future__ import annotations

from uaere.types import DeviceProfile

PROFILES: dict[str, DeviceProfile] = {
    "stm32l476": DeviceProfile(
        device_id="stm32l476",
        isa="ARM Cortex-M4",
        cpu_mhz=80.0,
        ram_kb=128.0,
        flash_kb=1024.0,
        idle_mw=0.3,
        active_mw=8.0,
        nj_per_mac_int8=2.5,
        tx_nj_per_bit=800.0,
        has_secure_element=True,
        has_secure_boot=True,
        has_tpm=False,
        has_npu=False,
    ),
    "esp32s3": DeviceProfile(
        device_id="esp32s3",
        isa="Xtensa LX7",
        cpu_mhz=240.0,
        ram_kb=512.0,
        flash_kb=8192.0,
        idle_mw=0.8,
        active_mw=40.0,
        nj_per_mac_int8=1.8,
        tx_nj_per_bit=400.0,
        has_secure_element=True,
        has_secure_boot=True,
        has_tpm=False,
        has_npu=False,
    ),
    "gd32vf103": DeviceProfile(
        device_id="gd32vf103",
        isa="RV32IMC",
        cpu_mhz=108.0,
        ram_kb=32.0,
        flash_kb=128.0,
        idle_mw=0.4,
        active_mw=12.0,
        nj_per_mac_int8=3.2,
        tx_nj_per_bit=900.0,
        has_secure_element=False,
        has_secure_boot=False,
        has_tpm=False,
        has_npu=False,
        supports_l2=True,
        supports_l3=False,
    ),
    "ice40up5k": DeviceProfile(
        device_id="ice40up5k",
        isa="FPGA (iCE40UP5K)",
        cpu_mhz=48.0,
        ram_kb=128.0,
        flash_kb=0.0,
        idle_mw=1.2,
        active_mw=25.0,
        nj_per_mac_int8=1.1,
        tx_nj_per_bit=800.0,
        has_secure_element=False,
        has_secure_boot=True,  # bitstream auth
        has_tpm=False,
        has_npu=False,
    ),
    "ethos_u55": DeviceProfile(
        device_id="ethos_u55",
        isa="Cortex-M55 + Ethos-U55",
        cpu_mhz=400.0,
        ram_kb=2048.0,
        flash_kb=2048.0,
        idle_mw=0.6,
        active_mw=30.0,
        nj_per_mac_int8=0.15,
        tx_nj_per_bit=800.0,
        has_secure_element=True,
        has_secure_boot=True,
        has_tpm=True,
        has_npu=True,
        npu_mac_per_s=64e9,
    ),
}


def load_profile(device_id: str) -> DeviceProfile:
    if device_id not in PROFILES:
        raise KeyError(f"unknown device profile {device_id}; known: {sorted(PROFILES)}")
    return PROFILES[device_id]
